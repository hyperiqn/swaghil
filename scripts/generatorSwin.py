import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel

class SwinEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super(SwinEncoder, self).__init__()
        self.encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.encoder.config.output_hidden_states = True
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        else:
            self.input_conv = nn.Identity() 

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        outputs = self.encoder(x)
        hidden_states = outputs.hidden_states 
        final_output = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = final_output.shape
        height = width = int(seq_len ** 0.5)
        final_output = final_output.permute(0, 2, 1).view(batch_size, hidden_size, height, width)
        reshaped_hidden_states = []
        for state in hidden_states[-4:]:
            _, seq_len, hidden_size = state.shape
            height = width = int(seq_len ** 0.5)
            state = state.permute(0, 2, 1).view(batch_size, hidden_size, height, width)
            reshaped_hidden_states.append(state)

        return reshaped_hidden_states, final_output

class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 3):
        super(UNetDecoder, self).__init__()

        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1536, 512, kernel_size=3, padding=1)  
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(1280, 256, kernel_size=3, padding=1)  
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(640, 128, kernel_size=3, padding=1)   
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, padding=1)    

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, encoder_features: list, input_skip: torch.Tensor):
        x = self.up1(x)
        x = torch.cat([x, F.interpolate(encoder_features[-1], size=x.shape[2:], mode='bilinear')], dim=1)
        x = self.conv1(x) 
        
        x = self.up2(x)
        x = torch.cat([x, F.interpolate(encoder_features[-2], size=x.shape[2:], mode='bilinear')], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, F.interpolate(encoder_features[-3], size=x.shape[2:], mode='bilinear')], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, F.interpolate(encoder_features[-4], size=x.shape[2:], mode='bilinear')], dim=1)
        x = self.conv4(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super(Generator, self).__init__()
        self.encoder = SwinEncoder(in_channels)
        self.channel_alignment = nn.Conv2d(1024, 768, kernel_size=1) 
        self.decoder = UNetDecoder(in_channels=768, out_channels=out_channels)

    def forward(self, x: torch.Tensor):
        input_skip = x
        encoder_features, final_output = self.encoder(x)
        if final_output.shape[1] != 768:
            final_output = self.channel_alignment(final_output)
        
        x = self.decoder(final_output, encoder_features, input_skip)
        return x

if __name__ == "__main__":
    model = Generator(in_channels=1, out_channels=3)
    x = torch.randn((1, 1, 256, 256)) 
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
