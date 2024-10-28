import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageToImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform_A=None, transform_B=None):
        """
        Args:
            root_A (str): Directory with input images (SAR).
            root_B (str): Directory with target images (RGB).
            transform_A (callable, optional): Optional transform to be applied on SAR images.
            transform_B (callable, optional): Optional transform to be applied on RGB images.
        """
        self.root_A = root_A
        self.root_B = root_B
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.images_A = sorted(os.listdir(root_A))
        self.images_B = sorted(os.listdir(root_B))
        assert len(self.images_A) == len(self.images_B), "Input and target directories must have the same number of images."

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.root_A, self.images_A[idx])).convert('L')  # 'L' for grayscale (single-channel)
        
        img_B = Image.open(os.path.join(self.root_B, self.images_B[idx])).convert('RGB')

        assert img_A.size == img_B.size, "Input (SAR) and target (RGB) images must have the same dimensions."

        if self.transform_A:
            img_A = self.transform_A(img_A)
        if self.transform_B:
            img_B = self.transform_B(img_B)

        return img_A, img_B
