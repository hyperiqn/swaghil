import os
import shutil

# Set source directories
source_dirs = {
    "opt": "/data/anirudh/sih/QXSLAB_SAROPT/opt_256_oc_0.2",
    "sar": "/data/anirudh/sih/QXSLAB_SAROPT/sar_256_oc_0.2"
}

# Set target directories
train_dirs = {
    "opt": "/data/anirudh/sih/QXSLAB_SAROPT/train/opt",
    "sar": "/data/anirudh/sih/QXSLAB_SAROPT/train/sar"
}
val_dirs = {
    "opt": "/data/anirudh/sih/QXSLAB_SAROPT/val/opt",
    "sar": "/data/anirudh/sih/QXSLAB_SAROPT/val/sar"
}

# Create target directories if they don't exist
for dir_dict in [train_dirs, val_dirs]:
    for path in dir_dict.values():
        os.makedirs(path, exist_ok=True)

# Number of files for training and validation sets
train_count = 18000
val_count = 2000

# List all files in one of the source directories (assuming both have matching filenames) and sort by filename
files = sorted(os.listdir(source_dirs["opt"]))

# Split files into train and validation sets
train_files = files[:train_count]
val_files = files[train_count:train_count + val_count]

# Function to move corresponding files
def move_files(file_list, split_type):
    for file_name in file_list:
        for key, source_dir in source_dirs.items():
            source_path = os.path.join(source_dir, file_name)
            if split_type == "train":
                dest_path = os.path.join(train_dirs[key], file_name)
            else:
                dest_path = os.path.join(val_dirs[key], file_name)
            shutil.move(source_path, dest_path)

# Move train and validation files
move_files(train_files, "train")
move_files(val_files, "val")

print("Data split with sorted filenames and correspondence completed successfully.")
