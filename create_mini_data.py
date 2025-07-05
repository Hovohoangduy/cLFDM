import os
import shutil

source_dir = r'datasets/UTD-MHAD/crop_image'

destination_dir = r'datasets/UTD-MHAD/crop_image_mini'
os.makedirs(destination_dir, exist_ok=True)
unique_prefixes = {}
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        parts = folder_name.split('_')
        if len(parts) == 3:
            prefix = f"{parts[0]}_{parts[1]}"  # aX_sY
            if prefix not in unique_prefixes:
                unique_prefixes[prefix] = folder_path
                dst_path = os.path.join(destination_dir, folder_name)
                shutil.copytree(folder_path, dst_path)
                print(f"Copied: {folder_name} → {dst_path}")

print(f"Total number of unique directories in the form aX_sY copied: {len(unique_prefixes)}")