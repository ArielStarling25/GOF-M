import os
from pathlib import Path
import shutil

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

dataset_dir = "dtu"
current_file_path = Path(__file__).resolve()
scripts_dir = current_file_path.parent
project_root = "."

def get_filenames(directory):
    filenames = os.listdir(directory)
    files_no_ext = []
    for f in filenames:
        if os.path.isfile(os.path.join(directory, f)):
            # name, extension = os.path.splitext(f)
            files_no_ext.append(f)
    return files_no_ext

def convert_dtu(image_n_mask):
    src_path = image_n_mask['mask_dir']
    src_parent = os.path.dirname(image_n_mask['mask_dir'])
    filename, ext = os.path.splitext(image_n_mask['img_name'])
    new_file_name = f"{filename}_mask{ext}"
    destination_path = os.path.join(src_parent, new_file_name)
    try:
        # Copy the file to the new location with the new name
        shutil.copy(src_path, destination_path)
        print(f"File copied from '{src_path}' and renamed to '{destination_path}' successfully!")
    except FileNotFoundError:
        print(f"Error: The source file '{src_path}' was not found.")
    except PermissionError:
        print("Error: Permission denied. Check file permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def check_directory(scene):
    dataset_path = os.path.join(project_root, "datasets", dataset_dir, f"scan{scene}")
    print("Checking Dataset: ", dataset_path)
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return False
    images_path = os.path.join(dataset_path, "images")
    if not os.path.exists(images_path):
        print("Dataset images not found!")
        return False
    masks_path = os.path.join(dataset_path, "mask")
    if not os.path.exists(masks_path):
        print("Dataset masks not found!")
        return False
    print(f"Dataset {dataset_path} found")
    return True

# Assuming: 
#   - Images start from '0000' to '0100'
#   - Masks start from '001' to '100'
def verify_directory(scene):
    dataset_path = os.path.join(project_root, "datasets", dataset_dir, f"scan{scene}")
    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "mask")
    print("Verifying Dataset: ", dataset_path)

    image_names = get_filenames(images_path)
    masks_names = get_filenames(masks_path)
    image_n_mask = [] # 2D array

    for i in range(len(image_names)):
        for j in range(len(masks_names)):
            img_name, _ = os.path.splitext(image_names[i])
            msk_name, _ = os.path.splitext(masks_names[j])

            if img_name[1:] == msk_name:
                image_n_mask.append({
                    "img_name": image_names[i],
                    "mask_name": masks_names[j],
                    "img_dir": os.path.join(images_path, image_names[i]),
                    "mask_dir": os.path.join(masks_path, masks_names[j])
                })
                print(f"Found that {image_names[i]} corresponds to {masks_names[j]}")

    print(f"Final image_n_mask count: {len(image_n_mask)} | original image count: {len(image_names)}")

    if len(image_n_mask) != len(image_names):
        return image_n_mask, False
    else:
        return image_n_mask, True

def main():
    for item in scenes:
        if check_directory(item):
            images_n_masks, result = verify_directory(item)
            if result:
                for item in images_n_masks:
                    convert_dtu(item)

main()