import os
import shutil
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import random

# --- 1. CONFIGURATION ---
SOURCE_DIR = './crops/pre-peduncle'
OUTPUT_DIR = './crops/peduncle'
# The final number of images you want in the OUTPUT_DIR
TARGET_IMAGE_COUNT = 385
# The final output size for the augmented images
FINAL_SIZE = 224

# --- 2. SETUP ---
print(f"--- Setting up Data Augmentation ---")
print(f"Source:      {SOURCE_DIR}")
print(f"Destination: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the set of augmentations to apply
augmentation_pipeline = transforms.Compose([
    transforms.Resize((int(FINAL_SIZE * 1.15), int(FINAL_SIZE * 1.15))),
    transforms.RandomRotation(15),
    transforms.CenterCrop(FINAL_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
])

# --- 3. COPY ORIGINAL FILES & CALCULATE HOW MANY TO AUGMENT ---
source_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
num_originals = len(source_files)

print(f"\nFound {num_originals} original images.")

# First, copy all original files to the destination
print(f"Copying {num_originals} original images to '{OUTPUT_DIR}'...")
for filename in tqdm(source_files, desc="Copying originals"):
    shutil.copy(os.path.join(SOURCE_DIR, filename), os.path.join(OUTPUT_DIR, filename))

# Calculate how many new augmented images we need to create
num_to_augment = TARGET_IMAGE_COUNT - num_originals

if num_to_augment <= 0:
    print(f"\nTarget count of {TARGET_IMAGE_COUNT} is already met or exceeded by the {num_originals} original images. No augmentation needed.")
else:
    print(f"Need to generate {num_to_augment} new augmented images to reach the target of {TARGET_IMAGE_COUNT}.")

    # --- 4. APPLY AUGMENTATIONS TO "TOP-UP" THE DATASET ---
    for i in tqdm(range(num_to_augment), desc="Generating Augmented Images"):
        # Pick a random image from the source directory to augment
        source_filename = random.choice(source_files)
        image_path = os.path.join(SOURCE_DIR, source_filename)
        
        try:
            original_image = Image.open(image_path).convert("RGB")
            
            # Apply the random transformations
            augmented_image = augmentation_pipeline(original_image)
            
            # Create a new unique filename
            base_name, extension = os.path.splitext(source_filename)
            new_filename = f"{base_name}_aug_{i+1}{extension}"
            save_path = os.path.join(OUTPUT_DIR, new_filename)
            
            # Save the new image
            augmented_image.save(save_path)
            
        except Exception as e:
            print(f"Could not process {source_filename}. Error: {e}")
            
    print(f"\nAugmentation complete! 🎉")

# --- 5. FINAL VERIFICATION ---
final_count = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"The output folder '{OUTPUT_DIR}' now contains {final_count} images.")