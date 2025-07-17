import os
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import random

# --- 1. CONFIGURATION ---
# The folder with the images you want to augment (e.g., your peduncle crops)
SOURCE_DIR = './crops/pre-peduncle'
# A new folder where the augmented images will be saved
OUTPUT_DIR = './crops/peduncle'
# How many new, augmented versions to create for each original image
NUM_COPIES_PER_IMAGE = 4
# The final output size of your augmented images
FINAL_SIZE = 224

# --- 2. SETUP ---
print(f"--- Setting up Realistic Data Augmentation ---")
print(f"Source:      {SOURCE_DIR}")
print(f"Destination: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the new, improved set of augmentations
# This pipeline is designed to avoid unrealistic artifacts.
augmentation_pipeline = transforms.Compose([
    # Step 1: Resize to be slightly larger than the final size
    transforms.Resize((int(FINAL_SIZE * 1.15), int(FINAL_SIZE * 1.15))),
    
    # Step 2: Perform rotation on the larger canvas. Black corners will be outside the final crop area.
    transforms.RandomRotation(15), # Rotates by up to 15 degrees
    
    # Step 3: Crop the center back down to the final desired size.
    transforms.CenterCrop(FINAL_SIZE),
    
    # Step 4: Apply a random horizontal flip.
    transforms.RandomHorizontalFlip(p=0.5),
    
    # Step 5: Apply a GENTLER color jitter.
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    
    # Optional: Add a very slight blur to simulate different focus, which is common in photos.
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
])

# --- 3. APPLY AUGMENTATIONS ---
source_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\nFound {len(source_files)} images to augment...")

for filename in tqdm(source_files, desc="Augmenting Images"):
    image_path = os.path.join(SOURCE_DIR, filename)
    try:
        # Open the original image
        original_image = Image.open(image_path).convert("RGB")
        
        # Create and save N augmented copies
        for i in range(NUM_COPIES_PER_IMAGE):
            # Apply the random transformations
            augmented_image = augmentation_pipeline(original_image)
            
            # Create a new unique filename
            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_aug_{i+1}{extension}"
            save_path = os.path.join(OUTPUT_DIR, new_filename)
            
            # Save the new image
            augmented_image.save(save_path)
            
    except Exception as e:
        print(f"Could not process {filename}. Error: {e}")
        
print(f"\nAugmentation complete! 🎉 Check the '{OUTPUT_DIR}' folder.")