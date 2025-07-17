import json
import random
import os
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

def bbox_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap
    bbox format: [x, y, width, height]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Check if boxes don't overlap
    if (x1 >= x2 + w2 or x2 >= x1 + w1 or 
        y1 >= y2 + h2 or y2 >= y1 + h1):
        return False
    return True

def get_random_bbox(img_width, img_height, min_size=32, max_size=256):
    """
    Generate a random bounding box within image dimensions
    """
    # Ensure max_size doesn't exceed image dimensions
    max_w = min(max_size, img_width - 10)
    max_h = min(max_size, img_height - 10)
    
    if max_w < min_size or max_h < min_size:
        return None
    
    # Random width and height
    w = random.randint(min_size, max_w)
    h = random.randint(min_size, max_h)
    
    # Random position ensuring bbox stays within image
    x = random.randint(0, img_width - w)
    y = random.randint(0, img_height - h)
    
    return [x, y, w, h]

def generate_complete_dataset(coco_json_path, images_dir='', output_dir='crops', 
                            target_negative_samples=680, min_size=32, max_size=256, 
                            max_attempts=50):
    """
    Generate complete dataset with both positive classes and negative samples
    """
    
    # Load COCO dataset
    coco = COCO(coco_json_path)
    
    # Define classes of interest
    real_categories = ["calyx", "fruitlet", "peduncle"]
    
    # Filter for classes of interest
    ids_to_keep = {
        cat['id'] for cat in coco.dataset['categories'] 
        if cat['name'] in real_categories
    }
    
    coco.dataset['categories'] = [
        cat for cat in coco.dataset['categories'] 
        if cat['id'] in ids_to_keep
    ]
    
    coco.dataset['annotations'] = [
        ann for ann in coco.dataset['annotations'] 
        if ann['category_id'] in ids_to_keep
    ]
    
    coco.createIndex()
    categories = coco.loadCats(coco.getCatIds())
    
    print("=== GENERATING POSITIVE SAMPLES ===")
    print("Categories:", [cat['name'] for cat in categories])
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for positive classes
    for cat in categories:
        os.makedirs(f'{output_dir}/{cat["name"]}', exist_ok=True)
    
    # Create directory for negative class
    os.makedirs(f'{output_dir}/negative', exist_ok=True)
    
    # STEP 1: Generate positive samples (your original code)
    positive_count = 0
    for ann in coco.dataset['annotations']:
        try:
            img_meta = coco.loadImgs(ann['image_id'])[0]
            img_path = os.path.join(images_dir, img_meta['file_name']) if images_dir else img_meta['file_name']
            img = Image.open(img_path)
            x, y, w, h = ann['bbox']
            crop = img.crop((x, y, x + w, y + h))
            label = coco.loadCats([ann['category_id']])[0]['name']
            crop.save(f'{output_dir}/{label}/{img_meta["file_name"]}_{ann["id"]}.jpg')
            positive_count += 1
        except Exception as e:
            print(f"Error processing positive annotation {ann['id']}: {e}")
            continue
    
    print(f"Generated {positive_count} positive samples")
    
    # STEP 2: Generate negative samples
    print("\n=== GENERATING NEGATIVE SAMPLES ===")
    
    # Get all image IDs that have annotations
    image_ids = list(set(ann['image_id'] for ann in coco.dataset['annotations']))
    total_images = len(image_ids)
    
    print(f"Target negative samples: {target_negative_samples}")
    print(f"Total images: {total_images}")
    
    # Calculate samples per image distribution
    base_samples_per_image = target_negative_samples // total_images
    extra_samples_needed = target_negative_samples % total_images
    
    print(f"Base samples per image: {base_samples_per_image}")
    print(f"Extra samples needed: {extra_samples_needed}")
    print(f"Distribution: {total_images - extra_samples_needed} images will get {base_samples_per_image} samples, {extra_samples_needed} images will get {base_samples_per_image + 1} samples")
    
    # Shuffle image IDs to randomly distribute the extra samples
    random.shuffle(image_ids)
    
    negative_count = 0
    failed_images = 0
    
    for idx, img_id in enumerate(image_ids):
        try:
            # Calculate how many samples this image should get
            samples_for_this_image = base_samples_per_image
            if idx < extra_samples_needed:  # First N images get an extra sample
                samples_for_this_image += 1
            
            # Load image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_dir, img_info['file_name']) if images_dir else img_info['file_name']
            
            # Skip if image doesn't exist
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            # Load image
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Get all annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            
            # Extract bounding boxes from annotations
            positive_bboxes = [ann['bbox'] for ann in annotations]
            
            # Generate negative samples for this image
            image_negatives = 0
            for sample_idx in range(samples_for_this_image):
                
                # Try to find a non-overlapping region
                found_valid_bbox = False
                for attempt in range(max_attempts):
                    
                    # Generate random bbox
                    candidate_bbox = get_random_bbox(img_width, img_height, min_size, max_size)
                    
                    if candidate_bbox is None:
                        continue
                    
                    # Check if it overlaps with any positive annotation
                    overlaps = False
                    for pos_bbox in positive_bboxes:
                        if bbox_overlap(candidate_bbox, pos_bbox):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Found a valid non-overlapping region
                        x, y, w, h = candidate_bbox
                        
                        # Crop the region
                        crop = img.crop((x, y, x + w, y + h))
                        
                        # Save the crop
                        crop_filename = f"{img_info['file_name'].split('.')[0]}_neg_{sample_idx}.jpg"
                        crop_path = os.path.join(output_dir, 'negative', crop_filename)
                        crop.save(crop_path)
                        
                        image_negatives += 1
                        negative_count += 1
                        found_valid_bbox = True
                        break
                
                if not found_valid_bbox:
                    print(f"Could not find non-overlapping region for {img_info['file_name']} (sample {sample_idx})")
            
            if image_negatives == 0:
                failed_images += 1
            
            # Progress update
            if (idx + 1) % 20 == 0:
                print(f"Processed {idx + 1}/{total_images} images. Generated {negative_count} negative samples. Target: {target_negative_samples}")
                
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            failed_images += 1
            continue
    
    print(f"\n=== DATASET GENERATION COMPLETE ===")
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count} (target was {target_negative_samples})")
    print(f"Failed to process {failed_images} images")
    
    # Print summary of each class
    print("\nSummary by class:")
    for cat in categories:
        crop_dir = f'{output_dir}/{cat["name"]}'
        if os.path.exists(crop_dir):
            count = len([f for f in os.listdir(crop_dir) if f.endswith('.jpg')])
            print(f"{cat['name']}: {count} samples")
    
    # Count negative samples
    neg_dir = f'{output_dir}/negative'
    if os.path.exists(neg_dir):
        neg_count = len([f for f in os.listdir(neg_dir) if f.endswith('.jpg')])
        print(f"negative: {neg_count} samples")
    
    total_samples = positive_count + negative_count
    print(f"\nTotal dataset size: {total_samples} samples")

# Example usage
if __name__ == "__main__":
    generate_complete_dataset(
        coco_json_path='_annotations.coco.json',
        images_dir='',  # Set to your images directory if needed
        output_dir='crops',
        target_negative_samples=85,  # Target number of negative samples
        min_size=64,
        max_size=256,
        max_attempts=100
    )