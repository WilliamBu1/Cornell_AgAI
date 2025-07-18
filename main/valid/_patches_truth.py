import json
import csv
import os

def generate_patch_annotations_from_coco(
    coco_file_path,
    patch_size,
    stride
):
    """
    Parses a COCO annotation file for specific images and generates a ground truth CSV
    for their patches using a sliding window approach. It only processes images that
    contain at least one of the target class annotations.
    """
    # --- 1. Load COCO Annotation File ---
    if not os.path.exists(coco_file_path):
        print(f"❌ Error: Annotation file not found at '{coco_file_path}'")
        return

    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # --- 2. Prepare Data Structures ---
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # --- 3. Identify Images That Contain Target Classes ---
    target_classes = {'calyx', 'peduncle', 'fruitlet'}
    relevant_image_ids = set()
    for ann in coco_data.get('annotations', []):
        category_name = categories.get(ann['category_id'])
        if category_name in target_classes:
            relevant_image_ids.add(ann['image_id'])

    if not relevant_image_ids:
        print("⚠️ Warning: No images with 'calyx', 'peduncle', or 'fruitlet' annotations were found. The output file will be empty.")
        return

    print(f"✅ Found {len(relevant_image_ids)} images containing the target classes. Processing them now...")

    # --- 4. Process Filtered Images and Generate Patches ---
    output_filename = '_patch_ground_truth_filtered.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'patch_id', 'patch_x', 'patch_y', 'calyx', 'peduncle', 'fruitlet', 'negative']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_info in coco_data['images']:
            image_id = image_info['id']
            image_filename = image_info['file_name']
            
            # **MODIFICATION**: Skip any image that is not in our set of relevant IDs.
            if image_id not in relevant_image_ids:
                print(f"⏭️  Skipping '{image_filename}' (no target annotations).")
                continue

            image_width = image_info['width']
            image_height = image_info['height']
            patch_counter = 0

            for y in range(0, image_height - patch_size + 1, stride):
                for x in range(0, image_width - patch_size + 1, stride):
                    patch_id = f"patch_{patch_counter}"
                    patch_labels = {'calyx': 0, 'peduncle': 0, 'fruitlet': 0}

                    # This check is now guaranteed to be on a relevant image
                    if image_id in annotations_by_image:
                        for ann in annotations_by_image[image_id]:
                            category_name = categories.get(ann['category_id'], 'unknown')
                            if category_name in target_classes:
                                ann_bbox = ann['bbox']

                                if (x < ann_bbox[0] + ann_bbox[2] and
                                    x + patch_size > ann_bbox[0] and
                                    y < ann_bbox[1] + ann_bbox[3] and
                                    y + patch_size > ann_bbox[1]):
                                    patch_labels[category_name] = 1

                    is_negative = 1 if all(label == 0 for label in patch_labels.values()) else 0

                    writer.writerow({
                        'image_id': image_filename,
                        'patch_id': patch_id,
                        'patch_x': x,
                        'patch_y': y,
                        'calyx': patch_labels['calyx'],
                        'peduncle': patch_labels['peduncle'],
                        'fruitlet': patch_labels['fruitlet'],
                        'negative': is_negative
                    })
                    patch_counter += 1
            print(f"Processed '{image_filename}' and generated {patch_counter} patches.")

    print(f"\n✅ Successfully generated ground truth file: '{output_filename}'")


# --- Configuration and Execution ---
# Note: The script now gets image dimensions directly from the COCO file,
# so you no longer need to set FULL_IMAGE_WIDTH and FULL_IMAGE_HEIGHT.

PATCH_SIZE = 224      # The size of the square image patches
STRIDE = 112          # The step size for the sliding window

# Set the path to YOUR COCO annotation file
# Make sure this file contains the CORRECT labels for 'calyx', 'peduncle', 'fruitlet'
COCO_ANNOTATION_FILE = r'C:\Users\William\Desktop\project\main\valid\_annotations.coco.json'

generate_patch_annotations_from_coco(
    COCO_ANNOTATION_FILE,
    PATCH_SIZE,
    STRIDE
)