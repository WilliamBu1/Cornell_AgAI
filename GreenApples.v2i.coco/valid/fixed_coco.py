import json
import os
from collections import Counter

# --- 1. CONFIGURATION ---

# The original annotation file you want to clean
SOURCE_ANNOTATION_FILE = '_annotations.coco.json'
# The name of the new, cleaned file that will be created
CLEANED_ANNOTATION_FILE = '_annotations_cleaned_positives_only.json'

# --- IMPORTANT: UPDATE THESE CLASS NAMES ---
# Define the mapping from the incorrect/redundant class names to the correct positive class names.
CLASS_MERGE_MAP = {
    "-": "fruitlet",
    "GreenApples - v1 2024-09-24 2-56pm": "calyx", #<-- REPLACE WITH YOUR EXACT CLASS NAME
    "GreenApples - v1 2024-09-24 2-58pm": "calyx"  #<-- REPLACE WITH YOUR EXACT CLASS NAME
}

# Define the final list of categories you want to keep
FINAL_CLASSES = ["calyx", "fruitlet", "peduncle"]

# --- 2. SCRIPT LOGIC ---

print(f"--- Starting COCO Annotation Cleaning ---")
print(f"Loading source file: {SOURCE_ANNOTATION_FILE}")

if not os.path.exists(SOURCE_ANNOTATION_FILE):
    raise FileNotFoundError(f"Source annotation file not found. Please make sure '{SOURCE_ANNOTATION_FILE}' is in the same directory.")

with open(SOURCE_ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)

# --- Step A: Create Mappings from the Original Categories ---
print("Building category maps...")
original_categories = coco_data['categories']
name_to_id = {cat['name']: cat['id'] for cat in original_categories}
id_to_name = {cat['id']: cat['name'] for cat in original_categories}

# Create a remapping dictionary for category IDs that need to be changed
id_remap = {}
for old_name, new_name in CLASS_MERGE_MAP.items():
    if old_name in name_to_id and new_name in name_to_id:
        old_id = name_to_id[old_name]
        new_id = name_to_id[new_name]
        id_remap[old_id] = new_id

print(f"Will remap the following category IDs: {id_remap}")

# --- Step B: Update the 'annotations' Section ---
print("Processing and remapping annotations...")
cleaned_annotations = []
original_counts = Counter()
cleaned_counts = Counter()

for ann in coco_data['annotations']:
    original_cat_id = ann['category_id']
    original_counts[id_to_name.get(original_cat_id, 'Unknown')] += 1
    
    # Remap the category ID if it's in our remap dictionary
    if original_cat_id in id_remap:
        ann['category_id'] = id_remap[original_cat_id]

    # Keep the annotation only if its (new) category is in our final list
    final_cat_name = id_to_name.get(ann['category_id'])
    if final_cat_name in FINAL_CLASSES:
        cleaned_annotations.append(ann)
        cleaned_counts[final_cat_name] += 1
        
coco_data['annotations'] = cleaned_annotations
print("Annotations remapped and filtered successfully.")

# --- Step C: Update the 'categories' Section ---
print("Cleaning up category definitions...")
# Keep only the final, desired categories
final_categories_data = [cat for cat in original_categories if cat['name'] in FINAL_CLASSES]
coco_data['categories'] = final_categories_data
print("Categories section cleaned.")


# --- Step D: Save the New Cleaned File ---
with open(CLEANED_ANNOTATION_FILE, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"\n--- Cleaning Complete! ✅ ---")
print(f"New cleaned file saved as: {CLEANED_ANNOTATION_FILE}")

print("\nOriginal Annotation Counts (before merge):")
for name, count in original_counts.items():
    print(f"- {name}: {count}")

print("\nCleaned Annotation Counts (after merge):")
for name, count in cleaned_counts.items():
    print(f"- {name}: {count}")