#PREPARING THE INITIAL DATA THAT WILL BE USED FOR VLM SPECIFIC DATASETS

#listing 1: parsing coco dataset

# Your existing code
from pycocotools.coco import COCO
coco = COCO('_annotations.coco.json')
real_categories = ["calyx", "fruitlet", "peduncle"]


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
print("Categories after filtering:", [cat['name'] for cat in categories])



#listing 2: saving cropped fruitlet parts

from PIL import Image
import os

os.makedirs('crops', exist_ok = True)
for cat in categories:
    os.makedirs(f'crops/{cat["name"]}', exist_ok = True)

'''
creates dirs in this structure
crops/
├── calyx/
├── fruitlet/
└── peduncle/
'''


for ann in coco.dataset['annotations']:
    img_meta = coco.loadImgs(ann['image_id'])[0]
    img = Image.open(f'{img_meta["file_name"]}')
    x, y, w, h = ann['bbox']
    crop = img.crop((x, y, x + w, y + h))
    label = coco.loadCats([ann['category_id']])[0]['name']
    crop.save(f'crops/{label}/{img_meta["file_name"]}_{ann["id"]}.jpg')

'''
populates the 3 subdirs above with cropped images of the corresponding part
used bbox coordinates from the json annotation file to crop
'''

