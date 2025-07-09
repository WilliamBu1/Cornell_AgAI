# PREPARING DATA FOR THE 3 VLMS

#1. TINYCLIP EXPECTS TEXT PROMPTS AND IMAGE FOLDERS

'''
i believe it will use the crops dir directly and the 3 subdirs
then a list of promtps
prompts = ["a photo of a calyx", "a photo of a fruitlet", "a photo of a peduncle"]
'''


#2. listing 3: creating caption JSONL file for MINIGPT4

import json, glob

entries = []
for label in ['calyx', 'fruitlet', 'peduncle']:
    for path in glob.glob(f'crops/{label}/*.jpg'):
        entries.append({
            "image" : path,
            "text" : f"This_is_the_{label}_part_of_a_green_apple."
        })

'''
just run once

with open("minigpt_dataset.json", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
'''


#3. listing4: creating JSON for LLaVA-Light

qa_data = []
idx = 0

for label in ['calyx', 'fruitlet', 'peduncle']:
    for path in glob.glob(f'crops/{label}/*.jpg'):
        qa_data.append({
            "id" : str(idx),
            "image" : path,
            "conversations" : [
                {"from" : "human", "value" : "What_part_of_the_fruit_is_this?"},
                {"from" : "gpt", "value" : f"This_is_the_{label}."}
            ]
        })
        idx += 1

with open("llava_data.json", "w") as f:
    json.dump(qa_data, f, indent = 2)