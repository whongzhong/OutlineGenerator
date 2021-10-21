import os
import json
from tqdm import tqdm


dir = "/users10/lyzhang/Datasets/chinese_bookcropus/bookcroups/json_1"
# file_name = "fec7e73f-9f13-4370-952a-5eb129232b57@8c407517-8e13-4856-bd35-600caddebfb1@65442@1.json"
# path = '/'.join([dir, file_name])
file_names = os.listdir(dir)
# raise ("Value")
outputs = []
error_files = []
for file in tqdm(file_names):
    path = '/'.join([dir, file])
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            list = data["region"]
            pages = ""
            for item in list:
                pages += item['recog']['content'].strip().replace(" ", "")
            outputs.append(pages)
    except:
        error_files.append(file)
with open("/users10/lyzhang/opt/tiger/polish/data/retrieval/chinese_qa.txt", "r", encoding="utf-8"):
    for item in outputs:
        f.write(item + '/n')

with open("/users10/lyzhang/opt/tiger/polish/data/retrieval/chinese_qa_error_filename.txt", "r", encoding="utf-8"):
    for item in error_files:
        f.write(item + '/n')
        # print(list[0]['recog']['content'])
        
