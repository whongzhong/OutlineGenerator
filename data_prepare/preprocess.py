
import glob
import os
import json
from tqdm import tqdm

def check_length(root):
    total_length = 0.0
    sample_nums = 0
    for file in tqdm(glob.glob(os.path.join(root, '*.json'))):
        with open(file, 'r', encoding="utf-8") as f:
            samples = json.load(f)
        sample_nums += len(samples)
        for sample in samples:
            total_length += len(sample['content'])
    avg_length = total_length / sample_nums
    
    return avg_length
            


if __name__ == '__main__':
    print(check_length("../data/datasets/wudao"))