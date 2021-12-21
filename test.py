import json
import os
import jieba
from tqdm import tqdm

def add_outline(root, file_name, output_filename):
    samples = []
    label = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ']
    with open(os.path.join(root, file_name), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            forbid_idx = set()
            replace_idx = {}
            sample = json.loads(line)
            
            for idx, outline in enumerate(sample['outline']):
                find_idx = sample['story'].find(outline)
                if find_idx != -1:
                    for i in range(len(outline)):
                        forbid_idx.add(find_idx + i)    
            for outline_idx, outline in enumerate(sample['outline']):
                if outline not in sample['story']:
                    outline_len = len(outline)
                    insert_flag = False
                    for idx in range(outline_len - 1, 0, -1):
                        find_idx = sample['story'].find(outline[:idx])
                        while find_idx != -1 and find_idx in forbid_idx:
                            find_idx = sample['story'].find(outline[:idx], find_idx + 1)
                        if find_idx != -1:
                            for i in range(idx):
                                forbid_idx.add(find_idx + i)   
                            replace_idx[str(outline_idx)] = idx    
                            sample['story'] = sample['story'][:find_idx] + label[outline_idx] + sample['story'][find_idx + 1:]
                            insert_flag = True
                        if insert_flag:
                            break
                        
                    if not insert_flag:
                        words = jieba.lcut(outline)
                        for word in words:
                            find_idx =  sample['story'].find(word)
                            while find_idx != -1 and find_idx in forbid_idx:
                                find_idx = sample['story'].find(word, find_idx + 1)
                            if find_idx != -1:
                                for i in range(len(word)):
                                   forbid_idx.add(find_idx + i)   
                                replace_idx[str(outline_idx)] = len(word)    
                                sample['story'] = sample['story'][:find_idx] + label[outline_idx] + sample['story'][find_idx + 1:]
                                insert_flag = True
                                break
                            
                    if not insert_flag:
                        for ch in outline:
                            find_idx =  sample['story'].find(ch)
                            while find_idx != -1 and find_idx in forbid_idx:
                                find_idx = sample['story'].find(ch, find_idx + 1)
                            if find_idx != -1:
                                forbid_idx.add(find_idx) 
                                replace_idx[str(outline_idx)] = 1    
                                sample['story'] = sample['story'][:find_idx] + label[outline_idx] + sample['story'][find_idx + 1:]
                                insert_flag = True
                        #sample['story'] = sample['story'] + outline
                    
            for outline_idx, outline in enumerate(sample['outline']): 
                if str(outline_idx) in replace_idx:
                    find_idx = sample['story'].find(label[outline_idx])
                    sample['story'] = sample['story'][:find_idx] + outline + \
                        sample['story'][find_idx+replace_idx[str(outline_idx)]:]
            samples.append(sample)
    
    with open(os.path.join(root, output_filename), 'w', encoding='utf-8') as f:
        for sample in samples: 
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
            

if __name__ == '__main__':
    add_outline('/users1/whzhong/code/OutlineGenerator/outputs/cpm_final', 'final_origin.txt', 'final_origin_no_fugai.txt')
    '''
    with open('fuck1.txt', 'r', encoding='utf-8') as f:
        item = json.load(f)
        
    
    with open('fuck1.txt', 'w', encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False, indent=4)
        '''