
import glob
import os
import json
from threading import current_thread
from tqdm import tqdm
import re
import jieba
from rake import extractzh
import argparse

punc = ["。","！", "？", "“", "”", "?"]
stop_punc = ["。","！", "？", "”", "?"]

def rerake(root, file_name, output_filename):
    samples = []
    
    stop_path = "/users1/whzhong/code/OutlineGenerator/data/stoplist/sp.txt"
    conj_path = "/users1/whzhong/code/OutlineGenerator/data/stoplist/spw.txt"
    # Construct Stopword Lib
    swLibList = [line.rstrip('\n') for line in open(stop_path,'r',encoding='utf-8')]
    # Construct Phrase Deliminator Lib
    conjLibList = [line.rstrip('\n') for line in open(conj_path,'r',encoding='utf-8')]
    
    with open(os.path.join(root, file_name), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)
            samples.append(sample)
            new_sample = {'story': sample['story']}
            original_outline, all_keywords = extractzh(sample['story'], 12, 0.37, swLibList, conjLibList)

            new_outlines = []
            for outline in original_outline:
                if len(new_outlines) < 4:
                    new_outlines.append(outline)
                elif len(new_outlines) < 8:
                    if outline not in sample['outline']:
                        new_outlines.append(outline)
                else:
                    break
            if len(new_outlines) < 8:
                for outline in sample['outline']:
                    if outline not in new_outlines:
                        new_outlines.append(outline)
                        if len(new_outlines) == 8:
                            break
            new_sample['outline']  = new_outlines
            samples.append(new_sample)
            
    with open(os.path.join(root, output_filename), 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
    
            
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def split_and_rake(root, basic_length, data_path):
    
    file_path = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            file_path.append(os.path.join(root, line.strip()))
    
    stop_path = "/users1/whzhong/code/OutlineGenerator/data/stoplist/sp.txt"
    conj_path = "/users1/whzhong/code/OutlineGenerator/data/stoplist/spw.txt"
    # Construct Stopword Lib
    swLibList = [line.rstrip('\n') for line in open(stop_path,'r',encoding='utf-8')]
    # Construct Phrase Deliminator Lib
    conjLibList = [line.rstrip('\n') for line in open(conj_path,'r',encoding='utf-8')]
    
    for file in tqdm(file_path):
        with open(file, 'r', encoding="utf-8") as f:
            samples = json.load(f)
            
            
        new_samples = []
        for sample in samples:
            content = sample['content'].replace(" ", "")
            content_words = jieba.lcut(content)[:basic_length]
            length = min(len(content_words), basic_length)
            for i in range(length-1, -1, -1):
                if content_words[i] in stop_punc:
                    content_words = content_words[:i+1]
                    break
            short_content = "".join(content_words)
            outline = extractzh(short_content, 8, 0.25, swLibList, conjLibList)
            
            '''
            outline = [phrase for phrase, score in outline_rating if len(phrase) > 1]
            
            outline_len = len(outline)
            remove_index = []
            for idx1, phrase_1 in enumerate(outline):
                if len(jieba.lcut(phrase_1)) > 8:
                    remove_index.append(idx)
                if idx1 not in remove_index:
                    for idx in range(0, outline_len):
                        if idx not in remove_index and idx != idx1 and outline[idx] in phrase_1:
                            remove_index.append(idx)

            new_outline = []
            for idx, phrase in enumerate(outline):
                if idx not in remove_index and len(phrase) > 1:
                    new_outline.append(phrase)
            new_outline = new_outline[:8]
            '''
            if len(outline) == 0:
                continue
            new_sample = {}
            new_sample['uniqueKey'] = sample['uniqueKey']
            new_sample['title'] = sample['title']
            new_sample['dataType'] = sample['dataType']
            new_sample['story'] = short_content
            new_sample['sent_cut'] = cut_sent(short_content)
            new_sample['outline'] = outline
            new_samples.append(new_sample)
        with open(file+".processed", 'w', encoding="utf-8") as f:
            json.dump(new_samples, f, ensure_ascii=False, indent=4)
            
def check_bisic(root):
    total_length = 0.0
    sample_nums = 0
    cataloge = set()
    for file in tqdm(glob.glob(os.path.join(root, '*.json'))):
        with open(file, 'r', encoding="utf-8") as f:
            samples = json.load(f)
        sample_nums += len(samples)
        for sample in samples:
            total_length += len(sample['content'])
            cataloge.add(sample['dataType'])
    avg_length = total_length / sample_nums
    
    return avg_length, cataloge

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

def check_cataloge(root):
    cataloge = {}
    sample_num = 0
    for file in tqdm(glob.glob(os.path.join(root, '*.json'))):
        with open(file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        sample_num += len(samples)
        for sample in samples:
            if sample['dataType'] not in cataloge:
                cataloge[sample['dataType']] =  1
            else:
                cataloge[sample['dataType']] += 1
    
    return cataloge, sample_num
            
            
def create_split(root, split_file):
    file_list = []
    with open(os.path.join(root, split_file), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            file_list.append(line.strip())
    j = 0
    count = 0
    while j < len(file_list):
        current_list = file_list[j:j+12]
        with open(os.path.join(root, f'split_{count}.txt'), 'w', encoding='utf-8') as f:
            for file_name in current_list:
                f.write(file_name+ "\n")
        j += 12
        count += 1

        

if __name__ == '__main__':
    rerake('/users1/whzhong/code/OutlineGenerator/data/datasets/LOTdatasets', 'train_val.jsonl', "train_val_rerake.jsonl")
    #parser = argparse.ArgumentParser(description='Process some integers.')
    #parser.add_argument("--data_path", type=str, default=None)
    #args = parser.parse_args()
    
    #split_and_rake('/users1/whzhong/code/OutlineGenerator/data/datasets/wudao1', 250, args.data_path)
    #create_split('data/datasets/wudao1', 'total_path.txt')
    #cataloge, sample_num = check_cataloge('/users1/whzhong/code/OutlineGenerator/data/datasets/wudao1')
    #print(sample_num)
   #print("\n")
    #print(json.dumps(cataloge))
    
    #with open('/users1/whzhong/code/OutlineGenerator/data/datasets/wudao1/#part-202101281a.json.processed', 'r', encoding='utf-8') as f:
    #    samples = json.load(f)
    #    import ipdb; ipdb.set_trace()
    