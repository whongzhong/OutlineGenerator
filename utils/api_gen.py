# encoding:utf-8

import requests
import json
import time
import os
from tqdm import tqdm

def generate(text): 
    API_KEY = "tj9NP+YhQ8nDEpSGjjTZ8nlRa1dmw2VuwreYzFttkWnqEpznaYFL2Q=="  # 从悟道开发平台获取
    API_SECRET = "thp9I75apCNtv6RlhzZvFKtLuyXUBe7ixbtiIoP3QOXYmH4Q612LDw=="
    KEY = "queue1"  # 队列名称，默认queue1
    CONTENT = text  # 文本内容
    CONCURRENCY = 1  # 并发数
    TYPE = "para"  # para,sentence
    request_url = "https://pretrain.aminer.cn/api/v1/"
    api = 'generate'

    # 指定请求参数格式为json
    headers = {'Content-Type': 'application/json'}
    request_url = request_url + api
    data = {
        "key": KEY,
        "content": CONTENT,
        "concurrency": CONCURRENCY,
        "type": TYPE,
        "apikey": API_KEY,
        "apisecret": API_SECRET,
        'max_length': 256
    }
    response = requests.post(request_url, headers=headers, data=json.dumps(data))
    if response:
        while "output" not in response.json()["result"]:
            time.sleep(10)
            response = requests.get(request_url)
        return response.json()['result']['output'][0]
    
def gen_for_files(root, file_name, output_file_name):
    samples = []
    count = 0
    with open(os.path.join(root, file_name), 'r', encoding='utf-8') as f, \
    open(os.path.join(root, output_file_name), 'w', encoding='utf-8') as f1:
        for line in tqdm(f.readlines()):
            sample = json.loads(line)
            outline_text = "#".join(sample['outline']) + "。"
            if count < 311:
                count += 1
                continue
            time.sleep(0.01)
            story = generate(outline_text)
            sample['story'] = story
            
            json.dump(sample, f1, ensure_ascii=False)
            f1.write('\n')
            

if __name__ == '__main__':
    gen_for_files("../data/datasets/LOTdatasets", 'test.jsonl', 'test_api_out_1.jsonl')