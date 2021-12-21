from whoosh.fields import TEXT, Schema, ID
from whoosh.index import open_dir, create_in
from whoosh.qparser import QueryParser
import jieba.analyse

import json
import os
import re
from tqdm import tqdm

class Seacher():
    def __init__(self, data_root) -> None:
        super().__init__()
        self.schema = Schema(sentences=TEXT(analyzer=jieba.analyse.ChineseAnalyzer(), stored=True), tokens = TEXT())
        self.data_root = data_root
        #with open("")
        #self.stop_list = 

    #def remove_stopwords(self, List):

    def index(self, data_root, file_name):
        idx = create_in(data_root, self.schema)
        writer = idx.writer()
        self.data_root = data_root
        with open(os.path.join(data_root, file_name), 'r') as f:
            #import ipdb; ipdb.set_trace()
            samples = json.load(f)
            for sample in tqdm(samples):
                #sample = json.loads(line)
                story_split_by_eos = re.split("。", sample['story'].strip())
                for single_sentence in story_split_by_eos:
                    writer.add_document(sentences=single_sentence, tokens=' '.join(jieba.lcut(single_sentence)))

        writer.commit()
    
    def search(self, idx, querys = []):
        results =  []
        with idx.searcher() as searcher:
            parser = QueryParser("sentences", idx.schema)
            for query in querys:
                parsed_query = parser.parse(' OR '.join(jieba.lcut(query)))
                #parsed_query = parser.parse(query)
                result = [hit['sentences'] for hit in searcher.search(parsed_query, limit=1)]
                if len(result) > 0:
                    results.append(result[0])
                else:
                    results.append("")
        return results

def create_searched_list(data_root, file_name, output_filename):
    searcher = Seacher(data_root)
    #searcher.index(data_root, file_name)
    processed_samples = []
    with open(os.path.join(data_root, file_name), 'r') as f:
        
        samples = json.load(f)
        idx = open_dir(data_root)
        for sample in tqdm(samples):
            #sample = json.loads(line)
            outlines = sample['outline']
            result = searcher.search(idx, outlines)
            sample['result'] = result
            processed_samples.append(sample)

    with open(os.path.join(data_root, output_filename), 'w') as f:
        for sample in processed_samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")
    



if __name__ == '__main__':
    create_searched_list('../data/datasets/wudao', 'part-2021009337.json.processed', 'train_searched_result.jsonl')