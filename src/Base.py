import torch
from config import *
from utils import utils
import logging
from transformers import AutoTokenizer, BartForConditionalGeneration
from train import *
import json
import metrics
import random
import datetime
logging.getLogger().setLevel(logging.INFO)


class Example(object):
    def __init__(self, story, outline, title):
        self.story = story
        self.outline = outline
        self.title = title


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.input_mask = []
        self.output_ids = []
        self.output_mask = []
        self.build(Examples, tokenizer)
 
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "output_ids": self.output_ids[idx],
            "output_mask": self.output_mask[idx]
        }
    
    def __len__(self):
        return len(self.input_ids)
    
    def build(Examples, tokenizer):
        for item in Examples:
            input = "[title]" + item.title + "[word]"
            for word in item.outline:
                input += "<w>"
                input += word
            input += "[EOS]"
            output = "[BOS]" + item.stroy
            input_ids = tokenizer.covert_tokens_to_ids(tokenizer.tokenize(input))
            input_mask = [1] * len(input_ids)
            output_ids = tokenizer.conver_tokens_to_ids(tokenizer.tokenize(output))
            output_mask = [1] * len(input_ids)


class Collection(object):
    def __init__(self, args):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = args.fix_length
        self.config["PAD_ID"] = args.pad_id

    def __call__(self, batch):
        out = {
            "input_ids": [],
            "input_mask": [],
            "output_ids": [],
            "output_mask": [],
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        input_max_pad = 0
        output_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["output_ids"]:
                output_max_pad = max(output_max_pad, len(p))
        else:
            input_max_pad = self.config["FIX_LENGTH"]
            output_max_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_ids"][i]))
            out["output_ids"][i] = out["output_ids"][i] + [-100] * (output_max_pad - len(out["output_ids"][i]))
            out["output_mask"][i] = out["output_mask"][i] + [0] * (output_max_pad - len(out["output_mask"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["input_mask"] = torch.tensor(out["input_mask"], dtype=torch.long)
        out["output_ids"] = torch.tensor(out["output_ids"], dtype=torch.long)
        out["output_mask"] = torch.tensor(out["output_mask"], dtype=torch.long)
        return out 


def read_json(path):
    data = utils.read_data(path)
    res = []
    for item in data:
        # utils.debug("item", item)
        res.append(json.loads(item))
    return res


def prepare_examples(path):
    data = read_json(path)
    Examples = []
    for item in data:
        Examples.append(Example(story=item["story"], outline=item["outline"], title=item["title"]))
    return Examples


def main(args):
    train_data = prepare_examples(args.train_path)
    valid_data = prepare_examples(args.valid_path)
    test_data = prepare_examples(args.test_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = BartForConditionalGeneration(args.pretrain_path)
    # special_token = {"additional_special_tokens": ["[titile]"] + ["EOS"] + ["BOS"] + [f"<w{i}>" for i in range(8)]}
    special_token = {"additional_special_tokens": ["[titile]"] + ["[EOS]"] + ["[BOS]"] + ["[word]"] + ["<w>"]}
    tokenizer.add_special_tokens(special_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.bos_token = "[BOS]"
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    args.pad_id = tokenizer.pad_token_id
    train_dataset = BaseDataset(train_data, tokenizer)
    valid_dataset = BaseDataset(valid_data, tokenizer)
    test_dataset = BaseDataset(test_data, tokenizer)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    Base_train(train_iter, valid_iter, model, args)
    

if __name__ == "__main__":
    args = Base_config()
    if args.train:
        main(args)