import torch
import os
import argparse
import random
import logging
import stanza
import datetime
from utils import utils
from config import *
import train
from transformers import AutoTokenizer, BartForConditionalGeneration
import csv
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)
seed = 19980917
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Example(object):
    def __init__(self, sentences=["", "", "", "", ""]):
        self.up_contents = sentences[0] + " " + sentences[1]
        self.dn_contents = sentences[3] + " " + sentences[4]
        self.target = sentences[2]
        self.concepts = []

    def add(self, up_contents, dn_contents, target, concepts):
        self.up_contents = up_contents
        self.dn_contents = dn_contents
        self.target = target
        self.concepts = concepts

    def build_concep(self, nlp):
        doc = nlp(self.target)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos == "PROPN" or word.upos == "NOUN":
                    self.concepts.append(word.lemma)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer):
        super(Dataset, self).__init__()
        self.label = []
        self.input_ids = []
        self.attention_mask = []
        self.build(Examples, tokenizer)

    def __getitem__(self, idx):
        return {
            "label": self.label[idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

    def __len__(self):
        return len(self.label)

    def build(self, Examples, tokenizer):
        for item in Examples:
            up_content_tk = tokenizer(item.up_contents)
            dn_content_tk = tokenizer(item.dn_contents)
            concepts_tks = [tokenizer(word) for word in item.concepts]
            label_tk = tokenizer(item.target)
            concepts_input_ids, concepts_attention_mask = [], []
            for concepts_tk in concepts_tks:
                concepts_input_ids.extend(concepts_tk.input_ids)
                concepts_attention_mask.extend(concepts_tk.attention_mask)
            self.input_ids.append(up_content_tk.input_ids + concepts_input_ids + dn_content_tk.input_ids)
            self.attention_mask.append(up_content_tk.attention_mask + concepts_attention_mask + dn_content_tk.attention_mask)
            self.label.append(label_tk.input_ids)


class Collection(object):
    def __init__(self, FIX_LENGTH):
        self.config = {}
        self.config["BUCKET"] = True
        self.config["FIX_LENGTH"] = FIX_LENGTH
    
    def __call__(self, batch):
        out = {
            "input_ids": [],
            "attention_mask": [],
            "label": []
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        max_pad = 0
        max_label_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                max_pad = max(max_pad, len(p))
            for p in out["label"]:
                max_label_pad = max(max_label_pad, len(p))
        else:
            max_pad = self.config["FIX_LENGTH"]
            max_label_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            txt_len = len(out["input_ids"][i])
            label_len = len(out["label"][i])
            out["input_ids"][i] += [0] * (max_pad - txt_len)
            out["attention_mask"][i] += [0] * (max_pad - txt_len)
            out["label"][i] += [-100] * (max_label_pad - label_len)
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        out["label"] = torch.tensor(out["label"], dtype=torch.long)
        return out


def draw_data(path, Examples):
    with open(path, "w", newline="") as f:
        tsv_w = csv.writer(f, delimiter="\t")
        tsv_w.writerow(["id", "up_contents", "dn_contents", "concepts", "target"])
        for idx, item in enumerate(Examples):
            tsv_w.writerow([idx, item.up_contents, item.dn_contents, utils.list2str(item.concepts), item.target])


def build_data(path, args):
    data = utils.read_data(path)
    Examples = []
    nlp = stanza.Pipeline("en")
    logging.info("len: {}".format(len(data)))
    for idx, item in enumerate(data):
        # logging.info("ids: {}".format(idx))
        utils.debug("ids", idx)
        if idx > 0:
            item = item.strip().split(",")
            Examples.append(Example(item[2:]))
            Examples[-1].build_concep(nlp)
    random.shuffle(Examples)
    train_Examples = Examples[:int(len(Examples) * 0.9)]
    valid_Examples = Examples[int(len(Examples) * 0.9):int(len(Examples) * 0.95)]
    test_Examples = Examples[int(len(Examples) * 0.95):]
    draw_data(args.train_save, train_Examples)
    draw_data(args.valid_save, valid_Examples)
    draw_data(args.test_save, test_Examples)
    # tokenizer = AutoTokenizer(args.pre_train)
    # train_dataset = Dataset(train_Examples)


def prepare_Examples(path):
    Examples = []
    data = utils.read_data(path)
    for idx, item in enumerate(data):
        if idx == 0:
            continue
        item = item.strip().split("\t")
        add_Example = Example()
        # utils.debug("item", item)
        add_Example.add(up_contents=item[0], dn_contents=item[1], target=item[2], concepts=item[3].split(","))
        # utils.debug("target", add_Example.target)
        # utils.debug("concepts", add_Example.concepts)
        Examples.append(add_Example)
    return Examples


def main(args):
    args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
    train_Examples = prepare_Examples(args.train_path)
    valid_Examples = prepare_Examples(args.valid_path)
    test_Examples = prepare_Examples(args.test_path)
    if args.mini_test:
        train_Examples = train_Examples[:100]
        valid_Examples = valid_Examples[:100]
        test_Examples = test_Examples[:100]
    logging.info("Finish read data")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_cp)
    train_dataset = Dataset(train_Examples, tokenizer)
    valid_dataset = Dataset(valid_Examples, tokenizer)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=Collection(args.fix_length), shuffle=True)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args.fix_length))
    logging.info("Finish prepare DataLoader")
    model = BartForConditionalGeneration.from_pretrained(args.pretrain_cp).cuda()
    reals = [item.target for item in valid_Examples]
    args.reals = reals
    train.ComGen_train(train_iter, valid_iter, model, tokenizer, args)


if __name__ == "__main__":
    args = ComGen_config()
    if args.build_data:
        build_data(args.train_path, args)
    if args.train:
        main(args)
