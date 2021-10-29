import torch
from config import *
from utils import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer, BartForConditionalGeneration, BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
from model import OrderBartForConditionalGeneration
from train import *
import json
import metrics
import eval
import random
import datetime
logging.getLogger().setLevel(logging.INFO)


class Example(object):
    def __init__(self, story="", outline=[], order=[]):
        self.story = story
        self.outline = outline
        self.un_cat_story = self.story
        self.order = order
        if len(self.story) >= 505:
            self.story = self.story[:500]
        # self.title = title


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, is_train):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.input_mask = []
        self.output_ids = []
        self.output_mask = []
        self.encoder_labels = []
        self.encoder_labels_idx = []
        self.encoder_labels_mask = []
        self.tokenizer = tokenizer
        self.is_train = is_train
        if is_train:
            self.build(Examples)
        else:
            self.build_eval(Examples)
 
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "input_mask": self.input_mask[idx],
            "output_ids": self.output_ids[idx],
            "output_mask": self.output_mask[idx],
            "encoder_labels": self.encoder_labels[idx],
            "encoder_labels_idx": self.encoder_labels_idx[idx],
            "encoder_labels_mask": self.encoder_labels_mask[idx]
        }
    
    def __len__(self):
        return len(self.input_ids)


    def _convert(self, s):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s))

    
    def build(self, Examples):
        for item in Examples:
            input_ids = self._convert("[word]")
            word_label_idx = []
            for idx, word in enumerate(item.outline):
                word_ids = self._convert(f"<w{idx}>" + word)
                word_label_idx.append(len(input_ids))
                input_ids += word_ids
            input_ids += self._convert("[SEP]")
            input_mask = [1] * len(input_ids)
            output_ids = self._convert("[CLS]" + item.story + "[SEP]")
            output_mask = [1] * len(output_ids)
            encoder_labels_idx = []
            encoder_labels = []
            for i in range(len(word_label_idx)):
                for j in range(i+1, len(word_label_idx)):
                    encoder_labels_idx.append((word_label_idx[i], word_label_idx[j]))
                    encoder_labels.append(1 if self.order[i]<self.order[j] else 0)
            encoder_labels_mask = [1] * len(encoder_labels)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.output_ids.append(output_ids)
            self.output_mask.append(output_mask)
            self.encoder_labels.append(encoder_labels)
            self.encoder_labels_idx.append(encoder_labels_idx)
            self.encoder_labels_mask.append(encoder_labels_mask)


    def build_eval(self, Examples):
        for item in Examples:
            input_ids = self._convert("[word]")
            for idx, word in enumerate(item.outline):
                word_ids = self._convert(f"<w{idx}>" + word)
                input_ids += word_ids
            input_ids += self._convert("[SEP]")
            input_mask = [1] * len(input_ids)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.output_ids.append([])
            self.output_mask.append([])
            self.encoder_labels.append([])
            self.encoder_labels_idx.append([])
            self.encoder_labels_mask.append([])


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
            "encoder_labels": [],
            "encoder_labels_ids": [],
            "encoder_labels_mask": []
        }
        for mini_batch in batch:
            for k, v in mini_batch.items():
                out[k].append(v)
        input_max_pad = 0
        output_max_pad = 0
        encoder_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["output_ids"]:
                output_max_pad = max(output_max_pad, len(p))
            for p in out["encoder_lables_ids"]:
                encoder_max_pad = max(encoder_max_pad, len(p))
        else:
            input_max_pad = self.config["FIX_LENGTH"]
            output_max_pad = self.config["FIX_LENGTH"]
            encoder_max_pad = self.config["FIX_LENGTH"]
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_mask"][i]))
            out["output_ids"][i] = out["output_ids"][i] + [-100] * (output_max_pad - len(out["output_ids"][i]))
            out["output_mask"][i] = out["output_mask"][i] + [0] * (output_max_pad - len(out["output_mask"][i]))
            out["encoder_labels"][i] = out["encoder_labels"][i] + [-100] * (encoder_max_pad - len(out["encoder_labels"][i]))
            out["encoder_labels_ids"][i] = out["encoder_labels_ids"][i] + [(-1, -1)] * (encoder_max_pad - len(out["encoder_labels_ids"][i]))
            out["encoder_labels_mask"][i] = out["encoder_labels_mask"][i] + [0] * (encoder_max_pad - len(out["encoder_labels_mask"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["input_mask"] = torch.tensor(out["input_mask"], dtype=torch.long)
        out["output_ids"] = torch.tensor(out["output_ids"], dtype=torch.long)
        out["output_mask"] = torch.tensor(out["output_mask"], dtype=torch.long)
        out["encoder_labels"] = torch.tensor(out["encoder_labels"], dtype=torch.long)
        out["encoder_labels_mask"] = torch.tensor(out["encoder_labels_mask"], dtype=torch.long)
        return out 


def read_json(path):
    data = utils.read_data(path)
    res = []
    for item in data:
        # utils.debug("item", item)
        res.append(json.loads(item))
    return res


def prepare_examples(path, is_train=True):
    data = read_json(path)
    Examples = []
    for item in data:
        if is_train:
            Examples.append(Example(story=item["story"], outline=item["outline"], order=item["order"]))
        else:
            Examples.append(Example(story=item["story"], outline=item["outline"]))
    return Examples


def main(args):
    logging.info("Load Data")
    train_data = prepare_examples(args.train_path)
    valid_data = prepare_examples(args.valid_path)
    args.gold = []
    args.outline = []
    for item in valid_data:
        args.gold.append(item.un_cat_story)
        args.outline.append(item.outline)
    # test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    args.n_gpu = torch.cuda.device_count()
    utils.debug("tokenizer", args.tokenizer_path)
    utils.debug("pretrain", args.pretrain_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    model = OrderBartForConditionalGeneration.from_pretrained(args.pretrain_path)
    special_token = {"additional_special_tokens": ["[titile]"] + ["[eos]"] + ["[bos]"] + ["[word]"]}
    word_token = [f"<w{i}>" for i in range(8)]
    special_token["additional_special_tokens"].extend(word_token)
    tokenizer.add_special_tokens(special_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.bos_token = "[CLS]"
    # tokenizer.eos_token = "[eos]"
    # tokenizer.bos_token = "[bos]"
    # model.config.decoder_start_token_id = tokenizer.bos_token_id
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.bos_token_id = tokenizer.bos_token_id
    # model.config.forced_eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    logging.info(f"gpu num:{args.n_gpu}")
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    args.pad_id = tokenizer.pad_token_id
    logging.info("Prepare Dataset")
    train_dataset = BaseDataset(train_data, tokenizer)
    valid_dataset = BaseDataset(valid_data, tokenizer)
    # test_dataset = BaseDataset(test_data, tokenizer)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=3, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start Training")
    OrderBase_train(train_iter, valid_iter, model, tokenizer, args)
    

if __name__ == "__main__":
    args = Base_config()
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        main(args)