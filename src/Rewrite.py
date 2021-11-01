from threading import Condition
import torch
from transformers.utils.dummy_pt_objects import AutoModel
from config import *
from utils import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer, BartForConditionalGeneration
from train import *
import json
import random
import datetime
import copy
logging.getLogger().setLevel(logging.INFO)


class Raw_Example(object):
    def __init__(self, story, outline):
        self.story = story
        self.outline = outline
    
    def build(self):
        Examples = []
        sentences = utils.sentence_cut(self.story)
        mp = {i: [] for i in range(len(sentences))}
        for word in self.outline:
            for idx, sentence in enumerate(sentences):
                if word in sentence:
                    mp[idx].append(word)
        for k, v in mp.items():
            mp[k] = utils.reorder(sentences[k], v)
            if len(mp[k]) > 0:
                Examples.append(Example(up_content=utils.cat(sentences[:k]), dn_content=utils.cat(sentences[k+1:]) if k+1 != len(sentences) else "", \
                    outline=mp[k], target=sentences[k]))
        return Examples


class Example(object):
    def __init__(self, up_content, dn_content, outline, target, old_outline=[]):
        self.up_content = up_content
        self.dn_content = dn_content
        self.outline = outline
        self.target = target
        self.old_outline = old_outline
    
    def output(self):
        utils.debug("up_content", self.up_content)
        utils.debug("dn_content", self.dn_content)
        utils.debug("outline", self.outline)
        utils.debug("target", self.target)


class Rewrite(object):
    def __init__(self, outline, gold_story, predict_story):
        self.outline = outline
        self.gold_story = gold_story
        self.predict_story = predict_story
    
    def build(self, step):
        sentences = utils.sentence_cut(self.predict_story)
        mp = {i: [] for i in range(len(sentences))}
        for word in self.outline:
            for idx, sentence in enumerate(sentences):
                if word in sentence:
                    mp[idx].append(word)
        condition = []
        for k, v in mp.items():
            mp[k] = utils.reorder(sentences[k], v)
            if len(mp[k]) > 0:
                condition.append(k)
        # utils.debug("mp", mp)
        if len(condition) == 0:
            return None
        sorted(condition)
        # utils.debug("condition", condition)
        pos = max(len(condition) - 1 - step, 0)
        # pos = random.randint(0, len(condition) - 1)
        return Example(up_content=utils.cat(sentences[:condition[pos]]), dn_content=utils.cat(sentences[condition[pos]+1: ]) \
            if condition[pos]+1 != len(sentences) else "", outline=mp[condition[pos]], target=self.gold_story, old_outline=self.outline)
    
    def output(self):
        utils.debug("outline", self.outline)
        utils.debug("gold_story", self.gold_story)
        utils.debug("predict_story", self.predict_story)


class RewriteDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer):
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
    
    def build(self, Examples, tokenizer):
        for item in Examples:
            word_input = "[rewrite] [MASK]"
            for word in item.outline:
                word_input += word + "[MASK]"
            word_input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_input))
            l = 0
            r = 0
            up_input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item.up_content))
            dn_input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item.dn_content))
            while len(word_input_ids) + len(up_input_ids) - l + len(dn_input_ids) - r + 3 >= 500:
                if len(up_input_ids) - l > len(dn_input_ids) - r:
                    l += 1
                else:
                    r += 1
            # up_input = "[up_content]" + item.up_content[l:]
            # utils.debug("up_input_ids", up_input_ids)
            up_input_ids = tokenizer.convert_tokens_to_ids(["[up_content]"]) + up_input_ids[l:]
            # utils.debug("up_input_ids_", up_input_ids)
            # utils.debug("dn_input_ids", dn_input_ids)
            # utils.debug("dn_content", tokenizer.convert_tokens_to_ids(["[dn_content]"]))
            dn_input_ids = tokenizer.convert_tokens_to_ids(["[dn_content]"]) + (dn_input_ids[:-r] if r != 0 else dn_input_ids) +\
                tokenizer.convert_tokens_to_ids(["[eos]"])
            # dn_input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dn_input))
            input_ids = up_input_ids + word_input_ids + dn_input_ids
            output = item.target + "[eos]"
            input_mask = [1] * len(input_ids)
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            if len(output_ids) > 500:
                logging.warning(f"target: {item.target}")
                logging.warning(f"outline: {item.outline}")
                logging.warning(f"up_content: {item.up_content}")
                logging.warning(f"dn_content: {item.dn_content}")
                continue
            output_mask = [1] * len(output_ids)
            self.input_ids.append(input_ids)
            self.input_mask.append(input_mask)
            self.output_ids.append(output_ids)
            self.output_mask.append(output_mask)


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
            "output_mask": []
        }
        for minibatch in batch:
            for k, v in minibatch.items():
                out[k].append(v)
        input_max_pad = 0
        output_max_pad = 0
        if self.config["BUCKET"]:
            for p in out["input_ids"]:
                input_max_pad = max(input_max_pad, len(p))
            for p in out["output_ids"]:
                output_max_pad = max(output_max_pad, len(p))
        for i in range(len(batch)):
            out["input_ids"][i] = out["input_ids"][i] + [self.config["PAD_ID"]] * (input_max_pad - len(out["input_ids"][i]))
            out["input_mask"][i] = out["input_mask"][i] + [0] * (input_max_pad - len(out["input_mask"][i]))
            out["output_ids"][i] = out["output_ids"][i] + [-100] * (output_max_pad - len(out["output_ids"][i]))
            out["output_mask"][i] = out["output_mask"][i] + [0] * (output_max_pad - len(out["output_mask"][i]))
        for k, v in out.items():
            out[k] = torch.tensor(v, dtype=torch.long)
        return out


def prepare_data(path):
    data = utils.read_data(path)
    Examples = []
    for item in data:
        item = json.loads(item)
        raw_example = Raw_Example(story=item["story"], outline=item["outline"])
        Examples.extend(raw_example.build())
    return Examples


def prepare_rewrite_data(path, step):
    data = utils.read_data(path)
    outline, gold_story, predict_story = [], "", ""
    gold_flag, predict_flag = False, False
    Examples = []
    Error_Examples = dict()
    for line in data:
        if line.startswith("outline"):
            outline = line.split(":")[1].strip()[:-1].replace(" ","").split(",")
        elif line.startswith("gold"):
            gold_flag = True
        elif line.startswith("predict"):
            predict_flag = True
        elif line.startswith("---"):
            add_example = Rewrite(outline=outline, gold_story=gold_story, predict_story=predict_story).build(step)
            if add_example is not None:
                Examples.append(add_example)
            else:
                if Error_Examples.get(len(Examples), None) is None:
                    Error_Examples[len(Examples)] = [Rewrite(outline=outline, gold_story=gold_story, predict_story=predict_story)]
                else:
                    Error_Examples[len(Examples)].append(Rewrite(outline=outline, gold_story=gold_story, predict_story=predict_story))
            outline = []
            gold_story = ""
            predict_story = ""
        elif line.startswith("bleu-1"):
            break
        else:
            if gold_flag:
                gold_flag = False
                gold_story = line.strip()
            elif predict_flag:
                predict_flag = False
                predict_story = line.strip()
    return Examples, Error_Examples


def main(args):
    logging.info("Prepare Data")
    train_examples = prepare_data(args.train_path)
    valid_examples = prepare_data(args.valid_path)
    train_examples[0].output()
    args.gold = []
    args.outline = []
    for item in valid_examples:
        args.gold.append(item.target)
        args.outline.append(item.outline)
    logging.info("Init Pre-train Model and Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = BartForConditionalGeneration.from_pretrained(args.pretrain_path)
    args.n_gpu = torch.cuda.device_count()
    special_token = {"additional_special_tokens": ["[up_content]", "[dn_content]", "[rewrite]"] + ["[eos]"] + ["[bos]"] + ["[MASK]"]}
    # word_token = [f"<w{i}>" for i in range(8)]
    # special_token["additional_special_tokens"].extend(word_token)
    tokenizer.add_special_tokens(special_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[eos]"
    tokenizer.bos_token = "[bos]"
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    logging.info(len(tokenizer))
    logging.info(model.config.vocab_size)
    model.resize_token_embeddings(len(tokenizer))
    logging.info(model.config.vocab_size)
    args.pad_id = tokenizer.pad_token_id
    logging.info(f"gpu num:{args.n_gpu}")
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
        utils.debug("model", model)
        # model.cpu()
    # tokenizer.save(args.toeknizer_save)
    logging.info("Prepare DataLoader")
    train_dataset = RewriteDataset(train_examples, tokenizer)
    valid_dataset = RewriteDataset(valid_examples, tokenizer)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info(f"Train Loader len:{len(train_iter)}")
    Rewrite_train(train_iter, valid_iter, model, tokenizer, args)


def rewrite(args):
    # args.rewrite_save = args.rewrite_path.split(".")[0] + "_new.txt"
    test_data, error_data = prepare_rewrite_data(args.rewrite_path, args.step)
    args.gold = copy.deepcopy(test_data)
    args.error_data = error_data
    args.gold[0].output()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = torch.load(args.model_path)
    args.n_gpu = torch.cuda.device_count()
    special_token = {"additional_special_tokens": ["[up_content]", "[dn_content]", "[rewrite]"] + ["[eos]"] + ["[bos]"] + ["[MASK]"]}
    # word_token = [f"<w{i}>" for i in range(8)]
    # special_token["additional_special_tokens"].extend(word_token)
    tokenizer.add_special_tokens(special_token)
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[eos]"
    tokenizer.bos_token = "[bos]"
    logging.info(len(tokenizer))
    args.pad_id = tokenizer.pad_token_id
    logging.info(f"gpu num:{args.n_gpu}")
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
        utils.debug("model", model)
        # model.cpu()
    logging.info("Prepare DataLoader")
    test_dataset = RewriteDataset(test_data, tokenizer)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    Rewrite_predict(test_iter, model, tokenizer, args)
    args.rewrite_path = args.rewrite_save

if __name__ == "__main__":
    utils.set_seed(42)
    args = Rewrite_config()
    time_now = utils.d2s(datetime.datetime.now(), time=True)
    # args.tokenizer_save = "/".join([args.tokenizer_save, "tokenizer" + time_now + ".json"])
    if args.train:
        args.model_save = "/".join([args.model_save, time_now])
        main(args)
    elif args.rewrite:
        args.step = 0
        while args.step < 5:
            rewrite(args)
            args.step += 1