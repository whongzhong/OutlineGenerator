import torch
from config import *
from utils import utils
import logging
# from torch._C import dtype
# from transformers.utils.dummy_pt_objects import BartForCausalLM, BartModel
from transformers import AutoTokenizer, GPT2LMHeadModel, BertTokenizer, AutoModelWithLMHead
from train import *
import json
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
import re

logging.getLogger().setLevel(logging.INFO)


def cut_story(story):
    end_set = {'”', "。", "！", "？", ""}
    story = story[:506]
    r_idx = story.rfind("<SENT>")
    if r_idx > 0:
        return story[:r_idx]
    
    #print(story)
    for i in range(505, -1, -1):
        if story[i] in end_set:
            return story[:i + 1]
        
    return story[:506]

class Example(object):
    def __init__(self, story, outline, replace_name=[], title=""):
        self.story = story
        self.outline = outline
        self.replace_name = replace_name
        self.un_cat_story = self.story
        if len(self.story) >= 506:
            self.story = cut_story(self.story)
        # self.title = title


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, Examples, tokenizer, args):
        super(BaseDataset, self).__init__()
        self.input_ids = []
        self.label = []
        self.output_ids = []
        self.output_mask = []
        self.args = args
        self.build(Examples, tokenizer)
 
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "label": self.label[idx],
            "output_ids": self.output_ids[idx],
            "output_mask": self.output_mask[idx]
        }
    
    def __len__(self):
        return len(self.input_ids)
    
    def build(self, Examples, tokenizer):
        for item in Examples:
            input = "[word]"
            # input = ""
            for idx, word in enumerate(item.outline):
                if self.args.inserted_keywords:
                    input += f"<WORD_{idx}>"
                else:
                    input += "<w>"
                input += word
            if self.args.replace_name:    
                replace_name = item.replace_name
                for idx, name in enumerate(replace_name):
                    real_name = list(name[0].keys())[0]
                    if real_name in input:
                        input = input.replace(real_name, f"<NAME_{idx}>")
                        item.story = item.story.replace(real_name, f"<NAME_{idx}>")
            input += "[SEP]"
            output = "<cls>" + input + item.story + "<eod>"
            # output = item.story + "[SEP]"
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            sep_idx = output_ids.index(tokenizer.convert_tokens_to_ids('[SEP]'))
            output_mask = [1] * len(output_ids)
            label = [-100] * (sep_idx + 1) + output_ids[sep_idx + 1:]
            self.input_ids.append(input_ids)
            self.label.append(label)
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
            "output_mask": [],
            "output_ids": [],
            "label": [],
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
            out["output_mask"][i] = out["output_mask"][i] + [0] * (output_max_pad - len(out["output_mask"][i]))
            out["output_ids"][i] = out["output_ids"][i] + [self.config["PAD_ID"]] * (output_max_pad - len(out["output_ids"][i]))
            out["label"][i] = out["label"][i] + [-100] * (output_max_pad - len(out["label"][i]))
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["output_mask"] = torch.tensor(out["output_mask"], dtype=torch.long)
        out["output_ids"] = torch.tensor(out["output_ids"], dtype=torch.long)
        out["label"] = torch.tensor(out["label"], dtype=torch.long)
        return out 


def read_json(path):
    data = utils.read_data(path)
    res = []
    for item in data:
        # utils.debug("item", item)
        res.append(json.loads(item))
    return res


def prepare_examples(path, args):
    data = read_json(path)
    Examples = []
    for item in data:
        if args.replace_name:
            Examples.append(Example(story=item["story"], outline=item["outline"], replace_name=item['names']))
        else:
            Examples.append(Example(story=item["story"], outline=item["outline"]))
    return Examples

def get_optimizer(args, model):

    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            # "lr": args.learning_rate * 0.1,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, args.learning_rate, correct_bias=False)
    
    return optimizer

def main(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
    logging.info("Load Data")
    train_data = prepare_examples(args.train_path, args)
    valid_data = prepare_examples(args.valid_path, args)
    args.gold = []
    args.gold_names = []
    args.outline = []
    for item in valid_data:
        args.gold.append(item.un_cat_story)
        args.outline.append(item.outline)
        if args.replace_name:
            args.gold_names.append(item.replace_name)
            
    # test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    args.n_gpu = torch.cuda.device_count()
    utils.debug("tokenizer", args.tokenizer_path)
    utils.debug("pretrain", args.pretrain_path)
    tokenizer = CpmTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = BartTokenizer.from_file(args.tokenizre_path)
    if args.model_load:
        model = torch.load(args.model_load)
    else:
        model = AutoModelWithLMHead.from_pretrained(args.pretrain_path)
    # utils.debug("model", model)
    # model = CPTForConditionalGeneration.from_pretrained(args.pretrain_path)
    # special_token = {"additional_special_tokens": ["[titile]"] + ["EOS"] + ["BOS"] + [f"<w{i}>" for i in range(8)]}
    if args.inserted_keywords:
        word_label = [f'<WORD_{idx}>' for idx in range(9)]
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] +["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"] + ["<SENT>"]+ word_label}
    elif args.replace_name:
        word_label = [f'<NAME_{idx}>' for idx in range(30)]
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] + ["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"] + word_label}
    else:
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] + ["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"]}
    tokenizer.add_special_tokens(special_token)
    word_token = ["“", "”"]
    tokenizer.add_tokens(word_token)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eod>"
    tokenizer.bos_token = "<cls>"
    # tokenizer.eos_token = "[eos]"
    # tokenizer.bos_token = "[bos]"
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.forced_eos_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.config.device = args.device
    logging.info(f"eos_token_id:{model.config.eos_token_id}")
    logging.info(f"bos_token_id:{model.config.bos_token_id}")
    logging.info(f"gpu num:{args.n_gpu}")
    # DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logging.info(f"local rank:{args.local_rank}")
    
    optimizer = get_optimizer(args, model)
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    #if args.local_rank != -1:
    #    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # if args.n_gpu > 1:
    #     model = DDP(model).
    # else:
    #     model = model.cuda()
    args.pad_id = tokenizer.pad_token_id
    logging.info("Prepare Dataset")
    train_dataset = BaseDataset(train_data, tokenizer, args)
    valid_dataset = BaseDataset(valid_data, tokenizer, args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = utils.SequentialDistributedSampler(valid_dataset, args.batch_size)
    args.valid_len = len(valid_dataset)
    # test_dataset = BaseDataset(test_data, tokenizer)
    
    test_data = prepare_examples(args.test_path, args)
    test_dataset = BaseDataset(test_data, tokenizer, args)
    args.test_len = len(test_dataset)
    test_sampler = utils.SequentialDistributedSampler(test_dataset, args.batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, collate_fn=Collection(args))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=Collection(args))
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=Collection(args))
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=Collection(args))
    logging.info("Start Training")
    
    args.parameter = utils.get_train_parameter()
    
        
    if args.cos_lr:
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter) // args.opt_step, num_training_steps=len(train_iter) * args.epoch // args.opt_step)
        
    
    model = model.to(args.device)
    model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     optimizer=optimizer,
                                                     lr_scheduler=scheduler)
    
    CPM_train(train_iter, valid_iter, test_iter, model, tokenizer, args, optimizer)


def predict(args):
    logging.info("Config Init")
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
    logging.info("Load Data")
    test_data = prepare_examples(args.test_path, args)
    args.gold = []
    args.outline = []
    args.gold_names = []
    for item in test_data:
        args.gold.append(item.un_cat_story)
        args.outline.append(item.outline)
        if args.replace_name:
            args.gold_names.append(item.replace_name)
    # test_data = prepare_examples(args.test_path)
    logging.info("Init Model and Tokenizer")
    args.n_gpu = torch.cuda.device_count()
    utils.debug("tokenizer", args.tokenizer_path)
    utils.debug("pretrain", args.pretrain_path)
    tokenizer = CpmTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = BartTokenizer.from_file(args.tokenizre_path)
    #model = deepspeed.DeepSpeedEngine.load_checkpoint(load_dir=args.model_load)
    model = torch.load(args.model_load).to(args.device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    #model = deepspeed.init_inference(model=model)
    
   # model = torch.load(args.model_load, map_location='cpus')
    #model = deepspeed.init_inference(model,
    #                             mp_size=2,
    #                             dtype=torch.half,
    #                             checkpoint=None,
    #                             replace_method='auto')
    #model, optimizer, _, _ = deepspeed.initialize(args=args,
    #                                                 model=model,
    #                                                 model_parameters=model.parameters())
    # utils.debug("model", model)
    # model = CPTForConditionalGeneration.from_pretrained(args.pretrain_path)
    # special_token = {"additional_special_tokens": ["[titile]"] + ["EOS"] + ["BOS"] + [f"<w{i}>" for i in range(8)]}
    if args.inserted_keywords:
        word_label = [f'<WORD_{idx}>' for idx in range(9)]
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] + ["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"] + ["<SENT>"] + word_label}
    elif args.replace_name:
        word_label = [f'<NAME_{idx}>' for idx in range(30)]
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] + ["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"] + word_label}
    else:
        special_token = {"additional_special_tokens": ["[titile]"] +['<eod>'] + ["[SEP]"] + ["<cls>"] + ["[word]"] + ["<w>"]}
    tokenizer.add_special_tokens(special_token)
    word_token = ["“", "”"]
    tokenizer.add_tokens(word_token)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eod>"
    tokenizer.bos_token = "<cls>"
    args.pad_id = tokenizer.pad_token_id
    test_dataset = BaseDataset(test_data, tokenizer, args)
    args.test_len = len(test_dataset)
    test_sampler = utils.SequentialDistributedSampler(test_dataset, args.test_batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, collate_fn=Collection(args))
    logging.info("Start predict")
    parameter_list = utils.get_CPM_parameter()
    continue_list = []
    args.output += f"_batch{args.test_batch_size}"
    args.step = 0
    with torch.no_grad():
        for idx, parameter in enumerate(parameter_list):
            if idx in continue_list:
                continue
            args.parameter = parameter
            args.step = idx
            Base_predict(test_iter, model, tokenizer, args)
    logging.info("END")


if __name__ == "__main__":
    args = Base_config()
    utils.set_seed(959794)
    if args.train:
        args.model_save = '/'.join([args.model_save, utils.d2s(datetime.datetime.now(), time=True)])
        args.output = '/'.join([args.output, utils.d2s(datetime.datetime.now(), time=True)])
        main(args)
    if args.predict:
        args.output = '/'.join([args.output, utils.d2s(datetime.datetime.now(), time=True)])
        predict(args)