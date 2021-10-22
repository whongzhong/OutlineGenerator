import torch
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
logging.getLogger().setLevel(logging.INFO)


class Example(object):
    def __init__(self, up_contents, dn_contents, outline):
        self.up_contents = up_contents
        self.dn_contents = dn_contents
        self.outline = outline


def prepare_data(path, args):
    
    raise


def main(args):
    raise


if __name__ == "__main__":
    utils.set_seed(19980917)
    args = Rewrite_config()
    if args.train:
        main(args)