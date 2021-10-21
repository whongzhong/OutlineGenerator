import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartModel


class Order_BART(nn.Module):
    def __init__(self, args):
        super(Order_BART, self).__init__()
        self.BART = BartModel.from_pretrained(args.pretrain_path)
        self.lm = nn.Linear(1024, 9)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        self.BART()