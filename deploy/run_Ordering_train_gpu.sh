#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

MODEL="Order_BART"
TOKENIZER="bert-base-chinese"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
# TOKENIZER="fnlp/bart-large-chinese"
# PRETRAIN="fnlp/bart-large-chinese"
PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"



python ../src/Ordering.py \
--train \
--train_path="$HOME/Datasets/LOT/data/train_order.jsonl" \
--valid_path="$HOME/Datasets/LOT/data/valid_order.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.0003 \
--epoch=30 \
# > ../log/Ordering.log 2>&1 &