#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
# TOKENIZER="fnlp/bart-large-chinese"
# PRETRAIN="fnlp/bart-large-chinese"
PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
MODEL="Base_BART"
python ../src/Base.py \
--train \
--train_path="$HOME/Datasets/outgen/train.jsonl" \
--valid_path="$HOME/Datasets/outgen/valid.jsonl" \
--test_path="$HOME/Datasets/outgen/test.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$HOME/model/bart_zyfeng/bart-zyfeng" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--batch_size=2 \
--epoch=30 \
> ../log/Base.log 2>&1 &