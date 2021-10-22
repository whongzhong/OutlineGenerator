#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

MODEL="Base_BART"
python ../src/Base.py \
--train \
--train_path="$HOME/Datasets/outgen/train.jsonl" \
--valid_path="$HOME/Datasets/outgen/valid.jsonl" \
--test_path="$HOME/Datasets/outgen/test.jsonl" \
--tokenizer_path="bert-base-chinese" \
--pretrain_path="$HOME/model/bart_zyfeng/bart-zyfeng" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--epoch=30 \
> ../log/Base.log 2>&1 &