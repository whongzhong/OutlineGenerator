#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

MODEL="Order_BART"
python ../src/Ordering.py \
--train \
--train_path="$HOME/opt/tiger/polish/data/LOT/ordering_train.jsonl" \
--valid_path="$HOME/opt/tiger/polish/data/LOT/ordering_valid.jsonl" \
--test_path="$HOME/opt/tiger/polish/data/LOT/ordering_test.jsonl" \
--tokenizer_path="bert-base-chinese" \
--pretrain_path="$HOME/model/bart_zyfeng/bart-zyfeng" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
> ../log/Ordering.log 2>&1 &