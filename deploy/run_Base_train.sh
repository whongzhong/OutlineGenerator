#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

MODEL="Base_BART"
# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
TRAIN_PATH="$HOME/Datasets/LOT/data/train.jsonl"

python ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--batch_size=2 \
--epoch=30 \
--opt_step=2 \
> ../log/Base.log 2>&1 &