#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1,2"

MODEL="Base_BART/LOT"
TOKENIZER="bert-base-chinese"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
# TOKENIZER="fnlp/bart-large-chinese"
# PRETRAIN="fnlp/bart-large-chinese"
PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"

python ../src/Base.py \
--train \
--train_path="$HOME/Datasets/LOT/data/train.jsonl" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.000003 \
--batch_size=8 \
--epoch=30 \
# > ../log/Base.log 2>&1 &