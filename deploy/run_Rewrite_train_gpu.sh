#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"
export CUDA_VISIBLE_DEVICES="0,1,2"

MODEL="Rewrite"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
TOKENIZER="hfl/chinese-roberta-wwm-ext"
# TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"

python ../src/Rewrite.py \
--train \
--train_path="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--tokenizer_save="$HOME/opt/tiger/polish/model/$MODEL/tokenizer" \
--pretrain_path="$PRETRAIN" \
--model_save="$HOME/opt/tiger/polish/model/$MODEL" \
--learning_rate=0.00003 \
--batch_size=6 \
--epoch=30 \
# > ../log/Base.log 2>&1 &