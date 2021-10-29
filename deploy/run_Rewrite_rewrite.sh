#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
TOKENIZER="hfl/chinese-roberta-wwm-ext"
# TOKENIZER="fnlp/bart-large-chinese"
PRETRAIN="fnlp/bart-large-chinese"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
MODEL="Rewrite"
python ../src/Rewrite.py \
--rewrite \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_path="$HOME/opt/tiger/polish/model/Rewrite/2021_10_25_22_45_epoch18.pkl" \
--rewrite_path="$HOME/opt/tiger/polish/model/Base_BART/LOT/2021_10_22_01_21_epoch15.txt" \
--rewrite_save="$HOME/opt/tiger/polish/log/rewrite/tmp.txt" \
--batch_size=2 \
> ../log/Rewrite_rewrite.log 2>&1 &