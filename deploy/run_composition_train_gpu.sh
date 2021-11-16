#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/userhome/whzhong/code/polish
export LC_ALL=C.UTF-8


exec 1>$PYTHONPATH/info/composition_open_correct.out
exec 2>$PYTHONPATH/info/composition_open_correct.error 

# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="$PYTHONPATH/data/models/CBART"
PRETRAIN="$PYTHONPATH/data/models/CBART"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
TRAIN_PATH="$PYTHONPATH/data/datasets/LOTdatasets/composition_open/train.jsonl"
# TRAIN_PATH="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train.jsonl"

/userhome/anaconda3/envs/lot10/bin/python -m torch.distributed.launch --nproc_per_node 8 ../src/Base.py \
--train \
--train_path="$TRAIN_PATH" \
--valid_path="$PYTHONPATH/data/datasets/LOTdatasets/val.jsonl" \
--tokenizer_path="$TOKENIZER" \
--pretrain_path="$PRETRAIN" \
--model_save="$PYTHONPATH/ckpts/composition_open" \
--learning_rate=0.00003 \
--batch_size=16 \
--epoch=30 \
--opt_step=1 \
# --model_load="$HOME/opt/tiger/polish/model/Base_BART/LOT/2021_10_27_23_25_epoch27.pkl" \
# > ../log/Base.log 2>&1 &