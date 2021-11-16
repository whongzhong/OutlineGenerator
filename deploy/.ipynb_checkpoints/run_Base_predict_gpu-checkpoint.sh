#!/bin/bash

# 输入要执行的命令，例如 ./hello 或 python test.py 等
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/userhome/whzhong/code/polish
export LC_ALL=C.UTF-8


exec 1>$PYTHONPATH/info/test_cmrc_keywords.out
exec 2>$PYTHONPATH/info/test_cmrc_keywords.error 

# TOKENIZER="fnlp/cpt-large"
# TOKENIZER="$HOME/model/bart_zyfeng/bart-zyfeng"
# TOKENIZER="hfl/chinese-roberta-wwm-ext"
TOKENIZER="$PYTHONPATH/data/models/CBART"
PRETRAIN="$PYTHONPATH/data/models/CBART"
# PRETRAIN="$HOME/model/bart_zyfeng/bart-zyfeng"
# PRETRAIN="fnlp/cpt-large"

# TRAIN_PATH="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline_2.jsonl"
TRAIN_PATH="$PYTHONPATH/data/datasets/LOTdatasets/permute_data/10x/train.jsonl"
# TRAIN_PATH="$HOME/Datasets/LOT_datasets_and_models/data/datasets/LOTdatasets/permute/train.jsonl"

/userhome/anaconda3/envs/lot10/bin/python -m torch.distributed.launch --nproc_per_node 2 ../src/Base.py \
--predict \
--inserted_keywords \
--test_path="$PYTHONPATH/data/datasets/LOTdatasets/test.jsonl" \
--tokenizer_path="$TOKENIZER" \
--model_load="$PYTHONPATH/ckpts/cmrc_keywords/2021_11_10_00_13_epoch39.pkl" \
--batch_size=7 \
--output="$PYTHONPATH/outputs/test/keywords" \
--ans_list="$PYTHONPATH/outputs/ans_list.txt" \
# > ../log/Base_predict.log 2>&1 &