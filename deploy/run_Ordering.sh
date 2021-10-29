#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../src/Ordering.py \
--build_data \
--train_path="$HOME/Datasets/LOT/data/train.jsonl" \
--valid_path="$HOME/Datasets/LOT/data/val.jsonl" \
--test_path="$HOME/Datasets/chinese_tonghua/chinese_tonghua_etstory_clean_2_outline.jsonl" \
--train_save="$HOME/Datasets/LOT/data/train_order.jsonl" \
--valid_save="$HOME/Datasets/LOT/data/valid_order.jsonl" \
--test_save="$HOME/Datasets/LOT/data/extral_order.jsonl" \
> ../log/Ordering.log 2>&1 &