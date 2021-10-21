#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../src/Ordering.py \
--build_data \
--train_path="$HOME/Datasets/outgen/train.jsonl" \
--valid_path="$HOME/Datasets/outgen/valid.jsonl" \
--test_path="$HOME/Datasets/outgen/test.jsonl" \
--train_save="$HOME/opt/tiger/polish/data/LOT/ordering_train.jsonl" \
--valid_save="$HOME/opt/tiger/polish/data/LOT/ordering_valid.jsonl" \
--test_save="$HOME/opt/tiger/polish/data/LOT/ordering_test.jsonl" \
> ../log/Ordering.log 2>&1 &