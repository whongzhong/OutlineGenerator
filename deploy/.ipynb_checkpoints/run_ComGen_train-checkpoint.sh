#!/bin/sh

export PYTHONPATH="$HOME/opt/tiger/polish"

python ../src/ComGen.py \
--train \
--train_path="$HOME/opt/tiger/polish/data/ROCStory_train.tsv" \
--valid_path="$HOME/opt/tiger/polish/data/ROCStory_valid.tsv" \
--test_path="$HOME/opt/tiger/polish/data/ROCStory_test.tsv" \
--model_save="$HOME/opt/tiger/polish/model/ComGen_BART" \
--learning_rate=0.00003 \
--batch_size=2 \
--epoch=16 \
--mini_test \
> ../log/Comgen.log 2>&1 &