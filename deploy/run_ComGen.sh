#!/bin/sh
# source ~/.bashrc
# source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../src/ComGen.py \
--build_data \
--train_path="$HOME/Datasets/ROCStory/ROCStoryies2017.csv" \
--train_save="$HOME/opt/tiger/polish/data/ROCStory_train.tsv" \
--valid_save="$HOME/opt/tiger/polish/data/ROCStory_valid.tsv" \
--test_save="$HOME/opt/tiger/polish/data/ROCStory_test.tsv" \
# > ../log/Comgen.log 2>&1 &