#!/bin/sh
MAX_LENGTH=50
LABEL="fiction"
export PYTHONPATH="$HOME/opt/tiger/polish"
python ../data_prepare/data_prepare.py \
--dir_path="$HOME/Datasets/bookcroups/books1/epubtxt" \
--cropus_type="bookcroups" \
--save_path="$HOME/opt/tiger/polish/data/bookcroups_$MAX_LENGTH.txt" \
--max_length=$MAX_LENGTH \
--prepare_rawdata \
> ../log/data_prepare.log 2>&1 &
