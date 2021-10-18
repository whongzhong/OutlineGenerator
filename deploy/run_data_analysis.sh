#!/bin/sh
MAX_LENGTH=50
LABEL="fiction"
NAME="bookcroups"
export PYTHONPATH="$HOME/opt/tiger/polish"
python ../data_prepare/data_analysis.py \
--data_path="$HOME/opt/tiger/polish/data/${NAME}_$MAX_LENGTH.txt" \
--savefig_path="$HOME/opt/tiger/polish/log/pictures/" \
> ../log/data_analysis.log 2>&1 &