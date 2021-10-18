#!/bin/sh
MAX_LENGTH=50
LABEL="news"
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../data_prepare/data_prepare.py \
--rawdata_path="$HOME/opt/tiger/polish/data/Brown_${LABEL}_$MAX_LENGTH.txt" \
--minidata_save_path="$HOME/opt/tiger/polish/data/minidata" \
--minidata_size=400 \
--prepare_minidata \
> ../log/data_prepare.log 2>&1 &