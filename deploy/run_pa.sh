#!/bin/sh
source ~/.bashrc
source activate telma
export PYTHONPATH="$HOME/opt/tiger/polish"

python ../data_prepare/pa.py