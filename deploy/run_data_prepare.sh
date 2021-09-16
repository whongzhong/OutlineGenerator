#!/bash/sh
python ../data_prepare/data_prepare.py \
--dir_path="/Users/zhanglingyuan/Datasets/Gutenberg/txt" \
--save_path="/Users/zhanglingyuan/opt/tiger/polish/data/Gutenberg.txt" \
> ../log/data_prepare.log 2>&1 &