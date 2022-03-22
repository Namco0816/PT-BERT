#! /bin/bash
path=$1
files=$(ls $path)

for filename in $files
do
    python evaluation.py \
    --model_name_or_path srl/$filename/ \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
done
