#!/bin/bash
python evaluation.py \
    --model_name_or_path result/poly128lr3e-5bs64ml64qs4tpbeta0.885 \
    --pooler cls_before_pooler \
    --task_set full \
    --mode test