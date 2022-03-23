lr=3e-5
poly=128
batch_size=64
max_len=64
queue_size=4
temp=0.05
mo_beta=0.885

checkpoint="poly${poly}lr${lr}bs${batch_size}ml${max_len}qs${queue_size}tp${temp}beta${mo_beta}"
python train.py \
--model_name_or_path bert-base-uncased \
--cl_type bert_poly\
--train_file data/wiki1m_for_simcse.txt \
--queue_size $queue_size \
--mo_beta $mo_beta \
--poly_m $poly \
--output_dir result/${checkpoint} \
--num_train_epochs 1 \
--per_device_train_batch_size $batch_size \
--learning_rate $lr \
--max_seq_length $max_len \
--evaluation_strategy steps \
--metric_for_best_model stsb_spearman \
--load_best_model_at_end \
--eval_steps 125 \
--pooler_type cls \
--mlp_only_train \
--overwrite_output_dir \
--temp $temp \
--do_train \
--do_eval \
--fp16 