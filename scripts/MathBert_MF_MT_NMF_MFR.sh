#!/bin/bash

base_dir=/root/transformer_pretraining

echo "Preparing parameters"

data_dir="$base_dir/data"
out_dir="$base_dir/out"
pretraining_obj="MF_MT_NMF_MFR"
base_path=data_dir
out="${base_dir}/out"
models="${base_dir}/models"
tokenizer_name="null"
base_bert="tbs17/MathBERT"
epochs="null"
opt_steps=250000
interval_len=25000
batch=16
trial="null"
workers=4
maxlen=512 # deprecated use
max_len="auto"
num_gpus=8
acc_steps=1
model_name="mathbert_mf_mt_nmf_mfr"
taurus="True"
run_number=1
lr="2e-5"
warmup="200"
epsilon="1e-8"
beta2="0.999"
mixed_precision="fp16"
multi_gpu="True"
one_by_one="False"
accelerate_config="/config/accelerate_basic_multi_gpu.yaml"

mkdir -p "$base_dir/models/$pretraining_obj"
echo "Created dirs"

conda activate transformer_pretraining

cmd="bash scripts/run_pretraining.sh --pretraining_obj $pretraining_obj --out $out --models $models --tokenizer_name $tokenizer_name --taurus $taurus --epochs $epochs --batch $batch --maxlen $maxlen --trial $trial --workers $workers --multi_gpu $multi_gpu --num_gpus $num_gpus --acc_steps $acc_steps --accelerate_config $accelerate_config --model_name $model_name --run_number $run_number --lr $lr --warmup $warmup --epsilon $epsilon --beta2 $beta2 --mixed_prec $mixed_precision --opt_steps $opt_steps --interval_len $interval_len --base_path $base_path --base_bert $base_bert --one_by_one $one_by_one --max_len $max_len"

echo $cmd

echo "Start"
$cmd

echo "Done"
