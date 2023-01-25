#!/bin/bash


while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi

  shift
done

if [ -z "$accelerate_config" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --accelerate_config $accelerate_config"
fi

if [ -z "$num_gpus" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --num_gpus $num_gpus"
fi



if [ -z "$port" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --port $port"
fi




if [ -z "$mixed_prec" ]
        then
                cmd_string=$cmd_string
        else
                mixed_precision=$mixed_prec
fi


accelerate_config="config/accelerate_basic_multi_gpu.yaml"
cmd_string="accelerate launch --config_file $accelerate_config --num_processes $num_gpus --mixed_precision $mixed_precision src/execution/training/execute_pretraining.py"



if [ -z "$pretraining_obj" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --pretraining_obj $pretraining_obj"
fi



if [ -z "$data" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --data $data"
fi

if [ -z "$out" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --out $out"
fi

if [ -z "$models" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --models $models"
fi



if [ -z "$tokenizer_name" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --tokenizer_name $tokenizer_name"
fi

if [ -z "$data1" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --data1 $data1"
fi

if [ -z "$data2" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --data2 $data2"
fi

if [ -z "$file1" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --file1 $file1"
fi


if [ -z "$file2" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --file2 $file2"
fi


if [ -z "$epochs" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --epochs $epochs"
fi

if [ -z "$batch" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --batch $batch"
fi

if [ -z "$trial" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --trial $trial"
fi

if [ -z "$maxlen" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --maxlen $maxlen"
fi

if [ -z "$workers" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --workers $workers"
fi

if [ -z "$taurus" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --taurus $taurus"
fi



if [ -z "$model_name" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --model_name $model_name"
fi



if [ -z "$acc_steps" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --acc_steps $acc_steps"
fi

if [ -z "$num_gpus" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --num_gpus $num_gpus"
fi



if [ -z "$run_number" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --run_number $run_number"
fi



if [ -z "$lr" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --lr $lr"
fi

if [ -z "$warmup" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --warmup $warmup"
fi

if [ -z "$epsilon" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --epsilon $epsilon"
fi

if [ -z "$beta2" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --beta2 $beta2"
fi


if [ -z "$opt_steps" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --opt_steps $opt_steps"
fi

if [ -z "$interval_len" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --interval_len $interval_len"
fi


if [ -z "$comet_api_key" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --comet_api_key $comet_api_key"
fi



if [ -z "$pretrained_model_path" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --pretrained_model_path $pretrained_model_path"
fi



if [ -z "$pretrained_objectives" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --pretrained_objectives $pretrained_objectives"
fi


if [ -z "$retrain" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --retrain $retrain"
fi

if [ -z "$one_by_one" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --one_by_one $one_by_one"
fi

if [ -z "$files" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --files $files"
fi

if [ -z "$max_len" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --max_len $max_len"
fi

if [ -z "$base_bert" ]
        then
                cmd_string=$cmd_string
        else
                cmd_string="$cmd_string --base_bert $base_bert"
fi

export PYTHONPATH=$PYTHONPATH:src



#accelerate launch --config_file $accelerate_config --num_processes $num_gpus /python/test_scripts/thread_test.py $workers $batch $num_gpus

echo "Executing command:"
echo $cmd_string

$cmd_string