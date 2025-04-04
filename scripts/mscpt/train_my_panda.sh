#!/bin/zsh

seeds="1 2 3 4 5"
num_shots="16"
base_models="conch"
n_tpro="2"
cuda_ids=$1

declare -A epochs
epochs["clip"]="100"
epochs["plip"]="50"
epochs["conch"]="50"
for base_model in $base_models
do
    for num_shot in $num_shots
    do 
        for seed in $seeds
        do
            echo "Running with seed=$seed and num_shot=$num_shot total epochs=${epochs[${base_model}]} base_model=$base_model"
            CUDA_VISIBLE_DEVICES=${cuda_ids} TOKENIZERS_PARALLELISM=true python main.py --seed $seed --num_shots $num_shot --base_model $base_model \
            --dataset_name "PANDA" --dataset "my_data" --model_name "mscpt_conch" --total_epochs ${epochs[${base_model}]} --n_tpro $n_tpro --n_vpro $n_tpro \
            --data_dir "path/to/root/dir" --feat_data_dir "path/to/pt/files"
        done
    done
done