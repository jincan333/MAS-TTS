#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate agent
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export temperature=0
export max_length=32768
export model_name_or_path="Can111/m1-32b"

# run agentverse on the model and dataset
dataset="aime2024"
judge_model=""
model="m1-32b"
task=tasksolving/${dataset}/${model}
output_path=""
dataset_path=data/${dataset}/test.jsonl

log_dir=logs/benchmark/${model}
mkdir -p $log_dir
date_str=$(date '+%d_%b_%H')
experiment_name=${dataset}_"${output_path}"_${date_str}

echo "Running agentverse benchmark..."
echo "Model: ${model}"
echo "Task: ${task}"
echo "Dataset path: ${dataset_path}"
echo "Log file: ${log_dir}/${experiment_name}.log"
nohup agentverse-benchmark \
    --task $task \
    --dataset_path $dataset_path \
    > ${log_dir}/${experiment_name}.log 2>&1 &
pid1=$!
wait $pid1
echo "----------------------------------------"

echo "Evaluating dataset..."
echo "Judge model: ${judge_model}"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo "Log file: logs/evaluation.log"
nohup python evaluate_dataset.py \
    --judge_model "${judge_model}" \
    --model $model \
    --dataset $dataset \
    --output_path "${output_path}" \
>> logs/evaluation.log 2>&1 &
pid2=$!
wait $pid2
echo "----------------------------------------"
echo "All processes have been completed."
echo "----------------------------------------"