#!/bin/bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate llamafactory
prefix="sft"
log_dir=$MY_PROJECT/logs/${prefix}
mkdir -p $log_dir
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export model_name_or_path="Qwen/Qwen2.5-32B-Instruct"
model_name=${model_name_or_path##*/}
export model_name=${model_name,,}
export dataset="m500"
export learning_rate=1e-5
export weight_decay=1e-4
export num_train_epochs=5
export cutoff_len=32768
export per_device_train_batch_size=1
export per_device_eval_batch_size=1
export gradient_accumulation_steps=2
export experiment_name=${model_name}_${dataset}_lr${learning_rate}_wd${weight_decay}_epo${num_train_epochs}_tbs${per_device_train_batch_size}_ga${gradient_accumulation_steps}
export output_dir=$MY_OUTPUT/mas_tts/ckpt/${experiment_name}
export WANDB_PROJECT="mas_tts"
mkdir -p $output_dir

echo "Running $prefix ..."
echo "model_name_or_path: $model_name_or_path"
echo "model_name: $model_name"
echo "dataset: $dataset"
echo "learning_rate: $learning_rate"
echo "weight_decay: $weight_decay"
echo "num_train_epochs: $num_train_epochs"
echo "cutoff_len: $cutoff_len"
echo "per_device_train_batch_size: $per_device_train_batch_size"
echo "per_device_eval_batch_size: $per_device_eval_batch_size"
echo "gradient_accumulation_steps: $gradient_accumulation_steps"
echo "experiment_name: $experiment_name"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "config_file: $log_dir/${experiment_name}.yaml"
echo "log_file: $log_dir/${experiment_name}.log"
echo "ckpt output_dir: $output_dir"

envsubst < $MY_PROJECT/LLaMA-Factory/examples/train_full/qwen_sft.yaml > $log_dir/${experiment_name}.yaml

nohup script -f -c "FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli train $log_dir/${experiment_name}.yaml" $log_dir/${experiment_name}.log > /dev/null 2>&1 &
pid=$!
wait $pid
echo "----------------------------------------"
echo "All processes have been completed."
echo "----------------------------------------"