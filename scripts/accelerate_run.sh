#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --time=0-24:00:00
#SBATCH --mem-per-cpu=40G
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/rlhf_%j.err
#SBATCH --gres=gpu:a100:4

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

conda activate nlp_work

accelerate launch --multi_gpu --num_machines 1  --num_processes 4 ../rlhf/tuning_lm_with_rl.py --log_with wandb --save_freq 1 --batched_gen True --ppo_epochs 4 --learning_rate 1.4e-5 --early_stopping True --model_name "meta-llama/Llama-2-7b-chat-hf" --batch_size=48 --gradient_accumulation_steps=8 --mini_batch_size=6 --adafactor False --dataset "../data/combined_q_values.csv" --tokenizer_name "meta-llama/Llama-2-7b-hf" --output_dir "../model/"