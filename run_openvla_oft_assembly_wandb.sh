#!/bin/bash
#SBATCH --job-name=openvla_assembly
#SBATCH --partition=gpu
#SBATCH --constraint=gpu2h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --output=/scratch/pioneer/users/exr343/logs/openvla_assembly_%j.out

module load Miniconda3/23.10.0-1
source activate /home/exr343/.conda/envs/openvla-oft

cd /home/exr343/CIRP_Project/openvla-oft

unset WANDB_DISABLED
export WANDB_MODE=online
export WANDB_ENTITY="exr343-case-western-reserve-university"
export WANDB_PROJECT="Robotic_Assembly"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/home/exr343/datasets/TFDS_front_side_wrist_cleaned" \
  --dataset_name "assembly_robot_data" \
  --run_root_dir "/home/exr343/checkpoints/openvla_assembly_robot" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 6 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 5000 \
  --max_steps 10005 \
  --use_val_set True \
  --val_freq 500 \
  --save_freq 5000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 64 \
  --lora_dropout 0.05 \
  --wandb_entity "exr343-case-western-reserve-university" \
  --wandb_project "Robotic_Assembly" \
  --run_id_note "assembly_robot_model"