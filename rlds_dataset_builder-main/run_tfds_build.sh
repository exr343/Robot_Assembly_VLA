#!/bin/bash
#SBATCH -J tfds_assembly_robot
#SBATCH -p gpu
#SBATCH -C gpu2h100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH -o /scratch/pioneer/users/exr343/logs/tfds_assembly_%j.out
#SBATCH -e /scratch/pioneer/users/exr343/logs/tfds_assembly_%j.err

module load Miniconda3/23.10.0-1
source activate /home/exr343/.conda/envs/rlds_env

# Where TFDS will write the built dataset
export TFDS_DATA_DIR=/scratch/pioneer/users/exr343/tensorflow_datasets

# Go to the dataset directory (where assembly_robot_data_dataset_builder.py lives)
cd /home/exr343/CIRP_Project/openvla-oft/rlds_dataset_builder-main/assembly_robot_data

# Build the dataset
tfds build --overwrite
