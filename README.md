# Fine-tuned OpenVLA-OFT Model for Robotic Assembly Tasks

This repository contains code and configurations for fine-tuning the OpenVLA-OFT model on engine-block based assembly tasks and subsequently evaluating the fine-tuned VLA model. 

Fine-tuning and evaluation was performed using Case Western Reserve University's High Performance Computing (HPC) cluster.

Specs for fine-tuning:
  - GPU type: NVIDIA H100 80GB
  - Resources per job: 1 GPU, 2 CPU cores, 200 GB RAM
  - Module stack: `module load Miniconda3/23.10.0-1`

The slurm batch file submitted to the HPC cluster is seen in run_openvla_oft_assembly_wandb.sh.

## 1. Configure Environment
### Set-up Conda Environment
The following commands create a conda environment and install this project and its dependencies as specified in pyproject.toml.
```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/exr343/Robot_Assembly_VLA.git
cd Robot_Assembly_VLA
pip install -e . 

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation # not essential
```

## 2. Fine-tune Base OpenVLA-OFT Model

The finetune.py script is run on a TensorFlow-based RLDS dataset formed from rlds_dataset_builder-main/assembly_robot_data. In particular, 86 demonstrations were recorded of a UR5 robot placing a valve cover assembly onto an engine cylinder head. Each demonstration consists of a variable number of timesteps. For each timestep, joint and gripper angles are recorded as the robot state, side image and wrist images are recorded as the observation, and joint and gripper control values are recorded as the action.
It is noted that that action chunking is defined within prismatic/vla/constants.py, which is set to 5 in this model. The following code snippet is used to fine-tune the base model.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \ 
  --data_root_dir "/PATH/TO/RLDS/DATASETS/DIR/" \
  --dataset_name "assembly_robot_data" \
  --run_root_dir "/PATH/TO/CHECKPOINT" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set False \
  --save_freq 5000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note "MODEL_NAME"

```

## 3. Use Fine-tuned Model

The fine-tuned OpenVLA-OFT model forms a policy based on the following inputs and outputs:

- **Input:** joint angles (dim 6), gripper state (dim 1), language instruction, side image (480, 640, 3), wrist image (480, 640, 3)  
- **Output:** 5-step action chunk: joint control (dim 6), gripper state (dim 1)

To evaluate the model, the following Python script may be used:

```python
import os
import numpy as np
from PIL import Image
from experiments.robot.assembly.run_assembly_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

CHECKPOINT = "/PATH/TO/CHECKPOINT" # included checkpoint based on finetuning script above
IMG_PATH_SIDE = "/PATH/TO/side_image.png" # (480, 640, 3) PNG file
IMG_PATH_WRIST = "/PATH/TO/wrist_image.png" # (480, 640, 3) PNG file
STATE_PATH = "/PATH/TO/states.npy" # (7,) NumPy array
INSTRUCTION = "put the valve cover assembly onto the cylinder head" # text string

cfg = GenerateConfig(
    pretrained_checkpoint=CHECKPOINT,
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="assembly_robot_data",
)

vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

def load_observation():
    states = np.load(STATE_PATH)
    state = states[0].astype(np.float32)
    full_image = np.array(Image.open(IMG_PATH_SIDE).convert("RGB"), dtype=np.uint8)
    wrist_image = np.array(Image.open(IMG_PATH_WRIST).convert("RGB"), dtype=np.uint8)
    obs = {
        "full_image": full_image,
        "wrist_image": wrist_image,
        "state": state,
        "task_description": INSTRUCTION,
    }
    return obs

if __name__ == "__main__":
    obs = load_observation()
    actions = get_vla_action(
        cfg,
        vla,
        processor,
        obs,
        obs["task_description"],
        action_head,
        proprio_projector,
    )
    print("Generated action chunk:")
    for a in actions:
        print(a)
```