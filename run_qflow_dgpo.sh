#!/bin/bash

#SBATCH --partition=unkillable
#SBATCH -c 6                                                           
#SBATCH --mem=32G                                        
#SBATCH --time=48:00:00    
#SBATCH --gres=gpu:a100l:1                           
#SBATCH -o /home/mila/l/luke.rowe/qflowoffline/slurm_logs/qflow_dgpo_hopper_medium_replay_v2_alpha0.05_seed1.out
module --quiet load miniconda/3
conda activate qflow_new
wandb login $WANDB_API_KEY
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mila/l/luke.rowe/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
python qflow_offline.py --track --env-id hopper-medium-replay-v2 --alpha 0.05 --seed 1