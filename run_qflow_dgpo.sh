#!/bin/bash

#SBATCH --partition=long
#SBATCH -c 6                                                           
#SBATCH --mem=32G                                        
#SBATCH --time=96:00:00    
#SBATCH --gres=gpu:rtx8000:1                          
#SBATCH -o /home/mila/l/luke.rowe/qflowoffline/slurm_logs/qflow_dgpo_hopper_medium_expert_v2_alpha0.5_seed0_lr1e-4_gradclip.out
module --quiet load miniconda/3
conda activate qflow_new
wandb login $WANDB_API_KEY
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mila/l/luke.rowe/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
python qflow_offline.py --track --env-id hopper-medium-expert-v2 --alpha 0.5 --seed 0 --lr 1e-4