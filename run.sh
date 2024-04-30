#!/bin/bash

#SBATCH --partition=unkillable
#SBATCH -c 6                                                           
#SBATCH --mem=32G                                        
#SBATCH --time=24:00:00    
#SBATCH --gres=gpu:a100l:1                           
#SBATCH -o /home/mila/l/luke.rowe/qflowoffline/slurm_logs/corrected_train_bc_halfcheetah_medium-expert-v2_rerun.out
module --quiet load miniconda/3
conda activate qflow_new
wandb login $WANDB_API_KEY
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mila/l/luke.rowe/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
python train_bc.py --track --env-id halfcheetah-medium-expert-v2