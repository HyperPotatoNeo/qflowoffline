# QFlow Offline

## Installation and Running on Mila Cluster

### Install Mujoco

Install Mujoco, following the instructions here: https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

### Conda Setup

```
salloc -c 2 --gres=gpu:1 --mem=12000 --partition unkillable
module load miniconda/3
conda create --name qflow python=3.9
conda activate qflow
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include # add to .bashrc to make permanent
pip install patchelf
pip3 install -U 'mujoco-py<2.2,>=2.1'
pip install "cython<3"
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip3 install torch
pip install tensorboard
pip install stable-baselines3
pip install 'shimmy>=0.2.1'
pip install wandb
```

### Run

```
python qflow_offline.py
```