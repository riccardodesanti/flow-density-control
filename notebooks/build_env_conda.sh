#!/bin/bash
set -e # this will stop the script on first error

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Building flowmol into the environment '$ENV_NAME'"

OPENBLAS=/cluster/apps/openblas/0.2.13_seq/x86_64/gcc_4.8.2/lib

conda install pytorch=2.3.0 torchvision torchaudio pytorch-cuda=12.1 pytorch-cluster -c pytorch -c nvidia -c pyg -y
pip install --force-reinstall --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
conda install -c dglteam/label/cu121 dgl -y
conda install -c conda-forge pystow einops -y
pip install rdkit==2025.3.2
pip install wandb useful_rdkit_utils py3Dmol --no-input
pip install -e ./
