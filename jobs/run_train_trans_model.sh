#!/bin/bash

#SBATCH --job-name=SuFGD_TF
#SBATCH --gres=gpu
#SBATCH --nodelist=cedar
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mo.jia@stonybrook.edu
#SBATCH --time=24:00:00

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

pythonVer=/home/mojia/anaconda3/envs/py3104_env/bin

workDir=$PWD
appDir=/home/mojia/SuperFGDML/trajectory_fitting-main

cd $appDir

${pythonVer}/python -m nn_training.train_transformer &> ${workDir}/LogTraining_transformer.txt
