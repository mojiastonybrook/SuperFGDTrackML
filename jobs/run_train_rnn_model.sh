#!/bin/bash

#SBATCH --job-name=SuFGD_RNN
#SBATCH --gres=gpu
#SBATCH --mem=30G

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mo.jia@stonybrook.edu

pythonVer=/home/mojia/anaconda3/envs/py3104_env/bin

workDir=$PWD
appDir=/home/mojia/SuperFGDML/trajectory_fitting-main

cd $appDir

${pythonVer}/python -m nn_training.train_rnn &> ${workDir}/LogTraining_rnn.txt
