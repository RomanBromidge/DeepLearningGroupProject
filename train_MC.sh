#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=1:0:0
#SBATCH --account=comsm0018
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

# hyperparameter value defined by default.
python group_cnn.py --mode MC --checkpoint-path checkpoint_MC.py --epochs 50
