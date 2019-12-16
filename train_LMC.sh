#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0018
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

# no arguments: hyperparameter value defined by default.
python group_cnn.py --mode LMC --checkpoint-path checkpoint_LMC.py --logdir logs_LMC --epochs 30
