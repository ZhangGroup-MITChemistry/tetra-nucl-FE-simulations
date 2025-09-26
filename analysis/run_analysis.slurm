#!/bin/bash
#SBATCH --job-name=Free_Energy_tetra
#SBATCH --gres=gpu:volta:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --time=4-00:00:00
#SBATCH --output=slurmouts/%A.out
cd $SLURM_SUBMIT_DIR
python 2d_pmf_calc.py
cd dot_clusters
### Below wont run because trajectories are not included in shared simulation data ###
#sbatch run_clustering.sh
