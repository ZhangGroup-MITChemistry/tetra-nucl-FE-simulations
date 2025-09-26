#!/bin/bash
#SBATCH --job-name=tetra_nucl
#SBATCH --array=0-65
#SBATCH -A bio240061p
#SBATCH --output=slurmouts/umbrella-%A_%a.out
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-16:1
#SBATCH --time=48:00:00
MY_TASK_ID=$(($SLURM_ARRAY_TASK_ID))
cd $SLURM_SUBMIT_DIR
micromamba activate openmm
module load cuda/11.7.1
echo "Task Id is ${MY_TASK_ID}"

e2e_at=($(python ./templates/make_pointgrid.py --curr_window $MY_TASK_ID --axis 1))
e2e_k=($(python ./templates/make_pointgrid.py  --curr_window $MY_TASK_ID --axis 1 --k))

d13_d24_at=($(python ./templates/make_pointgrid.py --curr_window $MY_TASK_ID --axis 0))
d13_d24_k=($(python ./templates/make_pointgrid.py --curr_window $MY_TASK_ID --axis 0 --k))

dir_name="./10bp_restraint_simulations_rerun/sim_e2e-center_${e2e_at}_d13-d24-center_${d13_d24_at}"

mkdir -p $dir_name
cd $dir_name
cat > ./sim_config.txt <<EOL
	e2e_c ${e2e_at}
	e2e_k ${e2e_k}
	d13_d24_c ${d13_d24_at}
	d13_d24_k ${d13_d24_k}
EOL

cat ./sim_config.txt

python ../../templates/run_umbrella.py  --output_dir "sim_output" --kappa1 $e2e_k --kappa2 $d13_d24_k --center1 $e2e_at --center2 $d13_d24_at --n_steps 200000000

echo "job done"
