#!/bin/bash
#SBATCH --job-name=tetra_nucl
#SBATCH --array=0-42%7
#SBATCH --output=slurmouts/umbrella-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=10GB
#SBATCH --time=4-00:00:00

cd $SLURM_SUBMIT_DIR
module load cuda/11.8

e2e_at=($(python ./templates/make_pointgrid.py --curr_window $SLURM_ARRAY_TASK_ID --axis 1))
e2e_k=($(python ./templates/make_pointgrid.py  --curr_window $SLURM_ARRAY_TASK_ID --axis 1 --k))

d13_d24_at=($(python ./templates/make_pointgrid.py --curr_window $SLURM_ARRAY_TASK_ID --axis 0))
d13_d24_k=($(python ./templates/make_pointgrid.py --curr_window $SLURM_ARRAY_TASK_ID --axis 0 --k))

dir_name="./new_simulations/sim_e2e-center_${e2e_at}_d13-d24-center_${d13_d24_at}"

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
