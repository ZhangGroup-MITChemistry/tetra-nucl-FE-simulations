# tetra-nucl-FE-simulations
Repository used for sharing scripts used in calculating free energy profiles for our collaborative work with the Poirer Group. (bioRxiv 2025.08.13.670082; doi: https://doi.org/10.1101/2025.08.13.670082)

# Details
./build_system -- Contains script (build_nonrigid_system.py) to prepare system files for simulation, including the CV restraints we used for umbrella sampling. 
./simulation -- Contains simulation scripts as well as the values for the relevant CVs during simulations. See run_umbrella.slurm for usage. Trajectories are excluded due file size limitations. 
./analysis -- Contains scripts used to calculate free energy profiles from umbrella simulations. As only CVs are required to compute the FE profile, this script can be run to reproduce the data shared in the article. See run_analysis.slurm for usage. 
./tetra-nucl-nrl-177-add-50bp-tail -- Contains inital structures used to prepare systems.
