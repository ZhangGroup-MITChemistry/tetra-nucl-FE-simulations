import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob
import os
from FastMBAR import FastMBAR
import openmm.unit as unit
import pandas as pd
import MDAnalysis as mda
import torch

print("Modules Loaded")

'''
C1 = E2E
C2 = D13-D24
'''

sim_folder = "../simulations/10bp_restraint_simulations_rerun"
colvar_files = glob.glob(f"{sim_folder}/sim_e2e-center_*/sim_output/COLVARS")
traj_files   = glob.glob(f"{sim_folder}/sim_e2e-center_*/sim_output/output.dcd")
config_files = glob.glob(f"{sim_folder}/sim_e2e-center_*/sim_config.txt")

top = "../build_system/cg_tetra_nucl.pdb"
sim_count = len(traj_files)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device for doing optimization

NCPP_IDX = np.arange(0,982*4).reshape((4,-1))

GAS_CONSTANT = (1.0 * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).in_units_of(unit.kilojoule_per_mole / unit.kelvin)
RT = (unit.Quantity(300, unit.kelvin)) * GAS_CONSTANT

l_PMF = 17
stride = 1

def align_and_sub_min(A,B):
    diff = (A - B)
    c = 0.5*diff[~np.isinf(diff)].mean()
    A -= A.min()
    B -= B.min()
    return A,B

def compute_bias_PE(colvars, c1, c2, k1, k2):
    u_c1 = 0.5 * k1 * (colvars[:,0] - c1)**2
    u_c2 = 0.5 * k2 * (colvars[:,1] - c2)**2
    return (u_c1+u_c2)

def parse_config_files(config_files):
    grid_points = []
    force_consts = []
    for file in config_files:
        with open(file, "r+") as f:
            data = f.read().split('\n')
            data = [s.lstrip('\t') for s in data]
            e2e_c = float(data[0].split()[1])
            e2e_k = float(data[1].split()[1])
            dd_c = float(data[2].split()[1])
            dd_k = float(data[3].split()[1])
            grid_points.append([dd_c,e2e_c])
            force_consts.append([dd_k,e2e_k])
    grid_points = np.array(grid_points)
    force_consts = np.array(force_consts)
    return grid_points, force_consts

def get_colvars(colvar_files):
    colvars = []
    sample_count = []
    for file in colvar_files:
        with open(file,"r+") as f:
            inner_colvar = []
            print(file)
            for line in f.readlines():
                tokens = line.split()
                if tokens[0] == "#!":
                    continue
                try:
                    float(tokens[0])
                except:
                    continue
                time = float(tokens[0])
                if len(inner_colvar) > 0 and time in inner_colvar[:][-1]:
                    continue
                e2e_c1 = float(tokens[-2])
                dd_c2 = float(tokens[-1])
                d13 = float(tokens[2])
                d24 = float(tokens[5])
                inner_colvar.append([e2e_c1,dd_c2,d13,d24,time])
            inner_colvar = inner_colvar[1:]
            print(len(inner_colvar))
            sample_count.append(len(inner_colvar))
            colvars += inner_colvar
    np.save("new_sim_colvars2.npy", np.array(colvars))
    np.save("new_sim_sample_count.npy", np.array(sample_count))
    return np.array(colvars), np.array(sample_count)

def calc_potentials(grid_points, force_consts, colvars):
    energy_matrix = []
    for i in range(len(grid_points)):
        c1 = grid_points[i,1]
        c2 = grid_points[i,0]
        k1 = force_consts[i,1]
        k2 = force_consts[i,0]
        print(f"Working on sim: {i}", flush=True)
        energy_matrix.append(compute_bias_PE(colvars, c1, c2, k1, k2) / RT)
    energy_matrix = np.array(energy_matrix)
    np.save("new_sim_energy_matrix2.npy", energy_matrix)

def calc_bias_potentials(l_PMF, c1_PMF, c1_width, c2_PMF, c2_width, colvars, energy_matrix):
    bias_matrix = np.zeros((l_PMF * l_PMF, energy_matrix.shape[1]))
    for index in range((l_PMF * l_PMF)):
        c1_index = index // l_PMF
        c2_index = index % l_PMF
        c1_c_PMF = c1_PMF[c1_index]
        c2_c_PMF = c2_PMF[c2_index]

        c1_low  = c1_c_PMF - 0.5*c1_width
        c1_high = c1_c_PMF + 0.5*c1_width

        c2_low  = c2_c_PMF - 0.5*c2_width
        c2_high = c2_c_PMF + 0.5*c2_width

        c1_indicator = ((colvars[:,0] > c1_low) & (colvars[:,0] <= c1_high))
        c2_indicator = ((colvars[:,1] > c2_low) & (colvars[:,1] <= c2_high))

        indicator = ~(c1_indicator & c2_indicator)
        bias_matrix[index, indicator] = np.inf
    return bias_matrix

def get_projected_F(cv_idx, cv_PMF, cv_width, colvars, energy_matrix, sample_count, n_blocks=5):
    PMF = []
    for i in range(n_blocks):
        blk_e_mat = np.empty((energy_matrix.shape[0],1))
        blk_cvars = np.empty((1,colvars.shape[1]))
        blk_samples = np.zeros_like(sample_count)
        for j in range(sample_count.shape[0]):
            baseline = sample_count[np.arange(j)].sum()
            lidx = int(baseline + (i * (sample_count[j]//n_blocks)))
            uidx = int(baseline + ((i+1) * (sample_count[j]//n_blocks)))
            if uidx > (baseline + sample_count[j]):
                uidx = int(baseline + sample_count[j])
            if i == n_blocks - 1 and (uidx < baseline + sample_count[j]):
                uidx = int(baseline + sample_count[j])
            print(i,j,lidx,uidx,energy_matrix.shape[-1])
            blk_e_mat = np.append(blk_e_mat,energy_matrix[:,lidx:uidx],axis=-1)
            blk_cvars = np.append(blk_cvars,colvars[lidx:uidx,:], axis=0)
            blk_samples[j] = uidx-lidx
        blk_e_mat = blk_e_mat[:,1:]
        blk_cvars = blk_cvars[1:]
        fastmbar = FastMBAR(energy = blk_e_mat, num_conf = blk_samples, cuda = True, bootstrap=False)
        bias_matrix = np.zeros((l_PMF, blk_e_mat.shape[1]))
        for index in range(l_PMF):
            cv_c_PMF = cv_PMF[index]

            cv_low  = cv_c_PMF - 0.5*cv_width
            cv_high = cv_c_PMF + 0.5*cv_width

            cv_indicator = ((blk_cvars[:,cv_idx] > cv_low) & (blk_cvars[:,cv_idx] <= cv_high))

            bias_matrix[index, ~cv_indicator] = np.inf
        results = fastmbar.calculate_free_energies_of_perturbed_states(bias_matrix)
        pmf = results['F']
        #pmf = pmf-pmf.min()
        PMF.append(pmf)

    PMF = np.array(PMF)
    PMFU = PMF.std(axis=0)
    PMF = PMF.mean(axis=0)
    return PMF,PMFU

def make_reduced_set(col_list, energy_matrix, colvars, sample_count):
    incl_mat    = np.empty((energy_matrix.shape[0],1))
    incl_cv     = np.empty((1,colvars.shape[1]))
    incl_sample = []
    row_remove_list = []
    for i in range(sample_count.shape[0]):
        lb = sample_count[np.arange(i)].sum()
        ub = lb + sample_count[i]
        cols = np.array(col_list)
        cols = cols[(cols >= lb) & (cols < ub)]
        if cols.shape[0] <= 5:
            row_remove_list.append(i)
            continue
        incl_sample.append(cols.shape[0])
        incl_mat = np.append(incl_mat,energy_matrix[:,cols],axis=-1)
        incl_cv = np.append(incl_cv,colvars[cols,:],axis=0)
    
    incl_sample = np.array(incl_sample)
    incl_mat = incl_mat[:,1:]
    incl_cv  = incl_cv[1:]
    incl_mat = np.delete(incl_mat, row_remove_list, 0)
    return incl_mat, incl_cv, incl_sample

def write_cpptraj_file(bins, bin_width, full_list, cluster_string, bin_path_header, sample_count, colvars, traj_files):
    for bin in bins:
        bl = bin - (bin_width/2)
        bu = bin + (bin_width/2)
        s_cv = np.where(full_list != -1, colvars[:,0],-1)
        ids_in_bin = np.where((s_cv >= bl) & (s_cv < bu), full_list,-1)
        bin_path = f"{bin_path_header}%3.2f" % bin
        if not os.path.exists(bin_path):
            os.makedirs(bin_path)
        with open(f"{bin_path}/cluster_bin.in", "w+") as f:
            print(f"parm ../../../{top}", file=f)
            for i in range(energy_matrix.shape[0]):
                row_lb = sample_count[:i].sum()
                row = ids_in_bin[row_lb:row_lb+sample_count[i]]
                row = row[row != -1] - row_lb
                if row.shape[0] == 0:
                    continue
                print(row)
                lb = row[0]
                ub = lb
                for j in range(1,row.shape[0]):
                    if row[j] == ub + 1:
                        ub = row[j]
                    else:
                        if lb == ub:
                            print(f"trajin ../../../{traj_files[i]} {stride*lb+1} {stride*ub+1}", file=f)
                        else:
                            print(f"trajin ../../../{traj_files[i]} {stride*lb+1} {stride*ub+1} {stride}", file=f)
                        lb = row[j]
                        ub = row[j]
                if lb == ub:
                    print(f"trajin ../../../{traj_files[i]} {stride*lb+1} {stride*ub+1}", file=f)
                else:
                    print(f"trajin ../../../{traj_files[i]} {stride*lb+1} {stride*ub+1} {stride}", file=f)
            print(cluster_string,file=f) 

def calculate_angles(traj_files, top, file):
    if os.path.exists(file):
        return np.load(file)
    block_count = 350
    u = mda.Universe(top, traj_files)
    angles = np.empty((4,len(u.trajectory)))
    def calc_angle(pos,i,j):
        ncp1 = pos[NCPP_IDX[i][0]:NCPP_IDX[i][-1]]
        ncp2 = pos[NCPP_IDX[j][0]:NCPP_IDX[j][-1]]
        norm1  = torch.linalg.svd(ncp1 - ncp1.mean(0))[-1][:,-1]
        norm2  = torch.linalg.svd(ncp2 - ncp2.mean(0))[-1][:,-1]
        angle = torch.arccos((norm1 * norm2).sum(-1))
        inter_vec = ncp2.mean(0) - ncp1.mean(0)
        inter_vec /= torch.linalg.norm(inter_vec)
        angle_b = torch.arccos((norm2 * inter_vec).sum(-1))
        angle_b = torch.maximum(angle_b,torch.arccos((norm1 * inter_vec).sum(-1)))
        return angle,angle_b
    for i in range(block_count):
        lb = i * len(u.trajectory)//block_count
        ub = (i+1) * len(u.trajectory)//block_count
        if ub > len(u.trajectory):
            ub = len(u.trajectory)
        traj = torch.tensor(np.stack([u.trajectory[j].positions for j in range(lb,ub)]), dtype=dtype, device=device)
        angle13,angle13b = torch.vmap(lambda x: calc_angle(x, 0, 2))(traj)
        angle24,angle24b = torch.vmap(lambda x: calc_angle(x, 1, 3))(traj)
        angles[0][lb:ub] = angle13.cpu().numpy()
        angles[1][lb:ub] = angle24.cpu().numpy()
        angles[2][lb:ub] = angle13b.cpu().numpy()
        angles[3][lb:ub] = angle24b.cpu().numpy()
        torch.cuda.empty_cache()
        print(f"block {i}")
    np.save(file, angles)
    return angles

'''
    This function is specifically for computing 2-Dot and 3-Dot Free Energies. Also generates a cpptraj file which clusters structures based on distance cutoffs used to determine
    whether or not a structure should be considered 2-Dot vs 3-Dot. Clusters are only there for me to validate if our metrics are reasonable.
'''
def get_projected_F_DRange(colvars, energy_matrix, sample_count, traj_files, d_bounds=(7,8)):
    #angles = calculate_angles(traj_files, top, "angles.npy")
    cluster_string = "cluster C0 dbscan minpoints 1 epsilon 20 rms !@CA sieve 4 out cnumvtime.dat summary summary.dat info info.dat cpopvtime cpopvtime.agr normframe singlerepout singlerep.nc singlerepfmt netcdf"
    angle_bound1 = np.pi * (60/180)
    angle_bound2 = np.pi * (90/180)
    #indicator = (colvars[:,2] <= d_bounds[0]) & (colvars[:,3] <= d_bounds[0]) & (angles[0] <= angle_bound1) & (angles[1] <= angle_bound1) & (angles[2] <= angle_bound2) & (angles[3] <= angle_bound2)
    indicator = (colvars[:,2] <= d_bounds[0]) & (colvars[:,3] <= d_bounds[1]) | (colvars[:,2] <= d_bounds[1]) & (colvars[:,3] <= d_bounds[0]) 
    col_list = np.arange(energy_matrix.shape[-1])[indicator]
    full_list = np.where(indicator,np.arange(energy_matrix.shape[-1]),-1)
    incl_mat, incl_cv, incl_sample = make_reduced_set(col_list, energy_matrix, colvars, sample_count)
    incl_pmf, incl_pmfu = get_projected_F(0, c1_PMF, c1_width, incl_cv, incl_mat, incl_sample)
    incl_pmf_rel = incl_pmf.copy()
    #incl_pmf -= incl_pmf.min()

    bin_width_str_fig = (80)/4
    bins_for_str_fig = np.array([bin_width_str_fig/2 + (bin_width_str_fig * i) for i in range(4)])
    #dot2_file = "./dot_clusters_H1/2dot/bin_"
    #dot3_file = "./dot_clusters_H1/3dot/bin_"
    dot2_file = "./dot_clusters/2dot/bin_"
    dot3_file = "./dot_clusters/3dot/bin_"
    write_cpptraj_file(bins_for_str_fig, bin_width_str_fig,full_list,cluster_string,dot2_file,sample_count, colvars, traj_files)

    exp_2dot = np.array([5,0.109368517 ,0.54763902 ,
                        10,-0.462284594,0.324600953,
                        15,-0.31427733 ,0.326059286,
                        20,-0.20589945 ,0.256838327,
                        25,-0.106530381,0.183167253,
                        30,-0.236685035,0.225610761,
                        35,0.266565122 ,0.194358352,
                        40,0.522517446 ,0.222321335,
                        45,0.998213771 ,0.324434442,
                        50,1.623096746 ,0.349762158,
                        55,1.854394261 ,0.305520739,
                        60,2.632281028 ,0.308110394,
                        65,3.552679732 ,0.463781521,
                        70,3.459369807 ,0.425868097,
                        75,3.596600552 ,0.555145202,
                        80,4.050218702 ,0.646701764,
                        85,np.inf,np.inf           ,
                         ]).reshape((-1,3))

    #exp_2dot[:,1] -= exp_2dot[:,1].min()
    exp_2dot_H1 = np.array([5,-1.586272983,0.780284727 ,
                            10,-1.353298156,0.30104542 ,
                            15,-0.520981421,0.223263974,
                            20,0.24251471,0.260184574  ,
                            25,0.783268035,0.254868413 ,
                            30,0.569122872,0.292556199 ,
                            35,1.076406558,0.229525098 ,
                            40,1.580914053,0.314326669 ,
                            45,1.62664932,0.245387928  ,
                            50,2.03962333,0.295617092  ,
                            55,2.612626128,0.295074827 ,
                            60,3.677119445,0.364928071 ,
                            65,4.130599657,0.411747836 ,
                            70,3.338145982,0.846561673 ,
                            75,5.115727536,0.722822184 ,
                            80,np.inf,np.inf           ,
                            85,5.271482065,1.441153384,
                            ]).reshape((-1,3))
    #exp_2dot_H1 -= exp_2dot_H1.min()
    incl_pmf, exp_2dot[:,1] = align_and_sub_min(incl_pmf, exp_2dot[:,1])
    incl_pmf, exp_2dot_H1[:,1] = align_and_sub_min(incl_pmf, exp_2dot_H1[:,1])
    
    plt.errorbar(c1_PMF, incl_pmf, yerr=incl_pmfu, fmt = '-o', ecolor = 'grey', capsize = 2, capthick = 1, markersize = 6, 
                 label='Simulation')
    #plt.errorbar(exp_2dot_H1[:,0], exp_2dot_H1[:,1], yerr=exp_2dot_H1[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6, label='Experiment')
    plt.errorbar(exp_2dot[:,0], exp_2dot[:,1], yerr=exp_2dot[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6,    label='Experiment')
    plt.ylabel("Free Energy ($k_B T$)")
    plt.xlabel("End to End Distance (nm)")
    plt.legend()
    plt.title("2 Dot")
    #plt.savefig("./PMF_e2e_2Dot_H1.png", dpi=300)
    plt.savefig("./PMF_e2e_2Dot.png", dpi=300)
    plt.close()

    indicatora = (colvars[:,2] > d_bounds[1]) & (colvars[:,3] <= d_bounds[0])
    indicatorb = (colvars[:,2] <= d_bounds[0]) & (colvars[:,3] > d_bounds[1])
    indicator = (colvars[:,2] <= d_bounds[0]) & (colvars[:,3] <= d_bounds[1]) | (colvars[:,2] <= d_bounds[1]) & (colvars[:,3] <= d_bounds[0]) 
    excl_list = np.arange(energy_matrix.shape[-1])[~indicator]
    full_list = np.where(~indicator,np.arange(energy_matrix.shape[-1]),-1)
    excl_mat, excl_cv, excl_sample = make_reduced_set(excl_list, energy_matrix, colvars, sample_count)
    excl_pmf, excl_pmfu = get_projected_F(0, c1_PMF, c1_width, excl_cv, excl_mat, excl_sample)
    excl_pmf_rel = excl_pmf.copy()
    #excl_pmf -= excl_pmf.min()

    bin_width_str_fig = (80)/4
    bins_for_str_fig = np.array([bin_width_str_fig/2 + (bin_width_str_fig * i) for i in range(4)])
    write_cpptraj_file(bins_for_str_fig, bin_width_str_fig,full_list,cluster_string,dot3_file, sample_count, colvars, traj_files)

    exp_3dot = np.array([
                         5,np.inf,np.inf           ,
                        10,1.867289818,1.154700538,
                        15,1.954689945,0.550072145,
                        20,2.506402679,0.529256124,
                        25,1.871748445,0.356236352,
                        30,1.257177119,0.497563409,
                        35,1.378850428,0.272605807,
                        40,1.635787996,0.312700407,
                        45,1.540286586,0.273081249,
                        50,1.715655061,0.302909302,
                        55,2.323512566,0.310785121,
                        60,3.061249729,0.373984483,
                        65,2.994177533,0.4536443  ,
                        70,3.870786942,0.513750592,
                        75,4.133890028,0.645705043,
                        80,2.937731229,0.746420027,
                        85,np.inf,np.inf           ,
                         ]).reshape((-1,3))
    #exp_3dot[:,1] -= exp_3dot[:,1].min()
    exp_3dot_H1 = np.array([ 5,1.21165972,1.154700538  ,
                            10,1.754882738,0.535479553,
                            15,1.505342185,0.39244796 ,
                            20,2.231651412,0.441382919,
                            25,2.485919241,0.491341337,
                            30,1.790202229,0.354936787,
                            35,2.513511517,0.466575582,
                            40,2.042673087,0.379728621,
                            45,2.639617333,0.343878579,
                            50,2.636378759,0.370087174,
                            55,3.726364119,0.463254862,
                            60,3.469322161,0.354964787,
                            65,4.733249512,0.589761978,
                            70,5.559164138,0.828189251,
                            75,np.inf,np.inf           ,
                            80,np.inf,np.inf           ,
                            85,np.inf,np.inf           ,
                            ]).reshape((-1,3))
    #exp_3dot_H1 [:,1] -= exp_3dot_H1[:,1].min()
    excl_pmf, exp_3dot[:,1] = align_and_sub_min(excl_pmf, exp_3dot[:,1])
    excl_pmf, exp_3dot_H1[:,1] = align_and_sub_min(excl_pmf, exp_3dot_H1[:,1])
    
    plt.errorbar(c1_PMF, excl_pmf, yerr=excl_pmfu, fmt = '-o', ecolor = 'grey', capsize = 2, capthick = 1, markersize = 6,
                 label='Simulation')
    plt.errorbar(exp_3dot_H1[:,0], exp_3dot_H1[:,1], yerr=exp_3dot_H1[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6, label='Experiment')
    #plt.errorbar(exp_3dot[:,0], exp_3dot[:,1], yerr=exp_3dot[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6, label='Experiment')
    plt.ylabel("Free Energy ($k_B T$)")
    plt.xlabel("End to End Distance (nm)")
    plt.legend()
    plt.title("3 Dot")
    plt.savefig("./PMF_e2e_3Dot_H1.png", dpi=300)
    #plt.savefig("./PMF_e2e_3Dot.png", dpi=300)
    plt.close()

    plt.errorbar(c1_PMF, excl_pmf_rel, yerr=excl_pmfu, fmt = '-o', ecolor = 'grey', capsize = 2, capthick = 1, markersize = 6,
                 label='3 Dot')
    plt.errorbar(c1_PMF, incl_pmf_rel, yerr=excl_pmfu, fmt = '-o', ecolor = 'grey', capsize = 2, capthick = 1, markersize = 6,
                 label='2 Dot')
    pd.DataFrame(np.stack((c1_PMF,excl_pmf_rel, excl_pmfu), axis=-1), columns=["End-to-End (nm)", "FE (kT)", "Uncertainty"]).to_csv("3dot_data.csv")
    pd.DataFrame(np.stack((c1_PMF,incl_pmf_rel, incl_pmfu), axis=-1), columns=["End-to-End (nm)", "FE (kT)", "Uncertainty"]).to_csv("2dot_data.csv")
    plt.ylabel("Free Energy ($k_B T$)")
    plt.xlabel("End to End Distance (nm)")
    plt.legend()
    plt.title("2 Dot vs. 3 Dot Simulation")
    plt.savefig("./PMF_e2e_2v3Dot_H1.png", dpi=300)
    #plt.savefig("./PMF_e2e_2v3Dot.png", dpi=300)
    plt.close()

def sieve_data(colvars, energy_matrix, sample_count, step_size):
    new_vars = np.empty((0,colvars.shape[1]))
    new_energy = np.empty((sample_count.shape[0], 0))
    for i in range(sample_count.shape[0]):
        lb = sample_count[:i].sum()
        new_vars = np.append(new_vars,colvars[lb:sample_count[i]+lb:step_size],axis=0)
        new_energy = np.append(new_energy, energy_matrix[:,lb:sample_count[i]+lb:step_size], axis=-1)

    energy_matrix = new_energy
    colvars = new_vars
    sample_count = np.ceil(sample_count/step_size).astype(int)

    assert sample_count.sum() == colvars.shape[0], f"Sample Count Sum: {sample_count.sum()}, Colvars Shape: {colvars.shape}"
    assert sample_count.sum() == energy_matrix.shape[1], f"Sample Count Sum: {sample_count.sum()}, Colvars Shape: {energy_matrix.shape}"
    return colvars, energy_matrix, sample_count

min_c1 =  5 #E2E
max_c1 =  85
min_c2 = -20 #DD
max_c2 =  20
c1_PMF = np.linspace(min_c1, max_c1, l_PMF)
c2_PMF = np.linspace(min_c2, max_c2, l_PMF)
c1_width = (c1_PMF.max()-c1_PMF.min())/l_PMF
c2_width = (c2_PMF.max()-c2_PMF.min())/l_PMF

calculate_potentials = True


if calculate_potentials:
    grid_points, force_consts = parse_config_files(config_files)
    colvars,sample_count = get_colvars(colvar_files)
    calc_potentials(grid_points, force_consts, colvars)

sample_count = np.load("new_sim_sample_count.npy")
colvars = np.load("new_sim_colvars2.npy")
energy_matrix = np.load("new_sim_energy_matrix2.npy")

bias_matrix = calc_bias_potentials(l_PMF, c1_PMF, c1_width, c2_PMF, c2_width, colvars, energy_matrix)

e2e_F,e2e_Fu = get_projected_F(0, c1_PMF, c1_width, colvars, energy_matrix, sample_count) 
dd_F, dd_Fu = get_projected_F(1, c2_PMF, c2_width , colvars, energy_matrix, sample_count) 
get_projected_F_DRange(colvars, energy_matrix, sample_count, traj_files)
print("Done with Proj")

colvars, energy_matrix, sample_count = sieve_data(colvars, energy_matrix, sample_count, step_size=3)
bias_matrix = calc_bias_potentials(l_PMF, c1_PMF, c1_width, c2_PMF, c2_width, colvars, energy_matrix)

fastmbar = FastMBAR(energy = energy_matrix, num_conf = sample_count, cuda = True)
print("Finished MBAR")
results = fastmbar.calculate_free_energies_of_perturbed_states(bias_matrix)
print("Finished F Calc")

PMF = results['F']
PMF_STD = results['F_std']
PMF = PMF - PMF.min()
PMF = PMF.reshape((l_PMF, l_PMF))

## plot the PMF
plt.rcParams['font.size'] = 20
fig=plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(2, 4, width_ratios=[1,4,0.2,0.1], height_ratios=[4,1], hspace=0, wspace=0)
ax = plt.subplot(gs[0,1])
axl = plt.subplot(gs[0,0],sharey=ax)
axb = plt.subplot(gs[1,1],sharex=ax)

c1_offset = 0.5*(c1_PMF.max()-c1_PMF.min())/l_PMF
c2_offset = 0.5*(c2_PMF.max()-c2_PMF.min())/l_PMF
im = ax.imshow(np.flipud(PMF.T), extent=(c1_PMF.min()-c1_offset, c1_PMF.max()+c1_offset, c2_PMF.min()-c2_offset, c2_PMF.max()+c2_offset), aspect='auto')

exp_data = np.array([   5,0.030825056  ,0.545334098,
                        10,-0.499803665,0.327478217,
                        15,-0.404825567,0.320287026,
                        20,-0.271015533,0.253556426,
                        25,-0.241979136,0.177993213,
                        30,-0.424556864,0.252722962,
                        35,-0.03090495 ,0.180558408,
                        40,0.236382737 ,0.20007733 ,
                        45,0.525598837 ,0.26109148 ,
                        50,0.965425268 ,0.271901268,
                        55,1.403486161 ,0.230934516,
                        60,2.041813637 ,0.237902479,
                        65,2.425049047 ,0.365143835,
                        70,2.955922931 ,0.338032436,
                        75,3.168642347 ,0.399055662,
                        80,3.173867273 ,0.47327025 ,
                        85,3.241885186 ,0.741767039
                     ]).reshape((-1,3))
exp_data = exp_data[(exp_data[:,0] <= c1_PMF.max())]

exp_H1_data = np.array([5,-1.638697775,0.766525607 ,
                        10,-1.411512617,0.287520564,
                        15,-0.660062412,0.215239246,
                        20,0.106439337,0.249665417 ,
                        25,0.685099044,0.26001126  ,
                        30,0.328551754,0.277244428 ,
                        35,0.859435165,0.227575972 ,
                        40,1.131292943,0.23210831  ,
                        45,1.328548689,0.226686918 ,
                        50,1.665593425,0.29937062  ,
                        55,2.314267128,0.262389045 ,
                        60,2.820145486,0.262506536 ,
                        65,3.711914353,0.346549218 ,
                        70,4.184178467,0.487746682 ,
                        75,4.892583984,0.649978392 ,
                        80,np.inf,np.inf           ,
                        85,4.578334885,1.037749043]).reshape((-1,3))

exp_H1_data = exp_H1_data[(exp_H1_data[:,0] <= c1_PMF.max())]


e2e_F, exp_data[:,1] = align_and_sub_min(e2e_F, exp_data[:,1])
e2e_F, exp_H1_data[:,1] = align_and_sub_min(e2e_F, exp_H1_data[:,1])

axl.hlines(c2_PMF, dd_F.min(), dd_F.max(), linestyles='dashed', color='gray', alpha=0.5)
axl.errorbar(dd_F, c2_PMF, xerr=dd_Fu, fmt = '-o', ecolor = 'gray', capsize = 2, capthick = 1, markersize = 6, label="Simulation")
axl.set(ylabel=r"$D_{13}-D_{24}$ (nm)", xlabel="Free Energy ($k_B T$)")
axl.invert_xaxis()

axb.vlines(c1_PMF, e2e_F.min(), e2e_F[~np.isinf(e2e_F)].max(), linestyles='dashed', color='gray', alpha=0.5)
axb.errorbar(c1_PMF, e2e_F, yerr=e2e_Fu, fmt = '-o', ecolor = 'gray', capsize = 2, capthick = 1, markersize = 6, label='Simulation')
axb.errorbar(exp_data[:,0], exp_data[:,1], yerr=exp_data[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6, label='Experiment')
#axb.errorbar(exp_H1_data[:,0], exp_H1_data[:,1], yerr=exp_H1_data[:,2], fmt = '-o', ecolor = 'black', capsize = 2, capthick = 1, markersize = 6, label='Experiment')
axb.set_ylabel("Free Energy ($k_B T$)",rotation=0)
axb.yaxis.set_label_coords(-0.25, 0.30)
axb.set_xlabel("End-to-End Distance (nm)")

axl.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelbottom=False)
axl.xaxis.set_label_position('top')
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
fig.colorbar(im, cax=plt.subplot(gs[0,3]), label="Free Energy ($k_B T$)")
axb.legend(fontsize=10)
ax.hlines(c2_PMF, c1_PMF.min(), c1_PMF.max(), linestyles='dashed', color='gray', alpha=0.5)
ax.vlines(c1_PMF, c2_PMF.min(), c2_PMF.max(), linestyles='dashed', color='gray', alpha=0.5)

pd.DataFrame(np.stack((c1_PMF,e2e_F, e2e_Fu), axis=-1), columns=["End-to-End (nm)", "FE (kT)", "Uncertainty"]).to_csv("full_data.csv")

#plt.savefig("./PMF_fastmbar2_uncertainty_blk_H1.png", dpi=300)
plt.savefig("./PMF_fastmbar2_uncertainty_blk.png", dpi=300)
plt.close()
