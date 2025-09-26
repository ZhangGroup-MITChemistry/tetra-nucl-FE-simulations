import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import argparse
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
from openmmplumed import PlumedForce
from openabc.forcefields.rigid import createRigidBodies
from openabc.utils.helper_functions import write_pdb, parse_pdb

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='test-output', help='output dcd path')
parser.add_argument('--output_interval', type=int, default=int(1e4), help='output interval')
parser.add_argument('--n_steps', type=int, default=int(1e7), help='simulation number of steps')
parser.add_argument('--kappa1', default=0.0, type=float)
parser.add_argument('--kappa2', default=0.0, type=float)
parser.add_argument('--center1', default=0.0, type=float)
parser.add_argument('--center2', default=0.0, type=float)
parser.add_argument('--ignore_rst', action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

build_dir = '../../../build_system'
nonrigid_system_xml = f'{build_dir}/nonrigid_system.xml'

pdb = f'{build_dir}/cg_tetra_nucl.pdb'

# prepare initial configuration
box_a, box_b, box_c = 1000.0, 1000.0, 1000.0 # in unit nm
tetra_nucl_atoms = parse_pdb(pdb)
coord = tetra_nucl_atoms[['x', 'y', 'z']].to_numpy()
#center coords and move to center of box (in Angstroms)
coord -= np.mean(coord, axis=0)
coord += 0.5 * 10 * np.array([box_a, box_b, box_c]) # in unit angstroms
tetra_nucl_atoms[['x', 'y', 'z']] = coord
write_pdb(tetra_nucl_atoms, 'aligned_tetra_nucl.pdb')
top = app.PDBFile('aligned_tetra_nucl.pdb').getTopology()
n_atoms = top.getNumAtoms()
init_coord = app.PDBFile('aligned_tetra_nucl.pdb').getPositions()
rigid_coord = init_coord

linker_bp = 30
extension_bp = 50
nrl = linker_bp + 147

n_histone_ca = 982 # the number of CA atoms in each histone
n_ca_atoms = 4 * n_histone_ca
histone_core_idx = np.array([[44, 135], [160, 237], 
                             [258, 352], [401, 487],
                             [531, 622], [647, 724], 
                             [745, 839], [888, 982]]) - 1 # start from 0

# set rigid bodies
n_bp = 147 * 4 + extension_bp * 2 + linker_bp * 3 # total number of base pairs
assert n_atoms == n_ca_atoms + 6 * n_bp - 2
bp_idx_to_atom_idx = {}
for i in range(n_bp):
    bp_idx_to_atom_idx[i] = []
    if i == 0:
        bp_idx_to_atom_idx[i] += (np.arange(2) + n_ca_atoms).tolist()
    else:
        bp_idx_to_atom_idx[i] += (np.arange(3) + n_ca_atoms + 3 * i - 1).tolist()
    if i == n_bp - 1:
        bp_idx_to_atom_idx[i] += (np.arange(2) + n_atoms - 3 * n_bp + 1).tolist()
    else:
        bp_idx_to_atom_idx[i] += (np.arange(3) + n_atoms - 3 * i - 3).tolist()

n_rigid_bp_per_nucl = 10
rigid_bodies = []
for i in range(4):
    rigid_body_i = []
    for each in histone_core_idx:
        rigid_body_i += (np.arange(each[0], each[1] + 1) + i * n_histone_ca).tolist()
    start_bp_idx = i * nrl + extension_bp + int((147 - n_rigid_bp_per_nucl) / 2)
    end_bp_idx = start_bp_idx + n_rigid_bp_per_nucl - 1
    rigid_body_i_dna_atoms = [] # for check
    for j in range(start_bp_idx, end_bp_idx + 1):
        rigid_body_i += bp_idx_to_atom_idx[j]
        rigid_body_i_dna_atoms += bp_idx_to_atom_idx[j]
    rigid_bodies.append(sorted(rigid_body_i))
    
    # check the DNA atom indices included in the rigid body
    rigid_body_i_dna_atoms = sorted(rigid_body_i_dna_atoms)
    a1 = rigid_body_i_dna_atoms[0]
    a4 = rigid_body_i_dna_atoms[-1]
    flag = False
    for j in range(len(rigid_body_i_dna_atoms) - 1):
        if rigid_body_i_dna_atoms[j + 1] != rigid_body_i_dna_atoms[j] + 1:
            if not flag:
                a2 = rigid_body_i_dna_atoms[j]
                a3 = rigid_body_i_dna_atoms[j + 1]
                flag = True
            else:
                print('Error: more than 2 gaps in DNA atom indices!')
    print(f'Rigid body {i}, DNA atom indices are {a1}-{a2} and {a3}-{a4}.')

with open(nonrigid_system_xml, 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())
createRigidBodies(system, rigid_coord, rigid_bodies)

temperature = 300 * unit.kelvin
friction_coeff = 0.01 / unit.picosecond
timestep = 10.0 * unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
platform_name = 'CUDA'
platform = mm.Platform.getPlatformByName(platform_name)
properties = {'Precision': 'mixed'}
simulation = app.Simulation(top, system, integrator, platform, properties)
simulation.context.setPositions(rigid_coord)
output_dcd = f'{args.output_dir}/output.dcd'
chk_points = f'{args.output_dir}/checkpoint.chk'
state_chkpoint = f'{args.output_dir}/state_checkpoint.xml'
state_out = f'{args.output_dir}/state.csv'


n_steps = args.n_steps

state_file = Path(state_out)
chckpoint_file = Path(chk_points)
state_chckpoint_file = Path(state_chkpoint)
warmup_steps = 1000 * 500 + 200 * 500 + 20000
restarting = False

if state_file.exists():
    with open(state_file, 'rb') as f:
        try:  
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        past_steps = int(f.readline().decode().split(',')[0])
        print(f"Goal Steps: {n_steps}, Past Steps: {past_steps}, Remaining Steps: {(n_steps + warmup_steps) - past_steps}")
        n_steps = (n_steps + warmup_steps) - past_steps
        restarting = True
        if n_steps <= 0:
            print("Simulation already complete...")
            exit()


simulation.context.setParameter('c1', args.center1)
simulation.context.setParameter('c2', args.center2)

restrain_c3:bool = False
restrain_c4:bool = False

if args.center2 < 0:
    simulation.context.setParameter('c3', 6.0)
    restrain_c3 = True
elif args.center2 > 0:
    simulation.context.setParameter('c4', 6.0)
    restrain_c4 = True
else:
    simulation.context.setParameter('c3', 6.0)
    simulation.context.setParameter('c4', 6.0)
    restrain_c3 = True
    restrain_c4 = True

if not restarting:
    simulation.minimizeEnergy()

    #Slowly engage restraints
    print("Starting slow restraint application.", flush=True)
    k1_step = np.linspace(0, args.kappa1, 200)
    k2_step = np.linspace(0, args.kappa2, 200)
    k3_step = np.linspace(0, 500,         200)
    k4_step = np.linspace(0, 500,         200)
    k3_s = 0.0
    k4_s = 0.0
    for i in range(k1_step.shape[0]):
        simulation.context.setParameter('k1', k1_step[i])
        simulation.context.setParameter('k2', k2_step[i])
        if restrain_c3:
            k3_s = k3_step[i]
            simulation.context.setParameter('k3', k3_s)
        if restrain_c4:
            k4_s = k4_step[i]
            simulation.context.setParameter('k4', k4_s)
        print(f"Step ({i+1}/{1000})Restraints: k1 = {simulation.context.getParameter('k1'):3.2f}, k2 = {simulation.context.getParameter('k2'):3.2f}, k3 = {simulation.context.getParameter('k3'):3.2f}, k4 = {simulation.context.getParameter('k4'):3.2f}", flush=True)
        simulation.step(500)

    print("Running intermediary equilibration", flush=True)
    simulation.step(10000)

    print("Restraint application finished, now releasing extra restraints.", flush=True)
    k3_step_r = np.linspace(500, 0, 200)
    k4_step_r = np.linspace(500, 0, 200)
    for i in range(k3_step_r.shape[0]):
        if restrain_c3:
            k3_s = k3_step_r[i]
            simulation.context.setParameter('k3', k3_s)
        if restrain_c4:
            k4_s = k4_step_r[i]
            simulation.context.setParameter('k4', k4_s)
        print(f"Step ({i+1}/{200})Restraints: k1 = {simulation.context.getParameter('k1'):3.2f}, k2 = {simulation.context.getParameter('k2'):3.2f}, k3 = {simulation.context.getParameter('k3'):3.2f}, k4 = {simulation.context.getParameter('k4'):3.2f}", flush=True)
        simulation.step(500)

    print("Restaining steps done, equilibrating again", flush=True)
    simulation.step(10000)
else:
    simulation.context.setParameter('k1', args.kappa1)
    simulation.context.setParameter('k2', args.kappa2)
    print("Warm-up bypassed!")

plumed_template = '../../templates/plumed_template.txt'
if restarting:
    plumed_template = '../../templates/plumed_template_restart.txt'

with open(plumed_template, 'r') as f:
    plumed_script = f.read()
plumed_script = plumed_script.replace('INPUT_COLVARS', f'{args.output_dir}/COLVARS')
plumed_script = plumed_script.replace('INPUT_COORD_XYZ', f'{args.output_dir}/coord.xyz')
plumed_script = plumed_script.replace('INPUT_STRIDE', f'{args.output_interval}')

with open(f'{args.output_dir}/plumed.txt', 'w') as f:
    f.write(plumed_script)
force = PlumedForce(plumed_script)
system.addForce(force)

simulation.context.reinitialize(preserveState=True)

if restarting:
    assert chckpoint_file.exists() or state_chckpoint_file.exists(), "Neither Checkpoint nor State exist when restarting!"
    try:
        simulation.loadCheckpoint(chk_points)
    except:
        print("LOADING CHECKPOINT FAILED! -- TRYING STATE INSTEAD...")
        simulation.loadState(state_chkpoint)

dcd_reporter = app.DCDReporter(output_dcd, args.output_interval, 
                               enforcePeriodicBox=True, append=restarting)
state_reporter = app.StateDataReporter(state_out, args.output_interval, step=True, 
                                       time=True, potentialEnergy=True, 
                                       kineticEnergy=True, totalEnergy=True, 
                                       temperature=True, speed=True, append=restarting)
checkpoint_reporter = app.CheckpointReporter(chk_points, args.output_interval)
state_checkpoint_reporter = app.CheckpointReporter(state_chkpoint, args.output_interval, writeState=True)

simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_reporter)
simulation.reporters.append(checkpoint_reporter)
simulation.reporters.append(state_checkpoint_reporter)

simulation.step(n_steps)

