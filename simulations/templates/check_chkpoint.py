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
from openabc.forcefields.rigid import createRigidBodies
from openabc.utils.helper_functions import write_pdb, parse_pdb
check_point_file = "/home/gridsan/iriveros/projects/cg_DNA/free_energy_chromatin_collab/tetra-nucl-stretch/simulations/10bp_restraint_simulations_rerun/sim_e2e-center_67.2500_d13-d24-center_-7.2708/sim_output/checkpoint.chk"
#check_point_file = "/home/gridsan/iriveros/projects/cg_DNA/free_energy_chromatin_collab/tetra-nucl-stretch/simulations/10bp_restraint_simulations/sim_e2e-center_38.2500_d13-d24-center_14.5625/sim_output/checkpoint.chk"

build_dir = '../build_system'
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


simulation.loadCheckpoint(check_point_file)
print(simulation.context.getParameter('k1'))
print(simulation.context.getParameter('k2'))
print(simulation.context.getParameter('k3'))
print(simulation.context.getParameter('k4'))
