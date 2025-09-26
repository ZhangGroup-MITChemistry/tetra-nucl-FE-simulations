import numpy as np
import os
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import parmed as pmd

from openabc.forcefields.parsers import SMOGParser, DNA3SPN2Parser
from openabc.forcefields import SMOG3SPN2Model
from openabc.utils.helper_functions import get_WC_paired_sequence, write_pdb

def make_topology(model):
    top = app.topology.Topology()
    chains = {}
    chain = None
    last_chain = None
    atoms = {}
    for idx, atom in model.atoms.iterrows():
        orig_chainID = atom['chainID']
        if orig_chainID != last_chain:
            if orig_chainID not in chains:
                chains[orig_chainID] = 0
            chains[orig_chainID] += 1
            chainID = f'{orig_chainID}{chains[orig_chainID]}'
            chain = top.addChain(chainID)
        residue = top.addResidue(atom['resname'], chain)
        element = app.Element.getBySymbol('C')
        assert idx not in atoms
        if atom['name'] == 'DN':
            name = 'NU'
        else:
            name = atom['name']
        atoms[idx] = top.addAtom(name, element, residue)
        last_chain = orig_chainID

    for idx, bond in model.dna_bonds.iterrows():
        top.addBond(atoms[int(bond['a1'])], atoms[int(bond['a2'])])

    for idx, bond in model.protein_bonds.iterrows():
        top.addBond(atoms[int(bond['a1'])], atoms[int(bond['a2'])])

    parmed_top:pmd.structure.Structure = pmd.openmm.load_topology(top)
    unique_names = np.unique(model.atoms['name'].to_numpy())
    type_dict = {}
    for i,n in enumerate(unique_names):
        mass = model.atoms[model.atoms['name'] == n].iloc[0]
        type_dict[n] = pmd.AtomType(n,i,mass)

    for i,atom in enumerate(parmed_top.atoms):
        parmed_top.view[i].atom_type = type_dict[parmed_top.view[i].name]

    parmed_top.save(f"cg_tetra_nucleosome.psf",overwrite=True)

"""
Build system xml file without any rigid bodies.
In this system, bonds and angles are all kept, and exclusions are all kept. 
"""

if not os.path.exists('intermediate-files'):
    os.makedirs('intermediate-files')

n_single_nucl = 26
box_a, box_b, box_c = 1000.0, 1000.0, 1000.0
chromatin = SMOG3SPN2Model()


res_map = {"G":"GLY", "A":"ALA", "L":"LEU", "M":"MET", "F":"PHE", "W":"TRP", "K":"LYS", "Q":"GLN", "E":"GLU", "S":"SER",
           "P":"PRO", "V":"VAL", "I":"ILE", "C":"CYS", "Y":"TYR", "H":"HIS", "R":"ARG", "N":"ASN", "D":"ASP", "T":"THR"}

res_ref = {'A': "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLAAIHAKRVTIMPKDIQLARRIRGERA",
           'E': "ARTKQTARKSTGGKAPRKQLATKAARKSAPATGGVKKPHRYRPGTVALREIRRYQKSTELLIRKLPFQRLVREIAQDFKTDLRFQSSAVMALQEACEAYLVGLFEDTNLAAIHAKRVTIMPKDIQLARRIRGERA",

           'B': "SGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG",
           'F': "SGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRLARRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG",

           'C': "SGRGKQGGKARAKAKSRSSRAGLQFPVGRVHRLLRKGNYAERVGAGAPVYLAAVLEYLTAEILELAGNAARDNKKTRIIPRHLQLAIRNDEELNKLLGRVTIAQGGVLPNIQAVLLPKCTESHHKAKGK",
           'G': "SGRGKQGGKARAKAKSRSSRAGLQFPVGRVHRLLRKGNYAERVGAGAPVYLAAVLEYLTAEILELAGNAARDNKKTRIIPRHLQLAIRNDEELNKLLGRVTIAQGGVLPNIQAVLLPKCTESHHKAKGK",

           'D': "PEPAKSAPAPKKGSKKAVTKAQKKDGKKRKRSRKESYSVYVYKVLKQVHPDTGISSKAMGIMNSFVNDIFERIAGEASRLAHYNKRSTITSREIQTAVRLLLPGELAKHAVSEGTKAVTKYTSSK",
           'H': "PEPAKSAPAPKKGSKKAVTKAQKKDGKKRKRSRKESYSVYVYKVLKQVHPDTGISSKAMGIMNSFVNDIFERIAGEASRLAHYNKRSTITSREIQTAVRLLLPGELAKHAVSEGTKAVTKYTSSK"}

def apply_res_map(s):
    seq = []
    for c in s:
        seq.append(res_map[c])
    return seq
pdb_dir = "../tetra-nucl-nrl-177-add-50bp-tails/build-fiber-pdb/fiber-177x4-50bp-tails"

# load tetranucleosome histones
for i in range(4):
    histone_i_parser = SMOGParser.from_atomistic_pdb(f'{pdb_dir}/histone_{i}_fix.pdb', 
                                                     f'intermediate-files/cg_tetra_nucl_histone_{i}_fix.pdb', 
                                                     default_parse=False)
    '''Modify Histone Sequences to match above sequences before parsing'''
    i = 0
    for chain in histone_i_parser.atoms['chainID'].unique():
        resnames = histone_i_parser.atoms.loc[histone_i_parser.atoms['chainID'] == chain]['resname'].values
        histone_i_parser.atoms.loc[histone_i_parser.atoms['chainID'] == chain, 'resname'] = apply_res_map(res_ref[chain])
        i = 0
        for row in histone_i_parser.atoms.loc[histone_i_parser.atoms['chainID'] == chain].iterrows():
            assert row[1]['resname'] == apply_res_map(res_ref[chain])[i], f"Not equal: {row[1]['resname'], apply_res_map(res_ref[chain])[i]}, idx: {i}, chain: {chain}"
            i += 1

    histone_i_parser.parse_mol(get_native_pairs=False)
    chromatin.append_mol(histone_i_parser)

# load tetranucleosome DNA
with open('tetra_nucl_nrl_177_dna_seq.txt', 'r') as f:
    seq1 = f.readlines()[0].strip()
full_seq1 = seq1 + get_WC_paired_sequence(seq1)

dna_parser = DNA3SPN2Parser.from_atomistic_pdb(f'{pdb_dir}/chromatin_dna.pdb', 
                                               'intermediate-files/cg_tetra_nucl_dna.pdb', new_sequence=full_seq1, 
                                               temp_name='dna1')
chromatin.append_mol(dna_parser)

chromatin.atoms_to_pdb('intermediate-files/cg_tetra_nucl_original_coord.pdb')
n_tetra_nucl_atoms = len(chromatin.atoms.index)
print(f'{n_tetra_nucl_atoms} CG atoms in tetranucleosome. ')

# put tetranucleosome geometric center at the center of the box
tetra_nucl_atoms = chromatin.atoms.copy()
coord = tetra_nucl_atoms[['x', 'y', 'z']].to_numpy()
coord -= np.mean(coord, axis=0)
coord += 10*np.array([box_a, box_b, box_c])/2
tetra_nucl_atoms[['x', 'y', 'z']] = coord
tetra_nucl_atoms.loc[:, 'charge'] = ''
write_pdb(tetra_nucl_atoms, 'cg_tetra_nucl.pdb')

'''Not needed since only the tetranucleosome is simulated'''
top = app.PDBFile('cg_tetra_nucl.pdb').getTopology()

chromatin.create_system(top, box_a=box_a, box_b=box_b, box_c=box_c)
chromatin.add_protein_bonds(force_group=1)
chromatin.add_protein_angles(force_group=2)
chromatin.add_dna_bonds(force_group=5)
chromatin.add_dna_angles(force_group=6)
chromatin.add_dna_stackings(force_group=7)
chromatin.add_dna_dihedrals(force_group=8)
chromatin.add_dna_base_pairs(force_group=9)
chromatin.add_dna_cross_stackings(force_group=10)
chromatin.parse_all_exclusions()
chromatin.add_all_vdwl(force_group=11)
chromatin.add_all_elec(salt_conc=200*unit.millimolar, force_group=12)

make_topology(chromatin)

'''
nucl1: CENTER ATOMS=44-135:3,160-237:3,258-352:3,401-487:3,531-622:3,647-724:3,745-839:3,888-982:3 
nucl2: CENTER ATOMS=1026-1117:3,1142-1219:3,1240-1334:3,1383-1469:3,1513-1604:3,1629-1706:3,1727-1821:3,1870-1956:3 
nucl3: CENTER ATOMS=2000-2091:3,2116-2193:3,2214-2308:3,2357-2443:3,2487-2578:3,2603-2680:3,2701-2795:3,2844-2930:3
nucl4: CENTER ATOMS=2974-3065:3,3090-3167:3,3188-3282:3,3331-3417:3,3461-3552:3,3577-3654:3,3675-3769:3,3818-3904:3 
end1:  CENTER ATOMS=3929,3930,8592,8593,8594
end2:  CENTER ATOMS=6259,6260,6261,6262,6263
'''

masses = np.array([chromatin.system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(len(chromatin.atoms))])
e2e = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
group1 = np.array([3929,3930,8592,8593,8594]) - 1
group2 = np.array([6259,6260,6261,6262,6263]) - 1
g1 = e2e.addGroup(group1, masses[group1])
g2 = e2e.addGroup(group2, masses[group2])
e2e.addBond([g1,g2])

d13 = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
nucl1 = np.concatenate([np.arange(44,135+1,3),np.arange(160,237+1,3),np.arange(258,352+1,3),np.arange(401,487+1,3),np.arange(531,622+1,3),np.arange(647,724+1,3),np.arange(745,839+1,3),np.arange(888,982+1,3)])- 1
nucl3 = np.concatenate([np.arange(2000,2091+1,3),np.arange(2116,2193+1,3),np.arange(2214,2308+1,3),np.arange(2357,2443+1,3),np.arange(2487,2578+1,3),np.arange(2603,2680+1,3),np.arange(2701,2795+1,3),np.arange(2844,2930+1,3)]) - 1
n1 = d13.addGroup(nucl1, masses[nucl1])
n3 = d13.addGroup(nucl3, masses[nucl3])
d13.addBond([n1,n3])

d24 = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
nucl2 = np.concatenate([np.arange(1026,1117+1,3),np.arange(1142,1219+1,3),np.arange(1240,1334+1,3),np.arange(1383,1469+1,3),np.arange(1513,1604+1,3),np.arange(1629,1706+1,3),np.arange(1727,1821+1,3),np.arange(1870,1956+1,3)]) - 1
nucl4 = np.concatenate([np.arange(2974,3065+1,3),np.arange(3090,3167+1,3),np.arange(3188,3282+1,3),np.arange(3331,3417+1,3),np.arange(3461,3552+1,3),np.arange(3577,3654+1,3),np.arange(3675,3769+1,3),np.arange(3818,3904+1,3)]) - 1
n2 = d24.addGroup(nucl2, masses[nucl2])
n4 = d24.addGroup(nucl4, masses[nucl4])
d24.addBond([n2,n4])

d13r = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
nucl1 = np.concatenate([np.arange(44,135+1,3),np.arange(160,237+1,3),np.arange(258,352+1,3),np.arange(401,487+1,3),np.arange(531,622+1,3),np.arange(647,724+1,3),np.arange(745,839+1,3),np.arange(888,982+1,3)])- 1
nucl3 = np.concatenate([np.arange(2000,2091+1,3),np.arange(2116,2193+1,3),np.arange(2214,2308+1,3),np.arange(2357,2443+1,3),np.arange(2487,2578+1,3),np.arange(2603,2680+1,3),np.arange(2701,2795+1,3),np.arange(2844,2930+1,3)]) - 1
n1 = d13r.addGroup(nucl1, masses[nucl1])
n3 = d13r.addGroup(nucl3, masses[nucl3])
d13r.addBond([n1,n3])

d24r = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
nucl2 = np.concatenate([np.arange(1026,1117+1,3),np.arange(1142,1219+1,3),np.arange(1240,1334+1,3),np.arange(1383,1469+1,3),np.arange(1513,1604+1,3),np.arange(1629,1706+1,3),np.arange(1727,1821+1,3),np.arange(1870,1956+1,3)]) - 1
nucl4 = np.concatenate([np.arange(2974,3065+1,3),np.arange(3090,3167+1,3),np.arange(3188,3282+1,3),np.arange(3331,3417+1,3),np.arange(3461,3552+1,3),np.arange(3577,3654+1,3),np.arange(3675,3769+1,3),np.arange(3818,3904+1,3)]) - 1
n2 = d24r.addGroup(nucl2, masses[nucl2])
n4 = d24r.addGroup(nucl4, masses[nucl4])
d24r.addBond([n2,n4])

CV1 = mm.CustomCVForce(f'0.5*k1*(e2e-c1)^2')
CV1.addGlobalParameter("k1", 0.0)
CV1.addGlobalParameter("c1", 0)
CV1.addCollectiveVariable("e2e", e2e)
CV1.setForceGroup(13)
chromatin.system.addForce(CV1)

CV2 = mm.CustomCVForce(f'0.5*k2*((d13 - d24) - c2)^2')
CV2.addGlobalParameter("k2", 0.0)
CV2.addGlobalParameter("c2", 0)
CV2.addCollectiveVariable("d13", d13)
CV2.addCollectiveVariable("d24", d24)
CV2.setForceGroup(14)
chromatin.system.addForce(CV2)

CV3 = mm.CustomCVForce(f'0.5*k3*(d13r - c3)^2')
CV3.addGlobalParameter("k3", 0.0)
CV3.addGlobalParameter("c3", 0)
CV3.addCollectiveVariable("d13r", d13r)
CV3.setForceGroup(15)
chromatin.system.addForce(CV3)

CV4 = mm.CustomCVForce(f'0.5*k4*(d24r - c4)^2')
CV4.addGlobalParameter("k4", 0.0)
CV4.addGlobalParameter("c4", 0)
CV4.addCollectiveVariable("d24r", d24r)
CV4.setForceGroup(16)
chromatin.system.addForce(CV4)
chromatin.save_system('nonrigid_system.xml')


