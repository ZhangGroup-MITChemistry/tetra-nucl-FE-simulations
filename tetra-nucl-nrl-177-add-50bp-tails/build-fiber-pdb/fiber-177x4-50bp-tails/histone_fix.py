from pdbfixer import PDBFixer
from openmm.app import PDBFile
for i in range(4):
    fixer = PDBFixer(filename=f'histone_{i}.pdb')
    print("PDB opened, fixer created")

    fixer.missingResidues = {
            (2, 121): ['HIS'],
            (6, 121): ['HIS'],
            (3, 0): ['PRO', 'GLU', 'PRO'],
            (7, 0): ['PRO', 'GLU', 'PRO'],
    }

    fixer.findMissingAtoms()
    print(fixer.missingAtoms)
    fixer.addMissingAtoms()
    print("Found missing residues")
    print(fixer.missingResidues)

    print("Added missing residues")

    PDBFile.writeFile(fixer.topology, fixer.positions, open(f'histone_{i}_fix.pdb', 'w'))
    print("New pdb created")

    print("Finished!")
