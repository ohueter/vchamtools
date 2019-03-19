#!/usr/bin/env python3

'''
Scan PEHS from starting structure defined in `start_aoforce.out``
to end structure defined in `end_aoforce.out` in an arbitrary number
of steps as defined by the `--steps` parameter.

Example:
$ ./Qeff_scan.py -i firefly-input.inp -p MOs.dat -f scans/q_eff -s S0_aoforce.out -e S1_aoforce.out -n 10
'''

from vchamtools import molecule, pesscan
import numpy as np
import argparse

# get command line options
parser = argparse.ArgumentParser(description='Print molecule coordinates after distortion(s)')
parser.add_argument('-f', '--folder', dest='folder', type=str, default='.', metavar='<folder>',
                    help='start calculation in <folder> (default = .)')
parser.add_argument("-s", "--start", dest="start_struct", metavar="<file>", default='start_aoforce.out', required=True,
                    help="read starting structure from turbomole aoforce output <file> (default = start_aoforce.out)")
parser.add_argument("-e", "--end", dest="end_struct", metavar="<file>", default='end_aoforce.out', required=True,
                    help="read final structure from turbomole aoforce output <file> (default = end_aoforce.out)")
parser.add_argument('-i', '--input', dest='inputfile', metavar='<input>', type=str, required=True,
                    help='input template file name')
parser.add_argument('-p', '--punch', dest='punchfile', metavar='<punchfile>', type=str, required=True,
                    help='initial punchfile name')
parser.add_argument('-n', '--steps', dest='steps', metavar='<n>', type=int, required=True,
                    help='number of linear steps')
parser.add_argument('-d', '--dry', dest='dry', default=False, action='store_true',
                    help='dry run: only create input files, do not submit to PBS')
args = parser.parse_args()

# load structures
mol_start = molecule.Molecule(args.start_struct)
mol_end = molecule.Molecule(args.end_struct)

# transform turbomole to firefly, if neccessary
# mol_start.reorder_atoms([3,1,4,2,11,12,7,5,8,6,10,9])
# mol_start.rotate(axis='z', degrees=90)
# mol_end.reorder_atoms([3,1,4,2,11,12,7,5,8,6,10,9])
# mol_end.rotate(axis='z', degrees=90)

# calculate linear displacement vectors from starting structure to final structure
mol = molecule.Molecule()
mol.atoms = mol_start.atoms
mol.modes_frequencies = [1]
mol.modes_reduced_masses = [1]
mol.modes_displacement_vectors = [ [atom_end.coords-atom_start.coords for atom_start, atom_end in zip(mol_start.atoms, mol_end.atoms)] ]

# prepare scan
scan = pesscan.PESScan(mol)
scan.inputfile = pesscan.make_absolute_path(args.inputfile)
scan.punchfile = pesscan.make_absolute_path(args.punchfile)
scan.modes = [1]
scan.coords = np.linspace(0, 1, num=args.steps+1)
scan.dry_run = args.dry
scan.working_directory = pesscan.make_sure_path_exists(args.folder)

# scan
print('PES linear scan at Qeff =', scan.coords)

scan.perform_scan()

print('scan finished!')
