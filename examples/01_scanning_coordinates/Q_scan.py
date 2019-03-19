#!/usr/bin/env python3

'''
Scan PEHS from starting structure defined in `aoforce.out``
along normal coordinates in an arbitrary number of steps 
and linear combinations of coordinates
as defined by the `--coords` or the `--qfile` parameter.

Example:
$ ./Q_scan.py -i firefly-input.inp -p MOs.dat -f scans/q14q13 -m 14 13 -Qf QQ.txt
'''

from vchamtools import molecule, pesscan
import argparse
import numpy as np

# get command line options
parser = argparse.ArgumentParser(description='Print molecule coordinates after distortion(s)')
parser.add_argument('-f', '--folder', dest='folder', type=str, default='.', metavar='<folder>',
                    help='start calculation in <folder> (default = .)')
parser.add_argument("-a", "--aoforce", dest="aoforce", metavar="<file>", default='aoforce.out',
                    help="read molecule data from turbomole aoforce output <file> (default = aoforce.out)")
parser.add_argument('-i', '--input', dest='inputfile', metavar='<input>', type=str, required=True,
                    help='input template file name')
parser.add_argument('-p', '--punch', dest='punchfile', metavar='<punchfile>', type=str, required=True,
                    help='initial punchfile name')
parser.add_argument('-d', '--dry', dest='dry', default=False, action='store_true',
                    help='dry run: only create input files, do not submit to PBS')
parser.add_argument('-m', '--modes', dest='modes', metavar='m_i', type=int, nargs='+', required=True,
                    help='scan PES along these mode(s)')
parser.add_argument('-Q', '--coords', dest='coords', metavar='Q_i', type=float, nargs='+', required=False,
                    help='coordinates Q used in the scan (same for all modes)')
parser.add_argument("-Qf", "--qfile", dest="qfile", metavar="<file>",
                    help="read Q coordinates to scan from <file>")

args = parser.parse_args()

if args.coords is None and args.qfile is None:
    raise ValueError('either -Q or -Qf must be specified')

# load structure
mol = molecule.Molecule(args.aoforce)

# check mode numbers
if any([i > mol.number_of_modes() for i in args.modes]):
    raise ValueError('non-existent mode given')

if args.qfile is not None:
    coords = np.genfromtxt(args.qfile, delimiter='\t')
    if coords.ndim < 2:
        coords_dim = 1
    else:
        coords_dim = coords.shape[-1]

    if not len(args.modes) == coords_dim:
        raise ValueError('number of modes given in -m and read from -Qf does not match')
else:
    coords = np.array(args.coords)

# transform turbomole to firefly, if neccessary
# mol.reorder_atoms([3,1,4,2,11,12,7,5,8,6,10,9])
# mol.reorder_modes([i-6 for i in [36,34,31,30,29,26,25,20,14,12,9,  # 11 a1
#                    23,17,15,13,7, # 5 a2 (v12 - v16)
#                    22,18,11,8, # 4 b1 (v17 - v20)
#                    35,33,32,28,27,24,21,19,16,10]]) # 10 b2
# mol.rotate(axis='z', degrees=90)

scan = pesscan.PESScan(mol)
scan.inputfile = pesscan.make_absolute_path(args.inputfile)
scan.punchfile = pesscan.make_absolute_path(args.punchfile)
scan.modes = args.modes
scan.coords = coords
scan.dry_run = args.dry
scan.working_directory = pesscan.make_sure_path_exists(args.folder)


print('PES scan along mode(s) v =', args.modes, '(w =', [mol.modes_frequencies[i-1] for i in args.modes], 'cm-1) at Q =', np.array2string(coords).replace('\n', '') )

scan.perform_scan()

print('scan finished!')
