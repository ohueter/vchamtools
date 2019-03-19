import numpy as np
import math

__all__ = ["Atom", "Molecule"]

a0 = 5.2917720859E-11   # Bohr radius
eV = 0.000123984        # cm-1 to eV; eV = hc 10^2 / e
Eh = 27.2114            # Hartree energy
u = 1.66053892E-27      # atomic mass unit
me = 9.10938291E-31     # electron mass
atom_numbers = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'F': 9.0}

class Atom(object):
    def __init__(self, symbol = 'H', charge=1.0, coords=[0,0,0]):
        self.basis = ''
        self.symbol = symbol
        self.charge = charge
        self.coords = np.array(coords)
        self.reset()

    def __str__(self):
        return repr(self.coords)

    def angstrom_coords(self):
        return repr(self.coords * 0.52917720859)

    def move(self, disp=[0,0,0]):
        self.current_coords += np.array(disp)

    def reset(self):
        self.current_coords = np.array(self.coords)

class Molecule(object):
    def __init__(self, filename=None):
        self.atoms = []
        self.modes_frequencies = []
        self.modes_displacement_vectors = []
        self.modes_reduced_masses = []
        if filename is not None:
            self.read_aoforce_output(filename)

    def number_of_atoms(self):
        return len(self.atoms)

    def number_of_modes(self):
        return len(self.modes_frequencies)

    def reorder_atoms(self, order=[]):
        if not len(order) == self.number_of_atoms():
            raise ValueError('order array does not match number of atoms')

        self.atoms = [self.atoms[i-1] for i in order]
        self.modes_displacement_vectors = [[mode[i-1] for i in order] for mode in self.modes_displacement_vectors]

    def reorder_modes(self, order=[]):
        if not len(order) == self.number_of_modes():
            raise ValueError('order array does not match number of modes')

        self.modes_frequencies = [self.modes_frequencies[i-1] for i in order]
        self.modes_reduced_masses = [self.modes_reduced_masses[i-1] for i in order]
        self.modes_displacement_vectors = [self.modes_displacement_vectors[i-1] for i in order]

    def rotate(self, axis='x', degrees=90):
        axes_names = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}
        axis = axes_names[axis.lower()]
        theta = degrees * np.pi / 180
        R = rotation_matrix(axis, theta)

        for atom in self.atoms:
            atom.coords = np.dot(R, atom.coords)
            atom.current_coords = np.dot(R, atom.current_coords)

        self.modes_displacement_vectors = [[np.dot(R, vector) for vector in mode] for mode in self.modes_displacement_vectors]

    def reset_to_equilibrium(self):
        for atom in self.atoms:
            atom.reset()

    def add_atom_displacement(self, mode, Q):
        # displace atoms by Q along mode
        # parameters may be lists
        if type(mode) is not list:
            mode = [mode]
        if type(Q) is not list:
            Q = [Q]
        if not len(mode) == len(Q):
            raise ValueError('mode and Q list lengths do not match')

        for m, q in zip(mode, Q):
            for atom, disp in zip(self.atoms, self.modes_displacement_vectors[m-1]):
                atom.move(q*disp)

    def set_atom_displacement(self, mode, Q):
        # displace atoms by Q along mode
        # parameters may be lists
        if type(mode) is not list:
            mode = [mode]
        if type(Q) is not list:
            Q = [Q]
        if not len(mode) == len(Q):
            raise ValueError('mode and Q list lengths do not match')

        self.reset_to_equilibrium()
        self.add_atom_displacement(mode, Q)

    def firefly_coords(self, eq=False):
        inplines = []
        for atom in self.atoms:
            line = []
            line.append(atom.symbol)
            line.append('\t')
            line.append('%.1f' % atom.charge)
            line.append('\t')
            if eq == True:
                line.append('\t'.join(['%.14f'%x for x in atom.coords]))
            else:
                line.append('\t'.join(['%.14f'%x for x in atom.current_coords]))
            line.append('\n')
            inplines.append(''.join(line))
        return ''.join(inplines)

    def firefly_coords_with_basis(self, eq=False):
        inplines = []
        for atom in self.atoms:
            line = []
            line.append(atom.symbol)
            line.append('\t')
            line.append('%.1f' % atom.charge)
            line.append('\t')
            if eq == True:
                line.append('\t'.join(['%.14f'%(x*0.52917720859) for x in atom.coords]))
            else:
                line.append('\t'.join(['%.14f'%(x*0.52917720859) for x in atom.current_coords]))
            line.append('\n')
            line.append(atom.basis)
            line.append('\n\n')
            inplines.append(''.join(line))
        return ''.join(inplines)

    def read_aoforce_output(self, filename='aoforce.out'):
        with open(filename) as f:
            for line in f:
                # find and read coordinates from aoforce output
                if 'actual cartesian coordinates' in line:
                    for line in f:
                        atom_line = line.strip().split()
                        if len(atom_line) == 5:
                            # parse coordinates
                            atom_symbol = atom_line[1].upper()
                            atom_charge = atom_numbers[atom_symbol]
                            atom_coords = [float(i) for i in atom_line[2:5]]
                            atom = Atom(atom_symbol, atom_charge, atom_coords)
                            self.atoms.append(atom)
                        elif len(atom_line) == 1:
                            # ignore separator line
                            continue
                        else:
                            # we have reached the end,
                            # coordinates are followed by an empty line
                            break

                # find and read normal modes
                if line.strip().startswith('mode '):
                    # read mode numbers
                    mode_numbers = line.split()
                    del mode_numbers[0] # delete label
                    mode_numbers = [int(i) for i in mode_numbers]
                    for _ in mode_numbers:
                        self.modes_displacement_vectors.append([])
                    #print mode_numbers

                    # nested loop for reading only this block of normal mode displacement vectors
                    for line in f:
                        stripped_line = line.strip()
                        # read mode frequencies
                        if stripped_line.startswith('frequency'):
                            frequencies = stripped_line.split()
                            del frequencies[0] # delete label
                            frequencies = [float(i) for i in frequencies]
                            self.modes_frequencies.extend(frequencies)
                            # continue with next line
                            continue

                        # read displacement vectors
                        if stripped_line[:4].strip().isdigit():
                            x_displacements = stripped_line.split()
                            y_displacements = next(f).split()
                            z_displacements = next(f).split()

                            # remove trailing items
                            del x_displacements[:3]
                            del y_displacements[0]
                            del z_displacements[0]

                            for mode, x_disp, y_disp, z_disp in zip(mode_numbers, x_displacements, y_displacements, z_displacements):
                                disp_vector = np.array([float(i) for i in [x_disp, y_disp, z_disp]])
                                self.modes_displacement_vectors[mode-1].append(disp_vector)

                            # continue with next line
                            continue

                        # read reduced masses
                        if stripped_line.startswith('reduced mass'):
                            reduced_masses = stripped_line.split()
                            del reduced_masses[:2] # delete label
                            reduced_masses = [float(i) for i in reduced_masses]
                            self.modes_reduced_masses.extend(reduced_masses)
                            # reduced masses are last line of block
                            # quit nested loop
                            break

        # delete rotational and translational modes
        if self.number_of_atoms() > 2:
            trans_rot_degrees_of_freedom = 6
        else:
            trans_rot_degrees_of_freedom = 5
        del self.modes_frequencies[:trans_rot_degrees_of_freedom]
        del self.modes_reduced_masses[:trans_rot_degrees_of_freedom]
        del self.modes_displacement_vectors[:trans_rot_degrees_of_freedom]

        # convert mode displacements from turbomole units to [amu^(1/2) a0]
        for idx, disp_vector in enumerate(self.modes_displacement_vectors):
            freq = self.modes_frequencies[idx]
            mred = self.modes_reduced_masses[idx]
            disp_au = disp_vector / np.sqrt(freq * eV / Eh * mred * u / me)
            self.modes_displacement_vectors[idx] = disp_au


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
