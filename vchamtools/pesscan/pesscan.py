#!/usr/bin/env python3

import os
import time
import datetime
from shutil import copyfile
import errno
import subprocess
import numpy as np
import numbers

from . import firefly

__all__ = ['PESScan', 'make_absolute_path', 'make_sure_path_exists']

def make_absolute_path(path):
    return os.path.normpath(os.path.join(os.getcwd(), path))

# creates path if not exists
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path

class PESScan(object):
    def __init__(self, molecule):
        self.molecule = molecule
        self.dry_run = False
        self.inputfile = self.punchfile = self.modes = self.coords = self.working_directory = None

    def modes_to_string(self, Qi=None):
        if Qi is None:
            return 'Q' + '_Q'.join([str(m) for m in self.modes])
        else:
            return 'Q' + '_Q'.join([str(m) + '_' + str('%.3f'%qi) for m, qi in zip(self.modes, Qi)])

    def perform_scan(self):
        if any(e is None for e in (self.inputfile, self.punchfile, self.modes, self.coords, self.working_directory)):
            raise ValueError('not all required parameters are set')

        os.chdir(self.working_directory)

        last_punchfile = self.punchfile
        for i, Q in enumerate(self.coords):
            # set molecule to coordinates for this step
            if isinstance(Q, (numbers.Integral, numbers.Real)):
                self.molecule.set_atom_displacement(self.modes, [Q]*len(self.modes))
                # create filenames
                this_step_inpfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.inp'
                this_step_punchfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.dat'
                this_step_outfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.out'
            elif isinstance(Q, (list, np.ndarray)):
                Qlist = Q.tolist()
                self.molecule.set_atom_displacement(self.modes, Qlist)
                this_step_inpfile = self.modes_to_string(Qlist) + '.inp'
                this_step_punchfile = self.modes_to_string(Qlist) + '.dat'
                this_step_outfile = self.modes_to_string(Qlist) + '.out'
            else:
                raise ValueError('unsupported type of Q')

            # print('### Q = ', '%.3f'%Q, ' ###')
            print('###  Q =', Q, ' (' + str(i+1) + '/' + str(len(self.coords)) + ')   ###')
            print('\tInput file:', this_step_inpfile)

            copyfile(self.inputfile, this_step_inpfile)
            with open(this_step_inpfile, 'a') as f:
                f.write(self.molecule.firefly_coords())
                f.write(' $end\n')
                # dry run: no vec block to copy, skip
                if not self.dry_run:
                    vec = firefly.get_vec_block(last_punchfile)
                    f.writelines(vec)

            # dry run: skip rest of the loop, continue with next Q
            if self.dry_run:
                continue

            # include temporary files from previous steps, if they exist
            # tempfolders = ''
            # if os.path.exists('fftemp.0'):
            #     tempfolders = ' fftemp.*'

            t_start = datetime.datetime.now()
            print('\tStart:', t_start)
            # os.system('ffsub ' + this_step_inpfile) # + tempfolders)
            cp = subprocess.run(['ffsub', this_step_inpfile], stdout=subprocess.PIPE, universal_newlines=True)
            print('\tPBS_ID:', cp.stdout.strip('\n'))

            # wait for calculation to finish
            while not os.path.exists(this_step_punchfile):
                time.sleep(1)

            t_end = datetime.datetime.now()
            print('\tEnd:', t_end)
            print('\tUsed Walltime:', t_end-t_start)
            last_punchfile = this_step_punchfile

            # safety delay to wait for file transfer from node
            time.sleep(10)

    def perform_scan_with_basis(self):
        if any(e is None for e in (self.inputfile, self.punchfile, self.modes, self.coords, self.working_directory)):
            raise ValueError('not all required parameters are set')

        os.chdir(self.working_directory)

        last_punchfile = self.punchfile
        for Q in self.coords:
            # set molecule to coordinates for this step
            self.molecule.set_atom_displacement(self.modes, [Q]*len(self.modes))

            # create filenames
            this_step_inpfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.inp'
            this_step_punchfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.dat'
            this_step_outfile = self.modes_to_string() + '_' + '%2.3f'%Q + '.out'

            print('### Q = ', '%.3f'%Q, ' ###')
            print('\tInput file:', this_step_inpfile)

            copyfile(self.inputfile, this_step_inpfile)
            with open(this_step_inpfile, 'a') as f:
                f.write(self.molecule.firefly_coords_with_basis())
                f.write(' $end\n')
                # dry run: no vec block to copy, skip
                if not self.dry_run:
                    vec = firefly.get_vec_block(last_punchfile)
                    f.writelines(vec)

            # dry run: skip rest of the loop, continue with next Q
            if self.dry_run:
                continue

            # include temporary files from previous steps, if they exist
            # tempfolders = ''
            # if os.path.exists('fftemp.0'):
            #     tempfolders = ' fftemp.*'

            print('\tStart:', datetime.datetime.now())
            # os.system('ffsub ' + this_step_inpfile) # + tempfolders)
            cp = subprocess.run(['ffsub', this_step_inpfile], stdout=subprocess.PIPE, universal_newlines=True)
            print('\tPBS_ID:', cp.stdout.strip('\n'))

            # wait for calculation to finish
            while not os.path.exists(this_step_punchfile):
                time.sleep(1)

            print('\tEnd:', datetime.datetime.now())
            last_punchfile = this_step_punchfile

            # safety delay to wait for file transfer from node
            time.sleep(10)
