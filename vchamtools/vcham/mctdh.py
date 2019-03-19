#!/usr/bin/env python3

import numpy as np
from vchamtools.vcham import vcham
from itertools import combinations

# TODO:
# load and save H from disk

# add mode numbers to H class
# enables also better plotting! make plotting package for H


def op_parameter_section(H, states=None):
    """states list includes all states to output, beginning with 1"""
    # if len(mode_numbers) != H.n_modes:
    #     raise ValueError('len(mode_numbers) != H.n_modes')
    n_states = len(H.E)

    if states is not None:
        if len(states) > n_states:
            # too many states given
            raise ValueError('len(states) > n_states')
        if len(states) != len(set(states)):
            # duplicate entries in states list
            # remove or throw error?
            raise ValueError('duplicate entries in states list')
        states.sort()
    else:
        # no states give -> include all states
        states = list(range(1, n_states+1))

    # indices of the coupling elements of the selected states in c_lambda and c_eta array
    # coupling_element_indices = [vcham.get_index_of_triangular_element_in_flattened_matrix(*state_pair, n_states) for state_pair in combinations(states, 2)]
    # example: states=[1, 3, 8]  --> indices = 1, 6, 25  -->  coupling_element_indices = {1: '12', 6: '13', 25: '23'}
    coupling_element_indices = {vcham.get_index_of_triangular_element_in_flattened_matrix(*state_pair, n_states):''.join([str(states.index(i)+1) for i in state_pair]) for state_pair in combinations(states, 2)}

    str_parameter_section = 'PARAMETER-SECTION\n'
    # ground state mode frequencies
    for i, w in enumerate(H.w):
        str_parameter_section += 'w_' + str(H.modes[i]) + ' = ' + str(w).replace('e', 'd') + ', ev\n'

    # vertical excitation energies
    for i in range(len(states)):
        str_parameter_section += 'E_' + str(i+1) + ' = ' + str(H.E[states[i]-1]).replace('e', 'd') + ', ev\n'

    it = np.nditer(H.c_kappa, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'kappa' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    it = np.nditer(H.c_gamma, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'gamma' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    it = np.nditer(H.c_rho, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'rho' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    it = np.nditer(H.c_sigma, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'sigma' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    it = np.nditer(H.c_lambda, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0] in coupling_element_indices:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'lambda' + str(coupling_element_indices[it.multi_index[0]]) + '_' + str(H.modes[it.multi_index[1]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    it = np.nditer(H.c_eta, flags=['multi_index'], order='C')
    while not it.finished:
        if not np.isclose(it.value, 0.) and it.multi_index[0] in coupling_element_indices:
            str_value = str(it.value).replace('e', 'd')
            str_parameter_section += 'eta' + str(coupling_element_indices[it.multi_index[0]])  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + ' = ' + str_value + ', ev\n'
        it.iternext()

    str_parameter_section += 'end-parameter-section'
    return str_parameter_section

def op_hamiltonian_section(H, states=None):
    """states list includes all states to output, beginning with 1"""
    # assumptions: first entry in states list = photoexcited state
    # add one state = last state = ground state
    # if len(mode_numbers) != H.n_modes:
    #     raise ValueError('len(mode_numbers) != H.n_modes')
    n_states = len(H.E)
    if states is not None:
        if len(states) > n_states:
            # too many states given
            raise ValueError('len(states) > n_states')
        if len(states) != len(set(states)):
            # duplicate entries in states list
            # remove or throw error?
            raise ValueError('duplicate entries in states list')
        states.sort()
    else:
        # no states give -> include all states
        states = list(range(1, n_states+1))

    # indices of the coupling elements of the selected states in c_lambda and c_eta array
    # coupling_element_indices = [vcham.get_index_of_triangular_element_in_flattened_matrix(*state_pair, n_states) for state_pair in combinations(states, 2)]
    # example: states=[1, 3, 8]  --> indices = 1, 6, 25  -->  coupling_element_indices = {1: '12', 6: '13', 25: '23'}
    coupling_element_indices = {vcham.get_index_of_triangular_element_in_flattened_matrix(*state_pair, n_states):''.join([str(states.index(i)+1) for i in state_pair]) for state_pair in combinations(states, 2)}

    str_hamiltonian_section = 'HAMILTONIAN-SECTION\n'

    # table header
    str_header = ' modes\t\t| '
    for i in H.modes:
        str_header += 'Q' + str(i) + '\t| '
    str_header += 'el\t| Time\t'
    str_header = str_header.expandtabs(8)
    delimiter = ''.join(['-']*len(str_header))
    str_hamiltonian_section += delimiter + '\n' + str_header + '\n' + delimiter + '\n'

    # kinetic energy terms
    for i in H.modes:
        # str_current_line = '-0.5*w_' + str(i) + '\t| '
        str_current_line = 'w_' + str(i) + '\t\t| '
        for j in H.modes:
            if i == j:
                # str_current_line += 'dq^2\t| '
                str_current_line += 'KE\t| '
            else:
                str_current_line += '1\t| '
        str_current_line += '1\t| 1\n'
        str_hamiltonian_section += str_current_line.expandtabs(8)
    str_hamiltonian_section += '\n'

    # potential energy terms
    for i in H.modes:
        str_current_line = '0.5*w_' + str(i) + '\t| '
        for j in H.modes:
            if i == j:
                str_current_line += 'q^2\t| '
            else:
                str_current_line += '1\t| '
        str_current_line += '1\t| 1\n'
        str_hamiltonian_section += str_current_line.expandtabs(8)
    str_hamiltonian_section += '\n'

    # vertical excitation energies
    for i in range(1, len(states)+1):
        str_current_line = 'E_' + str(i) + '\t\t| '
        for _ in H.modes: str_current_line += '1\t| '
        str_current_line += 'S' + str(i) + '&' + str(i) + '\t| 1\n'
        str_hamiltonian_section += str_current_line.expandtabs(8)
    str_hamiltonian_section += '\n'

    # kappa coefficients
    it = np.nditer(H.c_kappa, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_current_line += 'kappa' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1]:
                    str_current_line += 'q\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + str(states.index(it.multi_index[0]+1)+1) + '&' + str(states.index(it.multi_index[0]+1)+1) + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    # gamma coefficients
    it = np.nditer(H.c_gamma, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_current_line += 'gamma' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1] and i == it.multi_index[2]:
                    str_current_line += 'q^2\t| '
                elif i == it.multi_index[1] or i == it.multi_index[2]:
                    str_current_line += 'q\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + str(states.index(it.multi_index[0]+1)+1) + '&' + str(states.index(it.multi_index[0]+1)+1) + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    # rho coefficients
    it = np.nditer(H.c_rho, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_current_line += 'rho' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1] and i == it.multi_index[2]:
                    str_current_line += 'q^3\t| '
                elif i == it.multi_index[2]:
                    str_current_line += 'q^2\t| '
                elif i == it.multi_index[1]:
                    str_current_line += 'q\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + str(states.index(it.multi_index[0]+1)+1) + '&' + str(states.index(it.multi_index[0]+1)+1) + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    # sigma coefficients
    it = np.nditer(H.c_sigma, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0]+1 in states:
            str_current_line += 'sigma' + str(states.index(it.multi_index[0]+1)+1)  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1] and i == it.multi_index[2]:
                    str_current_line += 'q^4\t| '
                elif i == it.multi_index[1] or i == it.multi_index[2]:
                    str_current_line += 'q^2\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + str(states.index(it.multi_index[0]+1)+1) + '&' + str(states.index(it.multi_index[0]+1)+1) + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    # # lambda coefficients
    it = np.nditer(H.c_lambda, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0] in coupling_element_indices:
            str_current_line += 'lambda' + str(coupling_element_indices[it.multi_index[0]])  + '_' + str(H.modes[it.multi_index[1]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1]:
                    str_current_line += 'q\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + coupling_element_indices[it.multi_index[0]][:1] + '&' + coupling_element_indices[it.multi_index[0]][1:] + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    it = np.nditer(H.c_eta, flags=['multi_index'], order='C')
    while not it.finished:
        str_current_line = ''
        if not np.isclose(it.value, 0.) and it.multi_index[0] in coupling_element_indices:
            str_current_line += 'eta' + str(coupling_element_indices[it.multi_index[0]])  + '_' + str(H.modes[it.multi_index[1]]) + str(H.modes[it.multi_index[2]]) + '\t| '
            for i in range(len(H.modes)):
                if i == it.multi_index[1] and i == it.multi_index[2]:
                    str_current_line += 'q^3\t| '
                elif i == it.multi_index[2]:
                    str_current_line += 'q^2\t| '
                elif i == it.multi_index[1]:
                    str_current_line += 'q\t| '
                else:
                    str_current_line += '1\t| '
            str_current_line += 'S' + coupling_element_indices[it.multi_index[0]][:1] + '&' + coupling_element_indices[it.multi_index[0]][1:] + '\t| 1\n'
            str_hamiltonian_section += str_current_line.expandtabs(8)
        it.iternext()
    str_hamiltonian_section += '\n'

    # laser pulse
    str_current_line = '-1.0*e0\t\t| '
    for _ in H.modes: str_current_line += '1\t| '
    str_current_line += 'S1&2' #+ str(len(states)+1)
    str_current_line += '\t| carrier*env\n'
    str_hamiltonian_section += str_current_line.expandtabs(8)

    str_hamiltonian_section += delimiter + '\n' + 'end-hamiltonian-section'
    return str_hamiltonian_section
