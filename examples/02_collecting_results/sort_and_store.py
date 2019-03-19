#!/usr/bin/env python3

'''
Collect scan results created by Q_scan.py (see ../01_scanning_coordinates)
and store in SQLite database.

Very ugly code full of leftovers from previous versions, works but needs some love.

Example:
$ sort_and_store.py -o data.sqlite -m 30 -n 10 scans/q14q13
'''


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import argparse
from vchamtools import sqlitedb as db

# def create_results_table(conn, n_states, modes):
#     if conn is None:
#         raise ValueError('no sqlite connection given')

#     if n_states < 1 or len(modes) == 0:
#         raise ValueError('incorrect arguments')

#     table_name = 'model_' + str(len(modes)) + '_modes_' + str(n_states) + '_states'
#     columns_coords = ', '.join(['Q' + str(m) + ' REAL' for m in modes])
#     columns_energies = ', '.join(['V' + str(i) + ' REAL' for i in range(1,n_states+1,1)])
#     sql = 'CREATE TABLE IF NOT EXISTS ' + table_name + ' (calc_no INT, ' + columns_coords + ', ' + columns_energies + ')'
#     conn.execute(sql)

#     return table_name

parser = argparse.ArgumentParser(description='Evaluate performed PES scan, sort states and store data in sqlite DB.')
parser.add_argument("-o", "--output", dest="dbfile", metavar="<file>", required=True,
                    help="write data to sqlite db in <file>")
parser.add_argument('-m', '--modes', dest='modes', metavar='n_modes', type=int,
                    help='total number of modes of the system')
parser.add_argument('-n', '--states', dest='states', metavar='n_states', type=int,
                    help='number of states to save')
parser.add_argument('-e', '--refenergy', dest='e0', metavar='e0', type=float,
                    help='energetic reference in Hartree (only used if Q = 0 not found in data)')
parser.add_argument('-i', '--interactive', dest='interactive', default=False, action='store_true',
                    help='prompt for plotting options')
parser.add_argument('path', type=str, nargs='+', help='path of the folder to evaluate')

# argv = '-o scans_test/out.sqlite -m 20 19 18 17 16 15 -n 12 scans_test/q16q15'.split()
# args = parser.parse_args(argv)
args = parser.parse_args()

def get_modes_from_filename(filename):
    file, ext = os.path.splitext(os.path.basename(filename))
    split_file = [p.lstrip('Q') for p in file.split('_') if p.startswith('Q')]
    # if len(split_file) == 1:
    #     return int(split_file.pop())
    # else:
    return list(map(int, split_file))

def get_Q_from_filename(filename):
    file, ext = os.path.splitext(os.path.basename(filename))
    split_file = [p for p in file.split('_') if not p.startswith('Q')]
    n_modes = len(get_modes_from_filename(file))
    if len(split_file) == 1 and n_modes == 1:
        return float(split_file.pop())
    elif len(split_file) == 1 and n_modes > len(split_file):
        # backwards compability for old file name convention 'Q1_Q2_0.123.out'
        return list(map(float, split_file*n_modes))
    else:
        return list(map(float, split_file))

def plot_energies(Q, V, plot_limits, x_axis_label, img_filename):
    x_axis_limits, y_axis_limits = plot_limits
    plt.style.use('seaborn-colorblind')
    plt.close()
    plt.figure()
    for Vi in V:
        plt.plot(Q, Vi, marker='x', mew=2, ms=6)
    plt.xlim(x_axis_limits)
    plt.ylim(y_axis_limits)
    plt.xlabel(x_axis_label)
    plt.ylabel('energy / eV')
    plt.savefig(img_filename, dpi=150, bbox_inches='tight')



for scan_path in args.path:
    print('*', scan_path)

    outfiles = glob.glob(os.path.join(scan_path, 'Q*.out'))
    outfiles.sort(key=lambda norm: np.linalg.norm(get_Q_from_filename(norm)))

    if len(outfiles) == 0:
        print('no Firefly output files found in directory:', scan_path)
        continue
        # raise Exception('no Firefly output files found in directory')

    filename_modes = get_modes_from_filename(outfiles[0])

    # todo: put this loop into function
    RESULTS_MATCH_STRING = 'E(MP2)='
    Qx = []
    results_mcscf = []
    results_mp2 = []

    for outfile in outfiles:
        try:
            with open(outfile) as f:
                Q = get_Q_from_filename(outfile)
                Qx.append(Q)
                i = 0
                for line in f:
                    if RESULTS_MATCH_STRING in line:
                        i += 1
                        split_line = line.strip().split()
                        state = int(split_line[0])
                        mcscf_energy = float(split_line[2])
                        mp2_energy = float(split_line[4])

                        try:
                            results_mcscf[state-1].append(mcscf_energy)
                        except IndexError:
                            results_mcscf.append([mcscf_energy])

                        try:
                            results_mp2[state-1].append(mp2_energy)
                        except IndexError:
                            results_mp2.append([mp2_energy])
                if i == 0:
                    Qx.pop()
        except IOError:
            break

    Qx = np.array(Qx)
    results_mp2 = np.array(results_mp2)


    if Qx.ndim > 1:
        Q0 = np.all(Qx==0., axis=-1)
    else:
        Q0 = (Qx==0.)

    if np.all(Q0 == False):
        if not args.e0:
            raise ValueError('Q = 0 not included in scan, need to set E0! (-e switch)')
        E0 = args.e0
    else:
        E0 = results_mp2.T[Q0,0]

    # convert energy values to eV, set baseline
    results_mp2 -= E0
    results_mp2 *= 27.21138506

    # plot results
    if Qx.ndim > 1:
        Qeff = np.linalg.norm(Qx, axis=1)
    else:
        Qeff = Qx
    x_min = np.floor(np.amin(Qeff))
    x_max = np.ceil(np.amax(Qeff))
    y_min = np.floor(np.amin(results_mp2[1]) * 2.0) / 2.0
    x_axis_limits = (x_min, x_max)
    y_axis_limits = (y_min, 10)
    plot_limits = (x_axis_limits, y_axis_limits)
    x_axis_label = ''.join(['Q' + str(m) for m in filename_modes])
    img_filename = x_axis_label + '.png'
    n = 1
    while os.path.isfile(x_axis_label + '_' + str(n) + '.png'):
        n = n + 1
    img_filename = x_axis_label + '_' + str(n) + '.png'
    plot_energies(Qeff, results_mp2, plot_limits, x_axis_label, img_filename)

    if args.interactive:
        # get user input for reordering states
        print('exchange data points between states n and m (format: e [Q|Qi-Qj] n m), r to remove (r [Q|Qi-Qj] n), c to continue, q to quit:')
        while True:
            inp = input('> ')
            inp_split = inp.strip().split(' ')
            command = inp_split.pop(0).lower()

            # if command == 'e':
            #     try:
            #         Q, n, m = inp_split
            #         if '-' in Q:
            #             # evaluate range
            #             Qi, Qj = Q.split('-')
            #             Qi, Qj = float(Qi), float(Qj)
            #             indices = get_Q_range_indices(Qx, Qi, Qj)
            #         else:
            #             Q = float(Q)
            #             indices = np.where(Qx == Q)
            #         n, m = int(n), int(m)
            #     except:
            #         print('\twrong input format: ', inp)
            #         continue

            #     # todo: check for bounds of Q, n and m; and if they exist
            #     swap_state_energies(results_mp2, indices, n, m)
            #     print('\tswitched states', n, 'and', m, 'at Q =', Q)

            # if command == 'r':
            #     try:
            #         Q, n = inp_split
            #         if '-' in Q:
            #             # evaluate range
            #             Qi, Qj = Q.split('-')
            #             Qi, Qj = float(Qi), float(Qj)
            #             indices = get_Q_range_indices(Qx, Qi, Qj)
            #         else:
            #             Q = float(Q)
            #             indices = np.where(Qx == Q)
            #         n = int(n)
            #     except:
            #         print('\twrong input format: ', inp)
            #         continue

            #     # todo: check for bounds of Q, n and m; and if they exist

            #     remove_state_energies(results_mp2, indices, n)
            #     print('\tremoved points of state', n, 'at Q =', Q)

            if command == 'xlim':
                try:
                    x_lim_lower, x_lim_upper = inp_split
                    x_lim_lower, x_lim_upper = float(x_lim_lower), float(x_lim_upper)
                except:
                    print('\twrong input format: ', inp)
                    continue
                x_axis_limits = (x_lim_lower, x_lim_upper)

            if command == 'ylim':
                try:
                    y_lim_lower, y_lim_upper = inp_split
                    y_lim_lower, y_lim_upper = float(y_lim_lower), float(y_lim_upper)
                except:
                    print('\twrong input format: ', inp)
                    continue
                y_axis_limits = (y_lim_lower, y_lim_upper)

            if command == 'c':
                break

            if command == 'q':
                quit()

            plot_limits = (x_axis_limits, y_axis_limits)
            plot_energies(Qeff, results_mp2, plot_limits, x_axis_label, img_filename)


    # input: how many states to save?
    if args.states and args.states > 0:
        n_states = args.states
    else:
        while True:
            inp = input('How many states to save? ')
            inp = inp.strip()
            try:
                n_states = int(inp)
            except:
                print('\twrong input format:', inp)
                continue
            break

    if args.modes:
        modes = list(range(1, args.modes + 1))
    else:
        print('Modes read from filename:', sorted(filename_modes, reverse=True))
        inp = input('Enter total number of modes: ')
        if inp:
            modes = list(range(1, int(inp) + 1))
        else:
            raise ValueError('you must specify total number of vibrational modes')
    n_modes = len(modes)



    # store results in db
    conn = sqlite3.connect(args.dbfile)
    db.create_results_table(conn, n_states, n_modes)

    # get next calc_no
    cur = conn.execute('SELECT MAX(scan_no) AS max_scan_no FROM Q')
    res = cur.fetchone()[0]
    if res is None:
        next_scan_no = 1
    else:
        next_scan_no = int(res) + 1

    # column names
    cols_Q = 'scan_no, scan_folder, ' + ', '.join(['Q' + str(m) for m in filename_modes])
    cols_V = 'scan_no, scan_folder, ' + ', '.join(['V' + str(i) for i in range(1, n_states + 1)])

    # column placeholders
    val_Q = '?,?,' + ','.join(['?' for _ in filename_modes])
    val_V = '?,?,' + ','.join(['?' for _ in range(n_states)])

    # prepare data
    scan_nos = np.array([next_scan_no]*len(Qx))
    scan_folder = np.array([os.path.basename(scan_path)]*len(Qx))
    V = np.array(results_mp2[:n_states]).T
    sql_data_Q = np.c_[scan_nos, scan_folder, Qx]
    sql_data_V = np.c_[scan_nos, scan_folder, V]

    # sql insert queries
    sql_Q = 'INSERT INTO Q ({}) VALUES ({})'.format(cols_Q, val_Q)
    sql_V = 'INSERT INTO V ({}) VALUES ({})'.format(cols_V, val_V)

    conn.executemany(sql_Q, sql_data_Q)
    conn.executemany(sql_V, sql_data_V)

    conn.commit()
    conn.close()
