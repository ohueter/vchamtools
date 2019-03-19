#!/usr/bin/env python3

import sqlite3
import numpy as np
from vchamtools.vcham import vcham

# __all__ = ['get_single_calc_from_db', 'create_results_table', 'get_single_calc_single_coord_from_db']

def create_results_table(conn, n_states, n_modes):
    if conn is None:
        raise ValueError('no sqlite connection given')

    if n_states < 1 or n_modes < 1:
        raise ValueError('incorrect arguments')

    columns_coords = ', '.join(['Q' + str(m) + ' REAL DEFAULT 0.' for m in range(1, n_modes + 1)])
    columns_energies = ', '.join(['V' + str(i) + ' REAL DEFAULT 0.' for i in range(1, n_states + 1)])

    sql_coords = '''CREATE TABLE IF NOT EXISTS Q (
        calc_no INTEGER PRIMARY KEY,
        scan_no INTEGER NOT NULL,
        scan_folder TEXT, {},
        FOREIGN KEY (calc_no) REFERENCES V(calc_no)
            ON UPDATE NO ACTION ON DELETE CASCADE
        );'''.format(columns_coords)
    conn.execute(sql_coords)

    sql_energies = '''CREATE TABLE IF NOT EXISTS V (
        calc_no INTEGER PRIMARY KEY,
        scan_no INTEGER NOT NULL,
        scan_folder TEXT, {},
        FOREIGN KEY (calc_no) REFERENCES Q(calc_no)
            ON UPDATE NO ACTION ON DELETE CASCADE
        );'''.format(columns_energies)
    conn.execute(sql_energies)


def load_scans(db_path, H, n_vibrational_modes, exclude_scan_no=[]):
    # exclude_scan_no: list of int
    if len(exclude_scan_no) > 0:
        sql_scan_no = ', '.join([str(i) for i in exclude_scan_no])
        sql_exclude_scan_no = 'AND scan_no NOT IN ({})'.format(sql_scan_no)
    else:
        sql_exclude_scan_no = ''

    conn = sqlite3.connect(db_path)

    set_model_modes = set(H.modes)
    set_molecule_modes = set(range(1, n_vibrational_modes + 1))
    set_modes_not_in_model = set_molecule_modes.difference(set_model_modes)

    cols_Q_model_modes = ', '.join(['Q{}'.format(m) for m in H.modes]) # SELECT this mode(s)
    cols_Q_modes_not_in_model = ' AND '.join(['Q{} == 0.'.format(m) for m in set_modes_not_in_model]) # WHERE these modes are zero
    cols_Q_molecule_modes = ', '.join(['Q{}'.format(m) for m in set_molecule_modes]) # GROUP BY all modes, thereby eliminating duplicate Q's
    cols_V = ', '.join(['V{}'.format(i) for i in range(1, len(H.E) + 1)])

    sql_query = '''SELECT DISTINCT Q.scan_no, {}, {}
        FROM Q
        INNER JOIN V ON V.calc_no = Q.calc_no
        WHERE {} {}
        GROUP BY {} ORDER BY Q.calc_no;'''.format(cols_Q_model_modes, cols_V, cols_Q_modes_not_in_model, sql_exclude_scan_no, cols_Q_molecule_modes)

    cur = conn.execute(sql_query)
    sql_result = np.array(cur.fetchall())
    scan_no, Q, V = sql_result[:,0].astype(int), sql_result[:,1:len(set_model_modes)+1], sql_result[:,len(set_model_modes)+1:]

    conn.close()
    return scan_no, Q, V
