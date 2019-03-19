#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def approx_unique(ar, atol=1e-08, *args, **kwargs):
    decimals = int(np.abs(np.log10(atol)))
    ar = np.around(ar, decimals=decimals)
    return np.unique(ar, *args, **kwargs)

def vector_norm(vector):
    norm = np.linalg.norm(vector)
    with np.errstate(invalid='ignore', divide='ignore'):
        vector_norm = np.where(~np.isclose(vector, 0.), vector/norm, 0.)
    return vector_norm

def all_vector_elements_approx_equal(a):
    """Check if all elements of vector a (or list of vectors) are approximately equal, disregarding NaNs"""
    if np.all(np.isnan(a)):
        return False
    else:
        return np.isclose(np.nanmin(a, axis=-1), np.nanmax(a, axis=-1))

def line(u, v, t):
    return v + t * u

def line_parameter_form_from_points(a):
    """
    Parameters of line in parameter form: x = v + t * u, determined from points in a.
    Only the first and last element of a are used.
    """
    # u = a[0] - a[-1]
    u = a[-1] - a[0]
    v = a[0]
    return u, v

def point_in_line(p, u, v):
    """
    Tests if point(s) p (ndarray) are approximately in the line specified by u and v via x = v + t * u (parameter form)
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        t = (p - v) / u
    return all_vector_elements_approx_equal(t)
    # if np.all(np.isnan(t)):
    #     return False
    # else:
    #     return np.isclose(np.nanmin(t, axis=-1), np.nanmax(t, axis=-1))

def lines_identical(p1, p2):
    """
    Bedingungen für identische Geraden:
    Richtungsvektoren kollinear (= Vielfache voneinander)
    Aufpunkt der einen Geraden befindet sich auf der anderen Geraden
    """
    u1, v1 = p1
    u2, v2 = p2
    # print(u1, u2)
    # print()
    with np.errstate(invalid='ignore', divide='ignore'):
        u_parallel = all_vector_elements_approx_equal(u1/u2)
    return u_parallel and point_in_line(v2, u1, v1)

def line_plane_intersection(R, r, N):
    # line in parameter form: r(t): r + Rt; r,R: ndarray
    # plane in component form Ax + By + Cz = 0 with plane normal vector N = (A, B, C)
    # only valid for planes that contain the origin
    #
    # returns None if line and plane are parallel, ie. do not intersect
    # returns np.inf is line is in plane, ie. infinity number of intersections
    # else returns intersection point
    dir_prod = np.dot(R, N)
    pos = np.dot(r, N)
    if np.isclose(dir_prod, 0.):
        # line and plane are parallel
        if np.isclose(pos, 0.):
            # line in plane
            return np.inf
        else:
            # no intersection
            return None
    t = pos / dir_prod
    s = r - t*R # intersection point
    return s

def lines_identical2(p1, p2):
    """
    Bedingungen für identische Geraden:
    Richtungsvektoren kollinear (= Vielfache voneinander)
    Aufpunkt der einen Geraden befindet sich auf der anderen Geraden
    """
    u1, v1 = p1
    u2, v2 = p2
    print('lines_identical2')
    print(u1, u2)
    print()
    with np.errstate(invalid='ignore', divide='ignore'):
        u_parallel = all_vector_elements_approx_equal(u1/u2)
    return u_parallel and point_in_line(v2, u1, v1)

def equidirectional_scan_numbers(scan_no, Q):
    unique_scan_nos = np.unique(scan_no)
    scans_along_straight_line = [no for no in unique_scan_nos if np.all(point_in_line(Q[scan_no == no], *line_parameter_form_from_points(Q[scan_no == no]))) ]
    scans = list()
    for i, no in enumerate(scans_along_straight_line):
        this_scan = [no]
        p1 = line_parameter_form_from_points(Q[scan_no == no])

        for j, no_next in enumerate(scans_along_straight_line[i+1:]):
            p2 = line_parameter_form_from_points(Q[scan_no == no_next])
            if lines_identical(p1, p2):
                this_scan.append(no_next)
                del scans_along_straight_line[scans_along_straight_line.index(no_next)]

        # does the same thing, maybe more clear?
        # j = i + 1
        # while True:
        #     if j >= len(scans_along_straight_line):
        #         break
        #     no_next = scans_along_straight_line[j]
        #     p2 = line_parameter_form_from_points(Q[scan_no == no_next])
        #     if lines_identical(p1, p2):
        #         this_scan.append(no_next)
        #         del scans_along_straight_line[j]
        #     j += 1

        scans.append(this_scan)
    return scans


def plot_model(H, Q, V, scan_no, err_model, x_axis_limits_fixed=None, y_axis_limits=(4.3, 10), plot_points=200):
    plt.style.use('seaborn-colorblind')
    equidir_scan_no = equidirectional_scan_numbers(scan_no, Q)
    Q0 = np.atleast_2d(np.zeros_like(H.w))
    V0 = np.atleast_2d(H.E)

    # print(equidir_scan_no)

    for i in equidir_scan_no:
        # print(i)
        # get data that is to be plotted
        Q_data = Q[np.isin(scan_no, i),:]
        V_data = V[np.isin(scan_no, i),:]

        # if np.isin(3, i):
        # print('Q_data', Q_data)

        R, r = line_parameter_form_from_points(Q_data)
        R = vector_norm(R)

        num_ones = (np.sign(R) == 1).sum()
        if num_ones < len(R)-num_ones:
            R *= -1

        # print('R', R)
        # print('r', r)

        # if Q0 is point_in_line and not already in Q_data: then add Q0
        if not np.any(np.all(Q_data == Q0, axis=-1)) and point_in_line(Q0, R, r):
            Q_data = np.concatenate((Q0, Q_data), axis=0)
            V_data = np.concatenate((V0, V_data), axis=0)

        # plane through origin that is perpendicular to line has plane normal vector R/|R|
        isec = line_plane_intersection(R, r, R)
        # print('isec', isec)
        with np.errstate(divide='ignore', invalid='ignore'):
            # t parameter of the data points on the scanning line in vector form (rows have equal entries, or NaN)
            t_vec = (Q_data - isec) / R
            t = np.nanmean(t_vec, axis=-1) # like np.linalg.norm but with sign
            scan_modes = np.nan_to_num(R/R).astype(bool) # modes that are involved in the scan



        # sort all data
        sorted_indices = np.argsort(t)
        t = t[sorted_indices]
        Q_data = Q_data[sorted_indices]
        V_data = V_data[sorted_indices]

        # print('sorted_indices', sorted_indices)

        # determine plot range and generate a Q array for plotting the model
        Q_limits = np.stack((Q_data[0], Q_data[-1]), axis=1)
        # print('Q_limits', Q_limits)
        if x_axis_limits_fixed is None:
            t_min, t_max = t[0], t[-1]
        else:
            # sort x_axis_limits_fixed
            if x_axis_limits_fixed[1] < x_axis_limits_fixed[0]:
                x_axis_limits_fixed = (x_axis_limits_fixed[1], x_axis_limits_fixed[0])
            t_min, t_max = x_axis_limits_fixed

        x_axis_limits = t_min, t_max
        # print('x_axis_limits', x_axis_limits)
        # print('t', t)
        # print('np.floor(t).min()', np.floor(t).min())
        # print('np.ceil(t).max()', np.ceil(t).max())

        Q_limits = np.stack((line(R, isec, np.floor(t).min()), line(R, isec, np.ceil(t).max())))

        # print('Q_limits', Q_limits)
        Q_model_calc = np.array([np.linspace(qi[0], qi[1], num=plot_points) for qi in Q_limits.T]).T

        # t parameter of the data points on the scanning line in vector form (rows have equal entries, or NaN)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_vec = (Q_model_calc - isec) / R
            Q_model_plot = np.nanmean(t_vec, axis=-1) # like np.linalg.norm but with sign

        # generate x axis label and unique filename
        str_modes = '_'.join(['Q' + str(mode) for mode in H.modes[scan_modes]])
        # n = 1
        # while os.path.isfile(str_modes + '_' + str(n) + '.png'): n += 1
        img_filename = str_modes + '_' + str(i)

        V_diab = err_model.Vn_diab(Q_model_calc, H.w, H.E, H.c_kappa, H.c_gamma, H.c_rho, H.c_sigma)
        V_adiab = err_model.Vn_adiab(Q_model_calc, H.w, H.E, H.c_kappa, H.c_gamma, H.c_rho, H.c_sigma, H.c_lambda, H.c_eta)

        # print('t', t)
        # print('Q_model_calc', Q_model_calc)
        # print('Q_model_plot', Q_model_plot)

        # data
        plt.figure()
        plt.plot(t, V_data, ls='-', marker='x', lw=1, mew=1, ms=4)
        plt.xlim(x_axis_limits)
        plt.ylim(y_axis_limits)
        plt.xlabel(str_modes)
        plt.ylabel('energy / eV')
        # plt.legend()
        plt.savefig(img_filename + '_data.png', dpi=300)
        plt.close()

        # data and adiab states
        plt.figure()
        plt.plot(t, V_data, ls='', marker='x', lw=1, mew=1, ms=4)
        plt.gca().set_prop_cycle(None)
        plt.plot(Q_model_plot, V_adiab, '-', ls='-', lw=1, label="adiab.")
        plt.xlim(x_axis_limits)
        plt.ylim(y_axis_limits)
        plt.xlabel(str_modes)
        plt.ylabel('energy / eV')
        # plt.legend()
        plt.savefig(img_filename + '_fit.png', dpi=300)
        plt.close()

        # data, diab and adiab states
        plt.figure()
        plt.plot(t, V_data, ls='', marker='x', lw=1, mew=1, ms=4)
        plt.gca().set_prop_cycle(None)
        plt.plot(Q_model_plot, V_adiab, '-', ls='-', lw=1, label="adiab.")
        plt.gca().set_prop_cycle(None)
        plt.plot(Q_model_plot, V_diab, '-', ls='--', lw=1, label="diab.")
        plt.xlim(x_axis_limits)
        plt.ylim(y_axis_limits)
        plt.xlabel(str_modes)
        plt.ylabel('energy / eV')
        # plt.legend()
        plt.savefig(img_filename + '_fit_diab.png', dpi=300)
        plt.close()
