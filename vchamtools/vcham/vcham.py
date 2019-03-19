#!/usr/bin/env python3

from __future__ import division
import numpy as np
from scipy.optimize import minimize, least_squares, leastsq
from scipy.optimize._numdiff import _prepare_bounds
from numba import jit

from operator import itemgetter
# from collections import Iterable
from molsym import PointGroup

from timeit import default_timer as timer
from timeit import repeat

np.set_printoptions(threshold=np.nan, linewidth=200)

# TODO:
# andere symmetrien testen

# r_square etc -> vcham.statistics
# timefunc -> vcham.utils
# constants aus vcham.constants benutzen

# falsch, aber zur info bzgl anwendung von np.ix_
# ts_states_indices = np.where(self.state_symmetries == self.pg.ts)
# ts_modes_indices = np.where(self.mode_symmetries == self.pg.ts)
# idx = np.ix_(*ts_states_indices, *ts_modes_indices)
# self.c_kappa_fixed[idx] = np.nan

class VCHAM(object):
    def __init__(self, w, E, pg=None):
        if not E or not w:
            raise ValueError()

        # check if E or w is list of double -> make tuple, set symmetry to C1 and irrep to 'a'
        # check for pg=None -> pg = 'c1'

        # E are the VEEs of the states in eV
        # w are vibrational frequencies of normal modes in cm-1, converted in a.u.

        # possible formats for E and w:
        # E = {'a1': [0., 6.07943074064719, 6.55089818393788, 8.05323420934188],
        #      'a2': [6.61083035518918, 7.45626745074725, 7.59485074838269],
        #      'b1': [6.25687266786346, 7.12473335817744, 7.87094375780691],
        #      'b2': [4.73023878752642, 6.71117813718611, 7.90368196845609]}
        # w = {'a1': [445.47, 306.11, 580.70],
        #      'a2': [256.99],
        #      'b1': [138.19]}

        # E = [(0., 'a1'), (4.73023878752642, 'b2'), (6.07943074064719, 'a1'), (6.25687266786346, 'b1'), (6.55089818393788, 'a1'),
        #     (6.61083035518918, 'a2'), (6.71117813718611, 'b2'), (7.12473335817744, 'b1'), (7.45626745074725, 'a2'), (7.59485074838269, 'a2'),
        #     (7.87094375780691, 'b1'), (7.90368196845609, 'b2'), (8.05323420934188, 'a1')]
        # w = [(306.11, 'a1'), (445.47, 'a1'), (580.70, 'a1'), (138.19, 'b1'), (256.99, 'a2')]

        if isinstance(E, dict):
            E = [(vee, key) for key in E for vee in E[key]]

        if isinstance(w, dict):
            w = [(vib, key) for key in w for vib in w[key]]

        E.sort(key=itemgetter(0)) # sort by VEE
        w.sort(key=itemgetter(1, 0)) # sort by symmetry, then frequency

        self.pg = PointGroup(pg)
        self.E = np.array([state[0] for state in E])
        self.state_symmetries = np.array([self.pg(state[1]) for state in E])
        self.w = np.array([mode[0] for mode in w]) / 8065.54464481
        self.mode_symmetries = np.array([self.pg(mode[1]) for mode in w])

        # Mulliken numbers of the modes, only used for generating the MCTDH input files
        self._modes = np.arange(1, len(w)+1, dtype=int)

        # some total numbers often used
        n_states, n_modes, n_diagonal_matrix_coefficients, n_diagonal_vector_coefficients, n_offdiagonal_elements, n_coupling_vector_coefficients = get_model_size(self.w, self.E)
        # all numbers in one list
        # self.n_all = np.array([self.n_states, self.n_modes, self.n_diagonal_matrix_coefficients, self.n_diagonal_vector_coefficients, self.n_offdiagonal_elements, self.n_coupling_vector_coefficients])

        # coefficient matrices
        self.c_kappa = np.zeros((n_states, n_modes))
        self.c_gamma = np.zeros((n_states, n_modes, n_modes))
        self.c_rho = np.zeros((n_states, n_modes, n_modes))
        self.c_sigma = np.zeros((n_states, n_modes, n_modes))
        self.c_lambda = np.zeros((n_offdiagonal_elements, n_modes))
        self.c_eta = np.zeros((n_offdiagonal_elements, n_modes, n_modes))

        # coefficient bounds
        self.c_kappa_lower_bounds = np.full_like(self.c_kappa, -np.inf)
        self.c_gamma_lower_bounds = np.full_like(self.c_gamma, -np.inf)
        self.c_rho_lower_bounds = np.full_like(self.c_rho, -np.inf)
        self.c_sigma_lower_bounds = np.full_like(self.c_sigma, -np.inf) # np.zeros_like(self.c_sigma) # potential diverges to -inf if sigma becomes negative
        self.c_lambda_lower_bounds = np.full_like(self.c_lambda, -np.inf)
        self.c_eta_lower_bounds = np.full_like(self.c_eta, -np.inf)

        self.c_kappa_upper_bounds = np.full_like(self.c_kappa, np.inf)
        self.c_gamma_upper_bounds = np.full_like(self.c_gamma, np.inf)
        self.c_rho_upper_bounds = np.full_like(self.c_rho, np.inf)
        self.c_sigma_upper_bounds = np.full_like(self.c_sigma, np.inf)
        self.c_lambda_upper_bounds = np.full_like(self.c_lambda, np.inf)
        self.c_eta_upper_bounds = np.full_like(self.c_eta, np.inf)

        # fixed coefficient arrays hold the fixed coefficient values.
        # a lot of the coefficients are set to 0 due to symmetry,
        # while non-zero coefficients that should be included in the fit are set to np.nan.
        # kappa != 0 when "G_n x Q_i x G_n contains G_ts"   (G: greek Gamma symbol, n: n-th electronic state, i: i-th vib. mode), cf. Köppel1984 eq. (2.28b)
        self.c_kappa_fixed = np.zeros((n_states, n_modes))
        for n, state_sym in enumerate(self.state_symmetries):
            for i, mode_sym in enumerate(self.mode_symmetries):
                sym = state_sym * mode_sym * state_sym
                if self.pg.ts in sym:
                    self.c_kappa_fixed[n, i] = np.nan

        # only upper triangle of gamma and sigma are used
        # gamma != 0 when "G_n x Q_i x Q_j x G_n contains G_ts"
        # rho != 0 when "G_n x Q_i x (Q_j)^2 x G_n contains G_ts"
        # sigma != 0 when "G_n x (Q_i)^2 x (Q_j)^2 x G_n contains G_ts"
        self.c_gamma_fixed = np.zeros((n_states, n_modes, n_modes))
        self.c_rho_fixed = np.zeros((n_states, n_modes, n_modes))
        self.c_sigma_fixed = np.zeros((n_states, n_modes, n_modes))
        for n, state_sym in enumerate(self.state_symmetries):
            for i in range(n_modes):
                mode_i_sym = self.mode_symmetries[i]
                for j in range(i, n_modes):
                    mode_j_sym = self.mode_symmetries[j]

                    # gamma
                    sym = state_sym * mode_i_sym * mode_j_sym * state_sym
                    if self.pg.ts in sym:
                        self.c_gamma_fixed[n, i, j] = np.nan
                        # self.c_gamma_fixed[n, j, i] = np.nan

                    # rho
                    sym = state_sym * mode_i_sym * mode_j_sym**2 * state_sym
                    if self.pg.ts in sym:
                        self.c_rho_fixed[n, i, j] = np.nan
                    sym = state_sym * mode_i_sym**2 * mode_j_sym * state_sym
                    if self.pg.ts in sym:
                        self.c_rho_fixed[n, j, i] = np.nan

                    # sigma
                    sym = state_sym * mode_i_sym**2 * mode_j_sym**2 * state_sym
                    if self.pg.ts in sym:
                        self.c_sigma_fixed[n, i, j] = np.nan
                        # self.c_sigma_fixed[n, j, i] = np.nan

        # lambda != 0 when "G_n x Q_i x G_m contains G_ts"   (G: greek Gamma symbol, n,m: n-th (m-th) electronic state, i: i-th vib. mode), cf. Köppel1984 eq. (2.28a)
        # eta != 0 when "G_n x Q_i x Q_j x Q_j x G_m contains G_ts"
        self.c_lambda_fixed = np.zeros((n_offdiagonal_elements, n_modes))
        self.c_eta_fixed = np.zeros((n_offdiagonal_elements, n_modes, n_modes))
        for off_idx in range(n_offdiagonal_elements):
            n, m = triu_indices_offdiag_element(off_idx, n_states)
            state_n_sym, state_m_sym = self.state_symmetries[n], self.state_symmetries[m]
            for i in range(n_modes):
                mode_i_sym = self.mode_symmetries[i]
                # lambda
                sym = state_n_sym * mode_i_sym * state_m_sym
                if self.pg.ts in sym:
                    self.c_lambda_fixed[off_idx, i] = np.nan

                # eta
                for j in range(n_modes):
                    mode_j_sym = self.mode_symmetries[j]
                    sym = state_n_sym * mode_i_sym * mode_j_sym**2 * state_m_sym
                    if self.pg.ts in sym:
                        self.c_eta_fixed[off_idx, i, j] = np.nan

                    # # mu
                    # sym = state_n_sym * mode_i_sym * mode_j_sym * state_m_sym
                    # if self.pg.ts in sym:
                    #     print('mu', off_idx, 'ij', i, j)

    @property
    def params(self):
        return flatten_to_parameter_list(self.c_kappa, self.c_gamma, self.c_rho, self.c_sigma, self.c_lambda, self.c_eta)

    @property
    def params_fixed(self):
        return flatten_to_parameter_list(self.c_kappa_fixed, self.c_gamma_fixed, self.c_rho_fixed, self.c_sigma_fixed, self.c_lambda_fixed, self.c_eta_fixed)

    @property
    def params_reduced(self):
        return reduce_parameter_array(self.params, self.params_fixed)

    @property
    def params_bounds(self):
        p_lower_bounds = flatten_to_parameter_list(self.c_kappa_lower_bounds, self.c_gamma_lower_bounds, self.c_rho_lower_bounds, self.c_sigma_lower_bounds, self.c_lambda_lower_bounds, self.c_eta_lower_bounds)
        p_upper_bounds = flatten_to_parameter_list(self.c_kappa_upper_bounds, self.c_gamma_upper_bounds, self.c_rho_upper_bounds, self.c_sigma_upper_bounds, self.c_lambda_upper_bounds, self.c_eta_upper_bounds)
        p_lower_bounds = reduce_parameter_array(p_lower_bounds, self.params_fixed)
        p_upper_bounds = reduce_parameter_array(p_upper_bounds, self.params_fixed)
        bounds = (p_lower_bounds, p_upper_bounds)
        return bounds

    def set_coefficients_from_reduced_params(self, p_reduced):
        p_reduced[np.isclose(p_reduced, 0.)] = 0
        p_full = full_parameter_list(p_reduced, self.params_fixed)
        self.c_kappa, self.c_gamma, self.c_rho, self.c_sigma, self.c_lambda, self.c_eta = restore_parameters_from_flattened_list(p_full, self.w, self.E)

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, value):
        if len(value) == len(self.w):
            self._modes = np.array(value)
        else:
            raise ValueError('length of Mulliken mode numbers list is not equal to number of modes in VCHAM')

    def disable_coupling(self, states, disable_linear=True, disable_cubic=True):
        if not (disable_linear or disable_cubic):
            return
        states = np.atleast_2d(states)
        n_states = len(self.E)
        n_modes = len(self.w)
        for (n,m) in states:
            off_idx = get_index_of_triangular_element_in_flattened_matrix(n, m, n_states)
            if disable_linear:
                self.c_lambda[off_idx] = self.c_lambda_fixed[off_idx] = np.zeros((n_modes))
            if disable_cubic:
                self.c_eta[off_idx] = self.c_eta_fixed[off_idx] = np.zeros((n_modes, n_modes))

    def __str__(self):
        s = str()
        for i, coeff in enumerate(self.c_kappa):
            if not np.allclose(self.c_kappa_fixed[i], 0.):
                s += 'H.c_kappa[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_gamma):
            if not np.allclose(self.c_gamma_fixed[i], 0.):
                s += 'H.c_gamma[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_rho):
            if not np.allclose(self.c_rho_fixed[i], 0.):
                s += 'H.c_rho[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_sigma):
            if not np.allclose(self.c_sigma_fixed[i], 0.):
                s += 'H.c_sigma[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_lambda):
            if not np.allclose(self.c_lambda_fixed[i], 0.):
                s += 'H.c_lambda[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_eta):
            if not np.allclose(self.c_eta_fixed[i], 0.):
                s += 'H.c_eta[{!s}] = np.{!r}\n'.format(i, coeff)
        return s[:-1].replace('0.00000000e+00', '0.')

    def __repr__(self):
        s = str()
        for i, coeff in enumerate(self.c_kappa):
            s += 'H.c_kappa[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_gamma):
            s += 'H.c_gamma[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_rho):
            s += 'H.c_rho[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_sigma):
            s += 'H.c_sigma[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_lambda):
            s += 'H.c_lambda[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_eta):
            s += 'H.c_eta[{!s}] = np.{!r}\n'.format(i, coeff)

        for i, coeff in enumerate(self.c_kappa_fixed):
            s += 'H.c_kappa_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_gamma_fixed):
            s += 'H.c_gamma_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_rho_fixed):
            s += 'H.c_rho_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_sigma_fixed):
            s += 'H.c_sigma_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_lambda_fixed):
            s += 'H.c_lambda_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        for i, coeff in enumerate(self.c_eta_fixed):
            s += 'H.c_eta_fixed[{!s}] = np.{!r}\n'.format(i, coeff)
        return s[:-1].replace('0.00000000e+00', '0.')


@jit(nopython=True, cache=True)
def get_model_size(w, E):
    n_states = len(E)
    n_modes = len(w)
    n_diagonal_matrix_coefficients = n_states * n_modes**2
    n_diagonal_vector_coefficients = n_states * n_modes
    n_offdiagonal_elements = n_states*(n_states-1) // 2
    # n_offdiagonal_elements = int(0.5*n_states*(n_states-1))
    n_coupling_vector_coefficients = n_offdiagonal_elements * n_modes
    return n_states, n_modes, n_diagonal_matrix_coefficients, n_diagonal_vector_coefficients, n_offdiagonal_elements, n_coupling_vector_coefficients

@jit(['float64[:](float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:,:])'], nopython=True, cache=True)
def flatten_to_parameter_list(c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta):
    return np.concatenate((c_kappa.flatten(), c_gamma.flatten(), c_rho.flatten(), c_sigma.flatten(), c_lambda.flatten(), c_eta.flatten()))

@jit(nopython=True, cache=True)
def restore_parameters_from_flattened_list(p, w, E):
    n_states, n_modes, n_diagonal_matrix_coefficients, n_diagonal_vector_coefficients, n_offdiagonal_elements, n_coupling_vector_coefficients = get_model_size(w, E)

    c_kappa = np.zeros(n_diagonal_vector_coefficients)
    c_gamma = np.zeros(n_diagonal_matrix_coefficients)
    c_rho = np.zeros(n_diagonal_matrix_coefficients)
    c_sigma = np.zeros(n_diagonal_matrix_coefficients)
    c_lambda = np.zeros(n_coupling_vector_coefficients)
    c_eta = np.zeros(n_coupling_vector_coefficients * n_modes)

    for i in range(n_diagonal_vector_coefficients):
        c_kappa[i] = p[i]
    for i in range(n_diagonal_matrix_coefficients):
        c_gamma[i] = p[n_diagonal_vector_coefficients + i]
        c_rho[i] = p[n_diagonal_vector_coefficients + n_diagonal_matrix_coefficients + i]
        c_sigma[i] = p[n_diagonal_vector_coefficients + 2*n_diagonal_matrix_coefficients + i]
    for i in range(n_coupling_vector_coefficients):
        c_lambda[i] = p[n_diagonal_vector_coefficients + 3*n_diagonal_matrix_coefficients + i]
    for i in range(n_coupling_vector_coefficients * n_modes):
        c_eta[i] = p[n_diagonal_vector_coefficients + 3*n_diagonal_matrix_coefficients + n_coupling_vector_coefficients + i]

    c_kappa = c_kappa.reshape((n_states, n_modes))
    c_gamma = c_gamma.reshape((n_states, n_modes, n_modes))
    c_rho = c_rho.reshape((n_states, n_modes, n_modes))
    c_sigma = c_sigma.reshape((n_states, n_modes, n_modes))
    c_lambda = c_lambda.reshape((n_offdiagonal_elements, n_modes))
    c_eta = c_eta.reshape((n_offdiagonal_elements, n_modes, n_modes))

    return c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta

@jit(nopython=True, cache=True)
def reduce_parameter_array(p_full, p_fixed):
    if p_fixed is None:
        return p_full
    else:
        return p_full[np.isnan(p_fixed)]

@jit(nopython=True, cache=True)
def full_parameter_list(p, p_fixed):
    if p_fixed is None:
        p_full = p
    else:
        n_pars = len(p_fixed)
        p_full = np.zeros(n_pars)
        j = 0
        for i in range(n_pars):
            if np.isnan(p_fixed[i]):
                p_full[i] = p[j]
                j = j+1
            else:
                p_full[i] = p_fixed[i]
    return p_full

# todo: wenn über array geloopt wird, erwartet get_index_of_triangular_element_in_flattened_matrix die indices beginnend bei 1,
# dh. loop-indices i,j beginnend bei 0 -> i+1, j+1 müssen übergeben werden
# triu_indices_offdiag_element liefert diese zurück, und dann muss wieder -1 gerechnet werden.
# nicht so sinnvoll. lieber beide bei 0 beginnen lassen. viele anpassungen im restprogramm ...
@jit(nopython=True, cache=True)
def get_index_of_triangular_element_in_flattened_matrix(n, m, N):
    # check for nonsense arguments
    if n == m or m > N or n > N or n <= 0 or m <= 0 or N <= 0:
        raise ValueError('invalid off-diagonal index requested') #'invalid off-diagonal index requested: n={}, m={}, N={}'.format(str(n), str(m), str(N)))
    # no elements from lower triangle, ie. always n < m
    if m < n:
        n, m = m, n
    return int(n*m + (n-1)*(N-m) - 0.5*n*(n+1)) - 1

@jit(nopython=True, cache=True)
def triu_indices_offdiag_element(i, N):
    cs = np.cumsum(np.arange(N-1, 0, -1)) - 1
    idx, = np.where(cs < i)
    if len(idx) == 0:
        n = 0
    else:
        n = np.max(idx) + 1
    m = int(i + 1 + N - (n+1)*(N - ((n+1)+1)/2) - 1)
    return n, m





def fit_coefficients(H, err_func, Q, V, method='lm', bounds=None, **kwargs):
    # p_fixed = flatten_to_parameter_list(H.c_kappa_fixed, H.c_gamma_fixed, H.c_sigma_fixed, H.c_lambda_fixed, H.c_eta_fixed)
    # p_full = flatten_to_parameter_list(H.c_kappa, H.c_gamma, H.c_sigma, H.c_lambda, H.c_eta)
    # # coefficients whre p_fixed is np.nan are to be varied during fit, all other params are fixed
    # # and need not to be passed to least squares routine
    # p_guess = reduce_parameter_array(p_full, p_fixed)

    if method.lower() == 'lm':
        t_start = timer()
        p_best, cov_x, infodict, mesg, ier = leastsq(err_func, H.params_reduced, args=(Q, V, H.w, H.E, p_fixed), full_output=True)
        t_end = timer()
        if ier in [1,2,3,4]:
            H.set_coefficients_from_reduced_params(p_best)
            r2 = r_square(H, Q, V, err_func)
            adj_r2 = adj_r_square(H, Q, V, err_func)
            rmse = root_mean_square_error(H, Q, V, err_func)
            fit_goodness = {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'nfev': infodict['nfev'], 'message': mesg, 't_fit': t_end-t_start}
            # fit_goodness = {'r2': -1, 'adj_r2': -1, 'rmse': -1, , 'nfev': 0}
        else:
            print('leastsq() failure mesg:', mesg)
            fit_goodness = {'r2': -1, 'adj_r2': -1, 'rmse': -1, 'nfev': 0, 't_fit': t_end-t_start}
            raise RuntimeWarning('leastsq() did not converge')

    elif method.lower() in ('trf', 'dogbox'):
        if not bounds:
            bounds = (-np.inf, np.inf)
        t_start = timer()
        res = least_squares(err_func, H.params_reduced, bounds=bounds, method=method, args=(Q, V, H.w, H.E, H.params_fixed), **kwargs)
        t_end = timer()
        if res.success:
            H.set_coefficients_from_reduced_params(res.x)
            r2 = r_square(H, Q, V, err_func)
            adj_r2 = adj_r_square(H, Q, V, err_func)
            rmse = root_mean_square_error(H, Q, V, err_func)
            fit_goodness = {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'nfev': res.nfev, 'message': res.message, 't_fit': t_end-t_start, 'jac': res.jac}
        else:
            print('least_squares() failure mesg:', res.message)
            fit_goodness = {'r2': -1, 'adj_r2': -1, 'rmse': -1, 'nfev': 0, 't_fit': t_end-t_start}
            # raise RuntimeWarning('leastsq() did not converge')

    elif method.lower() in ('cg', 'nelder-mead', 'powell', 'cobyla', 'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'bfgs', 'l-bfgs-b', 'tnc', 'slsqp'):
        if bounds and method.lower() in ('l-bfgs-b', 'tnc', 'slsqp'):
            bounds = np.array(bounds).T
            bounds = [(None if np.isinf(lb) else lb, None if np.isinf(ub) else ub) for (lb, ub) in bounds]
        else:
            bounds = None
        t_start = timer()
        res = minimize(err_func, H.params_reduced, args=(Q, V, H.w, H.E, H.params_fixed), method=method, bounds=bounds, **kwargs)
        t_end = timer()
        if res.success:
            H.set_coefficients_from_reduced_params(res.x)
            r2 = r_square(H, Q, V, err_func)
            adj_r2 = adj_r_square(H, Q, V, err_func)
            rmse = root_mean_square_error(H, Q, V, err_func)
            fit_goodness = {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'nfev': res.nfev, 'message': res.message, 't_fit': t_end-t_start}
        else:
            print('least_squares() failure mesg:', res.message)
            fit_goodness = {'r2': -1, 'adj_r2': -1, 'rmse': -1, 'nfev': 0, 't_fit': t_end-t_start}
            raise RuntimeWarning('minimize() did not converge')

    else:
        raise ValueError('valid least squares methods are "lm", trf" and "dogbox", or methods supported by scipy.optimize.minimize()')

    return fit_goodness





# fit statistics:
# @jit(cache=True)
def summed_square_of_residuals(H, Q, V, err_func):
    err = err_func(H.params_reduced, Q, V, H.w, H.E, H.params_fixed)
    # SSE: summed square of errors (residuals)
    SSE = np.sum(err**2)
    return SSE

# @jit(cache=True)
def mean_square_error(H, Q, V, err_func):
    # n: number of response values (fitted data points)
    n = V.size

    # m: number of fitted coefficients m estimated from the response values.

    # fitted parameters are != 0 (fixed due to symmetry) and != 1 (initialization values)
    # p = flatten_to_parameter_list(H.c_kappa, H.c_gamma, H.c_sigma, H.c_lambda, H.c_eta)
    # m = len(np.where(np.logical_and(np.logical_not(p == 0.), np.logical_not(p == 1.)))[0])

    # better:
    # p_fixed = flatten_to_parameter_list(H.c_kappa_fixed, H.c_gamma_fixed, H.c_sigma_fixed, H.c_lambda_fixed, H.c_eta_fixed)
    # p_full = flatten_to_parameter_list(H.c_kappa, H.c_gamma, H.c_sigma, H.c_lambda, H.c_eta)
    # p = reduce_parameter_array(p_full, p_fixed)
    m = len(H.params_reduced)

    # v=n-m: residual degrees of freedom
    v = n - m

    # SSE: summed square of residuals
    SSE = summed_square_of_residuals(H, Q, V, err_func)

    # MSE: mean square error
    MSE = SSE / v

    return MSE

# @jit(cache=True)
def root_mean_square_error(H, Q, V, err_func):
    return np.sqrt(mean_square_error(H, Q, V, err_func))

# @jit(cache=True)
def total_sum_of_squares(H, Q, V, err_func):
    # average V for each state
    V_av = np.mean(V, axis=0)
    # SST: total sum of squares
    SST = np.sum((V-V_av)**2)
    return SST

# @jit(cache=True)
def r_square(H, Q, V, err_func):
    r_square = 1 - summed_square_of_residuals(H, Q, V, err_func) / total_sum_of_squares(H, Q, V, err_func)
    return r_square

# @jit(cache=True)
def adj_r_square(H, Q, V, err_func):
    # n: number of response values (fitted data points)
    n = V.size

    # m: number of fitted coefficients m estimated from the response values.
    # fitted parameters are != 0 (fixed due to symmetry) and != 1 (initialization values)
    # p = flatten_to_parameter_list(H.c_gamma, H.c_sigma, H.c_lambda, H.c_eta)
    # m = len(np.where(np.logical_and(np.logical_not(p == 0.), np.logical_not(p == 1.)))[0])
    # better:
    # p_fixed = flatten_to_parameter_list(H.c_kappa_fixed, H.c_gamma_fixed, H.c_sigma_fixed, H.c_lambda_fixed, H.c_eta_fixed)
    # p_full = flatten_to_parameter_list(H.c_kappa, H.c_gamma, H.c_sigma, H.c_lambda, H.c_eta)
    # p = reduce_parameter_array(p_full, p_fixed)
    m = len(H.params_reduced)

    adj_r_square = 1 - (1 - r_square(H, Q, V, err_func)) * (n-1)/(n-m-1)
    return adj_r_square



# for internal statistics
# time f(x,y) -> timefunc(f,x,y)
def timefunc(func, nexc, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    exc_time = min(repeat(lambda: func(*args, **kwargs), number=nexc, repeat=4)) * 1000 / nexc
    print(func.__name__, exc_time, 'ms')
    return exc_time, res
