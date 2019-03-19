#!/usr/bin/env python3

# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
from numba import jit, prange
from vchamtools.vcham import vcham
from scipy.optimize._numdiff import approx_derivative


@jit(nopython=True, cache=True)
def _indices_of_nonzero_elements(m):
    tol = 1.e-6
    return (np.abs(m) > tol).nonzero()


@jit(nopython=True, cache=True)
def Vnn(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma):
    n_states = len(E)
    QQ = Q**2
    Vnn = np.zeros(n_states)
    for n in range(n_states):
        Vnn[n] = E[n] + 0.5*np.dot(w, QQ) + np.dot(c_kappa[n], Q) + np.dot(Q, np.dot(c_gamma[n], Q)) + np.dot(Q, np.dot(c_rho[n], QQ)) + np.dot(QQ, np.dot(c_sigma[n], QQ))
    return Vnn


@jit(nopython=True, cache=True)
def Vnm(Q, c_lambda, c_eta):
    QQ = Q**2
    n_offdiagonal_elements = len(c_lambda)
    Vnm = np.zeros(n_offdiagonal_elements)
    nonzero_matrices_lambda = set(c_lambda.nonzero()[0])
    nonzero_matrices_eta = set(c_eta.nonzero()[0])
    coupling_indices = nonzero_matrices_lambda.union(nonzero_matrices_eta)
    # for n in range(n_offdiagonal_elements):
    for n in coupling_indices:
        Vnm[n] = np.dot(c_lambda[n], Q) + np.dot(Q, np.dot(c_eta[n], QQ))
    return Vnm


def Vn_adiab(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta):
    if Q.ndim != 2:
        Q = np.atleast_2d(Q)
    Sn, _ = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta)
    return Sn


def Vn_diab(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma):
    Sn = np.array([Vnn(Qi, w, E, c_kappa, c_gamma, c_rho, c_sigma) for Qi in Q])
    return Sn


@jit(nopython=True, parallel=True) # cache=True,
def diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, calc_ev=False):
    n_states = len(E)
    nq = len(Q)
    no = n_states*(n_states-1) // 2 # int(0.5*n_states*(n_states-1))
    ew = np.zeros((nq, n_states)) # eigenvalues
    ev = np.zeros((nq, n_states, n_states)) # eigenvectors

    for n in prange(nq):
        H = np.zeros((n_states, n_states))
        Hdiag = Vnn(Q[n], w, E, c_kappa, c_gamma, c_rho, c_sigma)
        Hoffdiag = Vnm(Q[n], c_lambda, c_eta)

        for i in range(n_states):
            for j in range(i, n_states):
                if i == j:
                    H[i,i] = Hdiag[i]
                else:
                    H[j,i] = Hoffdiag[vcham.get_index_of_triangular_element_in_flattened_matrix(i+1, j+1, n_states)]
                    # H[i,j] = Hoffdiag[vcham.get_index_of_triangular_element_in_flattened_matrix(i+1, j+1, n_states)]

        if calc_ev:
            # function signature: numpy.linalg.eigh(M, UPLO='L')
            # second argument defaults to 'L', ie. only the lower triangle of M is taken into account by the LAPACK routine.
            # changing the second parameter is not supported by Numba, but is not neccessary anyway.
            ew[n], ev[n] = np.linalg.eigh(H)
        else:
            ew[n] = np.linalg.eigvalsh(H)

    return ew, ev


@jit(['float64[:](float64[:],float64[:,:],float64[:,:],float64[:],float64[:],optional(float64[:]))'], nopython=True) # , cache=True
def err_weighted(p, Q, V, w, E, p_fixed=None):
    p_full = vcham.full_parameter_list(p, p_fixed)
    c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta = vcham.restore_parameters_from_flattened_list(p_full, w, E)
    S, _ = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, False)
    err = (V - S) * np.sqrt(np.exp(-np.absolute(V-E)))
    return err.flatten()

@jit(['float64[:](float64[:],float64[:,:],float64[:,:],float64[:],float64[:],optional(float64[:]))'], nopython=True)
def err(p, Q, V, w, E, p_fixed=None):
    p_full = vcham.full_parameter_list(p, p_fixed)
    c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta = vcham.restore_parameters_from_flattened_list(p_full, w, E)
    S, _ = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, False)
    err = V - S
    return err.flatten()

# def err_weighted_ev(p, Q, V, w, E, p_fixed=None):
#     p_full = vcham.full_parameter_list(p, p_fixed)
#     c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta = vcham.restore_parameters_from_flattened_list(p_full, w, E)
#     _, ev = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, True)
#     return ev.flatten()


@jit(nopython=True)
def err_weighted_lsq(p, Q, V, w, E, p_fixed=None):
    return np.sum(err_weighted(p, Q, V, w, E, p_fixed)**2)


@jit(nopython=True, cache=True)
def _pairwise_combinations(lst):
    n = len(lst)
    return [(lst[i], lst[j]) for i in range(n) for j in range(i+1, n)]


@jit(nopython=True, cache=True)
def _indices_of_nonzero_elements(m):
    tol = 1.e-6
    return (np.abs(m) > tol).nonzero()


@jit(nopython=True, parallel=True)
def err_weighted_jacobian(p, Q, V, w, E, p_fixed=None):
    # derivatives of eigenvalues see https://math.stackexchange.com/questions/2588473/derivatives-of-eigenvalues
    # p_fixed = p_fixed.copy()
    p_full = vcham.full_parameter_list(p, p_fixed)

    # compute eigenvectors
    c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta = vcham.restore_parameters_from_flattened_list(p_full, w, E)
    ww, ev = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, calc_ev=True)
    weights = np.sqrt(np.exp(-np.absolute(V-E)))
    nq = len(Q)
    npars = len(p)
    n_states = len(E)
    n_modes = len(w)
    n_offdiagonal_elements = n_states*(n_states-1) // 2 # int(0.5*n_states*(n_states-1))

    J = np.zeros((n_states*nq, npars))

    for q in prange(nq):
        # eigenvectoren sind spalten von ev -> ein ev: ev_n = ev[q].T[n] === ev[q,:,n]
        for n in prange(n_states):
            c_kappa_n = np.zeros((n_states, n_modes))
            c_gamma_n = np.zeros((n_states, n_modes, n_modes))
            c_rho_n = np.zeros((n_states, n_modes, n_modes))
            c_sigma_n = np.zeros((n_states, n_modes, n_modes))
            c_lambda_n = np.zeros((n_offdiagonal_elements, n_modes))
            c_eta_n = np.zeros((n_offdiagonal_elements, n_modes, n_modes))

            # eigenvector elements are nonzero for every state that contributes to eigenvalue
            state_indices, = _indices_of_nonzero_elements(ev[q,:,n])
            # print('n', n, 'state_indices', state_indices)
            for state_index in state_indices:
                for i in range(n_modes):
                    c_kappa_n[state_index,i] = -1 * Q[q,i] * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]
                    for j in range(i, n_modes):
                        c_gamma_n[state_index,i,j] = -1 * Q[q,i] * Q[q,j] * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]
                        # c_gamma_n[state_index,j,i] = -1 * Q[q,i] * Q[q,j] * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]
                        c_sigma_n[state_index,i,j] = -1 * Q[q,i]**2 * Q[q,j]**2 * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]
                        c_rho_n[state_index,i,j] = -1 * Q[q,i] * Q[q,j]**2 * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]
                        c_rho_n[state_index,j,i] = -1 * Q[q,i]**2 * Q[q,j] * ev[q,state_index,n] * ev[q,state_index,n] * weights[q,n]

            # Faktor 2 da H hermitesch ist
            # noch falsch: angenommen, S5 koppelt mit S1 und S3, dann wird auch das lambda
            # entsprechend k für die kopplung S1-S3 gesetzt. da wir nicht wissen, welcher
            # wert im eigenvector der für diagonaleintrag von H ist, da die reihenfolge der
            # eigenvektoren beliebig ist.
            # lambda-eintrag wird aber über reduce_parameter_array() wieder entfernt. geht das besser?
            for (idx1, idx2) in _pairwise_combinations(state_indices):
                k = vcham.get_index_of_triangular_element_in_flattened_matrix(idx1+1, idx2+1, n_states)
                for i in range(n_modes):
                    c_lambda_n[k,i] = -2 * Q[q,i] * ev[q,idx1,n] * ev[q,idx2,n] * weights[q,n]
                    for j in range(i, n_modes):
                        c_eta_n[k,i,j] = -2 * Q[q,i] * Q[q,j]**2 * ev[q,idx1,n] * ev[q,idx2,n] * weights[q,n]
                        c_eta_n[k,j,i] = -2 * Q[q,i]**2 * Q[q,j] * ev[q,idx1,n] * ev[q,idx2,n] * weights[q,n]

            p_n = vcham.flatten_to_parameter_list(c_kappa_n, c_gamma_n, c_rho_n, c_sigma_n, c_lambda_n, c_eta_n)
            # p_n2 = vcham.reduce_parameter_array(p_n, p_fixed.copy())
            p_n2 = vcham.reduce_parameter_array(p_n, p_fixed)
            J[q*n_states+n] = p_n2
    return J



@jit(nopython=True, parallel=True)
def err_jacobian(p, Q, V, w, E, p_fixed=None):
    # derivatives of eigenvalues see https://math.stackexchange.com/questions/2588473/derivatives-of-eigenvalues
    # p_fixed = p_fixed.copy()
    p_full = vcham.full_parameter_list(p, p_fixed)

    # compute eigenvectors
    c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta = vcham.restore_parameters_from_flattened_list(p_full, w, E)
    ww, ev = diagonalize_hamiltonian(Q, w, E, c_kappa, c_gamma, c_rho, c_sigma, c_lambda, c_eta, calc_ev=True)
    nq = len(Q)
    npars = len(p)
    n_states = len(E)
    n_modes = len(w)
    n_offdiagonal_elements = n_states*(n_states-1) // 2 # int(0.5*n_states*(n_states-1))

    J = np.zeros((n_states*nq, npars))

    for q in prange(nq):
        # eigenvectoren sind spalten von ev -> ein ev: ev_n = ev[q].T[n] === ev[q,:,n]
        for n in prange(n_states):
            c_kappa_n = np.zeros((n_states, n_modes))
            c_gamma_n = np.zeros((n_states, n_modes, n_modes))
            c_rho_n = np.zeros((n_states, n_modes, n_modes))
            c_sigma_n = np.zeros((n_states, n_modes, n_modes))
            c_lambda_n = np.zeros((n_offdiagonal_elements, n_modes))
            c_eta_n = np.zeros((n_offdiagonal_elements, n_modes, n_modes))

            # eigenvector elements are nonzero for every state that contributes to eigenvalue
            state_indices, = _indices_of_nonzero_elements(ev[q,:,n])
            # print('n', n, 'state_indices', state_indices)
            for state_index in state_indices:
                for i in range(n_modes):
                    c_kappa_n[state_index,i] = -1 * Q[q,i] * ev[q,state_index,n] * ev[q,state_index,n]
                    for j in range(i, n_modes):
                        c_gamma_n[state_index,i,j] = -1 * Q[q,i] * Q[q,j] * ev[q,state_index,n] * ev[q,state_index,n]
                        c_sigma_n[state_index,i,j] = -1 * Q[q,i]**2 * Q[q,j]**2 * ev[q,state_index,n] * ev[q,state_index,n]
                        c_rho_n[state_index,i,j] = -1 * Q[q,i] * Q[q,j]**2 * ev[q,state_index,n] * ev[q,state_index,n]
                        c_rho_n[state_index,j,i] = -1 * Q[q,i]**2 * Q[q,j] * ev[q,state_index,n] * ev[q,state_index,n]

            # Faktor 2 da H hermitesch ist
            # noch falsch: angenommen, S5 koppelt mit S1 und S3, dann wird auch das lambda
            # entsprechend k für die kopplung S1-S3 gesetzt. da wir nicht wissen, welcher
            # wert im eigenvector der für diagonaleintrag von H ist, da die reihenfolge der
            # eigenvektoren beliebig ist.
            # lambda-eintrag wird aber über reduce_parameter_array() wieder entfernt. geht das besser?
            for (idx1, idx2) in _pairwise_combinations(state_indices):
                k = vcham.get_index_of_triangular_element_in_flattened_matrix(idx1+1, idx2+1, n_states)
                for i in range(n_modes):
                    c_lambda_n[k,i] = -2 * Q[q,i] * ev[q,idx1,n] * ev[q,idx2,n]
                    for j in range(i, n_modes):
                        c_eta_n[k,i,j] = -2 * Q[q,i] * Q[q,j]**2 * ev[q,idx1,n] * ev[q,idx2,n]
                        c_eta_n[k,j,i] = -2 * Q[q,i]**2 * Q[q,j] * ev[q,idx1,n] * ev[q,idx2,n]

            p_n = vcham.flatten_to_parameter_list(c_kappa_n, c_gamma_n, c_rho_n, c_sigma_n, c_lambda_n, c_eta_n)
            # p_n2 = vcham.reduce_parameter_array(p_n, p_fixed.copy())
            p_n2 = vcham.reduce_parameter_array(p_n, p_fixed)
            J[q*n_states+n] = p_n2
    return J


def err_weighted_jacobian_numeric(p, Q, V, w, E, p_fixed=None):
    nq = len(Q)
    npars = len(p)
    n_states = len(E)
    n_modes = len(w)
    n_offdiagonal_elements = n_states*(n_states-1) // 2 # int(0.5*n_states*(n_states-1))

    c_kappa_scale = np.full((n_states, n_modes), 1.e-3)
    c_gamma_scale = np.full((n_states, n_modes, n_modes), 1.e-3)
    c_rho_scale = np.full((n_states, n_modes, n_modes), 1.e-4)
    c_sigma_scale = np.full((n_states, n_modes, n_modes), 1.e-5)
    c_lambda_scale = np.full((n_offdiagonal_elements, n_modes), 1.e-2)
    c_eta_scale = np.full((n_offdiagonal_elements, n_modes, n_modes), 1.e-5)
    p_scale = vcham.reduce_parameter_array(vcham.flatten_to_parameter_list(c_kappa_scale, c_gamma_scale, c_rho_scale, c_sigma_scale, c_lambda_scale, c_eta_scale), p_fixed)

    method = '3-point'

    EPS = np.finfo(np.float64).eps
    relative_step = {"2-point": EPS**0.5, "3-point": EPS**(1/3)}
    rel_step = relative_step[method] * p_scale

    J = approx_derivative(err_weighted, p, method=method, rel_step=rel_step, args=(Q, V, w, E, p_fixed))
    return J
