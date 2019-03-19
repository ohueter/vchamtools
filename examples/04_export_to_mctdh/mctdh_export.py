#!/usr/bin/env python3

import numpy as np
from vchamtools.vcham import vcham, mctdh

# molecular point group
pg = 'c2v'

# total number of vibrational degrees of freedom
n_vibrational_modes = 30

# vertical excitation energies
E = {'a1': [0., 4.74412327642684, 6.75995802149102, 7.98600046338684],
     'a2': [6.15547861028794],
     'b1': [6.43537288149314],
     'b2': [6.11882230057693, 6.66745869109101, 7.76702176293909]}

# harmonic vibrational frequencies
w = {'a1': [], # 276.28, 323.97, 457.14, 691.44, 1067.22, 1188.17, 1347.50, 1357.08, 1555.37, 1673.97, 3257.74
     'a2': [144.23, 365.18], # 144.23, 365.18, 487.15, 572.90, 922.85
     'b1': [], # 155.89, 295.01, 606.46, 815.81
     'b2': []} # 282.30, 487.41, 609.52, 753.74, 998.96, 1262.73, 1284.14, 1555.48, 1673.21, 3244.40

# mode numbers in Mulliken notation, only used for pretty-printing
modes = [16, 15]

# setting up the VCHAM model
H = vcham.VCHAM(w, E, pg)
H.modes = modes

# model coefficients
H.c_gamma[0] = np.array([[0.00100666, 0.00764394],
       [0.        , 0.00439196]])
H.c_gamma[1] = np.array([[-0.01030974,  0.00786163],
       [ 0.        , -0.00048934]])
H.c_gamma[2] = np.array([[-0.01589442,  0.01584065],
       [ 0.        , -0.00657488]])
H.c_gamma[3] = np.array([[-0.0091423 ,  0.00906303],
       [ 0.        , -0.00535392]])
H.c_gamma[4] = np.array([[-0.00291493,  0.01090703],
       [ 0.        , -0.01212118]])
H.c_gamma[5] = np.array([[-0.00725866,  0.00288015],
       [ 0.        , -0.02777383]])
H.c_gamma[6] = np.array([[-0.0070627 ,  0.007377  ],
       [ 0.        , -0.02669591]])
H.c_gamma[7] = np.array([[-0.01574861,  0.01092143],
       [ 0.        , -0.03177707]])
H.c_gamma[8] = np.array([[-0.00850089, -0.01087097],
       [ 0.        , -0.02034808]])
H.c_sigma[0] = np.array([[2.28342667e-05, 0.],
       [0., 1.42086097e-05]])
H.c_sigma[1] = np.array([[3.28050837e-05, 0.],
       [0., 0.]])
H.c_sigma[2] = np.array([[3.97722871e-05, 0.],
       [0., 0.]])
H.c_sigma[3] = np.array([[2.54563956e-05, 0.],
       [0., 0.]])
H.c_sigma[4] = np.array([[1.57975811e-05, 1.46426824e-08],
       [0., 0.]])
H.c_sigma[5] = np.array([[3.45394846e-05, 1.69213447e-05],
       [0., 7.90136830e-05]])
H.c_sigma[6] = np.array([[3.15968337e-05, 3.23286227e-06],
       [0., 7.81440537e-05]])
H.c_sigma[7] = np.array([[4.56069598e-05, 1.06314309e-07],
       [0., 9.67954455e-05]])
H.c_sigma[8] = np.array([[1.96030808e-05, 6.89494213e-05],
       [0., 3.26132653e-05]])
H.c_lambda[9] = np.array([-0.04000136,  0.2146144 ])
H.c_lambda[16] = np.array([-0.0168688 ,  0.10578677])
H.c_lambda[26] = np.array([-0.01850662,  0.05682587])
H.c_lambda[28] = np.array([0.00298991, 0.03596294])
H.c_eta[9] = np.array([[-4.88502771e-05,  4.84869439e-05],
       [ 4.84869439e-05, -6.32902894e-04]])
H.c_eta[16] = np.array([[-1.59602248e-05, -3.63339631e-05],
       [-3.63339631e-05, -3.84527179e-04]])
H.c_eta[26] = np.array([[ 2.80109992e-05,  1.54672659e-05],
       [ 1.54672659e-05, -1.64359380e-04]])
H.c_eta[28] = np.array([[-5.16494867e-05,  5.61996288e-06],
       [ 5.61996288e-06, -1.58438857e-04]])

# states to include in MCTDH export
states = [1,2,4]

print(mctdh.op_parameter_section(H, states))
print(mctdh.op_hamiltonian_section(H, states))
