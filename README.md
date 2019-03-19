# vchamtools

Framework for scanning molecular electronic potential energy hypersurfaces and parametrizing a vibronic coupling model Hamiltonian (VCHAM, cf. [1]) from the results.

Here, the model contains up to fourth-order terms for the diabatic states / on-diagonal elements:

```V_nn = E_n + 1/2 w_i Q_i^2 + kappa_n Q_i + gamma_n Q_i Q_j + rho_n Q_i Q_j^2 + sigma_n Q_i^2 Q_j^2```

and up to third-order terms for the vibronic coupling terms / off-diagonal elements:

```V_nm = lambda_n Q_i + c_eta_n Q_i Q_j^2```

where `V` denotes the electronic potential energy, `E` is the vertical electronic excitation energy, `w` the harmonic vibrational frequency, `Q` is a dimensionless normal coordinate, `n, m` indicate the excited-state number, and `i, j` the vibrational mode number. The naming of the coefficient matrices `kappa, gamma, sigma, lambda, eta` was chosen mostly according to the common literature conventions.

**Highlights:**
* Unattended scanning of the PEHS (just let it run on your cluster in a `screen` or `tmux` session until finished)
* Automated application of symmetry selection rules to the VCHAM model coefficients
* Super fast fitting of the VCHAM model to the data thanks to JIT compilation and analytic Jacobian calculation (matrix of first derivatives used in the least squares minimization), giving results in seconds
* Export of the parametrized VCHAM model in MCTDH operator file format (no more copy & paste)

This framework is based on the tool stack of the author:
* Turbomole 7.2[2] for optimization of the molecular ground-state equilibrium structure,
* Firefly[3] computational chemistry program for XMCQDPT2 excited-state electronic structure calculations,
* calculations done on a high-performance cluster accessed via Torque/`qsub`,
* quantum-dynamics simulations using the Heidelberg MCTDH package.

You will surely have to adapt it to your requirements.

### Features
The following features are currently implemented:

1. vchamtools.molecule
    * data structures for storing molecular structure and vibrational normal mode data as harmonic frequencies, reduced masses and cartesian displacement vectors
    * import data from Turbomole 7 `aoforce` calculations.
    * export to Firefly input files.

2. vchamtools.pesscan
    * interface to job-queueing systems (tested with Torque) via `qsub` for automated scanning of the electronic potential energy hypersurfaces along normal modes and arbitrary linear combinations thereof using Firefly.
    * collection of the calculation results and storage in an SQLite database.

3. vchamtools.sqlitedb
    * wrapper functions for creating the database, storing and loading calculation results.

4. vchamtools.vcham
    * abstract implementation of the VCHAM as Python class.
    * the `VCHAM` class:
        - can be initialized using an arbitrary number of excited states and vibrational modes,
        - symmetry of states and modes can be supplied,
        - contains all numerical coefficients of the model,
        - makes use of symmetry selection rules to automatically reduce the number of coefficients.
    * helper functions for fitting the model coefficients to PEHS data.
    * weighted or unweighted least-squares fitting routine supporting the following methods:
        - Levenberg-Marquard via `scipy.optimize.leastsq` (not recommended),
        - Trust Region Reflective (TRF) and Dogbox with parameter bounds via `scipy.optimize.least_squares` (TRF recommended),
        - all methods supported by `scipy.optimize.minimize`, also support for other methods than least squares.
    * evaluation of the fit results by R^2, Adjusted R^2 and RMSE measures.

5. vchamtools.vcham.err_nstate
    * highly performant implementation of the actual least squares calculation.
    * uses just-in-time compilation to machine code using [Numba](http://numba.pydata.org).
    * weighted and un-weighted analytic Jacobian implementation for the VCHAM model.

6. vchamtools.vcham.plot
    * automatic plotting of cuts through the data and fitted model.

7. vchamtools.vcham.mctdh
    * export of the VCHAM model as operator file in MCTDH format.


The `vchamtools` package/framework is still under development.

### Examples
Example scripts are included in the `examples` folder.
Using IPython/Jupyter notebooks is recommended for fitting the VCHAM model.

### Dependencies
* O. Hüter, [molsym](https://github.com/oh-fv/molsym) package, Analytic point group algebra for molecular symmetry operations.

### References
1. [H. Köppel, W. Domcke and L.S. Cederbaum, "Multimode Molecular Dynamics Beyond the Born‐Oppenheimer Approximation," Adv. Chem. Phys. 57, 59-246 (1984)](https://doi.org/10.1002/9780470142813.ch2)
2. TURBOMOLE V7.2 2017, a development of University of Karlsruhe and Forschungszentrum Karlsruhe GmbH, 1989-2007, TURBOMOLE GmbH, since 2007; available from [http://www.turbomole.com](http://www.turbomole.com).
3. Alex A. Granovsky, Firefly version 8, [http://classic.chem.msu.su/gran/gamess/index.html](http://classic.chem.msu.su/gran/gamess/index.html)