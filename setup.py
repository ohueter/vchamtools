import setuptools

setuptools.setup(
    name='vchamtools',
    version='0.1',
    description='Framework for scanning molecular electronic potential energy hypersurfaces and parametrizing a vibronic coupling model Hamiltonian from the results.',
    author='Ole Hüter',
    packages=['vchamtools', 'vchamtools.molecule', 'vchamtools.pesscan', 'vchamtools.sqlitedb', 'vchamtools.vcham'],
    install_requires=['numpy', 'scipy', 'numba', 'matplotlib', 'sqlite3', 'molsym'],
    keywords=[
        'molecule', 'symmetry', 'vibronic coupling', 'vcham', 'mctdh'
    ],
    url='https://github.com/oh-fv/vchamtools',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry'],
)
