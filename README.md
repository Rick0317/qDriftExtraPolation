# QDRIFT Extrapolation Project

## Directories
- data: contains byte-wise data of Hamiltonian objects used in the analysis (database)
- jupyter: contains jupyter notebook version of scripts for easier development
- scripts: contains python scripts used for the numerical analysis
  - algo: contains main algorithms including QDRIFT, operator-gate compiler, phase estimation, and interpolation.
  - database: contains data interface and database manager
  - utils: contains other useful classes and functions for the numerical analysis.
  
## Data Interface
You can manage the Hamiltonian-database located at `data` folder using the `DataManager` class in `database.py`.
The Hamiltonian objects are subclass of `Tensor` class, which has coefficient, and its own representations in
multidimensional and 1-dimensional matrix as attributes. For efficiency only the information of non-zero terms are stored in the 
1-dimensional matrix representation. The `Hamiltonian` class has its own attribute, `decomp` to store information of
its decomposition into other Hamiltonians. This decomposition is stored as an instance of `DecomposedHamiltonian` class.

Specifically, `Hubbard` class describes the Hubbard model by given the number of spatial orbitals. It can be further 
decomposed into one- and two-body terms with given coefficients. Again, each term is an instance of `Hamiltonian`.


## Implementation of QDRIFT algorithm

## Chevyshev Interpolation on Data
