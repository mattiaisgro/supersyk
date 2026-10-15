# SuperSYK

> The syckest system out there.

This repo contains code for numerical studies of quantum chaos in the supersimmetric Sachdev-Ye-Kitaev model.

## The $\mathcal{N}=2$ Super SYK Model
The $\mathcal{N} = 2$ supersymmetric SYK model with $\hat q = 3$ fermions is a system of N complex fermions with supercharge:

$$Q = i\sum_{i<j<k} C_{ijk} \hat\psi^i \hat\psi^j \hat\psi^k$$

where the fermion operators have the anticommuting relation:

$$\{ \psi^i, \psi^j \} = \delta^{ij}$$

while all other anticommutators are zero. The $C_{ijk}$ are complex numbers with a gaussian distribution with variance $2J/N$. The system has the two supercharges $Q, \bar Q = Q^\dagger$ which are used to define the Hamiltonian:

$$H = \{Q, \bar Q\}$$

## Fermion Clifford algebra
The first step in the study of the system is constructing a numerical representation of the fermion algebra defined by the anticommutation relation.
