# SuperSYK

> The syckest system out there.

This repo contains code for numerical studies of quantum chaos in the supersimmetric Sachdev-Ye-Kitaev model.


## The $\mathcal{N}=2$ Super SYK Model
The $\mathcal{N} = 2$ supersymmetric SYK model with $\hat q = 3$ fermions is a system of N complex fermions with supercharge:

$$
Q = i \sum_{i \< j \< k} C_{ijk} \psi^i \psi^j \psi^k
$$

where the fermion operators have the anticommuting relation:

$$
\\{ \psi_i, \bar \psi_j \\} = \delta_{ij}
$$

while all other anticommutators are zero. The $C_{ijk}$ are complex numbers with a Gaussian distribution with variance $\langle C_{ijk} \bar C^{ijk} \rangle = 2J/N$. The system has the two supercharges $Q, \bar Q = Q^\dagger$ which are used to define the Hamiltonian:

$$
H = \\{Q, \bar Q\\}
$$

## Fermion algebra
The first step in the study of the system is constructing a numerical representation of the fermion Clifford algebra defined by the anticommutation relations:

$$\\{ \psi_i, \bar \psi_j \\} = \delta_{ij}$$
$$\\{ \psi_i, \psi^j \\} = 0$$

One way to view the complex fermion modes is as composed of two real fermions each:

$$\psi_j = \frac{1}{\sqrt{2}} (\chi_j^{(1)} - i \chi_j^{(2)})$$
$$\\{ \chi_i, \chi_j \\} = \delta_{ij}$$

where the fermions are now real, so $\chi_j^\dagger = \chi_j$. If we have N complex fermions, we thus have 2N real fermions.
Our objective is to represent the algebra of complex fermion operators $\psi_j, \bar \psi_j$ as complex matrices acting over states.
Since the action of fermion creation and annihilation operators is to create and destroy a fermion at site $j$, with at most 1 fermion in each site, we can view a generic state as:

$$| n_1, n_2, n_3, \dots \rangle = (\psi_1^\dagger)^{n_1} (\psi_2^\dagger)^{n_2} (\psi_3^\dagger)^{n_3} \dots | 0 \rangle$$

where $|0\rangle$ is the vacuum state, with no fermions in any site, which is annihilated by all $\psi_j$. This is called the "occupation number basis".
Considering a system with N=2, we see that the number of possible states quickly grows with N:

$$|00\rangle, |01\rangle, |10\rangle, |11\rangle$$

In general, for N fermions the total number of states is $2^N$, which causes an exponential growth of states.
A possible approach to building a representation for the algebra of N fermions is based on the Jordan-Wigner transform,
which connects spin systems to fermion systems and vice versa. The base construction makes use of Pauli matrices $\sigma^i$:

$$\psi_j \sim \frac{1}{\sqrt{2}} (\sigma_j^x - i \sigma^y_j)$$

where the $j$-th sigma matrix acts on the space of site $j$. This construction respects the anticommuting relations for a single site,
but when different sites are considered it breaks down. This prompts the use of a "Wigner sign", an overall phase dependent on the previous occupation numbers, which corrects this behavior:

$$e^{i\pi \sum_k \psi_k^\dagger \psi_k} = \prod_k^{j-1} (-\sigma_z) $$

$$\psi_j = \frac{1}{\sqrt{2}} \Big[\prod_k^{j-1} (-\sigma_z)\Big] (\sigma_j^x - i \sigma^y_j)$$

This construction gives the correct representation of the fermion algebra, respecting all anticommutation relations.

## Bit strings
Another consideration should be taken into account: occupation numbers are either 0 or 1, and their collection writes out a number in the binary representation.
If we take the basis defined by representing a given occupation number basis state as the corresponding standard complex vector basis:

$$| n_1 n_2 \dots \rangle = \hat e_{1 + (n_1 n_2 \dots)_{10}}$$

where we are representing the ordered occupation numbers as a number in base 10. For example, for $N=2$ we have that:

$$|00\rangle = \hat e_1, |01\rangle = \hat e_2, |10\rangle = \hat e_3, |11\rangle = \hat e_4$$

This creates a natural representation of states which makes numerical implementations drastically easier, as for example counting occupation numbers reduces to counting bits before the current state, while checking the occupation of a certain site amounts to checking its corresponding bit. This representation of states is sometimes called "bit string" or simply "string".

Thus a possible approach to constructing the full fermion algebra could be to iteratively build the matrices by iterating over all states represented by bit strings, namely the integers $j \in \\{1, \dots, 2^N\\}$ and imposing the correct value according to the inner product over states, while remembering to introduce the Wigner sign to ensure correct anticommutation rules between different sites. We expect the matrices to be $2^N \times 2^N$, but they are generally sparse, as most elements are actually zero.
