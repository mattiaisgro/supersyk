
"""
Construction of fermionic operators as matrices respecting anticommutation relations.
"""

import numpy as np
import scipy.sparse as sparse


"""
Construct a 2^N x 2^N matrix representation of the fermion
annihilation operator in the occupation-number basis.

The parameter j is the mode index (0 <= j < N), and N is the total number of modes.
The matrix representation is stored as a sparse matrix (psi_j)_{ik}.
"""
def annihilate(j: int, N: int):

    dimH = 2**N
    psi = sparse.lil_matrix((dimH, dimH), dtype=complex)

    for state in range(dimH):

        # Check if the j-th mode is occupied in the current state.
        is_occupied = (state >> j) & 1

        # Skip zero elements to maintain sparsity.
        if not is_occupied:
            continue

        # Compute additional Wigner sign
        occupied_before = state & ((1 << j) - 1)
        sign = (-1) ** occupied_before.bit_count()

        # The new state is the current state with the
        # j-th mode either removed or added
        new_state = state ^ (1 << j)

        if sign != 0:
            psi[new_state, state] = sign

    return psi


"""
Construct the full representation of the fermionic algebra for N modes.

Returns a list of pairs of annihilation and creation operators,
where each operator is represented as a sparse matrix.
"""
def representation(N: int):

    annihilation = [ annihilate(j, N) for j in range(N) ]
    creation = [ operator.conj().T for operator in annihilation ]

    return annihilation, creation


"""
Compute the total error on the anticommutation relations for the
given fermion representation.
"""
def representation_error(annihilation, creation):

    N = len(annihilation)
    error = 0.0

    # Sum over all anticommutators
    for i in range(N):
        for j in range(N):
            res = annihilation[i] @ creation[j] + creation[j] @ annihilation[i] - (i == j) * sparse.identity(2**N)
            error += np.linalg.norm(res.todense())

    return error


"""
If the module is run as a script, run a simple test.
"""
if __name__ == "__main__":

    N = 4
    annihilation, creation = representation(N)

    print("psi_0:")
    print(annihilation[0])

    error = representation_error(annihilation, creation)
    print(f"Representation Error: {error}")
