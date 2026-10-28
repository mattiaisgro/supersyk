
"""
Construct the supercharge and Hamiltonian for the supersymmetric SYK model.
"""

import fermions
import numpy as np
import scipy.sparse as sparse


"""
Construct the random coefficients C_{ijk} for the couplings between fermions.
The coefficients are random complex numbers with a Gaussian distribution
with <C_ijk C_ijk^*> = 2J / N, where J is the coupling strength and N is the number of fermions.
"""
def coupling(N: int, J: float, seed: int):

    np.random.seed(seed)
    C = np.zeros((N, N, N), dtype=complex)
    stdev = np.sqrt(J / N)

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                C[i, j, k] = np.random.normal(0, stdev) + 1j * np.random.normal(0, stdev)

    return C


"""
Construct the supercharge Q = sum_{i<j<k} C_ijk psi_i psi_j psi_k
from the fermionic representation and the coupling coefficients.
"""
def supercharge(representation, couplings):

    N = len(representation[0])
    dimH = 2**N
    Q = sparse.lil_matrix((dimH, dimH), dtype=complex)

    annihilation = representation[0]

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                Q += couplings[i, j, k] * (annihilation[i] @ annihilation[j] @ annihilation[k])

    return sparse.csr_matrix(Q) * 1j


"""
Construct the supercharge Q = sum_{i<j<k} C_ijk psi_i psi_j psi_k
generating the random couplings progressively,
without storing the coupling tensor.
"""
def supercharge(representation, N: int, J: float, seed: int):

    np.random.seed(seed)
    N = len(representation[0])
    dimH = 2**N
    Q = sparse.lil_matrix((dimH, dimH), dtype=complex)
    stdev = np.sqrt(J / N)

    annihilation = representation[0]

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                Cijk = np.random.normal(0, stdev) + 1j * np.random.normal(0, stdev)
                Q += Cijk * (annihilation[i] @ annihilation[j] @ annihilation[k])

    return sparse.csr_matrix(Q) * 1j



"""
Construct the Hamiltonian H = {Q, Q^dagger} from the supercharge Q.
"""
def hamiltonian(Q):

    H = Q @ Q.conj().T + Q.conj().T @ Q

    # Enforce Hermiticity to account for numerical errors
    return 0.5 * (H + H.conj().T)


"""
If the module is run as a script, generate a random Hamiltonian and print it.
"""
if __name__ == "__main__":

    N = 8
    J = 1.0
    seed = 42

    couplings = coupling(N, J, seed)

    print("Couplings:")
    print(couplings)

    rep = fermions.representation(N)
    Q = supercharge(rep, couplings)

    print("Supercharge:")
    print(Q)

    print("Hamiltonian:")
    print(hamiltonian(Q))
