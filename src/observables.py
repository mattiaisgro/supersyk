
"""
Construction of observables and projectors onto the degenerate ground states.
"""

import scipy.sparse as sparse
import numpy as np


"""
Given the eigenvalues and eigenvectors of the Hamiltonian,
construct the projector onto the degenerate ground states
as the product G^\dagger G where G is the matrix of ground state eigenvectors.
"""
def projector(eigenvalues, eigenvectors, tol = 1e-12):
    
    # Indices of ground state eigenvalues
    ground_state_indices = np.where(np.abs(eigenvalues) < tol)[0]

    # Matrix of ground state eigenvectors
    G = eigenvectors[:, ground_state_indices]

    return G @ G.conj().T


"""
If the script is run directly, run a test case which constructs the projector
for N=6 and prints it.
"""
if __name__ == "__main__":

    import fermions
    import supersyk

    N = 4
    J = 1.0
    seed = 42

    couplings = supersyk.coupling(N, J, seed)
    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, couplings)
    H = supersyk.hamiltonian(Q)

    eigenvalues_res, eigenvectors = np.linalg.eigh(H)
    eigenvalues = np.array([E.real for E in eigenvalues_res])

    G = eigenvectors[:, np.where(np.abs(eigenvalues) < 1e-12)[0]]
    print("Ground state eigenvectors:")
    for i in range(G.shape[1]):
        print(G[:, i])

    P = projector(eigenvalues, eigenvectors)

    print(f"Projector (N={N}):")
    print(P)
