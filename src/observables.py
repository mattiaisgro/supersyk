
"""
Construction of observables and projectors onto the degenerate ground states.
"""

import scipy.sparse as sparse
import numpy as np


def projector(eigenvalues, eigenvectors, tol = 1e-12):

    """
    Given the eigenvalues and eigenvectors of the Hamiltonian,
    construct the projector onto the degenerate ground states
    as the matrix of ground state eigenvectors.
    """
    
    # Indices of ground state eigenvalues
    indices = np.where(np.abs(eigenvalues) < tol)[0]

    return eigenvectors[:, indices]


def project(operator, P):

    """
    Use the projector to project a certain operator onto the degenerate ground states.
    """

    return P.conj().T @ operator @ P


def symmetrize(operator):

    """
    Symmetrize the given operator to make it Hermitian with real eigenvalues.
    """

    return 0.5 * (operator + operator.conj().T)


def level_spacing(eigenvalues):

    """
    Compute the level spacing statistics of the Hamiltonian spectrum.
    """
    
    eigs_sorted = np.sort(eigenvalues)
    spacings = np.diff(eigs_sorted)

    return spacings / np.mean(spacings)


def fermion_number(psi, psi_bar):

    """
    Get a representation of the fermion number operator from
    the matrix representation of the complex fermions.
    """

    dimH = psi[0].shape[0]
    N = sparse.lil_matrix((dimH, dimH), dtype=complex)

    for i in range(len(psi)):
        N += psi_bar[i] * psi[i]

    return sparse.csr_matrix(N)


def r_charge(psi, psi_bar):

    """
    Get a representation of the R-charge operator from
    the matrix representation of the complex fermions.
    """

    N = len(psi)
    dimH = psi[0].shape[0]

    return fermion_number(psi, psi_bar) - N/2 * sparse.identity(dimH, dtype=complex)



if __name__ == "__main__":

    """
    If the script is run directly, run a test case which constructs the projector
    for the ground states of a random Hamiltonian.
    """

    import fermions
    import supersyk

    N = 10
    J = 1.0
    seed = 42

    couplings = supersyk.coupling(N, J, seed)
    rep = fermions.representation(N)
    Q = supersyk.supercharge(rep, N, J, seed)
    H = supersyk.hamiltonian(Q)

    # Check that the supercharge squares to zero
    Q2_residual = np.linalg.norm(Q @ Q)
    print("Q^2 =", Q2_residual.toarray()[0, 0].real)

    Number = fermion_number(rep[0], rep[1])
    vacuum = sparse.lil_matrix((2**N, 1), dtype=complex)
    vacuum[0, 0] = 1.0
    print("Particles in the vacuum: ", (vacuum.T @ Number @ vacuum).toarray()[0, 0].real)

    eigenvalues_, eigenvectors = np.linalg.eigh(H.toarray())
    eigenvalues = np.array([E.real for E in eigenvalues_])
    G = eigenvectors[:, np.where(np.abs(eigenvalues) < 1e-12)[0]]
    print(f"Ground state eigenvectors: {G.shape[1]} ({G.shape[1] / len(eigenvalues):.2%})")

    P = projector(eigenvalues, eigenvectors)
    print(f"Projector (N={N}):")
    print(P)
