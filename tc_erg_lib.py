import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
from tqdm import tqdm

def Hb_full_fun(ω0, j, M):
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    Hb = ω0 * (j+Jz)

    return Hb

def Hc_fun(ω, j, M):
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    Hc = ω * a.dag() * a

    return Hc
    
def Hb_fun(ω0, j, M):
    Jz = qt.jmat(j, 'z')
    Hb = ω0 * (j+Jz)

    return Hb

def TC_fun(ω, ω0, j, M, g):
    '''
    Tavis-Cummings Hamiltonian for the following parameters.
    Args:
    - ω : frequency of the bosonic field
    - ω0 : Energy difference in spin states
    - j : Pseudospin
    - M : Upper limit of bosonic fock states
    - g : Coupling strength
    '''
    a  = qt.tensor(qt.destroy(M), qt.qeye(int(2*j+1)))
    Jp = qt.tensor(qt.qeye(M), qt.jmat(j, '+'))
    Jm = qt.tensor(qt.qeye(M), qt.jmat(j, '-'))
    Jz = qt.tensor(qt.qeye(M), qt.jmat(j, 'z'))
    H0 = ω * a.dag() * a + ω0 * (j+Jz)
    H1 = 1.0 / np.sqrt(2*j) * (a * Jp+ a.dag() * Jm)
    H = H0 + g * H1
    # H_even = H[::2,::2]
    
    return H

def coh_state_fun(E: np.float128, M: np.int64, ω, j):

    '''
    Returns coherent state with given average energy 
    
    E: Average energy of the state
    
    M: Photon cutoff

    '''
    α = np.sqrt(E/ω)

    gnd_b_state = qt.fock(int(2*j+1), int(2*j))

    coh_state = qt.tensor(qt.coherent(M, α), gnd_b_state)

    return coh_state

def fock_state_fun(E: np.float128, M: np.int64, ω: np.float128, j):

    '''
    Returns Fock state with given average energy 

    E: Average energy of the state
    
    M: Photon cutoff

    '''
    gnd_b_state = qt.fock(int(2*j+1), int(2*j))

    fock_state = qt.tensor(qt.fock(M, int(E/ω)), gnd_b_state)

    return fock_state

def ρ_pass_fun(H, ρ0):
    '''
    Find passive state

    H : Hamiltonian of the system
    
    ρ0 : State of the system 

    '''
    r_vals, r_vecs =ρ0.eigenstates()
    
    N = len(r_vals)
        
    # r_list in descending order
    idx = np.argsort(r_vals)[::-1]   # descending
    r_vals = [r_vals[i] for i in idx]
    r_vecs = [r_vecs[i] for i in idx]
    # e_list in ascending order
    e_vals, e_vecs = H.eigenstates()
    idx = np.argsort(e_vals)   # ascending
    e_vals = [e_vals[i] for i in idx]
    e_vecs = [e_vecs[i] for i in idx]

    # unitary operator
    U = 0
    for j in range(N):
        U += e_vecs[j] * r_vecs[j].dag()

    U_dag = U.dag()

    # Passive state
    ρ_pass = U @ ρ0 @ U_dag

    return ρ_pass, U

def energy_projectors(H):
    '''
    Energy projectors of Hamiltonian
    '''
    e_vals, e_vecs = H.eigenstates()
    projectors = [ket*ket.dag() for ket in e_vecs]
    return e_vals, projectors

def pnm_fun(proj_n, proj_m, ρb0 , Hb):
    '''
    Use ρ0 and Hb to find passive state ρp and unitary to take there U. And then find joint probability of transition ρp to |Em> to Udag|Em> to |En>.
    '''

    ρ_pass, U = ρ_pass_fun(Hb,ρb0)

    pnm = (proj_n*U.dag()*proj_m*ρ_pass*proj_m*U*proj_n).tr()

    return pnm

def pnm_matrix_fun(ρb0, Hb):

    '''
    Use pnm_fun here to calculate the pnm matrix which will be useful to calculate ergotropy and ergotropy fluctuation.
    '''
    assert np.shape(ρb0) == np.shape(Hb)

    dim = np.shape(Hb)[0]
    pnm_matrix = np.zeros((dim, dim)) # Use pnm_fun here to calcualte for each possible transition
    
    evals, projectors = energy_projectors(Hb)

    for n in range(dim):
        proj_n = projectors[n]
        for m in range(dim):
            proj_m = projectors[m]
            pnm_matrix[n,m] = pnm_fun(proj_n, proj_m, ρb0, Hb)

    return pnm_matrix, evals

# def erg_fun(ρb0, Hb):
    
#     pnm_matrix, evals = pnm_matrix_fun(ρb0, Hb)

#     evals = np.array(evals)        # shape (N,)
#     evals_diff_mat = (evals[:, None] - evals[None, :])

#     erg = np.sum(evals_diff_mat.T * pnm_matrix)

#     return erg

def erg_fun(ρb0, Hb):

    dim = np.shape(Hb)[0]
    re_mat = np.zeros((dim, dim))
    iden = np.identity(dim)

    r_vals, r_vecs =ρb0.eigenstates()
    
    # r_list in descending order
    idx = np.argsort(r_vals)[::-1]   # descending
    r_vals = [r_vals[i] for i in idx]
    r_vecs = np.column_stack([r_vecs[i].full().ravel() for i in idx])
    # e_list in ascending order
    e_vals, e_vecs = Hb.eigenstates()
    idx = np.argsort(e_vals)   # ascending
    e_vals = [e_vals[i] for i in idx]
    e_vecs = np.column_stack([e_vecs[i].full().ravel() for i in idx])

    re_dot_mat_minus_iden = (np.abs(np.conjugate(r_vecs).T * e_vecs)**2 - iden)

    re_mat = np.outer(r_vals, e_vals)

    erg_mat = re_mat * re_dot_mat_minus_iden
    erg = np.sum(erg_mat)

    return erg

def erg_var_fun(pnm_matrix):

    erg_var = None # Some function of pnm_matrix

    return erg_var