import numpy as np
import qutip as qt
import os
import scipy.linalg as sl
from tqdm import tqdm

def Hb_fun(ω0, j, M):
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

def coh_state_fun(E: np.float128, M: np.int64, j):

    '''
    Returns coherent state with given average energy 
    
    E: Average energy of the state
    
    M: Photon cutoff

    '''
    α = np.sqrt(E)

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

    fock_state = qt.tensor(qt.fock(M, int(E/ω)),gnd_b_state)

    return fock_state

def pnm_fun(n, m, ρ , Hb):
    '''
    Use ρ and Hb to find passive state ρp and unitary to take there U. And then find joint probability of transition ρp to |Em> to Udag|Em> to |En>.
    '''
    pnm = None

    return pnm

def pnm_matrix_fun(j):

    '''
    Use pnm_fun here to calculate the pnm matrix which will be useful to calculate ergotropy and ergotropy fluctuation.
    '''

    dim = int(2*j+1)
    pnm_matrix = np.zeros((dim, dim)) # Use pnm_fun here to calcualte for each possible transition

    return pnm_matrix

def erg_fun(pnm_matrix):

    erg = None # Some function of pnm_matrix

    return erg