import math
import numpy as np
import pyquil.quil as pq
from numpy import pi as π
from pyquil.quil import Program

import scipy.linalg
from grove.qft.fourier import inverse_qft
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.gates import H, I, X, SWAP, CSWAP, CNOT, CCNOT, MEASURE, RY, RZ, PHASE
from pyquil.api._errors import QVMError


###########################################################################
# 2x2 HHL algorithm #######################################################
###########################################################################
def get_hhl_2x2(A, b, r, qubits):
    '''Generate a circuit that implements the full HHL algorithm for the case
    of 2x2 matrices.

    :param A: (numpy.ndarray) A Hermitian 2x2 matrix.
    :param b: (numpy.ndarray) A vector.
    :param r: (float) Parameter to be tuned in the algorithm.
    :param verbose: (bool) Optional information about the wavefunction.

    :return: A Quil program to perform HHL.
    '''
    p = pq.Program()
    p.inst(create_arbitrary_state(b, [qubits[3]]))
    p.inst(H(qubits[1]))
    p.inst(H(qubits[2]))
    p.defgate('CONTROLLED-U0', controlled(scipy.linalg.expm(2j*π*A/4)))
    p.inst(('CONTROLLED-U0', qubits[2], qubits[3]))
    p.defgate('CONTROLLED-U1', controlled(scipy.linalg.expm(2j*π*A/2)))
    p.inst(('CONTROLLED-U1', qubits[1], qubits[3]))
    p.inst(SWAP(qubits[1], qubits[2]))
    p.inst(H(qubits[2]))
    p.defgate('CSdag', controlled(np.array([[1, 0], [0, -1j]])))
    p.inst(('CSdag', qubits[1], qubits[2]))
    p.inst(H(qubits[1]))
    p.inst(SWAP(qubits[1], qubits[2]))
    uncomputation = p.dagger()
    p.defgate('CRy0', controlled(rY(2*π/2**r)))
    p.inst(('CRy0', qubits[1], qubits[0]))
    p.defgate('CRy1', controlled(rY(π/2**r)))
    p.inst(('CRy1', qubits[2], qubits[0]))
    p += uncomputation
    return p

def bit_code_H(p, q, qubits):
    """ Inserts bit correction code for one H gate
	p: program
        q: qubit to target
        qubits: all qubits """
    x1, x2 = qubits[4], qubits[5]
    p += [CNOT(q, qq) for qq in (x1, x2)]
    
    p += [H(qq) for qq in (q, x1, x2)] # applies H gate
    
    """ METHOD w/o ANCILLAS """
    p += [CNOT(q, qq) for qq in (x1, x2)]
    p += CCNOT(x1, x2, q)
    
    p.reset(x1)
    p.reset(x2)
    

def get_hhl_2x2_corrected(A, b, r, qubits):
    """ HHL program with bit code corrections on first two H gates """
    p = pq.Program()
    p.inst(create_arbitrary_state(b, [qubits[3]]))
    
    # PHASE ESTIMATION
    bit_code_H(p, qubits[1], qubits)  
    bit_code_H(p, qubits[2], qubits)    
    p.defgate('CONTROLLED-U0', controlled(scipy.linalg.expm(2j*π*A/4)))
    p.inst(('CONTROLLED-U0', qubits[2], qubits[3]))
    p.defgate('CONTROLLED-U1', controlled(scipy.linalg.expm(2j*π*A/2)))
    p.inst(('CONTROLLED-U1', qubits[1], qubits[3]))
    p.inst(SWAP(qubits[1], qubits[2]))
    p.inst(H(qubits[2]))
    p.defgate('CSdag', controlled(np.array([[1, 0], [0, -1j]])))
    p.inst(('CSdag', qubits[1], qubits[2]))
    p.inst(H(qubits[1]))
    p.inst(SWAP(qubits[1], qubits[2]))

    p.defgate('CRy0', controlled(rY(2*π/2**r)))
    p.inst(('CRy0', qubits[1], qubits[0]))
    p.defgate('CRy1', controlled(rY(π/2**r)))
    p.inst(('CRy1', qubits[2], qubits[0]))
    
    # HARD CODE THE INVERSE PHASE ESTIMATION :'(
    p.inst(SWAP(qubits[1], qubits[2]))
    p.inst(H(qubits[1]))
    p.defgate('CSdag-INV', controlled(np.array([[1, 0], [0, 1j]])))
    p.inst(('CSdag-INV', qubits[1], qubits[2]))
    p.inst(H(qubits[2]))
    p.inst(SWAP(qubits[1], qubits[2]))
    p.defgate('CONTROLLED-U1-INV', controlled(scipy.linalg.expm(-2j*π*A/2)))
    p.inst(('CONTROLLED-U1-INV', qubits[1], qubits[3]))
    p.defgate('CONTROLLED-U0-INV', controlled(scipy.linalg.expm(-2j*π*A/4)))
    p.inst(('CONTROLLED-U0-INV', qubits[2], qubits[3]))    
    p.inst(H(qubits[2]))
    p.inst(H(qubits[1]))

    # undoes create_arbitrary_state
    p.inst(RY(π/2, qubits[3]))
    p.inst(H(qubits[3]))
    return p

def verify_with_swap_test_2x2(reference_amplitudes, qubits):
    '''Generate a circuit for performing a swap test.
    :param program: (pyquil.quil.Program) Program storing the HHL algorithm.
    :param reference_amplitudes: (numpy.ndarray) Amplitudes of the state to be
                                  compared against.
    :return: A Quil program to do a swap test after inverting 2x2 matrices.
    '''
    swaptest = pq.Program()
    swaptest.inst(create_arbitrary_state(reference_amplitudes, [qubits[4]]))
    swaptest.inst(H(qubits[5]))
    swaptest.inst(CSWAP(qubits[5], qubits[4], qubits[3]))
    swaptest.inst(H(qubits[5]))
    c = swaptest.declare('ro', 'BIT', 2)
    swaptest += MEASURE(qubits[0], c[0]) # This is actually the post-selection in HHL
    swaptest += MEASURE(qubits[5], c[1])
    return swaptest


###############################################################################
# Helper functions ############################################################
###############################################################################
def fidelity(amplitudes1, amplitudes2):
    '''Helper function, computes the fidelity of two pure quantum states given
    their amplitude vectors. '''
    return np.abs(np.dot(amplitudes1.conj(), amplitudes2))**2

def rY(angle):
    '''Generate a rotation matrix over the Y axis in the Bloch sphere.
    :param angle: (float) The angle of rotation.
    :return: (numpy.ndarray) The rotation matrix
    '''
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                     [np.sin(angle/2), np.cos(angle/2)]])

