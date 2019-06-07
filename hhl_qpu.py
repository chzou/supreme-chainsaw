import numpy as np
import os
import pyquil.api as api
from pyquil import get_qc
from hhl_rigetti import get_hhl_2x2, verify_with_swap_test_2x2


if __name__ == '__main__':
    A = 0.5*np.array([[3, 1], [1, 3]])
    b = np.array([1., 0.])
    b /= np.linalg.norm(b)
    r = 4
    real_state = [0.9492929682, 0.3143928443]

    # TODO change this to match your desired lattice
    qc = get_qc('Aspen-4-6Q-A')
    qubits = qc.device.qubits()
    
    hhl = get_hhl_2x2(A, b, r, qubits)
    swaptest = verify_with_swap_test_2x2(real_state, qubits)
    complete_circuit = hhl + swaptest

    complete_circuit.wrap_in_numshots_loop(10000)
    binary = qc.compile(complete_circuit)
    results = qc.run(binary)
    results = np.array(results)
    successful_postselection = results[results[:, 0] == 1]
    N = len(successful_postselection)
    swap_test_after_succ_postselection = successful_postselection[:, 1]
    prob_swap_test_success = 1 - sum(swap_test_after_succ_postselection) / N
    print('swap test p(success)={}'.format(prob_swap_test_success))

