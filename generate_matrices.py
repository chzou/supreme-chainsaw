import numpy as np
import os
import pyquil.api as api
from pyquil import get_qc
from hhl_rigetti import get_hhl_2x2, verify_with_swap_test_2x2
import sys
import pickle as pkl

from random import gauss, uniform
def make_rand_vector(dims):
    """ Helper function to generate random vector """
    vec = [uniform(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])

def _run(num_trials = 20, num_shots = 100000):
    """ Generates random vectors and reports their swap test success probability
	:param: num_trials : number of matrices generated for each condition number
	:param: num_shots  : number of trials on quantum computer
			     note-- must be large to gen enough post-selection examples
	:returns: array of 2-element items [condition number, swap test P(success)] """
    sum_success = 0
    actual_trials = 0
    results_list = []
    for j in range(4):
        for i in range(num_trials):
            b = np.array([1., 0.])
            b /= np.linalg.norm(b)
            r = 4

            testVector = make_rand_vector(2)
            V = np.array([[testVector[0], testVector[1]],[-1*testVector[1], testVector[0]]]).T
            Vinv = np.linalg.inv(V)
            D = np.array([[1, 0],[0, 2**j]]) # Matrix of eigenvalues
            A = V.dot(D).dot(Vinv)

            if np.linalg.cond(A) < 1/sys.float_info.epsilon:
                i = np.linalg.inv(A)
            else: # skip in failure case
                print("uh oh")
                return

            # Helpful print statements!
            print("A: ", A)
            print("b: ", b)

            x = np.dot(i, b)
            print("x: ", x)
            x /= np.linalg.norm(x)
            x = np.abs(x)
            print("x (normalized): ", x)

            real_state = x
            qubits = list(range(6))  # Specify actual qubit topology here
            hhl = get_hhl_2x2(A, b, r, qubits)
            swaptest = verify_with_swap_test_2x2(real_state, qubits)
            complete_circuit = hhl + swaptest
            qc = get_qc('6q-qvm')

            complete_circuit.wrap_in_numshots_loop(num_shots)
            results = qc.run(complete_circuit)
            results = np.array(results)
            successful_postselection = results[results[:, 0] == 1]
            N = len(successful_postselection)
            swap_test_after_succ_postselection = successful_postselection[:, 1]
            if N > 5:
                prob_swap_test_success = 1 - sum(swap_test_after_succ_postselection) / N
                print("SUCCESS PROBABILITY:")
                print(prob_swap_test_success)
                sum_success += prob_swap_test_success
                actual_trials += 1
                results_list.append([np.linalg.cond(A), prob_swap_test_success])
        print("Average success:")
        print(sum_success/actual_trials)
        print("Summary:")
        print(results_list)
        np.savetxt('output/Rigetti_QPU', results_list)
        vals = np.array(results_list)
        vals[:,0] = np.round(vals[:,0]).astype(int)
        for val in np.unique(vals[:,0]):
            print("Condition number:")
            print(val)
            print("Success rate:")
            print(vals[vals[:,0] == val, 1].mean())
            
    pkl.dump(results_list, open('matrix_gen_results.p', 'wb'))
    return results_list

if __name__ == '__main__':
    # TODO see above function to run custom experiments
    _run(num_trials=5, num_shots=1000)
