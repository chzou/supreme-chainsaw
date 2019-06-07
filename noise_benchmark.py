import numpy as np
import os
import pyquil.api as api
from pyquil.quil import Program
from pyquil.gates import H, I, X, MEASURE
from pyquil import get_qc
from hhl_rigetti import get_hhl_2x2, get_hhl_2x2_corrected, verify_with_swap_test_2x2, fidelity
from collections import defaultdict
import pickle as pkl

# TODO can swap out for your own linear system here!
A = 0.5*np.array([[3, 1], [1, 3]])
b = np.array([1., 0.])
b /= np.linalg.norm(b)
r = 4


""" FUNCTIONS FOR BENCHMARKING NOISE """
def run_until_success(qvm, program):
    """ Runs until post-selection is successful (when qubit 0 is measured to 1) 
	:param: qvm     : the QVMConnection object to run the program on
	:param: program : the Program to be run until success
	:return: amplitudes of successful run, number of repetitions taken until success """
    num_reps = 0
    success  = False
    while not success:
        run = qvm.wavefunction(program)
        success = (abs(sum(run.amplitudes[::2])) < 1e-5)
        num_reps += 1
    return run.amplitudes, num_reps

def benchmark_noise(corrected = False, noise_type = 'X', measurement = False, num_iters = 5):
    """ Main function to benchmark noise
	:param: corrected   : (bool) True to use error-correcting code, False otherwise
	:param: noise_type  : ('X', 'Y', 'Z') specifies which Pauli gate to apply for noise
	:param:	measurement : (bool) True to test measurement noise, False for gate noise
	:param: num_iters   : Number of runs for each experiment 
        :return: map of fidelity ('fid') or repetitions until success ('rep') to 
		 map of noise level to average fidelity/repetitions 
		 e.g.: data['fid'][0.01] will give average fidelity at noise=0.01 """
    if corrected: # error corrected code
        qubits = list(range(6))
        hhl = get_hhl_2x2_corrected(A, b, r, qubits)
    else:         # original hhl code
        qubits = list(range(4))
        hhl = get_hhl_2x2(A, b, r, qubits)
    c = hhl.declare('ro', 'BIT', 1)
    hhl += MEASURE(qubits[0], c[0])
    
    qvm_clean = api.QVMConnection()
    clean_amplitudes, clean_reps = run_until_success(qvm_clean, hhl)
    
    data = {'fid': defaultdict(float), 'rep': defaultdict(float)}
    for noise_level in np.linspace(0.00, 0.1, num=11):
        print('****** {} ******'.format(noise_level))
        if noise_type is 'X':
            noise_params=[noise_level, 0.0, 0.0]
        elif noise_type is 'Y':
            noise_params=[0.0, noise_level, 0.0]
        else:
            noise_params=[0.0, 0.0, noise_level]
            
        if measurement:
            qvm_noise = api.QVMConnection(measurement_noise=noise_params)
        else:
            qvm_noise = api.QVMConnection(gate_noise=noise_params)
    
        denom = 0
        for i in range(num_iters):
            fid = 0
            while fid < 0.9:
                noisy_amplitudes, noisy_reps = run_until_success(qvm_noise, hhl)
                fid = fidelity(clean_amplitudes, noisy_amplitudes)
                data['fid'][noise_level] += fid
                data['rep'][noise_level] += noisy_reps
                denom += 1
            print('fid: {}, reps: {}'.format(fid, noisy_reps))

        data['fid'][noise_level] /= denom
        data['rep'][noise_level] /= num_iters
        
    pkl.dump(data, open('{}_measurement-{}_corrected-{}.p'.format(noise_type, measurement, corrected), 'wb'))
    print(data)
    return data

""" FUNCTIONS TO GENERATE PLOTS """
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def make_fid_rep_graphs(all_data):
    sns.set()
    sns.set_style('whitegrid', {'axes.grid': True})
    matplotlib.rcParams['legend.shadow'] = True
    matplotlib.rcParams['legend.frameon'] = True
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', ]
    linestyles = ['solid', 'dashed', 'dotted', (0, (3, 1, 1, 1))]
    textfontsize = 14
    fontsize = 14
    
    for k, l in [('fid', 'Fidelity'), ('rep', 'Repetitions until Success')]:
        plt.clf()
        plt.ylabel(l, fontsize=textfontsize, rotation=90)
        # plt.ylim((0.0, 1.1))
        plt.xlabel(r'Noise level', fontsize=textfontsize)
        plt.tick_params(axis='both', which='major', pad=3, labelsize=fontsize)

        for i, label in enumerate(all_data.keys()):
            data = all_data[label]
            color = colors[i]
            linestyle = linestyles[i]
            x, y = zip(*data[k].items())
            plt.plot(x, y, label=label, lw=1.1, color=color, linestyle=linestyle)
            
        leg = plt.legend(bbox_to_anchor=(1.0, 1.2), fancybox=True,
                         shadow=True, fontsize=fontsize, ncol=3)
        leg.set_alpha(0.5)
        plt.grid(linewidth=0.5)
        # plt.tight_layout()
    
        plt.show()
        plt.savefig('figures/' + k)

if __name__ == '__main__':
    # TODO see function header for benchmark_noise 
    # to see how to run custom experiments
    benchmark_noise()
