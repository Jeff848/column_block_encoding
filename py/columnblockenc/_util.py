import numpy as np
from qiskit import QuantumCircuit

def get_padded_matrix(a):
    n, m = a.shape
    if n != m:
        k = max(n, m)
        a = np.pad(a, ((0, k - n), (0, k - m)))
        n = k
    logn = int(np.ceil(np.log2(n)))
    if n < 2**logn:
        a = np.pad(a, ((0, 2**logn - n), (0, 2**logn - n)))
        n = 2**logn
    return a, n, logn

#0 to d
def gen_random_positive_int_matrix(n, d):
    #Generate matrix
    return np.random.randint(0, d, size=(2**n, 2**n)) 

#SNP random data matrix
snp_prob = np.array([14223/454207, 80632/454207, 359351/454207])
def gen_random_snp_matrix(n):
    #Construct a randomly via column snp distribution (use nonzero to retrieve if 0, 1 or 2)
    a= np.nonzero(np.random.multinomial(1, snp_prob, size=(2**n, 1, 2**n)).squeeze())[2] 
    return np.resize(a, (2**n, 2**n)).T

class QiskitMCWrapper():

    def __init__(self):
        pass
    
    @staticmethod
    def control(circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        if helper_qubit:
            circ.mcx(control_qubits, helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
            circ.mcx(control_qubits, helper_qubit)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)

        return circ

class QiskitPrepWrapper():

    def __init__(self):
        pass
    
    @staticmethod
    def initialize(circ, state, target_qubits):
        ancillas = QiskitPrepWrapper.get_ancillas(len(state), len(state))
        prep_circ = QuantumCircuit(len(target_qubits)-ancillas)
        prep_circ.initialize(state, list(range(len(target_qubits)-ancillas)))
        prep_circ = prep_circ.decompose(reps=5)
        prep_circ.data = [ins for ins in prep_circ.data if ins.operation.name != "reset"]
        circ = circ.compose(prep_circ, target_qubits[ancillas:])
        return circ

    @staticmethod
    def get_ancillas(sparsity, length): #Number of ancillas should be a function of the length/sparsity of state
        return 0