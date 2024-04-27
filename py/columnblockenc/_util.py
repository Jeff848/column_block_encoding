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
def gen_random_snp_matrix_prob(n, zero_prob=14223/454207, one_prob=80632/454207):
    snp_prob[0] = zero_prob
    snp_prob[1] = one_prob
    snp_prob[2] = 1. - zero_prob - one_prob
    #Construct matrix a randomly via column snp distribution (use nonzero to retrieve if 0, 1 or 2)
    a= np.nonzero(np.random.multinomial(1, snp_prob, size=(2**n, 1, 2**n)).squeeze())[2] 
    return np.resize(a, (2**n, 2**n)).T

def gen_random_snp_matrix_sparse_range(n, zero_range=None, one_range=None):
    if zero_range is None:
        zero_range = (1, int(n * 14223/454207))
    if one_range is None:
        one_range = (1, int(n * 80632/454207))
    zero_min, zero_max = zero_range
    one_min, one_max = one_range
    zero_count = np.random.randint(zero_min, zero_max)
    one_count = np.random.randint(one_min, one_max)
    return gen_random_snp_matrix_sparse(n, zero_count, one_count)

def gen_random_snp_matrix_sparse(n, zero_count=0, one_count=0):
    a = np.zeros((2**n, 2**n))
    if zero_count + one_count > n:
        return a
    snp_values = [0] * zero_count + [1] * one_count + [2] * (2**n - zero_count - one_count)
    for j in range(2**n):
        a[:, j] = np.random.permutation(snp_values)
    return a


class QiskitMCWrapper():

    def __init__(self):
        pass
    
    @staticmethod
    def control(circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        circ = QiskitMCWrapper.half_control(circ, unitary, control_qubits, target_qubits, helper_qubit)
        if helper_qubit:
            circ.mcx(control_qubits, helper_qubit)

        return circ

    @staticmethod
    def half_control(circ, unitary, control_qubits, target_qubits, helper_qubit):
        if helper_qubit:
            circ.mcx(control_qubits, helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)
        return circ

    @staticmethod
    def mcx(circ, control_qubits, target_qubit, helper_qubits=None):
        circ.mcx(control_qubits, target_qubit)
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
    def ctrl_initialize(self, circ, states, target_qubits, ctrl_qubit):
        init_circ = QuantumCircuit(len(target_qubits))
        QiskitPrepWrapper.initialize(init_circ, states, target_qubits)

        circ = circ.compose(init_circ.control(1), [ctrl_qubit] + target_qubits)
        return circ

    @staticmethod
    def get_ancillas(sparsity, length, wide_bin_state_prep=False): #Number of ancillas should be a function of the length/sparsity of state
        return 0

class SparsePrepWrapper():
    def __init__(self, wrapped_class, ancillas=0, print_sparsity=False, print_circuits=False):
        self.wrapped_class = wrapped_class
        self.ancillas = ancillas
        self.print_sparsity = print_sparsity
        self.print_circuits = print_circuits

    def initialize(self, circ, states, target_qubits):
        #Convert state to dict
        statedict = {}
        d = int(np.ceil(np.log2(len(states))))
        for i, state in enumerate(states):
            bin_string = bin(i)[2:].zfill(d)
            if state != 0:
                statedict[bin_string] = state

        if self.print_sparsity:
            print(len(statedict.values()))
        
        self.wrapped_class.initialize(circ, statedict, target_qubits)
        if self.print_circuits:
            print(circ.draw())
        return circ

    def ctrl_initialize(self, circ, states, target_qubits, ctrl_qubit):
        init_circ = QuantumCircuit(len(target_qubits))
        self.initialize(init_circ, states, target_qubits)

        circ = circ.compose(init_circ.control(1), [ctrl_qubit] + target_qubits)
        return circ

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=False): 
        return self.ancillas

class GeneralPrepWrapper():
    def __init__(self, wrapped_class, ancillas=0, return_circuit=False):
        self.wrapped_class = wrapped_class
        self.ancillas = ancillas
        self.return_circuit = return_circuit

    def initialize(self, circ, states, target_qubits):
        if self.return_circuit:
            circ = self.wrapped_class.initialize(circ, states, target_qubits)
        else:
            self.wrapped_class.initialize(circ, states, target_qubits)
        return circ

    def ctrl_initialize(self, circ, states, target_qubits, ctrl_qubit):
        init_circ = QuantumCircuit(len(target_qubits))
        if self.return_circuit:
            init_circ = self.initialize(init_circ, states, target_qubits)
        else:
            self.initialize(init_circ, states, target_qubits)

        circ = circ.compose(init_circ.control(1), [ctrl_qubit] + target_qubits)
        return circ


    def get_ancillas(self, sparsity, length, wide_bin_state_prep=False):
        return self.ancillas


class SwapPrepWrapper():
    def __init__(self, wrapped_class, ancillas=0, return_circuit=False):
        self.wrapped_class = wrapped_class
        self.ancillas = ancillas
        self.return_circuit = return_circuit

    def initialize(self, circ, states, target_qubits):
        logn = int(len(target_qubits)/2)
        init_circ = QuantumCircuit(logn)
        if self.return_circuit:
            circ = self.wrapped_class.initialize(init_circ, states, list(range(logn)))
        else:
            self.wrapped_class.initialize(circ, states, list(range(logn)))
        
        circ = circ.compose(init_circ, target_qubits[:logn])
        for i in range(logn):
            circ.swap(i, logn + i)
        circ = circ.compose(init_circ.inverse(), target_qubits[:logn])
        
        return circ

    def ctrl_initialize(self, circ, states, target_qubits, ctrl_qubit):
        logn = int(len(target_qubits)/2)
        init_circ = QuantumCircuit(logn)
        if self.return_circuit:
            circ = self.wrapped_class.initialize(init_circ, states, list(range(logn)))
        else:
            self.wrapped_class.initialize(circ, states, list(range(logn)))
        
        circ = circ.compose(init_circ, target_qubits[:logn])
        for i in range(logn):
            circ.cswap(ctrl_qubit, target_qubits[i], target_qubits[i+logn])
        circ = circ.compose(init_circ.inverse(), target_qubits[:logn])
        
        return circ

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=False):
        n = sparsity
        logn = int(np.log2(length))
        return logn + self.ancillas