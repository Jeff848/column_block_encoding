import numpy as np

class SNPWideBinPrepWrapper():
    def __init__(self, wrapped_class, ancillas=0, return_circuit=False):
        self.wrapped_class = wrapped_class
        self.ancillas = ancillas
        self.return_circuit = return_circuit

    def initialize(self, circ, states, target_qubits):
        #Convert states to array st nonzero values are at 2**power positions
        logd = len(target_qubits)
        wide_state = [0] * (2**logd)
        i = 1
        count = 1
        for state in states:
            if state == 0:
                continue
            wide_state[i] = state
            i = 2**count
            count = count + 1
        if self.return_circuit:
            circ = self.wrapped_class.initialize(circ, wide_state, target_qubits)
        else:
            self.wrapped_class.initialize(circ, wide_state, target_qubits)
        return circ

    def ctrl_initialize(self, circ, states, target_qubits, ctrl_qubit):
        init_circ = QuantumCircuit(len(target_qubits))
        if self.return_circuit:
            init_circ = self.initialize(init_circ, states, target_qubits)
        else:
            self.initialize(init_circ, states, target_qubits)

        circ = circ.compose(init_circ.control(1), [ctrl_qubit], target_qubits)
        return circ

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=True):
        if not wide_bin_state_prep:
            raise Exception("Should not be used in non-wide bin state prep mode")
        
        d = sparsity
        logd = int(np.log2(length))
        return d - logd + self.ancillas