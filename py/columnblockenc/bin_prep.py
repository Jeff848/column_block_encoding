import numpy as np

class SNPWideBinPrepWrapper():
    def __init__(self, wrapped_class, ancillas=0, return_circuit=False):
        self.wrapped_class = wrapped_class
        self.ancillas = ancillas
        self.return_circuit = return_circuit

    def initialize(self, circ, states, target_qubits):
        #Convert states to array st nonzero values are at 2**power positions
        logd = len(states)
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

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=True):
        if not wide_bin_state_prep:
            raise Exception("Should not be used in non-wide bin state prep mode")
        
        d = length
        logd = max(int(np.ceil(np.log2(d))), 1)
        return d - logd + self.ancillas