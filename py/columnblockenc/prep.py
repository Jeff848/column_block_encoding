
class BddSparsePrep:

    def __init__(self):
        pass

    def initialize(self, circ, states, target_qubits):
        #Convert state to sparse dict
        statedict = {}
        d = int(np.ceil(np.log2(len(states))))
        for i, state in enumerate(states):
            bin_string = bin(i)[2:].zfill(d)
            if state != 0:
                statedict[bin_string] = state

        #Get ROBDD representation of circuit
        robdd = ROBdd(statedict)

        #Initialze qc to 1 0 ... 0
        circ.x(target_qubits[0])



        return circ

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=False):
        return 1

class ROBdd:
    pass

class SimplePrep:

    def __init__(self):
        pass
    
     def initialize(self, circ, states, target_qubits):
        #Convert state to sparse dict
        statedict = {}
        d = int(np.ceil(np.log2(len(states))))
        for i, state in enumerate(states):
            bin_string = bin(i)[2:].zfill(d)
            if state != 0:
                statedict[bin_string] = state

        #Get ROBDD representation of circuit
        robdd = ROBdd(statedict)

        #Initialze qc to 1 0 ... 0
        circ.x(target_qubits[0])



        return circ

    def get_ancillas(self, sparsity, length, wide_bin_state_prep=False):
        return 1