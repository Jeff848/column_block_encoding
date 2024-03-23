from qiskit import QuantumCircuit
import numpy as np
from _util import get_padded_matrix


#Base approach
#Create column-based block encoding using qiskit built in amplitude encoding
def create_be_0(a):

    a, n, logn = get_padded_matrix(a)
    
    Oprep_list = []
    Octrl_list = []
    amps = []
    max_amp = 0
    
    for j in range(n): #Prepare state of columns
        state = a[:, j]
        amp = np.sqrt(np.sum(state**2)) # Shouldn't be any empty columns
        prep = QuantumCircuit(logn)
        if amp != 0:
            prep.initialize(state / amp, list(range(logn)))

        #Need to remove resets from preparation circuit
        prep = prep.decompose(reps=5)
        prep.data = [ins for ins in prep.data if ins.operation.name != "reset"]
        Oprep_list.append(prep)
        amps.append(amp)

        ctrl = QuantumCircuit(logn+1)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1

        ctrl.mcx(list(range(logn)), logn)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1
        Octrl_list.append(ctrl)

    max_amp = max(amps)

    #Block encoding circuit
    circ = QuantumCircuit(2*logn + 2)
    flag_qubit = logn * 2
    rotate_qubit = logn * 2 + 1
    for i in range(n):
        circ = circ.compose(Octrl_list[i], list(range(logn)) + [logn*2])
        circ = circ.compose(Oprep_list[i].control(), [flag_qubit] + list(range(logn, 2*logn)))
        rotate_angle = 2 * np.arccos(amps[i] / max_amp)
        circ.cry(rotate_angle, flag_qubit, rotate_qubit)
        circ = circ.compose(Octrl_list[i], list(range(logn)) + [logn*2])


    for i in range(logn):
        circ.swap(i, logn + i)

    for i in range(logn, logn*2):
        circ.h(i)
        
    alpha = max_amp * np.power(np.sqrt(2), logn)
    return circ, alpha

#Use multi-control unitary instead of mcx, eliminating extra qubit
def create_be_1(a):

    a, n, logn = get_padded_matrix(a)
    
    Oprep_list = []
    amps = []
    max_amp = 0
    
    #Pre calculate state and amp
    for j in range(n):
        state = a[:, j]
        amp = np.sqrt(np.sum(state**2)) # Shouldn't be any empty columns
        amps.append(amp)
    max_amp = max(amps)

    for j in range(n): #Prepare state of columns
        state = a[:, j]
        prep = QuantumCircuit(logn+1)
        prep.initialize(state / amps[j], list(range(logn)))

        #Need to remove resets from preparation circuit
        prep = prep.decompose(reps=5)
        prep.data = [ins for ins in prep.data if ins.operation.name != "reset"]

        rotate_angle = 2 * np.arccos(amps[j] / max_amp)
        prep.ry(rotate_angle, logn)

        ctrl = QuantumCircuit(logn+logn+1)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1

        ctrl = ctrl.compose(prep.control(logn), list(range(2*logn+1)))
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1
        Oprep_list.append(ctrl)

    #Block encoding circuit
    circ = QuantumCircuit(2*logn + 1)
    for i in range(n):
        circ = circ.compose(Oprep_list[i], list(range(2*logn+1)))

    for i in range(logn):
        circ.swap(i, logn + i)

    for i in range(logn, logn*2):
        circ.h(i)

    alpha = max_amp * np.power(np.sqrt(2), logn)
    return circ, alpha