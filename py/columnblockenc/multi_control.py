import numpy as np


class ItenMC():

    def __init__(self):
        pass
    
    def control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        if helper_qubit:
            circ = self.multicontrol(circ, control_qubits, target_qubits[0], helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
            circ = self.multicontrol(circ, control_qubits, target_qubits[0], helper_qubit)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)
        return circ

    def multicontrol(self, circ, control_qubits, helper, target_qubit):
        #Decompose into two k1 and two k2 half multicontrols
        n = len(control_qubits) + 1

        k1 = int(np.ceil(n/2))
        k2 = int(n - k1 - 1)

        circ = self.halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        circ = self.halfcontrol(circ, control_qubits[:k1], control_qubits[k1:] + [helper], target_qubit)
        circ = self.halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        circ = self.halfcontrol(circ, control_qubits[:k1], control_qubits[k1:] + [helper], target_qubit)
        
        return circ

    def halfcontrol(self, circ, additional_qubits, control_qubits, target_qubit):
        #Action part
        num_controls = len(control_qubits)
        num_additional = len(additional_qubits)
        if num_controls > np.ceil((num_additional + num_controls + 1)/2):
            return circ
        if num_controls == 0:
            return circ
        elif num_controls == 1:
            circ.cx(control_qubits[0], target_qubit)
            return circ
        elif num_controls == 2:
            circ.ccx(control_qubits[0], control_qubits[1], target_qubit)
            return circ
        
        qubit_tuples = list(zip(control_qubits[num_controls:1:-1] + [control_qubits[0]]
            , list(reversed(additional_qubits[:min(num_controls-2,num_additional)])) + [control_qubits[1]]
            , [target_qubit] + list(reversed(additional_qubits[:min(num_controls-2,num_additional)])))) 

        c1, c2, t = qubit_tuples[0]
        circ.ccx(c1,c2,t)

        for control1, control2, target in qubit_tuples[1:-1]:
            circ = self.toffoli_diagonal_first_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[-1]
        circ = self.toffoli_to_diagonal(circ, c1, c2, t)
        
        for control1, control2, target in reversed(qubit_tuples[1:-1]):
            circ = self.toffoli_diagonal_second_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[0]
        circ.ccx(c1,c2,t)

        #Reset part
        for control1, control2, target in qubit_tuples[1:-1]:
            circ = self.toffoli_diagonal_first_half(circ, control1, control2, target)
        
        c1, c2, t = qubit_tuples[-1]
        circ = self.toffoli_to_diagonal(circ, c1, c2, t)
    
        for control1, control2, target in reversed(qubit_tuples[1:-1]):
            circ = self.toffoli_diagonal_second_half(circ, control1, control2, target)

        return circ



    def toffoli_diagonal_first_half(self, circ, ctrl1, ctrl2, target):
        # circ.ccx(ctrl1, ctrl2, target)
        circ.ry(-np.pi/4, target)
        circ.cx(ctrl1, target)
        circ.ry(-np.pi/4, target)
        circ.cx(ctrl2, target)
        return circ

    def toffoli_diagonal_second_half(self, circ, ctrl1, ctrl2, target):
        # circ.ccx(ctrl1, ctrl2, target)
        circ.cx(ctrl2, target)
        circ.ry(np.pi/4, target)
        circ.cx(ctrl1, target)
        circ.ry(np.pi/4, target)
        return circ

    def toffoli_to_diagonal(self, circ, ctrl1, ctrl2, target):
        # circ.ccx(ctrl1, ctrl2, target)
        circ.ry(-np.pi/4, target)
        circ.cx(ctrl1, target)
        circ.ry(-np.pi/4, target)
        circ.cx(ctrl2, target)
        circ.ry(np.pi/4, target)
        circ.cx(ctrl1, target)
        circ.ry(np.pi/4, target)
        return circ




class HalfItenMC(ItenMC):
    def __init__(self):
        pass

    def control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        if helper_qubit:
            circ = self.halfcontrol(circ, target_qubits, control_qubits, helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
            circ = self.halfcontrol(circ, target_qubits, control_qubits, helper_qubit)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)
        return circ    
