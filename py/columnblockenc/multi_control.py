import numpy as np
from qiskit.circuit.library.standard_gates import RYGate


class ItenMC():

    def __init__(self):
        pass
    
    def control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        circ = self.half_control(circ, unitary, control_qubits, target_qubits, helper_qubit)
        if helper_qubit:
            circ = self.multicontrol(circ, control_qubits, target_qubits[0], helper_qubit)
        return circ

    def half_control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None, ctrl_initialize=None):
        if helper_qubit:
            circ = self.multicontrol(circ, control_qubits, target_qubits[0], helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)
        return circ

    def mcx(self, circ, control_qubits, target_qubit, helper_qubits=None):
        # if not helper_qubits:
        #     return circ
        circ.mcx(control_qubits, target_qubit)
        return circ
        # return self.multicontrol(circ, control_qubits, helper_qubits[0], target_qubit)

    def mcry(self, circ, rotate_angle, control_qubits, target_qubit, helper_qubits=None):
        # print(rotate_angle)
        # circ.ry(-rotate_angle / 2, target_qubit)
        # circ.mcry(rotate_angle, control_qubits, target_qubit)
        circ = circ.compose(RYGate(rotate_angle).control(len(control_qubits)), control_qubits + [target_qubit])
        # circ.ry(rotate_angle / 2, target_qubit)
        return circ

    def multicontrol(self, circ, control_qubits, helper, target_qubit):
        #Decompose into two k1 and two k2 half multicontrols
        n = len(control_qubits) + 1

        k1 = int(np.ceil(n/2))
        k2 = int(n - k1 - 1)

        circ = self.shor_halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        circ = self.shor_halfcontrol(circ, control_qubits[:k1], control_qubits[k1:] + [helper], target_qubit)
        circ = self.shor_halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        circ = self.shor_halfcontrol(circ, control_qubits[:k1], control_qubits[k1:] + [helper], target_qubit)
        
        return circ

    def multicontrolxry(self, circ, angle, control_qubits, helper, target_qubit, rotate_qubit):
        #Decompose into two k1 and two k2 half multicontrols
        n = len(control_qubits) + 1

        k1 = int(np.ceil(n/2))
        k2 = int(n - k1 - 1)
        circ.ry(angle/2, rotate_qubit)
        circ = self.shor_halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        circ = self.parallel_shor_halfcontrol(circ, control_qubits[:k1], 
            control_qubits[k1:] + [helper], target_qubit, rotate_qubit)
        circ = self.shor_halfcontrol(circ, control_qubits[k1:], control_qubits[:k1], helper)
        # circ = self.parallel_shor_halfcontrol(circ, control_qubits[:k1], 
        #     control_qubits[k1:] + [helper], target_qubit, rotate_qubit) 
        circ.ry(-angle/2, rotate_qubit)
        
        # print(circ.draw())
        return circ

    def parallel_mcxry(self, circ, angle_norm, control_qubits, target_qubit, rotate_qubit, helper_qubits=None):

        # if angle_norm == 0.0: #No need to parallelize
        #     return self.mcx(circ, control_qubits, target_qubit, helper_qubits=helper_qubits)

        
        # # circ.mcx(list(control_qubits), rotate_qubit)
        if angle_norm != 0:
            circ = self.mcx(circ, control_qubits, target_qubit, helper_qubits=helper_qubits)
            # circ = self.mcry(circ, angle_norm, control_qubits, rotate_qubit, helper_qubits=helper_qubits)
            circ = circ.compose(RYGate(angle_norm).control(len(list(control_qubits))), 
                list(control_qubits) + [rotate_qubit])
            # print(angle_norm)
            # circ = self.multicontrolxry(circ, angle_norm, control_qubits, helper_qubits[0], target_qubit, 
            #     rotate_qubit)
        else:
            circ = self.mcx(circ, control_qubits, target_qubit, helper_qubits=helper_qubits)

        return circ
        # return self.multicontrolxry(circ, angle_norm, control_qubits, helper_qubits[0], target_qubit, rotate_qubit)


    def shor_halfcontrol(self, circ, additional_qubits, control_qubits, target_qubit):
        #Action part
        num_controls = len(control_qubits)
        num_additional = len(additional_qubits)
        if num_controls > np.ceil((num_additional + num_controls + 1)/2):
            print("Not enough helpers")
            return circ
        if num_controls == 0:
            return circ
        elif num_controls == 1:
            circ.cx(control_qubits[0], target_qubit)
            return circ
        elif num_controls == 2:
            circ = self.efficient_toffoli(circ,control_qubits[0],control_qubits[1],target_qubit)
            return circ
        
        qubit_tuples = list(zip(control_qubits[num_controls:1:-1] + [control_qubits[0]]
            , list(reversed(additional_qubits[:min(num_controls-2,num_additional)])) + [control_qubits[1]]
            , [target_qubit] + list(reversed(additional_qubits[:min(num_controls-2,num_additional)])))) 

        c1, c2, t = qubit_tuples[0]
        circ = self.efficient_toffoli(circ,c1,c2,t)

        for control1, control2, target in qubit_tuples[1:-1]:
            circ = self.toffoli_diagonal_first_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[-1]
        circ = self.toffoli_to_diagonal(circ, c1, c2, t)
        
        for control1, control2, target in reversed(qubit_tuples[1:-1]):
            circ = self.toffoli_diagonal_second_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[0]
        circ = self.efficient_toffoli(circ,c1,c2,t)

        #Reset part
        for control1, control2, target in qubit_tuples[1:-1]:
            circ = self.toffoli_diagonal_first_half(circ, control1, control2, target)
        
        c1, c2, t = qubit_tuples[-1]
        circ = self.toffoli_to_diagonal(circ, c1, c2, t)
    
        for control1, control2, target in reversed(qubit_tuples[1:-1]):
            circ = self.toffoli_diagonal_second_half(circ, control1, control2, target)

        return circ

    #Parallel (initialize two qubits at once) with one being a rotate
    def parallel_shor_halfcontrol(self, circ, additional_qubits, control_qubits, target_qubit, rotate_qubit):
        #Action part
        num_controls = len(control_qubits)
        num_additional = len(additional_qubits)
        if num_controls > np.ceil((num_additional + num_controls + 1)/2):
            return circ
        if num_controls == 0:
            return circ
        elif num_controls == 1:
            circ.cx(control_qubits[0], target_qubit)
            circ.cry(angle,control_qubits[0], rotate_qubit)
            return circ
        elif num_controls == 2:
            circ = self.parallel_efficient_toffoli(circ,control_qubits[0],control_qubits[1],target_qubit,rotate_qubit)
            # circ = self.efficient_toffoli(circ,control_qubits[0],control_qubits[1],rotate_qubit)
            return circ
        
        qubit_tuples = list(zip(control_qubits[num_controls:1:-1] + [control_qubits[0]]
            , list(reversed(additional_qubits[:min(num_controls-2,num_additional)])) + [control_qubits[1]]
            , [target_qubit] + list(reversed(additional_qubits[:min(num_controls-2,num_additional)])))) 

        c1, c2, t = qubit_tuples[0]
        circ = self.parallel_efficient_toffoli(circ,c1,c2,t,rotate_qubit)

        # circ.ry(angle, rotate_qubit)

        for control1, control2, target in qubit_tuples[1:-1]:
            circ = self.toffoli_diagonal_first_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[-1]
        circ = self.toffoli_to_diagonal(circ, c1, c2, t)
        
        for control1, control2, target in reversed(qubit_tuples[1:-1]):
            circ = self.toffoli_diagonal_second_half(circ, control1, control2, target)

        c1, c2, t = qubit_tuples[0]
        circ = self.parallel_efficient_toffoli(circ,c1,c2,t,rotate_qubit)
        

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

    def efficient_toffoli(self, circ, ctrl1, ctrl2, target):
        circ.ccx(ctrl1, ctrl2, target)
        # circ.h(target)
        # circ.cx(ctrl2, target)
        # circ.tdg(target)
        # circ.cx(ctrl1, target)
        # circ.t(target)
        # circ.cx(ctrl2, target)
        # circ.tdg(target)
        # circ.cx(ctrl1, target)
        # circ.t(target)
        # circ.h(target)
        # circ.t(ctrl2)
        # circ.cx(ctrl1, ctrl2)
        # circ.t(ctrl1)
        # circ.tdg(ctrl2)
        # circ.cx(ctrl1, ctrl2)
        return circ

    def parallel_efficient_toffoli(self, circ, ctrl1, ctrl2, target1, target2):
        # circ.ccx(ctrl1, ctrl2, target1)
        # circ.ccx(ctrl1, ctrl2, target2)
        circ.h(target1)
        circ.h(target2)
        circ.cx(ctrl2, target1)
        circ.cx(ctrl2, target2)
        circ.tdg(target1)
        circ.tdg(target2)
        circ.cx(ctrl1, target1)
        circ.cx(ctrl1, target2)
        circ.t(target1)
        circ.t(target2)
        circ.cx(ctrl2, target1)
        circ.cx(ctrl2, target2)
        circ.tdg(target1)
        circ.tdg(target2)
        circ.cx(ctrl1, target1)
        circ.cx(ctrl1, target2)
        circ.t(target1)
        circ.h(target1)
        circ.t(target2)
        circ.h(target2)
        
        circ.t(ctrl2)
        circ.cx(ctrl1, ctrl2)
        circ.t(ctrl1)
        circ.tdg(ctrl2)
        circ.cx(ctrl1, ctrl2)
        return circ

    def efficient_ccry(self, rotate_angle, circ, ctrl1, ctrl2, target):
        circ.cry(rotate_angle / 2, ctrl2, target)
        circ.cx(ctrl1, ctrl2)
        circ.cry(-rotate_angle / 2, ctrl2, target)
        circ.cx(ctrl1, ctrl2)
        circ.cry(rotate_angle / 2, ctrl1, target)
        return circ

    #Only works if rotate_angle is positive
    def mcry_halfcontrol(self, rotate_angle, circ, additional_qubits, control_qubits, target_qubit):
        circ.ry(rotate_angle / 2, target_qubit)
        circ = self.shor_halfcontrol(circ, additional_qubits, control_qubits, target_qubit)
        circ.ry(-rotate_angle / 2,  target_qubit)
        return circ
    


class HalfItenMC(ItenMC):
    def __init__(self):
        pass

    def control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None):
        circ = self.half_control(circ, unitary, control_qubits, target_qubits, helper_qubit)
        if helper_qubit:
            circ = self.shor_halfcontrol(circ, target_qubits, control_qubits, helper_qubit)
        return circ

    def half_control(self, circ, unitary, control_qubits, target_qubits, helper_qubit=None, ctrl_initialize=None):
        if helper_qubit:
            circ = self.shor_halfcontrol(circ, target_qubits, control_qubits, helper_qubit)
            if ctrl_initialize is None:
                circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
            else:
                circ = ctrl_initialize.ctrl_initialize(circ, unitary, target_qubits, helper_qubit)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)
        return circ    

    def mcx(self, circ, control_qubits, target_qubit, helper_qubits=None):
        if not helper_qubits:
            return circ
        circ.mcx(control_qubits, target_qubit)
        return circ
        # return self.shor_halfcontrol(circ, helper_qubits, control_qubits, target_qubit)

    def mcry(self, circ, rotate_angle, control_qubits, target_qubit, helper_qubits=None):
        if not helper_qubits:
            return circ
        # circ.mcry(rotate_angle, control_qubits, target_qubit)
        # return circ
        return self.mcry_halfcontrol(rotate_angle, circ, helper_qubits, control_qubits, target_qubit)