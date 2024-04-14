class QiskitBinPrepWrapper():

    def __init__(self):
        pass

    @staticmethod
    def initialize(circ, state, target_qubits):
        circ.initialize(state, target_qubits)
        circ = circ.decompose(reps=5)
        circ.data = [ins for ins in circ.data if ins.operation.name != "reset"]
        return circ

    @staticmethod
    def control(circ, control_qubits, target_qubits, helper_qubit=None):
        if helper_qubit:
            circ.mcx(control_qubits, helper_qubit)
            circ = circ.compose(unitary.control(1), [helper_qubit] + target_qubits)
            circ.mcx(control_qubits, helper_qubit)
        else:
            circ = circ.compose(unitary.control(len(control_qubits)), control_qubits + target_qubits)

        return circ
