def state_decomposition(nqubits, data):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """
    new_nodes = []
    #Assuming data is in specific order
    # leafs
    for k in data:
        new_nodes.append(
            Node(
                k.index,
                nqubits,
                None,
                None,
                [],
                [],
                abs(k.amplitude),
                NodeType.VALUE
            )
        )

    # build state tree
    is_target = True
    while nqubits > 0:
        nodes = new_nodes
        new_nodes = []
        nqubits = nqubits - 1
        k = 0
        n_nodes = len(nodes)
        while k < n_nodes:
            if is_target:
                mag = np.sqrt(
                    nodes[k].mag ** 2 + nodes[k + 1].mag ** 2
                )
                ntype = NodeType.TARGET
            else:
                mag = max(
                    nodes[k].mag, nodes[k + 1].mag
                )
                ntype = NodeType.CTRL

            new_nodes.append(
                Node(nodes[k].index // 2, nqubits, nodes[k], nodes[k + 1], [], [], mag, ntype)
            )
            nodes[k].parents_left.append(new_nodes[-1])
            nodes[k+1].parents_right.append(new_nodes[-1])
            k = k + 2

        is_target = not is_target
    tree_root = new_nodes[0]
    return tree_root


#Last level contains subnorm of path
def create_angles_tree(state_tree, subnorm=1, end_level=1):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    if state_tree is None:
        return None, subnorm

    mag = 0.0
    if state_tree.ntype == NodeType.TARGET and state_tree.mag != 0.0 and state_tree.right:
        mag = state_tree.right.mag / state_tree.mag * np.sqrt(2)**(state_tree.right.level - state_tree.level-1)


    # Avoid out-of-domain value due to numerical error.
    if mag < -1.0:
        angle_y = 2*np.pi
    elif mag > 1.0:
        angle_y = 0
    else:
        angle_y = 2 * np.arccos(mag)

    is_ctrl = state_tree.ntype == NodeType.CTRL 
    node=None
    if state_tree.level <= end_level:
        node = NodeAngleTree(
            state_tree.index, state_tree.level, angle_y, subnorm, is_ctrl, None, None
        )

    if state_tree.level < end_level:

        subnorm_l = subnorm

        if state_tree.right and state_tree.mag != 0 :
            prev_mag = state_tree.right.mag
            #Go reverse from bottom to top
            for level in range(state_tree.right.level - 1, state_tree.level, -1):
                current_mag = state_tree.right.mag * np.sqrt(2)**(state_tree.right.level - level)
                if level % 2 == 0: #For ctrl nodes
                    subnorm_l = subnorm_l * prev_mag / current_mag
                prev_mag = current_mag
                
            if state_tree.ntype == NodeType.CTRL:
                subnorm_l = subnorm_l * prev_mag / state_tree.mag

        node.right, min_subnorm_l = create_angles_tree(state_tree.right, subnorm=subnorm_l, end_level=end_level)
        subnorm_r = subnorm


        if state_tree.left and state_tree.mag != 0 :
            prev_mag = state_tree.left.mag
            #Go reverse from bottom to top
            for level in range(state_tree.left.level - 1, state_tree.level, -1):
                current_mag = state_tree.left.mag * np.sqrt(2)**(state_tree.left.level - level)

                if level % 2 == 0: #For ctrl nodes
                    subnorm_r = subnorm_r * prev_mag / current_mag
                prev_mag = current_mag
                
            if state_tree.ntype == NodeType.CTRL:
                subnorm_r = subnorm_r * prev_mag / state_tree.mag

        node.left, min_subnorm_r = create_angles_tree(state_tree.left, subnorm=subnorm_r, end_level=end_level)
        subnorm = min(min_subnorm_l, min_subnorm_r)

    return node, subnorm

def bdd_based_sp(angle_tree, circuit, helper_qubit, path_qubit, sparse_qubit, qubit_order,
    multi_control=ItenMC(), current_path=[], last_1childs=[], end_level=1, extra_control=None,
):
    """pre order traversal"""
    #Get 1 branches taken along the current path
    last_1childs_levels = [last_1child.level for last_1child in last_1childs]

    if angle_tree is None: 
        return circuit

    
    last_level = None
    if len(current_path) > 0:
        last_node, _ = current_path[-1]
        last_level = last_node.level

    #If there are reduced nodes between last node on current path and current node
    if not last_level is None and angle_tree.level - last_level > 1:
        for level in range(last_level + 1, angle_tree.level): #Not counting last_level
            if level%2 != 0: #If not ctrl
                circuit = multi_control.mcry(circuit, np.pi/2, [path_qubit] + list(qubit_order[last_1childs_levels]), 
                    qubit_order[level], helper_qubits=[helper_qubit])
                
    #If last node in the path
    if angle_tree.level >= end_level:
        #Add any phase--not relevant for real numbers
        #Add completed path multi control
        control_qubits = []
        if not extra_control is None:
            control_qubits.append(extra_control)

        last_node, is_left = current_path[-1]
        is_terminal_last_node = False#(last_node.left and last_node.right) and (last_node.left.level == end_level) and (last_node.right.level == end_level)
        for node, is_left in current_path:
            if (not node.left is None and not node.right is None and not is_terminal_last_node) or node.is_ctrl:
                if not is_left:
                    circuit.x(qubit_order[node.level])
                control_qubits.append(qubit_order[node.level])

        # if is_terminal_last_node:
        #     if not is_left: #Don't double dip
        #         return circuit
        
        # last_node, _ = current_path[-1]
        # norm = last_node.angle_norm
        # for level in range(last_node.level, angle_tree.level): #Counting the current level
        #     print("Adjusting norm")
        #     print(level)
        #     if level % 2 == 0: #If splits a ctrl node
        #         norm = norm / np.sqrt(2)
        norm = np.clip(angle_tree.angle_norm, 0, 1) #Should be correctly calculated by angle tree
        angle = 2 * np.arccos(norm)

        circuit = multi_control.parallel_mcxry(circuit, angle, control_qubits, 
            path_qubit, rotate_qubit, helper_qubits=[helper_qubit])
        # circuit = multi_control.mcx(circuit, control_qubits, sparse_qubit)
        
        
        for node, is_left in current_path:
            if (node.left and node.right) or node.is_ctrl:
                if not is_left:
                    circuit.x(qubit_order[node.level])
        
        return circuit
    
    left_list = list(last_1childs)
    right_list = list(last_1childs)
    #Do nothing if is_ctrl
    if not angle_tree.is_ctrl and (angle_tree.left and angle_tree.right):
        #2-ctrl ry gate
        circuit = multi_control.mcry(circuit, angle_tree.angle_y, [path_qubit] + list(qubit_order[last_1childs_levels]), 
            qubit_order[angle_tree.level], helper_qubits=[helper_qubit])
        left_list = []
    elif not angle_tree.is_ctrl and angle_tree.left:
        #2-ctrl not gate
        circuit = multi_control.mcx(circuit, [path_qubit] + list(qubit_order[last_1childs_levels]), 
            qubit_order[angle_tree.level], helper_qubits=[helper_qubit])
        left_list = []
    left_list = left_list + [angle_tree]

    circuit = bdd_based(angle_tree.left, circuit, rotate_qubit, 
        helper_qubit, path_qubit, sparse_qubit, qubit_order, 
        multi_control=multi_control, current_path=current_path + [(angle_tree, True)], 
        last_1childs=left_list,end_level=end_level, extra_control=extra_control) #Do highest sv first
    circuit = bdd_based(angle_tree.right, circuit, rotate_qubit, 
        helper_qubit, path_qubit, sparse_qubit, qubit_order, 
        multi_control=multi_control, current_path=current_path + [(angle_tree, False)], 
        last_1childs=right_list,end_level=end_level, extra_control=extra_control)

    return circuit

class BDDPrep:
     
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
    def get_ancillas(sparsity, length, wide_bin_state_prep=False): #Number of ancillas should be a function of the length/sparsity of state
        return 1