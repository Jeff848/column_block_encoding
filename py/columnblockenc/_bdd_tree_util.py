# from ._angle_tree_util import NodeType
# from .multi_control import ItenMC
from _angle_tree_util import NodeType
from multi_control import ItenMC
import numpy as np
from qiskit.circuit.library.standard_gates import RYGate



def leavesAndParents(tree):
    parents = []
    children = [tree]
    while tree.ntype != NodeType.VALUE:
        nodes = children
        children = []
        for node in nodes:
            if node.left:
                children.append(node.left)
            if node.right:
                children.append(node.right)
        parents = nodes
        tree = children[0]

    return children, parents

def leavesBDD(tree):
    leaves = []
    stack = [(tree, 1)]
    leafToFreq = {}
    while len(stack) > 0:
        current_node, num_ways = stack.pop()
        # print(str(current_node))
        # visited.add(str(current_node.level) + ":" + str(current_node.index))
        node_hash = str(current_node.level) + "_" + str(current_node.index)
        if current_node.ntype == NodeType.VALUE and not node_hash in leafToFreq:
            leaves.append(current_node)
            leafToFreq[node_hash] = num_ways
        elif current_node.ntype == NodeType.VALUE:
            leafToFreq[node_hash] += num_ways
        
        if current_node.right:
            stack.append((current_node.right, num_ways * 2**(current_node.right.level - current_node.level - 1)))
        
        if current_node.left:
            stack.append((current_node.left,  num_ways * 2**(current_node.left.level - current_node.level - 1)))

    return leaves, leafToFreq


def equal_trees(tree1, tree2):
    if tree1 is None or tree2 is None:
        if tree1 is None and tree2 is None:
            return True
        else:
            return False

    if tree1.ntype != tree2.ntype:
        return False
    
    if tree1.mag != tree2.mag:
        return False
    
    if tree1.level != tree2.level:
        return False
    
    if not equal_trees(tree1.left, tree2.left):
        return False

    if not equal_trees(tree1.right, tree2.right):
        return False

    return True

def convert_tree_to_bdd(tree, parent_nodes=None, current_nodes=None):
    #Go to leaf
    current_nodes, parent_nodes = leavesAndParents(tree)

    i = 0
    #Apply rules in bottom-up fashion
    while len(current_nodes) > 0:
        parent_nodes = []
        #Get parent nodes
        parent_nodes.extend(current_nodes[0].parents_left)
        for node in current_nodes:
            parent_nodes.extend(node.parents_right)

        if len(parent_nodes)==0:
            break

        #Merge Rule
        merge_nodes = []
        while len(current_nodes) > 0:
            current_node = current_nodes.pop(0)
            merge_nodes.append(current_node)

            #Merge into current_node
            temp = []
            for node in current_nodes:
                if node.ntype == current_node.ntype and equal_trees(node, current_node):
                    for parent_l in node.parents_left:
                        parent_l.left = current_node
                        if parent_l not in current_node.parents_left:
                            current_node.parents_left.append(parent_l)

                    for parent_r in node.parents_right:
                        parent_r.right = current_node
                        if parent_r not in current_node.parents_right:
                            current_node.parents_right.append(parent_r)

                    if node.left:
                        node.left.parents_left.remove(node)
                        node.left.parents_left.append(current_node)

                    if node.right:
                        node.right.parents_right.remove(node)
                        node.right.parents_right.append(current_node)
                else:
                    temp.append(node)
            current_nodes = temp

        current_nodes = merge_nodes

        #Deletion Rule
        while len(current_nodes) > 0:
            current_node = current_nodes.pop(0)
            if current_node.left and current_node.right and current_node.left == current_node.right:
                #Remove this from child parents list
                # print("Deleting " + str(current_node.level) + "_" + str(current_node.index))
                current_node.left.parents_left.remove(current_node)
                current_node.left.parents_right.remove(current_node)

                for parent_l in current_node.parents_left:
                    parent_l.left = current_node.left
                    current_node.left.parents_left.append(parent_l)

                for parent_r in current_node.parents_right:
                    parent_r.right = current_node.left
                    current_node.left.parents_right.append(parent_r)
       

        current_nodes = parent_nodes
        # i=i+1

    return tree

def common_case_centering(tree):
    """post order traversal"""
    #Go to leaves
    leaves, leafFreq = leavesBDD(tree)

    # for leaf in leaves:
    #     print(str(leaf))
    # print(leafFreq)

    #Get most common case

    common_leaf_ind = max(leafFreq, key=leafFreq.get)
    # print(common_leaf_ind)
    for leaf in leaves:
        if common_leaf_ind == str(leaf.level) + "_" + str(leaf.index):
            common_leaf = leaf
            break

    #Remove most common case from bdd
    #for other leaves Center around common case
    for parent_l in common_leaf.parents_left:
        parent_l.left = None
    for parent_r in common_leaf.parents_right:
        parent_r.right = None

    for leaf in leaves:
        if leaf != common_leaf:
            leaf.mag = np.abs(leaf.mag - common_leaf.mag)

    #Do postorder traversal of tree to fix corresponding magnitudes
    tree = recalculate_mag_bdd(tree)

    return common_leaf, tree

def recalculate_mag_bdd(tree):
    #If null node do nothing
    if not tree:
        return None
    #Don't change leaf node
    if tree.ntype == NodeType.VALUE:
        return tree

    left = recalculate_mag_bdd(tree.left)
    right = recalculate_mag_bdd(tree.right)
    mag = 0
    if left:
        left_mag = left.mag**2 * 2**(left.level - tree.level - 1)
        if tree.ntype == NodeType.TARGET:
            mag += left_mag
        else:
            mag = max(left_mag, mag)
    if right:
        right_mag = right.mag**2 * 2**(right.level - tree.level - 1)
        if tree.ntype == NodeType.TARGET:
            mag += right_mag
        else:
            mag = max(right_mag, mag)

    tree.mag = np.sqrt(mag)
    return tree

def bdd_based(angle_tree, circuit, rotate_qubit, helper_qubit, path_qubit, sparse_qubit, qubit_order,
    multi_control=ItenMC(), current_path=[], last_1childs=[], end_level=1, extra_control=None):
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
