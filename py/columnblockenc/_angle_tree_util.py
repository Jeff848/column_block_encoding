
# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
https://arxiv.org/abs/2108.10182
"""

import math
import cmath
from dataclasses import dataclass
from typing import NamedTuple
from graphviz import Digraph
from enum import Enum
from typing import List
# from ._fable_util import compressed_uniform_rotation, sfwht, gray_permutation
from _fable_util import compressed_uniform_rotation, sfwht, gray_permutation
import numpy as np
from qiskit import QuantumCircuit
from bitstring import BitArray
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library.standard_gates import RYGate

# NodeType
class NodeType(Enum):
    TARGET = 1
    CTRL = 2
    VALUE = 3


class Amplitude(NamedTuple):
    """
    Named tuple for amplitudes
    """

    index: int
    amplitude: float

    def __str__(self):
        return f"{self.index}:{self.amplitude:.2f}"


@dataclass
class Node:
    """
    Binary tree node used in state_decomposition function
    """

    index: int
    level: int
    left: "Node"
    right: "Node"
    parents_left: List["Node"]
    parents_right: List["Node"]
    mag: float
    ntype: int

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.mag:.2f}_"
            f"{self.ntype}"
        )

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

def is_leaf(tree):
    """
    :param tree: a tree node
    :return: True if tree is a leaf
    """
    if tree.left is None and tree.right is None:
        return True

    return False

def is_last_ctrl(tree):
    """
    :param tree: an angle_tree node
    :return: True if tree is the last ctrl node before leaves
    """
    if tree.is_ctrl and is_leaf(tree.left) and is_leaf(tree.right):
        return True
    return False


@dataclass
class NodeAngleTree:
    """
    Binary tree node used in function create_angles_tree
    """

    index: int
    level: int
    angle_y: float
    angle_norm: float
    is_ctrl: bool
    left: "NodeAngleTree"
    right: "NodeAngleTree"

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.angle_y:.2f}_"
            f"{self.angle_norm:.2f}"
            f"{self.is_ctrl}"
        )
        return txt


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
            # #For ctrl nodes
            # if state_tree.ntype == NodeType.CTRL:
            #     #Get the subnormalization for the child of the current node
            #     subnorm_l = subnorm_l * state_tree.right.mag / state_tree.mag

            prev_mag = state_tree.right.mag
            #Go reverse from bottom to top
            for level in range(state_tree.right.level - 1, state_tree.level, -1):
                current_mag = state_tree.right.mag * np.sqrt(2)**(state_tree.right.level - level)
                if level % 2 == 0: #For ctrl nodes
                    subnorm_l = subnorm_l * prev_mag / current_mag
                prev_mag = current_mag
                
            if state_tree.ntype == NodeType.CTRL:
                subnorm_l = subnorm_l * prev_mag / state_tree.mag


        # if state_tree.ntype == NodeType.CTRL:
        #     if state_tree.mag != 0 and state_tree.right:
        #         for level in range(state_tree.level + 2, state_tree.right.level, 2):
        #             subnorm_l = subnorm_l / np.sqrt(2)
        #         subnorm_l = subnorm_l * state_tree.right.mag / state_tree.mag
        # elif state_tree.ntype == NodeType.TARGET:
        #     if state_tree.mag != 0 and state_tree.right:
        #         for level in range(state_tree.level + 1, state_tree.right.level, 2):
        #             subnorm_l = subnorm_l / np.sqrt(2)

        # print(subnorm_l)

        # if state_tree.ntype == NodeType.CTRL and state_tree.right and state_tree.mag != 0:
        #     subnorm_l = subnorm_l * state_tree.right.mag/state_tree.mag
            
                
                
            
            # for level in range(state_tree.level + 1, state_tree.right.level):
            #     if level % 2 == 0: #is ctrl
            #         subnorm_l = subnorm_l / np.sqrt(2)
            


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


        # if state_tree.left and state_tree.mag != 0 :
        #     #For ctrl nodes
        #     if state_tree.ntype == NodeType.CTRL:
        #         #Get the subnormalization for the child
        #         subnorm_r = subnorm_r * state_tree.left.mag / state_tree.mag

        #         #If there are hidden layers in between parent and child
        #         if (state_tree.left.level - state_tree.level) > 1:
        #             #Discount subnormalization from hidden target nodes (targets introduce sqrt(2) subnormalization)
        #             for level in range(state_tree.level+1, state_tree.left.level):
        #                 if level %2 != 0:
        #                     subnorm_l = subnorm_r * np.sqrt(2)
        #     #Otherwise if the current node is target and there are hidden ctrl nodes
        #     elif (state_tree.left.level - state_tree.level) > 1:
        #         subnorm_l = subnorm_l * 1 / (np.sqrt(2)**(state_tree.left.level - state_tree.level-1))

        #         #If there are hidden layers in between parent and child
        #         if (state_tree.left.level - state_tree.level) > 1:
        #             #Discount subnormalization from hidden target nodes (targets introduce sqrt(2) subnormalization)
        #             for level in range(state_tree.level, state_tree.left.level):
        #                 if level %2 != 0:
        #                     subnorm_l = subnorm_l * np.sqrt(2)

        # if state_tree.left and state_tree.mag != 0 :
        #     if state_tree.ntype == NodeType.CTRL:
        #         subnorm_r = subnorm_r * state_tree.left.mag / state_tree.mag
        #     elif (state_tree.left.level - state_tree.level) > 1:
        #         subnorm_r = subnorm_r * state_tree.left.mag * np.sqrt(2)**(state_tree.left.level - state_tree.level-1) / state_tree.mag
            
        #     if (state_tree.left.level - state_tree.level) > 1:
        #         print("HelloL")
                
        #         print(str(state_tree))
        #         print(str(state_tree.left))
        #         #Discount subnormalization from hidden target gates


        node.left, min_subnorm_r = create_angles_tree(state_tree.left, subnorm=subnorm_r, end_level=end_level)
        subnorm = min(min_subnorm_l, min_subnorm_r)

    return node, subnorm


def tree_visual_representation(tree, dot=None):
    """
    :param tree: A binary tree, with str(tree) defined
    """

    if dot is None:
        dot = Digraph()
        dot.node(str(tree))

    if tree.left:
        dot.node(str(tree.left))
        dot.edge(str(tree), str(tree.left))
        dot = tree_visual_representation(tree.left, dot=dot)

    if tree.right:
        dot.node(str(tree.right))
        dot.edge(str(tree), str(tree.right), style='dotted')
        dot = tree_visual_representation(tree.right, dot=dot)

    # for node in tree.parents_left:
    #     dot.edge(str(tree), str(node))

    # for node in tree.parents_right:
    #     dot.edge(str(tree), str(node))

    return dot

def sfwht(a):
    '''Scaled Fast Walsh-Hadamard transform of input vector a.

    Args:
        a: vector
            1D NumPy array of size 2**n.
    Returns:
        vector:
            Scaled Walsh-Hadamard transform of a.
    '''
    n = int(np.log2(a.shape[0]))
    for h in range(n):
        for i in range(0, a.shape[0], 2**(h+1)):
            for j in range(i, i+2**h):
                x = a[j]
                y = a[j + 2**h]
                a[j] = (x + y) / 2
                a[j + 2**h] = (x - y) / 2
    return a

def children(nodes):
    """
    Search and list all the nodes childs.
    :param nodes: a list with tree nodes
    :return: a list with nodes childs
    """
    child = []
    for node in nodes:
        if node.left:
            child.append(node.left)
        if node.right:
            child.append(node.right)

    return child

def get_min_subnorm(tree):
    nodes = [tree]
    while not is_leaf(nodes[0]):
        new_nodes = []
        for node in nodes:
            if node.left:
                new_nodes.append(node.left)
            if node.right:
                new_nodes.append(node.right)
        nodes = new_nodes
    

def generate_matrix_order(a, arr, target_level, num_set, bit_string, is_ctrl, override=True):
    # print(bit_string)
    if num_set == target_level:
        arr.append(a[BitArray(bin=''.join(bit_string)).uint])
        return
    i = int(np.floor(num_set/2))
    divider = int(np.floor(target_level/2))
    if is_ctrl:
        bit_string[i] = '1'
        generate_matrix_order(a, arr, target_level, num_set + 1, bit_string, not is_ctrl, override)
        bit_string[i] = '0'
        generate_matrix_order(a, arr, target_level, num_set + 1, bit_string, not is_ctrl, override)
    else:
        bit_string[i+divider] = '1'
        generate_matrix_order(a, arr, target_level, num_set + 1, bit_string, not is_ctrl, override)
        bit_string[i+divider] = '0'
        generate_matrix_order(a, arr, target_level, num_set + 1, bit_string, not is_ctrl, override)

def top_down(angle_tree, circuit, start_level, rotate_qubit, qubit_order, control_nodes=None, target_nodes=None):
    """top down state preparation"""
    if angle_tree:
        if angle_tree.level < start_level:
            top_down(angle_tree.left, circuit, start_level, rotate_qubit, qubit_order)
            top_down(angle_tree.right, circuit, start_level, rotate_qubit, qubit_order)
        else:
            if target_nodes is None:
                control_nodes = []  # initialize the controls
                target_nodes = [angle_tree]  # start by the subtree root
            else:
                target_nodes = children(
                    target_nodes
                )  # all the nodes in the current level

            angles_y = np.array([node.angle_y for node in target_nodes])
            angles_norm = 2 * np.arccos([node.angle_norm for node in target_nodes])
            target_qubit = qubit_order[target_nodes[0].level]
            target_is_ctrl = target_nodes[0].is_ctrl            
            control_qubits = qubit_order[[node.level for node in control_nodes]]
            
            if not target_is_ctrl:
                order = list(range(len(angles_y)-1, -1, -1))
                if is_leaf(target_nodes[0]):
                    #Do additional ucr
                    # ucry = uniform_rotation(angles_norm[order], None, ry=True, use_b=True)
                    # circuit.append(ucry, list(control_qubits) + [rotate_qubit])
                    ucry = uniform_rotation(angles_y[order], angles_norm[order], ry=True, use_b=True)
                    circuit.append(ucry, list(control_qubits) + [target_qubit]#)
                        + [rotate_qubit])
                    
                else:
                    ucry = uniform_rotation(angles_y[order], None, ry=True)
                    circuit.append(ucry, list(control_qubits) + [target_qubit])
                

            control_nodes.append(angle_tree)  # add current node to the controls list

            # walk to the first node of the next level.
            top_down(
                angle_tree.left,
                circuit,
                start_level,
                rotate_qubit,
                qubit_order,
                control_nodes=control_nodes,
                target_nodes=target_nodes,
            )

def uniform_rotation(a_angles, b_angles, ry=True, use_b=False):
    if not ry:
        return None #Not implemented

    # return None
    if use_b:
        b_m = gray_permutation(sfwht(np.array(b_angles)))
    else:
        b_m = None
    a_m = gray_permutation(sfwht(np.array(a_angles)))
    return compressed_uniform_rotation(a_m, b_m, ry=ry)