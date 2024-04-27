#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from qiskit import QuantumCircuit


def gray_code(b):
    '''Gray code of b.
    Args:
        b: int:
            binary integer
    Returns:
        Gray code of b.
    '''
    return b ^ (b >> 1)


def gray_permutation(a):
    '''Permute the vector a from binary to Gray code order.

    Args:
        a: vector
            1D NumPy array of size 2**n
    Returns:
        vector:
            Gray code permutation of a
    '''
    b = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        b[i] = a[gray_code(i)]
    return b


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


def compute_control(i, n):
    '''Compute the control qubit index based on the index i and size n.'''
    if i == 2**n:
        return 1
    return n - int(np.log2(gray_code(i-1) ^ gray_code(i)))


def compressed_uniform_rotation(a, b, ry=True):
    '''Compute a compressed uniform rotation circuit based on the thresholded
    vector a.

    Args:
        a: vector:
            A thresholded vector a a of dimension 2**n
        ry: bool
            uniform ry rotation if true, else uniform rz rotation
    Returns:
        circuit
            A qiskit circuit representing the compressed uniform rotation.
    '''
    n = int(np.log2(a.shape[0]))
    b_qubit = int(not b is None)
    circ = QuantumCircuit(n+1 +b_qubit)
    if not b is None and b.shape[0] != a.shape[0]:
        return None

    i = 0
    target_qubit = n
    target_qubit_b = n+1
    while i < a.shape[0]:
        #Separate parity checks
        parity_check = 0

        # add the rotation gate
        if a[i] != 0:
            if ry:
                circ.ry(a[i], target_qubit)
            else:
                circ.rz(a[i], target_qubit)

        if not b is None and b[i] != 0:
            if ry:
                circ.ry(b[i], target_qubit_b)
            else:
                circ.rz(b[i], target_qubit_b)

        # loop over sequence of consecutive zeros
        if not b is None:
            while True:
                ctrl = compute_control(i+1, n)
                # toggle control bit
                parity_check = (parity_check ^ (1 << (ctrl-1)))
                i += 1
                if i >= a.shape[0] or a[i] != 0 or b[i] != 0:
                    break
        else:
             while True:
                ctrl = compute_control(i+1, n)
                # toggle control bit
                parity_check = (parity_check ^ (1 << (ctrl-1)))
                i += 1
                if i >= a.shape[0] or a[i] != 0:
                    break
    # add CNOT gates
        for j in range(0, n):
            if parity_check & (1 << (j)):
                circ.cnot(j, target_qubit)
                if not b is None:
                    circ.cnot(j, target_qubit_b)

    return circ
