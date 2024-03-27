from block_enc import create_be_0, create_be_1, create_be_2, create_be_3
from fable import fable
from qiskit_aer import AerSimulator
from qiskit import transpile
import numpy as np


def test_be_0():
    n = 3
    a = np.random.randn(2**n, 2**n)

    simulator = AerSimulator(method="unitary")

    circ, alpha = create_be_0(a)
    transpiled = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be = result.get_unitary(transpiled)

    np.testing.assert_array_almost_equal(
        a/alpha,  np.asarray(u_be)[:2**n, :2**n]
    )

def test_be_1():
    n = 3
    a = np.random.randn(2**n, 2**n)

    simulator = AerSimulator(method="unitary")

    circ, alpha = create_be_1(a)
    transpiled = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be = result.get_unitary(transpiled)

    np.testing.assert_array_almost_equal(
        a/alpha,  np.asarray(u_be)[:2**n, :2**n]
    )

def test_be_2():
    n = 3
    #Test with all positive array of random ints
    # d = np.random.randint(n, 1000) #Random number of unique ints
    d=n
    a = np.random.randint(0, d, size=(2**n, 2**n)) 

    simulator = AerSimulator(method="unitary")

    circ, alpha = create_be_2(a, None)
    transpiled = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be = result.get_unitary(transpiled)

    np.testing.assert_array_almost_equal(
        a/alpha,  np.asarray(u_be)[:2**n, :2**n]
    )

def test_be_3():
    n = 3
    #Test with all positive array of random ints
    # d = np.random.randint(n, 1000) #Random number of unique ints
    d=n
    a = np.random.randint(0, d, size=(2**n, 2**n)) 

    simulator = AerSimulator(method="unitary")

    circ, alpha = create_be_3(a, None)
    transpiled = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be = result.get_unitary(transpiled)

    np.testing.assert_array_almost_equal(
        a/alpha,  np.asarray(u_be)[:2**n, :2**n]
    )

test_be_3()