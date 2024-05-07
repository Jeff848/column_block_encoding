# from .block_enc import create_be_0, create_be_1, create_be_2, create_be_3
# from .block_enc import column_block_encoding, simple_block_encoding, direct_block_encoding, topdown_block_encoding
# from ._util import QiskitPrepWrapper, QiskitMCWrapper, gen_random_snp_matrix_prob
# from .multi_control import ItenMC, HalfItenMC
# from .bin_prep import SNPWideBinPrepWrapper
# from ._angle_tree_util import top_down
# from ._angle_tree_util import state_decomposition
# from ._angle_tree_util import Amplitude
# from ._angle_tree_util import create_angles_tree
# from ._angle_tree_util import tree_visual_representation\
# from ._bdd_tree_util import convert_tree_to_bdd
from block_enc import create_be_0, create_be_1, create_be_2, create_be_3
from block_enc import column_block_encoding, simple_block_encoding, direct_block_encoding, topdown_block_encoding, bdd_based_block_encoding
from _util import QiskitPrepWrapper, QiskitMCWrapper, gen_random_snp_matrix_prob, SwapPrepWrapper
from multi_control import ItenMC, HalfItenMC
from bin_prep import SNPWideBinPrepWrapper
from _angle_tree_util import top_down
from _angle_tree_util import state_decomposition
from _angle_tree_util import Amplitude
from _angle_tree_util import create_angles_tree
from _angle_tree_util import tree_visual_representation
from _bdd_tree_util import convert_tree_to_bdd, common_case_centering, leavesBDD
from fable import fable
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit import QuantumCircuit
import numpy as np
from sympy import Matrix
from bitstring import BitArray


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

def test_be_4():
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

def generate_matrix_order(a, arr, target_level, bit_string, is_ctrl):
    if len(bit_string) == target_level:
        arr.append(a[BitArray(bin=bit_string).uint])
        return
    h = int(len(bit_string)/2)
    if is_ctrl:
        generate_matrix_order(a, arr, target_level, bit_string + '1', not is_ctrl)
        generate_matrix_order(a, arr, target_level, bit_string + '0', not is_ctrl)
    else:
        generate_matrix_order(a, arr, target_level, bit_string[:h] +'1'+ bit_string[h:], not is_ctrl)
        generate_matrix_order(a, arr, target_level, bit_string[:h] +'0'+ bit_string[h:], not is_ctrl)


def test_general(n, a, circ, alpha):
    simulator = AerSimulator(method="unitary")

    transpiled = transpile(circ, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be = result.get_unitary(transpiled)
    print(np.asarray(u_be)[:2**n, :2**n])
    print(transpiled.depth())
    print(transpiled.count_ops().get('cx', 0))
    np.testing.assert_array_almost_equal(
        a/alpha,  np.asarray(u_be)[:2**n, :2**n]
    )

def test_equiv(circ1, circ2):
    simulator = AerSimulator(method="unitary")

    transpiled = transpile(circ1, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be1 = result.get_unitary(transpiled)

    transpiled = transpile(circ2, basis_gates=['u', 'cx'], optimization_level=0)
    transpiled.save_state()
    result = simulator.run(transpiled).result()
    u_be2 = result.get_unitary(transpiled)

    np.testing.assert_array_almost_equal(
       np.asarray(u_be1),  np.asarray(u_be2)
    )

    
def run_all():
    n = 3
    #Test with SNP data matrix
    d=n
    a = gen_random_snp_matrix_prob(n)
    print(a)
    #General framework version
    circ, alpha = column_block_encoding(a, mc_helper_qubit=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, mc_helper_qubit=False)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, bin_state_prep=QiskitPrepWrapper)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, bin_state_prep=QiskitPrepWrapper, freq_center=True)
    test_general(n, a, circ, alpha)

def test_iten():
    n = 3
    a = gen_random_snp_matrix_prob(n)
    print(a)
    circ, alpha = column_block_encoding(a, multi_control=ItenMC(), mc_helper_qubit=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        bin_state_prep=QiskitPrepWrapper, freq_center=True)
    test_general(n, a, circ, alpha)

def test_optim():
    n = 3
    a = gen_random_snp_matrix_prob(n)
    print(a)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=QiskitPrepWrapper, optimal_control=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=QiskitPrepWrapper, optimal_control=True, freq_center=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=SNPWideBinPrepWrapper(QiskitPrepWrapper, return_circuit=True), wide_bin_state_prep=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=SNPWideBinPrepWrapper(QiskitPrepWrapper, return_circuit=True), wide_bin_state_prep=True, freq_center=True)
   
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=SNPWideBinPrepWrapper(QiskitPrepWrapper, return_circuit=True), wide_bin_state_prep=True, optimal_control=True)
    test_general(n, a, circ, alpha)
    circ, alpha = column_block_encoding(a, multi_control=HalfItenMC(), mc_helper_qubit=True, 
        prepare=QiskitPrepWrapper, bin_state_prep=SNPWideBinPrepWrapper(QiskitPrepWrapper, return_circuit=True), wide_bin_state_prep=True, optimal_control=True, freq_center=True)
    test_general(n, a, circ, alpha)
    # print(circ.draw())


def test_simple():
    n = 3
    a = gen_random_snp_matrix_prob(n)
    print(a)
    circ, alpha = simple_block_encoding(a)
    # print(circ.draw()) 
    test_general(n, a, circ, alpha)


def test_direct():
    n = 4
    a = gen_random_snp_matrix_prob(n)
    print(a)
    circ, alpha = direct_block_encoding(a)
    # print(circ.draw()) 
    test_general(n, a, circ, alpha)

def test_topdown():
    n = 4
    a = gen_random_snp_matrix_prob(n)
    print(a)

    circ, alpha = topdown_block_encoding(a)
    # print(circ.decompose(reps=2).draw()) 
    test_general(n, a, circ, alpha)

def test_bdd():
    n = 2
    # a = np.array([
    #     [2, 0, 1, 2],
    #     [0, 0, 2, 2],
    #     [1, 2, 2, 1],
    #     [2, 2, 2, 2]
    # ])
    a = np.array([[2, 2, 2, 2],
        [2, 2, 1, 2],
        [1, 2, 2, 1],
        [2, 2, 2, 1]])
    # a = gen_random_snp_matrix_prob(n)
    print(a)

    circ, alpha = bdd_based_block_encoding(a)
    # print(circ.decompose(reps=2).draw()) 
    test_general(n, a, circ, alpha)
        
def test_swap():
    n = 2
    a = gen_random_snp_matrix_prob(n)   
    print(a)
    circ, alpha = column_block_encoding(a, multi_control=QiskitMCWrapper, mc_helper_qubit=True, 
        prepare=SwapPrepWrapper(QiskitPrepWrapper, return_circuit=True), bin_state_prep=QiskitPrepWrapper, 
        optimal_control=True, freq_center=True, ctrl_initialize=True)
    # print(circ.draw())
    test_general(n, a, circ, alpha)




# run_all()
# test_iten()
# test_optim()
# test_simple()

# test_direct()
test_bdd()
# test_swap()

# data = [Amplitude(i, a) for i, a in enumerate([2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 2])]
# tree = state_decomposition(4, data)
# bdd = convert_tree_to_bdd(tree)
# centered_bdd = common_case_centering(bdd)
# leaves_bdd, _ = leavesBDD(centered_bdd)
# angle_tree, _ = create_angles_tree(centered_bdd, end_level=leaves_bdd[0].level, is_ctrl=True, subnorm=1)
# print(tree_visual_representation(angle_tree))