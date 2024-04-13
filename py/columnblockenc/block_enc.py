from qiskit import QuantumCircuit
import numpy as np
from _util import get_padded_matrix, QiskitPrepWrapper, QiskitMCWrapper



#Unified function
def column_block_encoding(a, prepare=None, multi_control=None, 
    mc_helper_qubit=False, bin_state_prep=None, freq_center=False, 
    optimal_control=False, wide_bin_state_prep=False):

    if prepare == None:
        prepare = QiskitPrepWrapper
    if multi_control == None:
        multi_control = QiskitMCWrapper
    

    a, n, logn = get_padded_matrix(a)

    helper = int(mc_helper_qubit)
    Oprep_list = []
    Cprep_list = []
    

    #Number of ancillas such that prep circuit # qubits = logn + prep_ancillas
    #lcu prep circuit # qubits = logd + lcu_prep_ancillas
    prep_ancillas = 0
    lcu_prep_ancillas = 0
    logd=0
    
    if bin_state_prep is None:
        logd = 0
        amps = []
        s = 0
        #Solely prepare state of column directly
        for j in range(n):
            state = a[:, j]
            amp = np.sqrt(np.sum(state**2)) # Shouldn't be any empty columns
            s = max(np.count_nonzero(state), s)
            amps.append(amp)

        max_amp = np.max(amps)
        prep_ancillas = prepare.get_ancillas(s, n)

        for j in range(n): #Create circuit to prepare column with prepare call
            state = a[:, j]
            prep = QuantumCircuit(prep_ancillas+logn+1)
            prep = prepare.initialize(prep, state/amps[j], list(range(prep_ancillas+logn)))
            target_qubit = prep_ancillas+logn
            
            rotate_angle = 2 * np.arccos(amps[j] / max_amp)
            prep.ry(rotate_angle, target_qubit)
            
            Cprep_list.append(prep)

    else:
        preps = []
        amps = []
        vals = []
        freqs = []
        most_freq_inds = []
        d = 0
        s = n

        #Pre calculate state and amp
        for j in range(n):
            state = a[:, j]
            values, inv, frequencies = np.unique(state, return_inverse=True, return_counts=True)

            #Id list of elements to val in values list
            ids = np.arange(len(values))

            #Get binary state preparation lists
            bin_state_preps = (inv == ids.reshape((-1, 1))).astype(int)
            
            if freq_center:
                #Get most frequent element
                most_freq_ind = np.argmax(frequencies)
                most_freq_ele = values[most_freq_ind]

                #Make the most frequent element the baseline
                frequencies[most_freq_ind] = n 
                most_freq_inds.append(most_freq_ind)
                values = values - most_freq_ele #Values are now deltas from baseline
                values[most_freq_ind] = most_freq_ele

            #By Default 0 is the center
            #Drop 0 value if it exists
            elif 0 in values: #0 value shouldn't exist for frequency based
                #Drop 0 prep
                ind = values.tolist().index(0)
                values = np.delete(values,ind)
                bin_state_preps = np.delete(bin_state_preps, ind, axis=0)
                frequencies = np.delete(frequencies, ind)
                
            preps.append(bin_state_preps)
            amps.append(np.sqrt(np.sum(frequencies * values**2)))
            vals.append(values * np.sqrt(frequencies))
            freqs.append(frequencies)

            d = max(d, len(values)) #Get max number of unique data elements

        logd = max(int(np.ceil(np.log2(d))), 1)
        max_amp = np.max(amps)

        if optimal_control:
            #Add padding if not enough
            if 2**logd == d:
                logd = logd+1

        lcu_prep_ancillas = bin_state_prep.get_ancillas(d, 2**logd) 
        prep_ancillas = prepare.get_ancillas(s, n)

        LCUprep_list = []

        rotate_qubit = logd+lcu_prep_ancillas+prep_ancillas+logn

        for j in range(n): #Prepare state of columns

            #Pad array if not enough
            padding = 2**logd - len(vals[j])
            if len(vals[j]) != 2**logd:
                temp_vals = np.pad(vals[j], (padding, 0)) 
            else:
                temp_vals = vals[j]

            #Prep LCU
            lcu_prep = QuantumCircuit(logd+lcu_prep_ancillas+1)
            lcu_prep = bin_state_prep.initialize(lcu_prep, temp_vals / amps[j], list(range(lcu_prep_ancillas + logd)))
            
            rotate_angle = 2 * np.arccos(amps[j] / max_amp) #Rotate to normalize
            lcu_prep.ry(rotate_angle, logd+lcu_prep_ancillas)
            
            prep = QuantumCircuit(logn+logd+lcu_prep_ancillas+prep_ancillas+1) 

            if optimal_control:
                LCUprep_list.append(lcu_prep)
            else:
                prep = prep.compose(lcu_prep, list(range(lcu_prep_ancillas + logd)) + [rotate_qubit])

            for i in range(len(vals[j])): 

                if freqs[j][i] == 0:
                    continue
                
                    
                bin_prep = QuantumCircuit(logn+prep_ancillas)
                
                bit_mask=1
                if not wide_bin_state_prep:
                    for r in range(lcu_prep_ancillas, logd+lcu_prep_ancillas):
                        if (i + padding) & bit_mask == 0:
                            prep.x(r)
                        bit_mask = bit_mask << 1

                if freq_center and i == most_freq_inds[j]: #Means that this is the most frequent element
                    for r in range(prep_ancillas, prep_ancillas+logn):
                        bin_prep.h(r)
                else:
                    #Get binary prep of specified values
                    
                    bin_prep = prepare.initialize(bin_prep, preps[j][i] / np.sqrt(freqs[j][i]), 
                        list(range(prep_ancillas+logn)))

                    
                
                if not wide_bin_state_prep:
                    prep = multi_control.control(prep, bin_prep, list(range(lcu_prep_ancillas, logd+lcu_prep_ancillas)), 
                        list(range(logd+lcu_prep_ancillas, logd+lcu_prep_ancillas+prep_ancillas+logn)), False)
                else:
                    if i > logd + lcu_prep_ancillas:
                        continue
                    prep = prep.compose(bin_prep.control(1), [i] + 
                        list(range(logd+lcu_prep_ancillas, logd+lcu_prep_ancillas+prep_ancillas+logn)))
                
                bit_mask=1
                if not wide_bin_state_prep:
                    for r in range(lcu_prep_ancillas, logd+lcu_prep_ancillas):
                        if (i + padding) & bit_mask == 0:
                            prep.x(r)
                        bit_mask = bit_mask << 1
            if not optimal_control:
                for i in range(lcu_prep_ancillas, logd + lcu_prep_ancillas):
                    prep.h(i)

            Cprep_list.append(prep)

    #Construct multicontrols of prep circuits
    for j in range(n):
        ctrl = QuantumCircuit(logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1
        if mc_helper_qubit:
            flag_qubit = logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper
        else:
            flag_qubit = None

        if optimal_control:
            multi_control_prep = QuantumCircuit(logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)
            multi_control_prep = multi_control.half_control(multi_control_prep, LCUprep_list[j], list(range(logn)), list(range(logn, logn+logd+lcu_prep_ancillas)) + [logn + rotate_qubit], flag_qubit)
            multi_control_prep = multi_control_prep.compose(Cprep_list[j], list(range(logn, logn+logn+logd+lcu_prep_ancillas+prep_ancillas+1))) 
            for i in range(logn+lcu_prep_ancillas,logn+logd+lcu_prep_ancillas):
                multi_control_prep.ch(flag_qubit, i)
            #Does multicontrol if flag qubit exists
            multi_control_prep = multi_control.mcx(multi_control_prep, list(range(logn)), flag_qubit, list(range(logn, logn+logd+lcu_prep_ancillas+prep_ancillas+logn+1)))
            ctrl = ctrl.compose(multi_control_prep, list(range(logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)))
        else:  
            multi_control_prep = QuantumCircuit(logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)
            multi_control_prep = multi_control.control(multi_control_prep, Cprep_list[j], list(range(logn)), list(range(logn, logn+logd+lcu_prep_ancillas+prep_ancillas+logn+1)), flag_qubit)
            ctrl = ctrl.compose(multi_control_prep, list(range(logn+logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)))

        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1

        Oprep_list.append(ctrl)

    #Assemble Block encoding circuit
    circ = QuantumCircuit(2*logn + logd + lcu_prep_ancillas + prep_ancillas + helper + 1)
    for i in range(n):
        circ = circ.compose(Oprep_list[i], list(range(2*logn+logd+lcu_prep_ancillas+prep_ancillas+helper+1)))

    for i in range(logn):
        circ.swap(i, lcu_prep_ancillas + logd + prep_ancillas + logn + i)

    for i in range(lcu_prep_ancillas + logd + prep_ancillas + logn, lcu_prep_ancillas + logd + prep_ancillas + 2 * logn):
        circ.h(i)

    alpha = max_amp * np.power(np.sqrt(2), logn+logd)
    return circ, alpha
    

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

    max_amp = np.max(amps)

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

    max_amp = np.max(amps)

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

#Linear Combination of Unitaries for low entropy data
#For data, int range (0, d)
def create_be_2(a, prep):

    a, n, logn = get_padded_matrix(a)
    
    Oprep_list = []
    preps = []
    amps = []
    vals = []
    freqs = []
    max_amp = 0
    d = 0

    #Pre calculate state and amp
    for j in range(n):
        state = a[:, j]
        values, inv, frequencies = np.unique(state, return_inverse=True, return_counts=True)

        #Id list of elements to val in values list
        ids = np.arange(len(values))

        #Get binary state preparation lists
        bin_state_preps = (inv == ids.reshape((-1, 1))).astype(int)

        if values[0] == 0:
            #Drop 0 prep
            values = values[1:]
            bin_state_preps = bin_state_preps[1:]
            frequencies = frequencies[1:]
            
        preps.append(bin_state_preps)
        amps.append(np.sqrt(np.sum(frequencies * values**2)))
        vals.append(values * np.sqrt(frequencies))
        freqs.append(frequencies)

        d = max(d, len(values)) #Get max number of unique data elements

    logd = max(int(np.ceil(np.log2(d))), 1)
    max_amp = np.max(amps)

    for j in range(n): #Prepare state of columns
        #Prep LCU
        lcu_prep = QuantumCircuit(logd)
        #Pad vals if not enough
        if len(vals[j]) != 2**logd:
            temp_vals = np.pad(vals[j], (0, 2**logd-len(vals[j])))
        else:
            temp_vals = vals[j]

        lcu_prep.initialize(temp_vals / amps[j], list(range(logd)))
        lcu_prep = lcu_prep.decompose(reps=5)
        lcu_prep.data = [ins for ins in lcu_prep.data if ins.operation.name != "reset"]



        prep = QuantumCircuit(logn+logd+1)
        prep = prep.compose(lcu_prep, list(range(logd)))

        for i in range(len(vals[j])): 
            bin_prep = QuantumCircuit(logn)
            bin_prep.initialize(preps[j][i] / np.sqrt(freqs[j][i]), list(range(logn)))
            bin_prep = bin_prep.decompose(reps=5)
            bin_prep.data = [ins for ins in bin_prep.data if ins.operation.name != "reset"]
            bit_mask = 1
            for r in range(logd):
                if i & bit_mask == 0:
                    prep.x(r)
                bit_mask = bit_mask << 1
            prep = prep.compose(bin_prep.control(logd), list(range(logn+logd)))
            bit_mask = 1
            for r in range(logd):
                if i & bit_mask == 0:
                    prep.x(r)
                bit_mask = bit_mask << 1

        for i in range(logd):
            prep.h(i)

        rotate_angle = 2 * np.arccos(amps[j] / max_amp) #Rotate to normalize
        prep.ry(rotate_angle, logn+logd)

        ctrl = QuantumCircuit(logn+logn+logd+1)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1

        ctrl = ctrl.compose(prep.control(logn), list(range(2*logn+logd+1)))
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1
        Oprep_list.append(ctrl)

    #Block encoding circuit
    circ = QuantumCircuit(2*logn + logd + 1)
    for i in range(n):
        circ = circ.compose(Oprep_list[i], list(range(2*logn+logd+1)))

    for i in range(logn):
        circ.swap(i, logn + logd + i)

    for i in range(logn + logd, logn*2 + logd):
        circ.h(i)

    alpha = max_amp * np.power(np.sqrt(2), logn+logd)#Extra two hadamards
    return circ, alpha

#Linear Combination of Unitaries for low entropy data, using most common as baseline
#For data, int range (0, d)
def create_be_3(a, prep):

    a, n, logn = get_padded_matrix(a)
    Oprep_list = []
    preps = []
    amps = []
    vals = []
    freqs = []
    most_freq_inds = []
    max_amp = 0
    d = 0

    #Pre calculate state and amp
    for j in range(n):
        state = a[:, j]
        values, inv, frequencies = np.unique(state, return_inverse=True, return_counts=True)

        #Id list of elements to val in values list
        ids = np.arange(len(values))

        #Get binary state preparation lists
        bin_state_preps = (inv == ids.reshape((-1, 1))).astype(int)
        
        #Get most frequent element
        most_freq_ind = np.argmax(frequencies)
        most_freq_ele = values[most_freq_ind]

        #Make the most frequent element the baseline
        frequencies[most_freq_ind] = n 
        most_freq_inds.append(most_freq_ind)
        values = values - most_freq_ele #Values are now deltas from baseline
        values[most_freq_ind] = most_freq_ele

        preps.append(bin_state_preps)
        amps.append(np.sqrt(np.sum(frequencies * values**2)))
        vals.append(values * np.sqrt(frequencies))
        freqs.append(frequencies)

        d = max(d, len(values)) #Get max number of unique data elements

    logd = max(int(np.ceil(np.log2(d))), 1)
    max_amp = np.max(amps)

    for j in range(n): #Prepare state of columns
        #Prep LCU
        lcu_prep = QuantumCircuit(logd)
        #Pad vals if not enough
        if len(vals[j]) != 2**logd:
            temp_vals = np.pad(vals[j], (0, 2**logd - len(vals[j])))
        else:
            temp_vals = vals[j]

        lcu_prep.initialize(temp_vals / amps[j], list(range(logd)))
        lcu_prep = lcu_prep.decompose(reps=5)
        lcu_prep.data = [ins for ins in lcu_prep.data if ins.operation.name != "reset"]

        prep = QuantumCircuit(logn+logd+1)
        prep = prep.compose(lcu_prep, list(range(logd)))

        for i in range(len(vals[j])): 
            bin_prep = QuantumCircuit(logn)
            
            if i == most_freq_inds[j]: #Means that this is the most frequent element
                bit_mask=1
                for r in range(logd):
                    if i & bit_mask == 0:
                        prep.x(r)
                    bit_mask = bit_mask << 1

                
                for r in range(logn):
                    bin_prep.h(r)
                prep = prep.compose(bin_prep.control(logd), list(range(logn+logd)))
                
                bit_mask=1
                for r in range(logd):
                    if i & bit_mask == 0:
                        prep.x(r)
                    bit_mask = bit_mask << 1
            else:

                if freqs[j][i] == 0:
                    continue

                bin_prep.initialize(preps[j][i] / np.sqrt(freqs[j][i]), list(range(logn)))
                bin_prep = bin_prep.decompose(reps=5)
                bin_prep.data = [ins for ins in bin_prep.data if ins.operation.name != "reset"]
                bit_mask = 1
                for r in range(logd):
                    if i & bit_mask == 0:
                        prep.x(r)
                    bit_mask = bit_mask << 1
                prep = prep.compose(bin_prep.control(logd), list(range(logn+logd)))
                bit_mask = 1
                for r in range(logd):
                    if i & bit_mask == 0:
                        prep.x(r)
                    bit_mask = bit_mask << 1

        for i in range(logd):
            prep.h(i)

        rotate_angle = 2 * np.arccos(amps[j] / max_amp) #Rotate to normalize
        prep.ry(rotate_angle, logn+logd)

        ctrl = QuantumCircuit(logn+logn+logd+1)
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1

        ctrl = ctrl.compose(prep.control(logn), list(range(2*logn+logd+1)))
        bit_mask = 1
        for i in range(logn):
            if j & bit_mask == 0:
                ctrl.x(i)
            bit_mask = bit_mask << 1
        Oprep_list.append(ctrl)

    #Block encoding circuit
    circ = QuantumCircuit(2*logn + logd + 1)
    for i in range(n):
        circ = circ.compose(Oprep_list[i], list(range(2*logn+logd+1)))

    for i in range(logn):
        circ.swap(i, logn + logd + i)

    for i in range(logn + logd, logn*2 + logd):
        circ.h(i)

    alpha = max_amp * np.power(np.sqrt(2), logn+logd)#Extra two hadamards
    return circ, alpha