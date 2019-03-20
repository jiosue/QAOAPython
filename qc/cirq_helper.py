import cirq
from numpy import pi, eye
# define pi so that in string gates we can have pi as an angle.
# Because we use eval for string gates. For example, gate = "rz(pi/2, 1)".

#TODO: add all.
PARAMETER_FREE_GATES = {"h", "cnot", "x", "y", "z", "cz", "ccx", "ccz"}


def get_gate_info(gate):
    """
    gate: str, string gate. ie H(0), or "cx(1, 0)".
    returns: tuple, (gate_name (str), gate_args (tuple)).
    """
    g = gate.strip().lower()
    i = g.index("(")
    gate_name, gate_args = g[:i], eval(g[i:])
    try: len(gate_args)
    except TypeError: gate_args = gate_args,
    if gate_name == "cx": gate_name = "cnot"
    return gate_name, gate_args
    

def get_num_qubits(algorithm):
    """
    Determine the max qubit value used in the algorithm.
    
    algorithm: iterable, each element must be a string gate, as in 
                     apply_string_gate above.
                     ie, algorithm = ["h(0)", "cx(0, 1)", "rx(pi/4, 1)",..]
                     
    returns: int, max qubit value in algorithm.
    """
    #TODO: support more of the cirq gate set
    m = 0
    for gate in algorithm:
        gate_name, gate_args = get_gate_info(gate)
        
        if gate_name in PARAMETER_FREE_GATES:
            m = max((m,) + gate_args)
        elif gate_name in ("rx", "ry", "rz"):
            _, qubit = gate_args
            m = max((m, qubit))
        elif gate_name == "measure":
            qubit = gate_args[0]
            m = max((m, qubit))
        else:
            raise NotImplementedError("%s gate not supported" % gate_name)
    return m + 1
    
    
def make_gate(gate, qubits):
    """
    Convert str gate to cirq gate.
    
    gate: str, gate of the form "cx(0, 1)" for example.
    qubits: list of ints, qubits that algorithm is running on.
    
    returns: cirq.Gate object applied on the correct qubits to be appended 
             to a cirq.Circuit object.
    """
    #TODO: support more of the cirq gate set
    gate_name, gate_args = get_gate_info(gate)
    
    if gate_name in PARAMETER_FREE_GATES:
        args = "(%s)" % ", ".join("qubits[%d]" for _ in range(len(gate_args)))
        return eval("cirq." + gate_name.upper() + args % gate_args)
    elif gate_name in ("rx", "ry", "rz"):
        angle, qubit = gate_args
        r = eval(
            "cirq.%sPowGate(exponent=%g)" % (gate_name[1].upper(), angle/pi)
        )
        return r(qubits[qubit])
    elif gate_name == "measure":
        return cirq.measure(qubits[gate_args[0]], key=str(gate_args[1]))
    else:
        raise NotImplementedError("%s gate not supported" % gate_name)

    
def make_circuit(algorithm, num_qubits):
    """
    Make cirq.Circuit object from algorithm. If measure is not in the algorithm
    then by default all qubits will be measured.
    
    algorithm: list of strs, gates to apply.
    num_qubits: int, number of qubits to run the algorithm on.
    returns: cirq.Circuit object.
    """
    qs = [cirq.GridQubit(i, 0) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    measure = False
    for gate in algorithm:
        circuit.append(make_gate(gate, qs), 
                       strategy=cirq.InsertStrategy.EARLIEST)
        if "measure" in gate.lower(): measure = True
        
    if not measure:
        for i in range(len(qs)):
            circuit.append(make_gate("measure(%d, %d)" % (i, i), qs), 
                           strategy=cirq.InsertStrategy.EARLIEST)
    return circuit
        
    
class Result(dict):
    """ Just a dictionary that automatically gives default values = 0.0 """
    def __getitem__(self, key):
        """ Return 0.0 if key not in result dictionary """
        return self.get(key, 0.0)
    

def cirq_output_to_Result(cirq_output):
    """
    Take the output of cirq.simulator.run and convert it to Result dictionary.
    For example, if the cirq_output is
        res.measurements = {"0": [True, False], "1": [True, True]}
    This means that the 0th qubit was 1 then 0, and the 1st qubit was 1 then 1.
    So this function should return
        Result({"11": 0.5, "01": 0.5})
    """
    qubits = sorted(int(x) for x in cirq_output.measurements.keys())
    counts = {}
    for i in range(cirq_output.repetitions):
        state = ""
        for j in qubits: 
            state += "1" if cirq_output.measurements[str(j)][i] else "0"
        if state in counts: counts[state] += 1
        else: counts[state] = 1
    for state in counts: counts[state] /= cirq_output.repetitions
    return Result(counts)

    
def run(algorithm, num_qubits=None, num_samples=8000):
    """
    Create a quantum circuit, run the algorithm, return the resulting
    probability distribution.
    
    algorithm: algorithm (list of strings) or list of algorithms, 
               each string is a gate, ie "cx(0, 1)" or rz(pi/2, 0)
    num_qubits: int, number of qubits to run each algorithm on. Can be None,
                     in which case the algorithm will be run on the minimum
                     number of qubits required.
    num_samples: int, number of samples to take from the quantum computer in
                      in order to determine the probabilities for each state.
                  
    returns: dict (common.Result), keys are states, values are probabilities 
                                   found to be in that state.
    """
    multiple = bool(algorithm and isinstance(algorithm[0], list))
    if not multiple: algorithm = [algorithm]
    
    if num_qubits is None: 
        num_qubits = max(get_num_qubits(a) for a in algorithm)
        
    circuits = [make_circuit(a, num_qubits) for a in algorithm]
    
    sim = cirq.Simulator()
    results = [sim.run(c, repetitions=num_samples) for c in circuits]
    
    if multiple: return [cirq_output_to_Result(r) for r in results]
    else: return cirq_output_to_Result(results[0])
    
    
def algorithm_unitary(algorithm, num_qubits=None):
    """
    Find the unitary corresponding to the algorithm.
    
    algorithm: list of strings, each string is a gate in GATE_ARGUMENTS.keys()
                                with whatever arguments required to define the 
                                gate.
    num_qubits: int, number of qubits to run the algorithm on.
                  
    returns: np.array, unitary matrix corresponding to the algorithm.
    """
    if num_qubits is None: num_qubits = get_num_qubits(algorithm)
    if not algorithm: return eye(2**num_qubits)
    circuit = make_circuit(algorithm, num_qubits)
    return circuit.to_unitary_matrix()
    
    
if __name__ == "__main__":
    ## Examples
    
    # `run` returns a dictionary mapping states to probabilities, ie
    # run(["h(0)", "cx(0, 1)"]) should return {"00":0.5, "11": 0.5}.
    
    # if no "measure" is included in alg, then by default everything will
    # be measured.
    alg = ["H(0)", "CX(0, 1)"]
    print(run(alg, 3, num_samples=10000))
    
    # since a measure is included, only that register will be measured.
    alg = ["H(0)", "CX(0, 1)", "measure(0, 0)"]
#    print(run(alg, 3, num_samples=1000, backend="ibmqx4"))
    
    # run multiple circuits at once
    alg0 = ["h(0)", "cx(0, 1)", "measure(1, 0)"]
    alg1 = ["x(0)", "H(1)", "ccx(0, 1, 2)", "rz(pi/2, 2)"]
    print(run([alg0, alg1]))
    
    # convert alg to its unitary respresentation.
    alg = ["h(0)", "cx(0, 1)", "rx(0, 1)", "rz(pi, 0)"]
    print(algorithm_unitary(alg, 2))