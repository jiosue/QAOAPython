from projectq import MainEngine, ops
from projectq.backends import CircuitDrawer
import random
from numpy import pi
# define pi so that in string gates we can have pi as an angle.
# Because we use eval for string gates. For example, gate = "rz(pi/2, 1)".


### Using this is probably very inefficient. To do fast simulation using 
### projectq, don't use this string algorithm technique. Use only for 
### convenience.


# dict, convert lowercase to projectq case for gate.
# TODO: add support for all gates.
CASE_CONVERSION = {
    "cx": "CNOT", "cnot": "CNOT", "h": "H", "x": "X", "y": "Y", "z": "Z",
    "rx": "Rx", "ry": "Ry", "rz": "Rz", "qft": "QFT",
    "measure": "Measure", # might need to change this
}

# dict, map gate to number of parameters
PARAMETER_GATES = {"Rx": 1, "Ry": 1, "Rz": 1}


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
    if gate_name not in CASE_CONVERSION:
        raise ValueError(gate_name + " not supported")
    return CASE_CONVERSION[gate_name], gate_args


def get_num_qubits(algorithm):
    """
    Determine the max qubit value used in the algorithm.
    
    algorithm: iterable, each element must be a string gate, as in 
                     apply_string_gate above.
                     ie, algorithm = ["h(0)", "cx(0, 1)", "rx(pi/4, 1)",..]
                     
    returns: int, max qubit value in algorithm.
    """
    m = 0
    for gate in algorithm:
        gate_name, gate_args = get_gate_info(gate)
        if gate_name == "CNOT": m = max((m,) + gate_args)
        elif gate_name in PARAMETER_GATES:
            m = max((m,) + gate_args[PARAMETER_GATES[gate_name]:])
        else:
            m = max((m,) + gate_args)
    return m + 1
        
    
def apply_gate(gate, qureg):
    """
    gate: str, gate to apply.
    qureg: eng.allocate_qureg object to apply gate to.
    return: None.
    """
    gate_name, gate_args = get_gate_info(gate)
    if gate_name not in PARAMETER_GATES:
        args = (("qureg[%d], " * len(gate_args))[:-2]) % gate_args
        eval("ops." + gate_name + " | (" + args + ")")
    else:
        i = PARAMETER_GATES[gate_name]
        if i == 1: a = "(" + str(gate_args[0]) + ")"
        else: a = str(gate_args[:i])
        g = "ops." + gate_name + a
        args = (("qureg[%d], " * len(gate_args[i:]))[:-2]) % gate_args[i:]
        eval(g + " | (" + args + ")")


class Result(dict):
    """ Just a dictionary that automatically gives default values = 0.0 """
    def __getitem__(self, key):
        """ Return 0.0 if key not in result dictionary """
        return self.get(key, 0.0)

    
def run_single_algorithm(algorithm, num_qubits, num_samples=8000):
    """
    Create a quantum circuit, run the algorithm, return the resulting
    probability distribution.
    
    algorithm: algorithm (list of strings), 
               each string is a gate, ie "cx(0, 1)" or "rx(pi/2, 0)"
    num_qubits: int, number of qubits to run each algorithm on. Can be None,
                     in which case the algorithm will be run on the minimum
                     number of qubits required.
    num_samples: int, number of samples to take from the quantum computer in
                      in order to determine the probabilities for each state.
                  
    returns: dict (common.Result), keys are states, values are probabilities 
                                   found to be in that state.
    """
    
    eng = MainEngine()
    qureg = eng.allocate_qureg(num_qubits)
    measure = []
    for gate in algorithm:
        if "measure" not in gate.lower(): apply_gate(gate, qureg)
        else: 
            _, gate_args = get_gate_info(gate)
            measure.append(gate_args[0])
    if not measure: measure = list(range(num_qubits))
    eng.flush()
    
    to_measure = [qureg[i] for i in measure]
    
    #TODO: find supported projectq way to do this with out iterating O(2^n)!
    res = {}
    for x in range(2**len(measure)):
        state = ("{0:0%db}" % len(to_measure)).format(x)
        res[state] = eng.backend.get_probability(state, to_measure)
        
    # so projectq doesn't throw an error
    ops.All(ops.Measure) | qureg
    
    # To have a realistic simulator, we need to sample, not just return the
    # probabilities.
    samples = {}
    
    for _ in range(num_samples):
        r = random.random()
        for state, prob in res.items():
            r -= prob
            if r <= 0:
                if state in samples: samples[state] += 1
                else: samples[state] = 1
                break
    
    return Result(
        {state: counts / num_samples for state, counts in samples.items()}
    )
    

def run(algorithm, num_qubits=None, num_samples=8000):
    """
    Create a quantum circuit, run the algorithm, return the resulting
    probability distribution.
    
    algorithm: algorithm (list of strings) or list of algorithms, 
               each string is a gate, ie "cx(0, 1)" or "rx(pi/2, 0)"
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
        
    if multiple:
        return [
            run_single_algorithm(a, num_qubits, num_samples) for a in algorithm
        ]
    else:
        return run_single_algorithm(algorithm[0], num_qubits, num_samples)
    
    
def draw_algorithm(algorithm, num_qubits=None, name="test"):
    """
    Draw circuit of corresponding algorithm.
    
    algorithm: list of strs, gate sequence.
    num_qubits: int, number of qubits for algorithm. Can be None,
                     in which case the algorithm will be run on the minimum
                     number of qubits required.
    name: str, the resulting tex file will be written to name.tex.
    
    return: None.
    """
    #TODO: include drawing of measure gates.
    if num_qubits is None: num_qubits = get_num_qubits(algorithm)
    drawing_engine = CircuitDrawer()
    eng = MainEngine(drawing_engine)
    qureg = eng.allocate_qureg(num_qubits)
    for gate in algorithm:
        if "measure" not in gate.lower(): apply_gate(gate, qureg)
        else:
            _, gate_args = get_gate_info(gate)
            ops.Measure | qureg[gate_args[0]]
    eng.flush()
    with open("%s.tex" % name, "w") as f:
        f.write(drawing_engine.get_latex())
    
    
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
    alg1 = ["x(0)", "H(1)", "cx(0, 1)", "rz(pi/2, 2)"]
    print(run([alg0, alg1]))