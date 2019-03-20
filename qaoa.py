from scipy.optimize import minimize
import qc


BACKENDS = {
    "cirq": qc.cirq_helper,
    "qiskit": qc.qiskit_helper,
    "projectq": qc.projectq_helper
}


def exponentiate_zs(qubits, angle):
    """
    Let q0 = qubits[0], ..., qn = qubits[n].
    Then this function finds the gate sequence (up to a phase) for
        exp(-i angle z(q0)...z(qn) / 2)

    qubits: list of ints, qubit that the z's act on.
    angle: float, angle of rotation.
    
    returns list of strs, gate sequence for exp(-i angle z(q0)...z(qn) / 2).
                
    Turns out that
        exp(-i angle z(i)z(j)...z(k)z(m) / 2) = 
        a cx(i, j)cx(j, .)...cx(k, m)rz(angle, m)cx(k, m)...cx(j, .)cx(i, j)
    (see pyquil.paulis docs.rigetti.com/en/latest/_modules/pyquil/paulis.html)
    (for intuitive explanation see 
        https://quantumcomputing.stackexchange.com/questions/5155/
        gate-sequence-for-exponential-of-product-of-pauli-z-operators)    
    """
    # Get rid of even duplicates. For example, 
    #   z(0) z(1) z(0) z(1) z(1) = z(1)
    qs = list(set(x for x in qubits if qubits.count(x) % 2))
    if not qs: return []
    
    # TODO: Since Zs commute, we can optimize the cnot sequence to the
    # connectivity of the hardware.
    cnots = ["cx(%d, %d)" % (qs[i], qs[i+1]) for i in range(len(qs)-1)]
    
    return cnots + ["rz(%g, %d)" % (angle, qs[-1])] + list(reversed(cnots))


class QAOA:
    
    def __init__(self, C, backend="cirq", num_samples=8000):
        """
        C: list of tuples, for example 
                C = 4 + 2 z(0) - 3 z(0) z(1) + 4 z(0) z(2) z(3) 
           would be
                C = [(4, []), (2, [0]), (-3, [0, 1]), (4, [0, 2, 3])]
        backend: str, what backend to run on. Must be in BACKENDS.keys()
        num_samples: int, number of samples of the qc to make.
        """
        if backend not in BACKENDS:
            raise ValueError("Backend must be in " + str(BACKENDS.keys()))
        run = BACKENDS[backend].run
        
        # cost function is C = sum w_alpha[i] c_alpha[i]
        try:
            self.w_alphas, self.c_alphas = [x[0] for x in C], [x[1] for x in C]
        except TypeError: 
            raise TypeError("`C` input incorrectly formatted; see docstring.")
            
        try: self.num_qubits = max(x for y in self.c_alphas for x in y) + 1
        except ValueError: raise ValueError("Cost function is a constant.")
        
        self.verbose, self.iters = True, 0
        
        self.initial_state = ["h(%d)" % i for i in range(self.num_qubits)]
        self.run_qc = lambda alg: run(alg, num_qubits=self.num_qubits, 
                                           num_samples=num_samples)
    def UB(self, beta):
        """
        beta: float, angle.
        return: list of strs, gate sequence for U(B, beta).
        """
        return ["rx(%g, %d)" % (beta, i) for i in range(self.num_qubits)]
    
    def UC(self, gamma):
        """
        gamma: float, angle.
        return: list of strs, gate sequence for U(C, gamma).
        """
        alg = []
        for m in range(len(self.c_alphas)):
            alg.extend(
                exponentiate_zs(self.c_alphas[m], self.w_alphas[m]*gamma)
            )
        return alg
        
    def prepare_ansatz(self, betas, gammas):
        """
        betas: list of floats, angles for U(B, beta).
        gammas: list of angles, angles for U(C, gamma).
        
        return: list of strs, gate sequence.
        """
        alg = self.initial_state.copy()
        for i in range(len(betas)):
            alg.extend(self.UC(gammas[i]))
            alg.extend(self.UB(betas[i]))
        return alg

    def expectation_value(self, ansatz, c_alpha):
        """
        ansatz: list of strs, gate sequence to prepare |gamma, beta>.
        c_alpha: list of ints, qubits for z(q1)...z(qn) for qi = c_alpha[i].
        
        return: float, <gamma, beta | c_alpha | gamma, beta>
        """
        if not c_alpha: return 1 # identity

        alg = ansatz + [ # measure each qubit that c_alpha has a z on.
            "measure" + str(tuple(reversed(x))) for x in enumerate(c_alpha)
        ]
        
        res = self.run_qc(alg)
        # `res` is a dictionary mapping states to probabilities (where the
        # probabilities were determined from sampling). So, ie if we were
        # measuring c_alpha = [0, 2], ie < z(0) z(2) >, then `res` would 
        # look like {"00": float, "01": float, "10": float, "11": float},
        # where the first bit in the string corresponds to the measurement
        # of the 0th qubit, and the second bit in the string corresponds
        # to the measurement of the second qubit.
        
        # compute the expectation value from the probabilities.
        
        ev = 0.0
        for state, prob in res.items():
            if state.count("1") % 2: ev -= prob  # odd number of ones gives neg
            else: ev += prob
        return ev
    
    def objective_function(self, params):
        """
        maximizing what we want => minimizing negative
        params: betas + gammas.
        return: float, result of running circuit.
        """
        self.iters += 1
        n = len(params) // 2
        ansatz = self.prepare_ansatz(params[:n], params[n:])
        f = -1 * sum(
            self.expectation_value(ansatz, self.c_alphas[i]) * self.w_alphas[i]
            for i in range(len(self.c_alphas))
        )
        if self.verbose: print("%d obj func eval: %g" % (self.iters, -1*f))
        return f
            
    def find_max(self, betas_0, gammas_0, 
                       method="Powell", maxiter=None, verbose=True):
        """
        betas_0: list of floats, 
                    initial angles for U(B, beta). beta in [0, pi]
        gammas_0: list of floats, initial angles for U(C, gamma). 
                    gamma in [0, 2pi]. 
                  NOTE: len(betas_0) = len(gammas_0) = p.
        method: str, minimization method to use (for scipy.minimize).
        maxiter: int, maximum number of iterations.
        verbose: bool, whether to print updates.
        
        return: tuple; (max C (float), betas (list), gammas (list))
        """
        self.iters, self.verbose = 0, verbose
        p = len(betas_0)
        if p != len(gammas_0):
            raise ValueError(
                "Must give the same number of beta and gamma parameters"
            )
        if self.verbose: print("Finding maximum of", self, "with p =", p, "\n")
        
        if maxiter is not None: kwargs = dict(options=dict(maxiter=maxiter))
        else: kwargs = {}
        
        res = minimize(
            self.objective_function,
            list(betas_0) + list(gammas_0), # in case they are np.arrays.
            method=method,
            **kwargs
        )
        if self.verbose: print("Found maximum to be", -1*res.fun, "\n")
        return -1*res.fun, list(res.x[:p]), list(res.x[p:])
    
    def get_state(self, betas, gammas):
        """
        betas: list of floats, angle for U(B, beta).
        gammas: list of floats, angles for U(C, beta).
        
        return: dict, result of running ansatz preparation on qc.
        """
        return self.run_qc(self.prepare_ansatz(betas, gammas))
    
    def __str__(self):
        """ Return string representation of objective function """
        s = ""
        for m in range(len(self.c_alphas)):
            c_alpha, w_alpha = self.c_alphas[m], self.w_alphas[m]
            if w_alpha < 0: 
                if s: s = s[:-3] + " - "
                else: s += "-"
            if not c_alpha: s += str(abs(w_alpha))
            elif abs(w_alpha) != 1: s += str(abs(w_alpha)) + " "
            for i in c_alpha: s += "z(%d)" % i
            s += " + "
        s = "C = " + s[:-3]
        return s

        
if __name__ == "__main__":
    import numpy as np
    
    # C = 2 + z(0)z(1) - z(1)z(2) + z(0) - 2 z(0)z(1)z(2)  
    C = [
        (2, []),
        (1, [0, 1]),
        (-1, [1, 2]),
        (1, [0]),
        (-2, [0, 1, 2])
    ]
    
    system = QAOA(C, backend="cirq", num_samples=8000)
    # system = QAOA(C, backend="qiskit", num_samples=8000)
    # system = QAOA(C, backend="projectq", num_samples=8000)
    
    print("Cost function:", system, "\n\n")
    
    p = 1
    # initialize 2p random parameters.
    betas_0 = np.random.random(p) * np.pi
    gammas_0 = np.random.random(p) * 2 * np.pi
    
    m, b, g = system.find_max(betas_0, gammas_0, 
                              method="Powell", maxiter=100, verbose=True)
    
    print("Max of", system, "found to be:", m)
    print("Betas found:", b)
    print("Gammas found:", g)
    print("Resulting basis state probabilites of |gamma, beta>:", 
          system.get_state(b, g))