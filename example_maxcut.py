from qaoa import QAOA
import numpy as np
#import pylab

# For AWS
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pylab


def solve_maxcut(edges, p=1, backend="cirq", method="Powell"):
    """
    This will randomly initialize two starting angles and optimize with
    QAOA with p=1. Using those two angles, it will initialize two more to 0 and
    run QAOA with p=2, and so on until p = the p provided.
    
    The objective function being maximized is:
        C = - sum_{(i, j) \in edges} z(i) z(j)
    
    edges: set of tuples, edge connections. No duplicates, ie (0, 1) and (1, 0)
             will not be treated as two different edges. 
    p: int, number of gamma and beta parameters.
    backend: str, which backend to run on.
    
    return: tuple; (max obj func (float), state (dict))
    """
    edges = set(tuple(sorted(x)) for x in edges)
    exact = exact_solution(edges)
    C = [(-1, x) for x in edges]

    system = QAOA(C, backend=backend, num_samples=8000)
    
    print("\nFinding the max of the cost function:", system, "\n")

    # start off with p = 1, 2p random parameters.
#    betas_0 = [np.random.random()*np.pi]
#    gammas_0 = [np.random.random()*2*np.pi]
    betas_0 = [0.4513979795536029]
    gammas_0 = [5.622737558750011]
    
    xs, ys = [], [] # for plotting
    
    for pp in range(p):
        print("Starting with p =", pp+1)
        m, b, g = system.find_max(betas_0, gammas_0, 
                                  method=method, maxiter=1000, verbose=False)
        print("Found max to be", m, "\n")
        
        xs.append(1 / (pp+1))
        ys.append(m / exact)
        
        # Add parameter to go the next p value. By initializing to zero, we
        # should not see the cost function immediately decrease.
#        betas_0 = b + [0.0]
#        gammas_0 = g + [0.0]
    
        # Add parameter to go the next p value, use INTERP method.
        betas_0, gammas_0 = [0]*(pp+1), [0]*(pp+1)
        betas_0[0], gammas_0[0] = b[0], g[0]
        betas_0[-1], gammas_0[-1] = b[-1], g[-1]
        for i in range(2, pp+1):
            betas_0[i-1] = ((i-1)/pp)*b[i-2] + ((pp-i+1)/pp)*b[i-1]
            gammas_0[i-1] = ((i-1)/pp)*g[i-2] + ((pp-i+1)/pp)*g[i-1]
        
    print("Max of", system, "found with p =", p, "to be:", m, "\n")
    print("Betas found:", b)
    print("Gammas found:", g, "\n")
    
    state = system.get_state(b, g)
    print("Resulting state, |gamma, beta>:", state)
    
    xs.append(0)
    ys.append(1)
    
    pylab.plot(xs, ys, "o-", label=method)
    
    return m, state


def exact_solution(edges):
    """ Brute force find max cut """
    def test_solution(z):
        """ z: str, bit string """
        f, zs = 0, [1 if x == "0" else -1 for x in z]
        for e in edges:
            prod = 1
            for i in e: prod *= zs[i]
            f -= prod
        return f
    n = max(max(x) for x in edges) + 1
    s = "{0:0%db}" % n
    m = -np.inf
    for i in range(2**n): m = max(m, test_solution(s.format(i)))
    return m


if __name__ == "__main__":
#    n = 10
#    edges = set(
#        (i, j) for i in range(n) for j in range(i+1, n)
#        if not ((i+j) % 3)
#    )

#    n = 6
#    edges = set(
#        (i, j) for i in range(n) for j in range(i+1, n)
#        if not ((i+j) % 2)
#    )
    
    n = 12
    edges = set(
        (i, j) for i in range(n) for j in range(i+1, n)
        if not ((i+j) % 4)
    )
    
#    print("\nExact solution:", exact_solution(edges))
    
    p = 3
    
    pylab.figure()
    solve_maxcut(edges, p, "cirq", "Powell")
    solve_maxcut(edges, p, "cirq", "BFGS")
    pylab.xlabel("1 / p")
    pylab.ylabel("found max/ exact max")
    pylab.title(
        "Max cut on %d qubits with %d edges with INTERP"
        % (n, len(edges))
    )
    pylab.legend()
    pylab.show()