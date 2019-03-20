from qaoa import QAOA
import numpy as np


def solve(C, p=1, backend="cirq"):
    """
    This will randomly initialize two starting angles and optimize with
    QAOA with p=1. Using those two angles, it will initialize two more to 0 and
    run QAOA with p=2, and so on until p = the p provided.
    
    C: list of tuples, each tuple is (int, list). See QAOA.__init__ docstring.
    p: number of beta and gamma parameters.
    backend: str, backend to run on.
    
    return: None.
    """
    system = QAOA(C, backend=backend, num_samples=8000)
    
    print("\nFinding the max of the cost function:", system, "\n")

    # start off with p = 1, 2p random parameters.
    betas_0 = [np.random.random()*np.pi]
    gammas_0 = [np.random.random()*2*np.pi]
    
    for _ in range(p):
        print("Starting with p =", _+1)
        m, b, g = system.find_max(betas_0, gammas_0, 
                                  method="Powell", maxiter=1000, verbose=False)
        print("Found max to be", m, "\n")
        # Add parameter to go the next p value. By initializing to zero, we
        # should not see the cost function immediately decrease.
        betas_0 = b + [0.0]
        gammas_0 = g + [0.0]
        
    
    print("Max of", system, "found with p =", p, "to be:", m, "\n")
    print("Betas found:", b)
    print("Gammas found:", g, "\n")
    print("Resulting state, |gamma, beta>:", system.get_state(b, g))
    
    
if __name__ == "__main__":
    # Find max of C = -1 + z(0)z(2) - 2 z(0)z(1)z(2) - 3 z(1)
    C = [
        (-1, []),
        (1, [0, 2]),
        (-2, [0, 1, 2]),
        (-3, [1])
    ]
    # Should find max C of 5 with 111 50% of the time and 010 50% of the time.
    
    p = 3
    
    solve(C, p=p, backend="cirq")
    # solve(C, p=p, backend="qiskit")
    # solve(C, p=p, backend="projectq")