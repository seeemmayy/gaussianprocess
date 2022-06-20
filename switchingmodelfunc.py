import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def arpoissontest(alpha, rho, sigma, n, p, q, seed=None):
    """
    n = number of simulations
    seed = seed number
    rho = rho value - must be 0<rho<1
    q = probability of leaving epi state
    p
    
    """
    
    etas = np.zeros(n)
    yt_1 = np.zeros(n)
    rand = np.random.default_rng(seed)
    zt = np.zeros(n)

    for i in range(2,n-1):
        if zt[i] == 0:
            zt[i+1] = rand.binomial(1,p)
        else:
            zt[i+1] = rand.binomial(1,1-q)
            
        if zt[i] == 0:
            etas[i] = 0
            lamt_1 = np.exp(alpha)
            yt_1[i] = rand.poisson(n*lamt_1)
        else:
            epsilon = rand.normal(loc=0.0, scale=sigma, size=1)
            delta = rho*(etas[i-1] - etas[i-2]) + epsilon
            etas[i] = etas[i-1] + delta
            lamt_1 = np.exp(alpha + etas[i])
            yt_1[i] = rand.poisson(n*lamt_1)
        
    
    return dict(y=yt_1, etas=etas, zt=zt)
