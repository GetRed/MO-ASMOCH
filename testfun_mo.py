# test function for multi-objective optimization
# Test Problems of Tanabe and Ishibuchi(2020) DOI: 10.1016/j.asoc.2020.106078
# # Author: sunrc  2021-04

import numpy as np
import math

def evaluate(x):
    
    f = np.zeros(2)
    
    ## RE22
    g = np.zeros(2)
    x1 = x[2]
    x2 = x[0]
    x3 = x[1]
    f[0] = (29.4 * x1) + (0.6 * x2 * x3)
    # Original constraint functions 	
    g[0] = (x1 * x3) - 7.735 * ((x1 * x1) / x2) - 180.0
    g[1] = 4.0 - (x3 / x2)
    g = np.where(g < 0, -g, 0)          
    f[1] = g[0] + g[1]

    ## RE23
    # g = np.zeros(3)
    # x3 = x[0]
    # x4 = x[1]
    # x1 = x[2]
    # x2 = x[3]
    # f[0] = (0.6224 * x1 * x3* x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
    # # Original constraint functions 	
    # g[0] = x1 - (0.0193 * x3)
    # g[1] = x2 - (0.00954 * x3)
    # g[2] = (np.pi * x3 * x3 * x4) + ((4.0/3.0) * (np.pi * x3 * x3 * x3)) - 1296000
    # g = np.where(g < 0, -g, 0)            
    # f[1] = g[0] + g[1] + g[2]

    ## RE25
    # g = np.zeros(6)
    # x1 = x[1]
    # x2 = x[0]
    # x3 = x[2]
    # # first original objective function
    # f[0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0
    # # constraint functions
    # Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
    # Fmax = 1000.0
    # S = 189000.0
    # G = 11.5 * 1e+6
    # K  = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
    # lmax = 14.0
    # lf = (Fmax / K) + 1.05 *  (x1 + 2) * x3
    # dmin = 0.2
    # Dmax = 3
    # Fp = 300.0
    # sigmaP = Fp / K
    # sigmaPM = 6
    # sigmaW = 1.25

    # g[0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
    # g[1] = -lf + lmax
    # g[2] = -3 + (x2 / x3)
    # g[3] = -sigmaP + sigmaPM
    # g[4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
    # g[5] = sigmaW- ((Fmax - Fp) / K)

    # g = np.where(g < 0, -g, 0)            
    # f[1] = g[0] + g[1] + g[2] + g[3] + g[4] + g[5]
    
   
    return f
