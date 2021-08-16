# Multi-Objective Adaptive Surrogate Modelling-based Optimization
from __future__ import division, print_function, absolute_import
import sampling
import gp
import gwgp
import SM
import NSGA2_mis
import numpy as np


def optimization(model, nInput, nOutput, xlb, xub, intstart, discrete, niter, pct, \
                 Xinit = None, Yinit = None, pop = 100, gen = 100, \
                 crossover_rate = 0.9, mu = 20, mum = 20):
    """ 
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    model: the evaluated model function
    nInput: number of model input
    nOutput: number of output objectives
    xlb: lower bound of input
    xub: upper bound of input
    niter: number of iteration
    pct: percentage of resampled points in each iteration
    Xinit and Yinit: initial samplers for surrogate model construction
    ### options for the embedded NSGA-II of MO-ASMO
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    """
    N_resample = int(pop*pct)
    
    xl = xlb.copy()
    xu = xub.copy()

    xl[intstart-1:] = xl[intstart-1:] - (0.5 - 1e-16)
    xu[intstart-1:] = xu[intstart-1:] + (0.5 - 1e-16)

    print(discrete)

    if (Xinit is None and Yinit is None):
        
        Ninit = 100
        Xinit = sampling.glp(Ninit, nInput)
        Xinit = Xinit * (xub - xlb) + xlb
        Xinit[:,intstart-1:] = np.rint(Xinit[:,intstart-1:]) 
        Xinit_ori = Xinit.copy()
        for i in range(Ninit):
            for j in range(intstart-1,nInput):
            # for j in range(intstart,nInput): # for RE25
                Xinit_ori[i,j] = discrete[Xinit[i,j].astype(np.int)-1]
        
        Yinit = np.zeros((Ninit, nOutput))
        for i in range(Ninit):
            Yinit[i,:] = model.evaluate(Xinit_ori[i,:])
    else:
        Ninit = Xinit.shape[0]

    icall = Ninit
    x = Xinit.copy()
    y = Yinit.copy()


    for i in range(niter):
        print('Surrogate Opt loop: %d' % i)
        
        sm = gwgp.MOGPR('CovMatern3', x, y, nInput, nOutput, xlb, xub, mean = np.zeros(nOutput))
        # sm = gp.GPR_Matern(x, y, nInput, nOutput, x.shape[0], xlb, xub)
        # sm = SM.train(x, y, nInput, nOutput, x.shape[0], xlb, xub)

        bestx_sm, besty_sm, x_sm, y_sm, ranktmp = \
            NSGA2_mis.optimization(sm, nInput, nOutput, xlb, xub, intstart,\
                               pop, gen, crossover_rate, mu, mum)

        x_resample = bestx_sm[:N_resample,:]
        y_resample = np.zeros((N_resample,nOutput))
        x_tmp = x_resample.copy()
        for m in range(N_resample):
            for n in range(intstart-1,nInput):
            # for n in range(intstart,nInput): # for RE25
                x_tmp[m,n] = discrete[x_resample[m,n].astype(np.int)-1]
            
        for j in range(N_resample):
            y_resample[j,:] = model.evaluate(x_tmp[j,:])
        icall += N_resample
        x = np.vstack((x, x_resample))
        y = np.vstack((y, y_resample))

    xtmp = x.copy()
    ytmp = y.copy()
    xtmp, ytmp, rank, crowd = NSGA2_mis.sortMO(xtmp, ytmp, nInput, nOutput)
    idxp = (rank == 0)
    bestx = xtmp[idxp,:]
    besty = ytmp[idxp,:]

    return bestx, besty, x, y
