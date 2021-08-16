# Multi-Objective Adaptive Surrogate Modelling-based Optimization
# mixed-integer version with constraint handling
# modified by sunrc 2021-04
from __future__ import division, print_function, absolute_import
import sampling
import gp
import gwgp
import SM
import NSGA2_miscon
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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

    if (Xinit is None and Yinit is None):
        
        Ninit = 1000
        Xinit = sampling.glp(Ninit, nInput)
        Xinit = Xinit * (xub - xlb) + xlb
        Xinit[:,intstart-1:] = np.rint(Xinit[:,intstart-1:]) 
        Xinit_ori = Xinit.copy()
        print(nInput)
        for i in range(Ninit):
            for j in range(intstart-1,nInput):
                Xinit_ori[i,j] = discrete[Xinit[i,j].astype(np.int)-1]

        Yinit = np.zeros((Ninit, nOutput+1))
        for i in range(Ninit):
            Yinit[i,:-1], Yinit[i,-1] = model.evaluate(Xinit_ori[i,:])
    else:
        Ninit = Xinit.shape[0]

    icall = Ninit
    x = Xinit.copy()
    y = Yinit.copy()

    constrtmp = y[:,-1]
    constrtmp[constrtmp == 0.0] = 1.0
    constrtmp[constrtmp < 0.0] = 0.0
    y[:,-1] = constrtmp


    for i in range(niter):
        print('Surrogate Opt loop: %d' % i)
        
        sm = gwgp.MOGPR('CovNN', x, y[:,:-1], nInput, nOutput, xlb, xub, mean = np.zeros(nOutput))
        # sm = gp.GPR_Matern(x, y[:,:-1], nInput, nOutput, x.shape[0], xlb, xub)
        # sm = SM.train(x, y[:,:-1], nInput, nOutput, x.shape[0], xlb, xub)

        xs = (x - xlb) / (xub - xlb)
        smc = GaussianProcessClassifier(kernel=1.0 * RBF(1.0))
        # smc =  SVC(kernel="rbf", C=1)
        # smc = MLPClassifier(max_iter=2000)
        smc.fit(xs, y[:,-1])

        bestx_sm, besty_sm, bestc_sm, x_sm, y_sm, rank_sm = \
            NSGA2_miscon.optimization(sm, smc, nInput, nOutput+1, xlb, xub, intstart,\
                               pop, gen, crossover_rate, mu, mum)

        ### adaptive sampling
        xpt = (bestx_sm - xlb) / (xub - xlb)  
        idxr = (rank_sm == 0.0)
        x_resample = bestx_sm[idxr,:]
        if N_resample > sum(idxr):
            N_rem = N_resample - sum(idxr)
            idx, mdist = maxmindist(xs,xpt)
            count = 0
            for j in range(bestx_sm.shape[0]):
                if rank_sm[idx[j]] > 0.0:
                    count += 1
                    if mdist[j] < 1e-4:
                        randtmp = np.random.random([10000,nInput])
                        idxtmp, mdistmp = maxmindist(xs,randtmp)
                        xatmp = randtmp[idxtmp[0],:] * (xub - xlb) + xlb
                        xatmp[intstart-1:] = np.rint(xatmp[intstart-1:])
                        x_resample = np.vstack((x_resample, xatmp))
                    else:
                        x_resample = np.vstack((x_resample, bestx_sm[idx[j],:]))
                
                if count == N_rem:
                    break
        else:
            x_resample = x_resample[:N_resample,:]
     

        x_tmp = x_resample.copy()
        y_resample = np.zeros((x_tmp.shape[0],nOutput+1))
        for m in range(x_tmp.shape[0]):
            for n in range(intstart-1,nInput):
                x_tmp[m,n] = discrete[x_tmp[m,n].astype(np.int)-1]
            
        for j in range(x_tmp.shape[0]):
            y_resample[j,:-1], y_resample[j,-1] = model.evaluate(x_tmp[j,:])
            
            if y_resample[j,-1] == 0.0:
                y_resample[j,-1] = 1.0
            else:
                y_resample[j,-1] = 0.0

        # icall += N_resample
        x = np.vstack((x, x_resample))
        y = np.vstack((y, y_resample))


    xtmp = x.copy()
    ytmp = y.copy()
    yt = ytmp[:,:-1] 
    constr = ytmp[:,-1]
    xtmp, yt, constr, rank, crowd = NSGA2_miscon.sortMO(xtmp, yt, constr, nInput, nOutput)
    idxp = (rank == 0)
    bestx = xtmp[idxp,:]
    besty = yt[idxp,:]
    bestc = constr[idxp]

    return bestx, besty, bestc, x, y
    

def maxmindist(A, B):
    """ 
    maximize the minimum distance from point set B to A
    A is the referene point set
    for each point in B, compute its distance to its nearest neighbor of A
    find the point in B that has largest min-dist
    P: the coordinate of point
    D: the maxmin distance
    """
    T1 = A.shape[0]
    T2 = B.shape[0]
    
    Dist = np.zeros([T1,T2])
    for i in range(T1):
        for j in range(T2):
            Dist[i,j] = np.sqrt(np.sum((A[i,:]-B[j,:])**2))

    mindist = np.min(Dist, axis = 0)
    idx = mindist.argsort()[::-1]
    mindist = mindist[idx]

    return idx, mindist