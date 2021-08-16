# build surrogate model with Gaussian Processes Regression (c version)
# written by gongwei, BNU
# Sep 18, 2018
import numpy as np
import copy
import cgp

class MOGPR:
    ''' Multi-objective Gaussian Processes Regression
        for multiple objective functions
    '''
    def __init__(self, covname, xin, yin, nInput, nOutput, xlb, xub, \
            mean = None, noise = 1e-6):
        ''' Initialize and train Gaussian Processes Regression surrogate model
            covname: name of covariance function
            xin: input parameter values, nSample*nInput, should be a 2d array
            yin: output objective function values, nSample*nOutput, should be a 2d array
            nInput: number of input parameters
            nOutput: number of output objectives
            xlb: lower bound of x
            xub: upper bound of x
            mean: mean value or mean function of GPR
            noise: a small value added to the diagonal to increase stability
        '''
        x = copy.deepcopy(xin)
        y = copy.deepcopy(yin)
        if len(x.shape) == 1:
            raise RuntimeError('The training data x in GPR should be a 2d array!')
        if len(y.shape) == 1:
            raise RuntimeError('The training data y in GPR should be a 2d array!')

        self.x = x
        self.y = y
        self.nInput = nInput
        self.nOutput = nOutput
        self.covname = covname
        self.xlb = xlb
        self.xub = xub
        if mean is None:
            self.mean = np.mean(self.y, axis=0)
        else:
            self.mean = mean
        self.noise = noise
        self.smlist = []

        for i in range(self.nOutput):
            self.smlist.append(GPR(self.covname, self.x, self.y[:,i], self.nInput, \
                    self.xlb, self.xub, self.mean[i], self.noise))

    def evaluate(self, xpred):
        ''' in order to be compatible with other dynamical and surrogate models
            'evaluate' only return the objective function value, not the variance
        '''
        x2 = copy.deepcopy(xpred)
        if len(x2.shape) == 1:
            # only 1 sample input
            nSample2 = 1
            if x2.shape[0] != self.nInput:
                raise RuntimeError('Dimension of xpred should be nInput!')
            x2 = x2.reshape((1,self.nInput))
        else:
            nSample2 = x2.shape[0]
        f = np.zeros([nSample2, self.nOutput])
        pv = np.zeros([nSample2, self.nOutput])
        for i in range(self.nOutput):
            f[:,i], pv[:,i] = self.smlist[i].predict(xpred)
        return f

class GPR:
    ''' Gaussian Processes Regression
        for single objective function
    '''
    def __init__(self, covname, xin, yin, nInput, xlb, xub, mean = 0, noise = 1e-6):
        ''' Initialize and train Gaussian Processes Regression surrogate model
            covname: name of covariance function
            xin: input parameter values, nSample*nInput, should be a 2d array
            yin: output objective function values, should be a 1d array
            nInput: number of input parameters
            xlb: lower bound of x
            xub: upper bound of x
            mean: mean value or mean function of GPR
            noise: a small value added to the diagonal to increase stability
        '''
        x = copy.deepcopy(xin)
        y = copy.deepcopy(yin)
        if len(x.shape) == 1:
            raise RuntimeError('The training data x in GPR should be a 2d array!')

        covdict = {'CovMatern3': 1, \
                   'CovMatern5': 2, \
                   'CovSE': 3, \
                   'CovSEnoisefree': 4, \
                   'CovNN': 5 }
        hypdict = {'CovMatern3': 2, \
                   'CovMatern5': 2, \
                   'CovSE': 3, \
                   'CovSEnoisefree': 2, \
                   'CovNN': 2 }

        self.nSample = x.shape[0]
        self.nInput = int(nInput)
        self.xlb = xlb
        self.xub = xub
        self.xrg = xub - xlb
        self.covfunc = covdict[covname]
        self.nhyp = hypdict[covname]
        self.hyplb = -10 * np.ones(self.nhyp)
        self.hypub = 10 * np.ones(self.nhyp)
        # self.hyplb = -2 * np.ones(self.nhyp) # The coefficient of 2 is suggested by Dr. Gong.
        # self.hypub = 2 * np.ones(self.nhyp)
        self.mean = mean
        self.noise = noise
        self.x = x
        self.y = y
        for i in range(self.nSample):
            self.x[i,:] = (self.x[i,:] - self.xlb) / self.xrg

        besthyp, bestm = self.hypopt()
        realhyp = np.exp(besthyp)
        print('hyper-parameters of GPR: %s' % str(realhyp))
        m, K, L, alpha = self.fit(realhyp)

        self.hyp = realhyp
        self.m = m
        self.K = K
        self.L = L
        self.alpha = alpha

    def evaluate(self, xpred):
        ''' in order to be compatible with other dynamical and surrogate models
            'evaluate' only return the objective function value, not the variance
        '''
        f, pv = self.predict(xpred)
        return f

    def fit(self, hyp):
        ''' Algorithm 2.1 of GPML page19
            training/fitting Gaussian Processes Regression
            x: input matrix, nSample*nInput
            y: output vectors, nSample
            covfunc: covariance function
            hyp: list of hyperparameters
            mean: mean function of GP
            noise: std variance of observation noise, sigma^2
            m: predictive marginal likelihood log(p(y|X))
        '''
        y = self.y - self.mean
        x = self.x
        CovIdx = self.covfunc
        noise = self.noise
        K = np.zeros([self.nSample, self.nSample])
        L = np.zeros([self.nSample, self.nSample])
        alpha = np.zeros(self.nSample)
        m = cgp.callGPtrain(x, y, CovIdx, hyp, noise, K, L, alpha)
        return m, K, L, alpha

    def predict(self, xpred):
        ''' algorithm 2.1 of GPML page19
            predicting of Gaussian Processes Regression
            f: predictive mean
            pv: predictive varaince
            xpred: input vector nSample2*nInput for prediction
        '''
        x1 = self.x
        x2 = copy.deepcopy(xpred)
        if len(x2.shape) == 1 and x2.shape[0] != self.nInput :
            raise RuntimeError('The input xpred should be an nSample2*nInput 2d array!')
        elif len(x2.shape) == 1:
            x2 = x2.reshape((1,self.nInput))
            nSample2 = 1
        else:
            nSample2 = x2.shape[0]
        for i in range(nSample2):
            x2[i,:] = (x2[i,:] - self.xlb) / self.xrg
        f = np.zeros(nSample2)
        pv = np.zeros(nSample2)

        CovIdx = self.covfunc
        L = self.L
        hyp = self.hyp
        alpha = self.alpha

        cgp.callGPpredict(x1, x2, CovIdx, hyp, L, alpha, f, pv)

        f += self.mean
        return f, pv

    def hypopt(self):
        '''
        SCE-UA optimizer for optimizing hyper parameters of GPR
        Input:
          * 'hyplb' and 'hypub' are log transformed upper and lower
            bounds of hyper-parameters
          * 'x, y, covfunc, noise, mean' are parameters for 'fit' function
        Returned:
          * 'bestx' is the best found hyperparameters theta
          * 'bestf' is the corresponding value of the target function
            (marginal likelihood).
        '''
        nopt = self.nhyp
        ngs = nopt
        maxn = 3000
        kstop = 10
        pcento = 0.1
        peps = 0.001
        verbose = False

        [bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list] = \
            self.sceua(self.MML, self.hyplb, self.hypub, nopt, ngs, maxn, kstop, pcento, peps, verbose)
        return bestx, bestf

    def MML(self,hyp):
        ''' temporal function for maximizing marginal likelihood
        '''
        realhyp = np.exp(hyp)
        m,_,_,_ = self.fit(realhyp)
        return -m

    def sceua(self, func, bl, bu, nopt, ngs, maxn, kstop, pcento, peps, verbose):
        """
        This is the subroutine implementing the SCE algorithm,
        written by Q.Duan, 9/2004
        translated to python by gongwei, 11/2017

        Parameters:
        func:   optimized function
        bl:     the lower bound of the parameters
        bu:     the upper bound of the parameters
        nopt:   number of adjustable parameters
        ngs:    number of complexes (sub-populations)
        maxn:   maximum number of function evaluations allowed during optimization
        kstop:  maximum number of evolution loops before convergency
        pcento: the percentage change allowed in kstop loops before convergency
        peps:   relative size of parameter space

        npg:  number of members in a complex
        nps:  number of members in a simplex
        npt:  total number of points in an iteration
        nspl:  number of evolution steps for each complex before shuffling
        mings: minimum number of complexes required during the optimization process

        LIST OF LOCAL VARIABLES
        x[.,.]:    coordinates of points in the population
        xf[.]:     function values of x[.,.]
        xx[.]:     coordinates of a single point in x
        cx[.,.]:   coordinates of points in a complex
        cf[.]:     function values of cx[.,.]
        s[.,.]:    coordinates of points in the current simplex
        sf[.]:     function values of s[.,.]
        bestx[.]:  best point at current shuffling loop
        bestf:     function value of bestx[.]
        worstx[.]: worst point at current shuffling loop
        worstf:    function value of worstx[.]
        xnstd[.]:  standard deviation of parameters in the population
        gnrng:     normalized geometri%mean of parameter ranges
        lcs[.]:    indices locating position of s[.,.] in x[.,.]
        bound[.]:  bound on ith variable being optimized
        ngs1:      number of complexes in current population
        ngs2:      number of complexes in last population
        criter[.]: vector containing the best criterion values of the last 10 shuffling loops
        """

        # Initialize SCE parameters:
        npg  = 2 * nopt + 1
        nps  = nopt + 1
        nspl = npg
        npt  = npg * ngs
        bd   = bu - bl

        # Create an initial population to fill array x[npt,nopt]
        x = np.random.random([npt,nopt])
        for i in range(npt):
            x[i,:] = x[i,:] * bd + bl

        xf = np.zeros(npt)
        for i in range(npt):
            xf[i] = func(x[i,:])
        icall = npt

        # Sort the population in order of increasing function values
        idx = np.argsort(xf)
        xf = xf[idx]
        x = x[idx,:]

        # Record the best and worst points
        bestx  = copy.deepcopy(x[0,:])
        bestf  = copy.deepcopy(xf[0])
        worstx = copy.deepcopy(x[-1,:])
        worstf = copy.deepcopy(xf[-1])

        bestf_list = []
        bestf_list.append(bestf)
        bestx_list = []
        bestx_list.append(bestx)
        icall_list = []
        icall_list.append(icall)

        if verbose:
            print('The Initial Loop: 0')
            print('BESTF  : %f' % bestf)
            print('BESTX  : %s' % np.array2string(bestx))
            print('WORSTF : %f' % worstf)
            print('WORSTX : %s' % np.array2string(worstx))
            print(' ')

        # Computes the normalized geometric range of the parameters
        gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
        # Check for convergency
        if verbose:
            if icall >= maxn:
                print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                print('ON THE MAXIMUM NUMBER OF TRIALS ')
                print(maxn)
                print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
                print(icall)
                print('OF THE INITIAL LOOP!')

            if gnrng < peps:
                print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

        # Begin evolution loops:
        nloop = 0
        criter = []
        criter_change = 1e+5
        cx = np.zeros([npg,nopt])
        cf = np.zeros(npg)

        while (icall < maxn) and (gnrng > peps) and (criter_change > pcento):
            nloop += 1

            # Loop on complexes (sub-populations)
            for igs in range(ngs):

                # Partition the population into complexes (sub-populations)
                k1 = np.int64(np.linspace(0, npg-1, npg))
                k2 = k1 * ngs + igs
                cx[k1,:] = copy.deepcopy(x[k2,:])
                cf[k1] = copy.deepcopy(xf[k2])

                # Evolve sub-population igs for nspl steps
                for loop in range(nspl):

                    # Select simplex by sampling the complex according to a linear
                    # probability distribution
                    lcs = np.zeros(nps, dtype=np.int64)
                    lcs[0] = 0
                    for k3 in range(1,nps):
                        for itmp in range(1000):
                            lpos = int(np.floor(
                                    npg + 0.5 - np.sqrt((npg + 0.5)**2 -
                                    npg * (npg + 1) * np.random.rand())))
                            if len(np.where(lcs[:k3] == lpos)[0]) == 0:
                                break
                        lcs[k3] = lpos
                    lcs = np.sort(lcs)

                    # Construct the simplex:
                    s = copy.deepcopy(cx[lcs,:])
                    sf = copy.deepcopy(cf[lcs])

                    snew, fnew, icall = self.cceua(func, s, sf, bl, bu, icall)

                    # Replace the worst point in Simplex with the new point:
                    s[nps-1,:] = snew
                    sf[nps-1] = fnew

                    # Replace the simplex into the complex
                    cx[lcs,:] = copy.deepcopy(s)
                    cf[lcs] = copy.deepcopy(sf)

                    # Sort the complex
                    idx = np.argsort(cf)
                    cf = cf[idx]
                    cx = cx[idx,:]

                # End of Inner Loop for Competitive Evolution of Simplexes

                # Replace the complex back into the population
                x[k2,:] = copy.deepcopy(cx[k1,:])
                xf[k2] = copy.deepcopy(cf[k1])

            # End of Loop on Complex Evolution;

            # Shuffled the complexes
            idx = np.argsort(xf)
            xf = xf[idx]
            x = x[idx,:]

            # Record the best and worst points
            bestx  = copy.deepcopy(x[0,:])
            bestf  = copy.deepcopy(xf[0])
            worstx = copy.deepcopy(x[-1,:])
            worstf = copy.deepcopy(xf[-1])
            bestf_list.append(bestf)
            bestx_list.append(bestx)
            icall_list.append(icall)

            if verbose:
                print('Evolution Loop: %d - Trial - %d' % (nloop, icall))
                print('BESTF  : %f' % bestf)
                print('BESTX  : %s' % np.array2string(bestx))
                print('WORSTF : %f' % worstf)
                print('WORSTX : %s' % np.array2string(worstx))
                print(' ')

            # Computes the normalized geometric range of the parameters
            gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
            # Check for convergency;
            if verbose:
                if icall >= maxn:
                    print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                    print('ON THE MAXIMUM NUMBER OF TRIALS %d HAS BEEN EXCEEDED!' % maxn)
                if gnrng < peps:
                    print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')


            criter.append(bestf)
            if nloop >= kstop:
                criter_change = np.abs(criter[nloop-1] - criter[nloop-kstop])*100
                criter_change /= np.mean(np.abs(criter[nloop-kstop:nloop]))
                if criter_change < pcento:
                    if verbose:
                        print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f%%' % (kstop, pcento))
                        print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')

        # End of the Outer Loops

        if verbose:
            print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' % icall )
            print('NORMALIZED GEOMETRIC RANGE = %f' % gnrng )
            print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f%%' % (kstop, criter_change))

        # END of Subroutine SCEUA_runner
        return bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list

    def cceua(self, func, s, sf, bl, bu, icall):
        """
        This is the subroutine for generating a new point in a simplex
        func:   optimized function
        s[.,.]: the sorted simplex in order of increasing function values
        sf[.]:  function values in increasing order

        LIST OF LOCAL VARIABLES
        sb[.]:   the best point of the simplex
        sw[.]:   the worst point of the simplex
        w2[.]:   the second worst point of the simplex
        fw:      function value of the worst point
        ce[.]:   the centroid of the simplex excluding wo
        snew[.]: new point generated from the simplex
        iviol:   flag indicating if constraints are violated
                 = 1 , yes
                 = 0 , no
        """

        nps, nopt = s.shape
        n = nps
        alpha = 1.0
        beta = 0.5

        # Assign the best and worst points:
        sw = s[-1,:]
        fw = sf[-1]

        # Compute the centroid of the simplex excluding the worst point:
        ce = np.mean(s[:n-1,:],axis=0)

        # Attempt a reflection point
        snew = ce + alpha * (ce - sw)

        # Check if is outside the bounds:
        ibound = 0
        s1 = snew - bl
        if sum(s1 < 0) > 0:
            ibound = 1
        s1 = bu - snew
        if sum(s1 < 0) > 0:
            ibound = 2
        if ibound >= 1:
            snew = bl + np.random.random(nopt) * (bu - bl)

        fnew = func(snew)
        icall += 1

        # Reflection failed; now attempt a contraction point
        if fnew > fw:
            snew = sw + beta * (ce - sw)
            fnew = func(snew)
            icall += 1

        # Both reflection and contraction have failed, attempt a random point
            if fnew > fw:
                snew = bl + np.random.random(nopt) * (bu - bl)
                fnew = func(snew)
                icall += 1

        # END OF CCE
        return snew, fnew, icall


if __name__ == "__main__":
    # main program of GP test
    x = np.linspace(-10,10,20)
    y = x**2
    nInput = 1
    nOutput = 1
    xlb = np.array([-20.0])
    xub = np.array([20.0])
    x = x.reshape((-1,nInput))
    y = y.reshape((-1,nOutput))
    covname = 'CovMatern3'
    #gpmodel = GPR(covname, x, y, nInput, xlb, xub)
    gpmodel = MOGPR(covname, x, y, nInput, nOutput, xlb, xub, mean=np.ones(1)*100)

    #x2 = np.linspace(-2000,20,1000)
    x2 = np.linspace(-10,10,100)
    x2 = x2.reshape((-1,nInput))
    y2 = gpmodel.evaluate(x2)

    import matplotlib.pyplot as plt
    plt.plot(x.reshape(-1),y.reshape(-1),'ro')
    plt.plot(x2.reshape(-1),y2.reshape(-1),'b-')
    plt.show()

