# define the uq workflow
import numpy as np
import os
import pickle
import copy
import time
from multiprocessing import Pool
# import matlab.engine

class Model:
    ''' Define a class for dynamic models
    '''
    def __init__(self, modelcfg):
        ''' Initialize a dynamic model object with the dictionary modelcfg
        '''
        self.workpath = modelcfg['workpath']
        os.chdir(self.workpath)
        self.driver = modelcfg['driver']
        driver = __import__(modelcfg['driver'])
        self.driverfunc = driver.evaluate

        # future = matlab.engine.start_matlab(background=True)
        # self.eng = future.result()

        self.nInput = len(modelcfg['parameters'])
        self.xlb = np.zeros(self.nInput)
        self.xub = np.zeros(self.nInput)
        self.xdf = np.zeros(self.nInput)
        self.xname = []
        self.dist = []

        if 'discretevalue' in modelcfg:
            self.discrete = modelcfg['discretevalue']
        
        if 'intstart' in modelcfg:
            self.intstart = modelcfg['intstart']
        
        for i,ipara in enumerate(modelcfg['parameters']):
            self.xname.append(ipara['name'])
            self.xlb[i] = ipara['lb']
            self.xub[i] = ipara['ub']
            if ipara['dft'] is None:
                self.xdf[i] = 0.5 * (ipara['lb'] + ipara['ub'])
            else:
                self.xdf[i] = ipara['dft']
            if ipara['dist'] is None:
                self.dist.append('uniform')
            else:
                self.dist.append(ipara['dist'])

        self.nOutput = len(modelcfg['objectives'])
        self.ydf = np.zeros(self.nOutput)
        self.yname = []
        self.screened = []
        for i,iobj in enumerate(modelcfg['objectives']):
            self.yname.append(iobj['name'])
            self.ydf[i] = iobj['dft']
            self.screened.append(iobj['screened'])

        self.parallel = modelcfg['parallel']
        self.processes = modelcfg['processes']
        self.usescreened = modelcfg['usescreened']
        self.usemultiobj = modelcfg['usemultiobj']

        if self.usescreened:
            self.paraidx = np.zeros(self.nInput, dtype = bool)
            if self.usemultiobj and self.nOutput > 1:
                for i in range(self.nOutput):
                    for j in range(len(self.screened[i])):
                        self.paraidx[self.screened[i][j]] = True
            else:
                for i in range(len(self.screened[0])):
                    self.paraidx[self.screened[0][i]] = True
        else:
            self.paraidx = np.ones(self.nInput, dtype = bool)
        self.nInputS = np.sum(self.paraidx)
        self.xlbS = self.xlb[self.paraidx]
        self.xubS = self.xub[self.paraidx]

    def evaluate(self,x):
        ''' evaluate the model with given input vector x
        '''
        if len(x.shape) == 1:
            if self.usescreened:
                xtmp = copy.deepcopy(self.xdf)
                c = 0
                for i in range(self.nInput):
                    if self.paraidx[i]:
                        xtmp[i] = x[c]
                        c += 1
            else:
                xtmp = copy.deepcopy(x)
            return self.driverfunc(xtmp)
            # return self.driverfunc(xtmp,self.eng)
        else:
            n = x.shape[0]
            if self.usescreened:
                xtmp = np.zeros([n,self.nInput])
                for i in range(n):
                    xtmp[i,:] = copy.deepcopy(self.xdf)
                    c = 0
                    for j in range(self.nInput):
                        if self.paraidx[j]:
                            xtmp[i,j] = x[i,c]
                            c += 1
            else:
                xtmp = copy.deepcopy(x)
            ytmp = np.zeros([n,self.nOutput])
            if self.parallel and n >= self.processes:
                xpara = []
                for i in range(n):
                    xpara.append(xtmp[i,:])
                p = Pool(processes = self.processes)
                res = p.map(self.driverfunc, xpara)
                for i in range(n):
                    ytmp[i,:] = res[i]
            else:
                for i in range(n):
                    ytmp[i,:] = self.driverfunc(xtmp[i,:])
                    # ytmp[i,:] = self.driverfunc(xtmp[i,:],self.eng)
            return ytmp

def workflow(modelcfg, uqcfg):
    ''' define uq workflow with the python object cfg
    '''
    model = Model(modelcfg)
    resultpath = uqcfg['resultpath']
    resultname = uqcfg['resultname']

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    fname = resultpath + '/' + resultname

    uqres = {}
    uqres['modelcfg'] = modelcfg
    uqres['uqcfg'] = uqcfg
    uqres['uqres'] = {}

    if 'sampling' in uqcfg:
        smpcfg = uqcfg['sampling']
        method = smpcfg['method']
        config = smpcfg['config']

        sampling = __import__('sampling')

        time_start = time.clock()

        # general options
        nSample = model.nInputS * 10 if 'nSample' not in config else config['nSample']
        maxiter = 5 if 'maxiter' not in config else config['maxiter']

        # MC: Monte Carlo
        if method == 'MC':
            x = sampling.mc(nSample, model.nInputS)
        # LH: Latin Hypercube
        elif method == 'LH':
            x = sampling.lh(nSample, model.nInputS, maxiter)
        # SLH: Symmetric Latin Hypercube
        elif method == 'SLH':
            x = sampling.slh(nSample, model.nInputS, maxiter)
        # GLP: Good Lattice Points
        elif method == 'GLP':
            x = sampling.glp(nSample, model.nInputS, maxiter)
        # method not found
        else:
            raise RuntimeError('Sampling method %s not supported, exit!' % method)

        if 'discretevalue' in modelcfg:
            xl = model.xlbS.copy()
            xu = model.xubS.copy()
            xl[model.intstart-1:] = xl[model.intstart-1:] - (0.5 - 1e-16)
            xu[model.intstart-1:] = xu[model.intstart-1:] + (0.5 - 1e-16) 
            x = x * (xu - xl) + xl
            x[:,model.intstart-1:] = np.rint(x[:,model.intstart-1:]) 
            x_ori = x.copy()
            for i in range(model.intstart-1,model.nInputS):
                for j in range(0,nSample):
                    x_ori[j,i] = model.discrete[x[j,i].astype(np.int)-1]
            y = model.evaluate(x_ori)
        else:
            x = x * (model.xubS - model.xlbS) + model.xlbS
            y = model.evaluate(x)
        result = {}
        result['x'] = x
        result['y'] = y

        timing = time.clock() - time_start

        uqres['uqres']['sampling'] = {}
        uqres['uqres']['sampling']['timing'] = timing
        uqres['uqres']['sampling']['result'] = result

        # store tmp result
        with open(fname + '.tmp', 'wb') as f:
            pickle.dump(uqres, f)

    if 'optimization' in uqcfg:
        optcfg = uqcfg['optimization']
        method = optcfg['method']
        config = optcfg['config']

        time_start = time.clock()

        if method == 'NSGA2_mixint':
            pop = 100 if 'pop' not in config else config['pop']
            gen = 100 if 'gen' not in config else config['gen']
            crossover_rate = 0.9 if 'crossover_rate' not in config else config['crossover_rate']
            mu = 20 if 'mu' not in config else config['mu']
            mum = 20 if 'mum' not in config else config['mum']
            NSGA2 = __import__('NSGA2_mixint')
            bestx, besty, x, y = \
                NSGA2.optimization(model, model.nInputS, model.nOutput, \
                    model.xlbS, model.xubS, model.intstart, model.discrete,\
                    pop, gen, crossover_rate, mu, mum)
            result = {}
            result['bestx'] = bestx
            result['besty'] = besty
            result['x'] = x
            result['y'] = y
        
        elif method == 'NSGA2_mixintcon':
            pop = 100 if 'pop' not in config else config['pop']
            gen = 100 if 'gen' not in config else config['gen']
            crossover_rate = 0.9 if 'crossover_rate' not in config else config['crossover_rate']
            mu = 20 if 'mu' not in config else config['mu']
            mum = 20 if 'mum' not in config else config['mum']
            NSGA2 = __import__('NSGA2_mixintcon')
            bestx, besty, bestc, x, y, constr  = \
                NSGA2.optimization(model, model.nInputS, model.nOutput, \
                    model.xlbS, model.xubS, model.intstart, model.discrete,\
                    pop, gen, crossover_rate, mu, mum)
            result = {}
            result['bestx'] = bestx
            result['besty'] = besty
            result['bestc'] = bestc
            result['x'] = x
            result['y'] = y
            result['constr'] = constr

        elif method == 'MOASMO_mixint':
            Xinit, Yinit = loadSamples(config, uqres, model)
            niter = 5 if 'niter' not in config else config['niter']
            pct = 0.2 if 'pct' not in config else config['pct']
            pop = 100 if 'pop' not in config else config['pop']
            gen = 100 if 'gen' not in config else config['gen']
            crossover_rate = 0.9 if 'crossover_rate' not in config else config['crossover_rate']
            mu = 20 if 'mu' not in config else config['mu']
            mum = 20 if 'mum' not in config else config['mum']
            MOASMO = __import__('MOASMO_mixint')
            bestx, besty, x, y = \
                MOASMO.optimization(model, model.nInputS, model.nOutput, \
                    model.xlbS, model.xubS, model.intstart, model.discrete, niter, pct, \
                    Xinit, Yinit, pop, gen, crossover_rate, mu, mum)
            result = {}
            result['bestx'] = bestx
            result['besty'] = besty
            result['x'] = x
            result['y'] = y

        elif method == 'MOASMO_mixintcon':
            Xinit, Yinit = loadSamples(config, uqres, model)
            niter = 5 if 'niter' not in config else config['niter']
            pct = 0.2 if 'pct' not in config else config['pct']
            pop = 100 if 'pop' not in config else config['pop']
            gen = 100 if 'gen' not in config else config['gen']
            crossover_rate = 0.9 if 'crossover_rate' not in config else config['crossover_rate']
            mu = 20 if 'mu' not in config else config['mu']
            mum = 20 if 'mum' not in config else config['mum']
            MOASMO = __import__('MOASMO_mixintcon')
            bestx, besty, bestc, x, y = \
                MOASMO.optimization(model, model.nInputS, model.nOutput, \
                    model.xlbS, model.xubS, model.intstart, model.discrete, niter, pct, \
                    Xinit, Yinit, pop, gen, crossover_rate, mu, mum)
            result = {}
            result['bestx'] = bestx
            result['besty'] = besty
            result['bestc'] = bestc
            result['x'] = x
            result['y'] = y

        # method not found
        else:
            raise RuntimeError('Optimization method %s not supported, exit!' % method)

        timing = time.clock() - time_start

        uqres['uqres']['optimization'] = {}
        uqres['uqres']['optimization']['timing'] = timing
        uqres['uqres']['optimization']['result'] = result

        # store tmp result
        with open(fname + '.tmp', 'wb') as f:
            pickle.dump(uqres, f)

    # after all workflow, store the result in a bin file
    with open(fname + '.bin', 'wb') as f:
        pickle.dump(uqres, f)

    # delete tmp file
    if os.path.exists(fname + '.tmp'):
        os.remove(fname + '.tmp')

    return uqres

def loadSamples(config, uqres, model):
    if ('Xinit' not in config) or ('Yinit' not in config):
        Xinit = None
        Yinit = None
    else:
        Xinit = config['Xinit']
        Yinit = config['Yinit']

    if config['Xinit'] == 'this':
        Xtmp = copy.deepcopy(uqres['uqres']['sampling']['result']['x'])
        if Xtmp.shape[1] == model.nInputS:
            Xinit = Xtmp
        elif Xtmp.shape[1] == model.nInput:
            Xinit = Xtmp[:,model.paraidx]
        else:
            raise RuntimeError('ERROR: the dimension of Xinit is inconsistent with nInputS!')
    if config['Yinit'] == 'this':
        Ytmp = copy.deepcopy(uqres['uqres']['sampling']['result']['y'])
        if Ytmp.shape[1] == model.nOutput:
            Yinit = Ytmp
        else:
            raise RuntimeError('ERROR: the dimension of Yinit is inconsistent with nOutput!')
            
    if (Xinit is not None) and (Yinit is not None) and (Xinit.shape[0] != Yinit.shape[0]):
        raise RuntimeError('ERROR: the length of Xinit and Yinit must be the same!')
    return Xinit, Yinit
