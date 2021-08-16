# Nondominated Sorting Genetic Algorithm II (NSGA-II)
# An multi-objective optimization algorithm
# mixed-integer version
# modified by sunrc  2021-04
from __future__ import division, print_function, absolute_import
import sampling
import numpy as np
import copy

def optimization(model, nInput, nOutput, xlb, xub, intstart, discrete, pop, gen, \
                 crossover_rate = 0.9, mu = 20, mum = 20):
    ''' Nondominated Sorting Genetic Algorithm II, An multi-objective algorithm
        model: the evaluated model function
        nInput: number of model input
        nOutput: number of output objectives
        xlb: lower bound of input
        xub: upper bound of input
        pop: number of population
        gen: number of generation
        crossover_rate: ratio of crossover in each generation
        mu: distribution index for crossover
        mum: distribution index for mutation
    '''
    poolsize = int(round(pop/2.)); # size of mating pool;
    toursize = 2;                  # tournament size;

    xl = xlb.copy()
    xu = xub.copy()

    xl[intstart-1:] = xl[intstart-1:] - (0.5 - 1e-16)
    xu[intstart-1:] = xu[intstart-1:] + (0.5 - 1e-16)

    x = sampling.lh(pop, nInput)
    x = x * (xub - xlb) + xlb
    x[:,intstart-1:] = np.rint(x[:,intstart-1:])
    x_ori = x.copy()
    for i in range(intstart-1,nInput):
    # for i in range(intstart,nInput):  # for RE25
        for j in range(0,pop):
            x_ori[j,i] = discrete[x[j,i].astype(np.int)-1]

    y = np.zeros((pop, nOutput))
    for i in range(pop):
        y[i,:] = model.evaluate(x_ori[i,:])
    icall = pop

    x, y, rank, crowd = sortMO(x, y, nInput, nOutput)
    population_para = x.copy()
    population_obj  = y.copy()

    for i in range(gen):
         ## tournament selection
        perms = []
        for i in range(2):
            perms.append(np.random.permutation(pop))
        P = np.concatenate(perms)
        P = P.reshape(pop,2)
        S = P.min(axis = 1)
        parentidx = S.reshape(poolsize, 2)
        
        ## crossover
        popchild = population_para[S]
        poptmp = []
        for j in range(parentidx.shape[0]):
            parent1   = population_para[parentidx[j,0],:]
            parent2   = population_para[parentidx[j,1],:]
            child1, child2 = crossover(parent1, parent2, mu, xlb, xub, intstart)
            poptmp.append(child1)
            poptmp.append(child2)
        popX = np.array(poptmp)
        do_crossover = np.random.random(pop) < crossover_rate
        popchild[do_crossover,:] = popX[do_crossover,:]

        ## mutation
        popchild = mutation(popchild, mum, xlb, xub, intstart)

        ##  evaluation new children
        for k in range(pop):
            child  = popchild[k,:]
            child_o   = child.copy()
            for j in range(intstart-1,nInput):
            # for j in range(intstart,nInput):  # for RE25
                child_o[j] = discrete[child[j].astype(np.int)-1]
            y1 = model.evaluate(child_o)
            x  = np.vstack((x,child))
            y  = np.vstack((y,y1))
            population_para = np.vstack((population_para,child))
            population_obj  = np.vstack((population_obj,y1))

        ## remove the duplicates
        population_para, idx=np.unique(population_para, axis=0, return_index=True)
        population_obj = population_obj[idx]


        population_para, population_obj, rank, crowd = \
            remove_worst(population_para, population_obj, pop, nInput, nOutput)

        population_para, population_obj, ranktmp, crowdtmp = sortMO(population_para, \
                                                    population_obj, nInput, nOutput)
    
    bestx = population_para.copy()
    besty = population_obj.copy()

    return bestx, besty, x, y

def sortMO(x, y, nInput, nOutput):
    ''' Non domination sorting for multi-objective optimization
        x: input parameter matrix
        y: output objectives matrix
        nInput: number of input
        nOutput: number of output
    '''
    rank, dom = fast_non_dominated_sort(y)
    idxr = rank.argsort(kind='stable')
    rank = rank[idxr]
    x = x[idxr,:]
    y = y[idxr,:]
    T = x.shape[0]

    crowd = np.zeros(T)
    rmax = int(rank.max())
    idxt = np.zeros(T, dtype = np.int)
    c = 0
    for k in range(rmax+1):
        rankidx = (rank == k)
        D = crowding_distance(y[rankidx,:])
        idxd = D.argsort()[::-1]
        crowd[rankidx] = D[idxd]
        idxtt = np.array(range(len(rank)))[rankidx]
        idxt[c:(c+len(idxtt))] = idxtt[idxd]
        c += len(idxtt)
    x = x[idxt,:]
    y = y[idxt,:]
    rank = rank[idxt]

    return x, y, rank, crowd

def fast_non_dominated_sort(Y):
    ''' a fast non-dominated sorting method
        Y: output objective matrix
    '''
    N, d = Y.shape
    Q = [] # temp array of Pareto front index
    Sp = [] # temp array of points dominated by p
    S = [] # temp array of Sp
    rank = np.zeros(N) # Pareto rank
    n = np.zeros(N)  # domination counter of p
    dom = np.zeros((N, N))  # the dominate matrix, 1: i doms j, 2: j doms i

    # compute the dominate relationship online, much faster
    for i in range(N):
        for j in range(N):
            if i != j:
                if dominates(Y[i,:], Y[j,:]):
                    dom[i,j] = 1
                    Sp.append(j)
                elif dominates(Y[j,:], Y[i,:]):
                    dom[i,j] = 2
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            Q.append(i)
        S.append(copy.deepcopy(Sp))
        Sp = []

    F = []
    F.append(copy.deepcopy(Q))
    k = 0
    while len(F[k]) > 0:
        Q = []
        for i in range(len(F[k])):
            
            p = F[k][i]
            for j in range(len(S[p])):
                q = S[p][j]
                n[q] -= 1
                if n[q] == 0:
                    rank[q]  = k + 1
                    Q.append(q)
        k += 1
        F.append(copy.deepcopy(Q))

    return rank, dom

def dominates(p,q):
    ''' comparison for multi-objective optimization
        d = True, if p dominates q
        d = False, if p not dominates q
        p and q are 1*nOutput array
    '''
    if sum(p > q) == 0:
        d = True
    else:
        d = False
    return d

def crowding_distance(Y):
    ''' compute crowding distance in NSGA-II
        Y is the output data matrix
        [n,d] = size(Y)
        n: number of points
        d: number of dimentions
    '''
    n,d = Y.shape
    lb = np.min(Y, axis = 0)
    ub = np.max(Y, axis = 0)

    idx = Y.argsort(axis = 0, kind='stable')    
    Y   = Y[idx, np.arange(d)]
    # calculate the distance from each point to the last and next
    dist = np.row_stack([Y, np.full(d, np.inf)]) - np.row_stack([np.full(d, -np.inf), Y])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = ub - lb
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(idx, axis=0)
    D = np.sum(dist_to_last[J, np.arange(d)] + dist_to_next[J, np.arange(d)], axis=1) 

    return D

def mutation(X, mum, xlb, xub, intstart):
    ''' Polynomial Mutation in Genetic Algorithm
        For more information about PMut refer the NSGA-II paper.
        mum: distribution index for mutation, default = 20
            This determine how well spread the child will be from its parent.
        parent: sample point before mutation
	'''

    Y = np.full(X.shape, np.inf)
    Y = X

    do_mutation = np.random.random(X.shape) < 0.1 
    xl = np.repeat(xlb[None, :], X.shape[0], axis=0)[do_mutation]
    xu = np.repeat(xub[None, :], X.shape[0], axis=0)[do_mutation]
    X = X[do_mutation]
    delta1 = (X - xl) / (xu - xl)
    delta2 = (xu - X) / (xu - xl)

    mut_pow = 1.0 / (mum + 1.0)
    rand = np.random.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)
    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (mum + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (mum + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (xu - xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < xl] = xl[_Y < xl]
    _Y[_Y > xu] = xu[_Y > xu]

    # set the values for output
    Y[do_mutation] = _Y
    Y[:,intstart-1:] = np.rint(Y[:,intstart-1:])
    return Y

def crossover(parent1, parent2, mu, xlb, xub, intstart):
    ''' SBX (Simulated Binary Crossover) in Genetic Algorithm
        For more information about SBX refer the NSGA-II paper.
        mu: distribution index for crossover, default = 20
        This determine how well spread the children will be from their parents.
    '''
    n      = len(parent1)
    beta   = np.ndarray(n)
    child1 = np.ndarray(n)
    child2 = np.ndarray(n)
    u = np.random.rand(n)
    for i in range(n):
        if (u[i] <= 0.5):
            beta[i] = (2.0*u[i])**(1.0/(mu+1))
        else:
            beta[i] = (1.0/(2.0*(1.0 - u[i])))**(1.0/(mu+1))
        child1[i] = 0.5*((1-beta[i])*parent1[i] + (1+beta[i])*parent2[i])
        child2[i] = 0.5*((1+beta[i])*parent1[i] + (1-beta[i])*parent2[i])
    child1[intstart-1:] = np.rint(child1[intstart-1:])
    child2[intstart-1:] = np.rint(child2[intstart-1:])
    child1 = np.clip(child1, xlb, xub)
    child2 = np.clip(child2, xlb, xub)
    return child1, child2

def selection(population_para, population_obj, rank, crowd, nInput, pop, poolsize, toursize):
    ''' tournament selecting the best individuals into the mating pool'''
    pool    = np.zeros([poolsize,nInput])
    poolidx = np.zeros(poolsize)
    count   = 0
    while (count < poolsize-1):
        candidate = np.random.choice(pop, toursize, replace = False)
        idx = candidate.min()
        if not(idx in poolidx):
            poolidx[count] = idx
            pool[count,:]  = population_para[idx,:]
            count += 1
    return pool

def remove_worst(population_para, population_obj, pop, nInput, nOutput):
    ''' remove the worst individuals in the population '''
    population_para, population_obj, rank, crowd = \
        sortMO(population_para, population_obj, nInput, nOutput)
    return population_para[0:pop,:], population_obj[0:pop,:], rank[0:pop], crowd[0:pop]

