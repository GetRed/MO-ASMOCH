# This python file stores config information for simulation model
# by: sunrc, 2021-04
import numpy as np
import random

# 'modelcfg' is a dict stores the config information for dynamic model.
modelcfg = {
    # 'workpath' is the working path of the dynamic model,
    # each model evaluation have a workdir in the workpath,
    # which stores the evaluation results of the dynamic model with different parameters.
    # i.e. 'workdir.125' stores the dynamic model outputs of the 125th evaluation.
    # usually we set 'workpath' to current path '.'
    # WARNING: if use relative path, this is relative to the current dir of running python, not current path of this file
    # use absolute path if possible
    'workpath': '.',

    # 'driver' is the python module name that drives with the dynamic model.
    # (no .py file name extention)
    # It puts the parameter values in the model, launch the model,
    # and compute the objective function(s) after model completed.
    # this file is suggested to store in the 'workpath'
    'driver': 'testfun_mo',

    # 'parameters' stores the information about parameters, including:
    'parameters': [
        # 'idx':  parameter index (must have)
        # 'name': parameter name (must have)
        # 'lb':   lower bound (must have)
        # 'ub':   upper bound (must have)
        # 'dft':  default value (optional)
        #  'None' means not set, 'dft' will be set to mean value of lb and ub
        # 'dist': priot parameter distribution (optional)
        #  should be a python file returns the probability density function
        #  'None' means not set, 'dist' will be set to uniform distribution

        ## RE22
        {'idx': 0, 'name': 'x1', 'lb': 0.001, 'ub': 20.0, 'dft': None, 'dist': 'uniform'},
        {'idx': 1, 'name': 'x2', 'lb': 0.0, 'ub': 40.0, 'dft': None, 'dist': 'uniform'},
        {'idx': 2, 'name': 'x3', 'lb': 1.0, 'ub': 77.0, 'dft': None, 'dist': 'uniform'}

        ## RE23
        # {'idx': 0, 'name': 'x1', 'lb': 10.0, 'ub': 200.0, 'dft': None, 'dist': 'uniform'},
        # {'idx': 1, 'name': 'x2', 'lb': 10.0, 'ub': 240.0, 'dft': None, 'dist': 'uniform'},
        # {'idx': 2, 'name': 'x3', 'lb': 1.0, 'ub': 100.0, 'dft': None, 'dist': 'uniform'},
        # {'idx': 3, 'name': 'x4', 'lb': 1.0, 'ub': 100.0, 'dft': None, 'dist': 'uniform'}

        ## RE25
        # {'idx': 0, 'name': 'x1', 'lb': 0.6, 'ub': 30.0, 'dft': None, 'dist': 'uniform'},
        # {'idx': 1, 'name': 'x2', 'lb': 1.0, 'ub': 70.0, 'dft': None, 'dist': 'uniform'},
        # {'idx': 2, 'name': 'x3', 'lb': 1.0, 'ub': 42.0, 'dft': None, 'dist': 'uniform'}
        ],
        
    ## RE22
    'discretevalue': np.array([0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0, 1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0, 2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08, 3,10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20, 4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72, 6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69, 9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0]),
    ## RE23
    # 'discretevalue': 0.0625*np.linspace(1,100,100),
    ## RE25
    # 'discretevalue': np.array([0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162, 0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047, 0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162, 0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362, 0.394, 0.4375, 0.5]),

    'intstart': 3, # for RE22 and RE23
    # 'intstart': 2,  # for RE25

    # 'objectives' stores the information about objective functions, including:
    'objectives': [
        # 'idx':  objective function index (must have)
        # 'name': objective function name (must have)
        # 'dft':  defualt objective value (optional)
        # 'None' means not set
        # 'screened': screened sensitive (important) parameters for each objective (optional)
        # 'None' means not set, 'screened' will be set to all parameters
        {'idx': 0, 'name': 'f1', 'dft': 200, 'screened': None},
        {'idx': 1, 'name': 'f2', 'dft': 100, 'screened': None}
        ],

    # 'parallel' is the option about parallel evaluation
    # should be True or False
    # if True, dynamic model will be evaluated parallelly with multiprocessing tool
    'parallel': False,
    # 'processes' is the number of processes, activated if 'parallel' is True
    'processes': 4,

    # 'usescreened' is the option about use or not use screened parameters
    # if True, only use the screened parameter; if False, use all parameters
    'usescreened': False,

    # 'usemultiobj' is the option about multi-objective optimization
    # if True, use multiple objectives; if False, use only one objective
    'usemultiobj': True
}

