# This python file stores config information for uq workflow

uqcfg = {
    # 'sampling': {
    #     'method': 'LH',
    #     'config': {
    #         'nSample': 100
    #     }
    # },

    # 'optimization' stores optimization method name and configration
    # 'method': optimization method name (must have)
    # 'config': config information of the optimization method (optional)
    #  will be different for different optimization methods
    'optimization': {
        'method': 'MOASMO_mixint',
        'config': {
            # Xinit/Yinit: initial samplers for surrogate model construction (optional, default = None)
            # set to 'this': use the samples generated in the 'sampling' section of this config file
            # set to a customized numpy array: use samples generated from other programs
            # 'Xinit': 'this',
            # 'Yinit': 'this',
            'Xinit': None,
            'Yinit': None,
            # niter: number of iteration (optional, defualt = 5)
            'niter': 5,
            # pct: percentage of resampled points in each iteration (optional, default = 0.2 (means 20%))
            'pct': 0.2,
            # pop: number of population (optional, default = 100)
            'pop': 100,
            # gen: number of generation (optional, default = 100)
            'gen': 100,
            # crossover_rate: ratio of crossover in each generation (optional, default = 0.9)
            'crossover_rate': 0.9,
            # mu: distribution index for crossover (optional, default = 20)
            'mu': 20,
            # mum: distribution index for mutation (optional, default = 20)
            'mum': 20
            }
        },

    # 'resultpath' is the path for storing uq results
    # WARNING: if use relative path, this is relative to the 'workpath' of model, not current path of this file
    # use absolute path if possible
    'resultpath': '.',

    # 'resultname' is the name of bin file dumped by pickle
    'resultname': 'MOASMO'
}
