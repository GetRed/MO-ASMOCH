# This python file stores config information for uq workflow

uqcfg = {
    
    # 'optimization' stores optimization method name and configration
    # 'method': optimization method name (must have)
    # 'config': config information of the optimization method (optional)
    #  will be different for different optimization methods
    'optimization': {
        'method': 'NSGA2_mixint',
        'config': {
            # pop: number of population (optional, default = 100)
            'pop': 50,
            # gen: number of generation (optional, default = 100)
            'gen': 49,
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
    'resultname': 'NSGAm'
}
