import pandas as pd
from itertools import product
def call_paramset(filename,id):
    # basic processing
    data = pd.read_csv(filename, header=None)
    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(0)   
    paramdf = data.iloc[id].to_dict()
    paramdflist = []
    # if there are semicolons, separate them and make all combinations of paramdf
    keys = paramdf.keys()
    tunekeys = []
    tunekeyvals = []
    # get which keys have multiple values to try out for tuning
    for key in keys:
        if key=='notes' or key=='score':
            continue
        if ';' in paramdf[key]:
            tunekeys.append(key)
            vals = paramdf[key].split(';')
            tunekeyvals.append(vals)
    # Generate all combinations of tuning parameters
    for combination in product(*tunekeyvals):
        temp_paramdf = paramdf.copy()
        for i, key in enumerate(tunekeys):
            temp_paramdf[key] = combination[i]
        paramdflist.append(temp_paramdf)
    return paramdflist


from pprdyn1 import pprdyn1
def call_env(param):
    config = eval(param['envconfig'])
    if param['envid'] == 'metapop1':
        return metapop1(config)
    elif param['envid'] == 'pprdyn1':
        return pprdyn1(config)
    else:
        raise ValueError("Unknown environment ID")
        