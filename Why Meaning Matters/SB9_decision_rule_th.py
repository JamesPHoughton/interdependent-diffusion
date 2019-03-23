import model
import pandas as pd
import numpy as np
import multiprocessing
import itertools


n_seeds = np.unique(np.floor(np.logspace(np.log10(10), np.log10(150), 40)))
thresholds = [1, .99, .8, .6, .4]
repeat = 5
argslist = [('random', 40, 4, n, 30, 2, th, 'average', 'shuffle') 
            for (th, n) in itertools.product(thresholds, n_seeds)] * repeat

rand_key = np.random.randint(100000)

for i in range(100000):
    print(i, end=', ')
    with multiprocessing.Pool() as pool:
        res = pool.map(model.point, argslist)

    df = pd.DataFrame(res).join(
        pd.DataFrame(argslist, columns=['Network', 'N Agents', 'Mean Degree', 
                                        'N Starting Beliefs', 'N Concepts',
                                        'Pathlength', 'Threshold', 'Clustering', 'Randomization']))

    pd.to_pickle(df, 'data/sensitivity_to_decision_rule_threshold_df_%i_%s.pickle' % (rand_key, str(i).zfill(4)))
    
