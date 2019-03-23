import model
import pandas as pd
import numpy as np
import multiprocessing
import itertools

clustering = ['shuffle', 'concepts', 'beliefs']
n_seeds = np.unique(np.floor(np.logspace(np.log10(10), np.log10(150), 40)))
repeat = 50
argslist = [('random', 40, 4, n, 30, 2, 1.1, 'average', 'shuffle') 
            for (cluster, n) in itertools.product(clustering, n_seeds)] * repeat

rand_key = np.random.randint(100000)

for i in range(100000):
    print(i, end=', ')
    with multiprocessing.Pool() as pool:
        res = pool.map(model.point, argslist)

    df = pd.DataFrame(res).join(
        pd.DataFrame(argslist, columns=['Network', 'N Agents', 'Mean Degree', 
                                        'N Starting Beliefs', 'N Concepts',
                                        'Pathlength', 'Threshold', 'Clustering', 'Randomization']))

    pd.to_pickle(df, 'data/sensitivity_to_randomization_df_%i_%s.pickle' % (rand_key, str(i).zfill(4)))