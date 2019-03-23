import model
import pandas as pd
import numpy as np
import multiprocessing
import itertools

repeat = 500
argslist = [('random', 40, 4, 25, 30, 2, 1.1, 'average', 'shuffle')] * repeat

rand_key = np.random.randint(100000)

for i in range(100000):
    print(i, end=', ')
    with multiprocessing.Pool() as pool:
        res = pool.map(model.sim, argslist)

    pd.to_pickle(res, 'data/main_article_df_list_%i_%s.pickle' % (rand_key, str(i).zfill(4)))