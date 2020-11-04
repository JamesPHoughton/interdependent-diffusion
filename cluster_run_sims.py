import model
import pandas as pd
import numpy as np
import multiprocessing
import itertools

repeat = 500


network = 'random'
n_agents = 60
deg = 3
n_beliefs = 25
n_concepts = 25
pathlength = 2
threshold = 1.1
clustering = 'average'
randomization = 'shuffle'

argslist = [(network, n_agents, deg, n_beliefs, n_concepts, pathlength, threshold, clustering, randomization)] * repeat

res = model.sim(argslist[0])
print(res) # check that its working before you run the whole set

rand_key = np.random.randint(100000)

for i in range(100000):
    print(i, end=', ')
    with multiprocessing.Pool() as pool:
        res = pool.map(model.sim, argslist)

    pd.to_pickle(res, 'data/main_article_df_list_%s.pickle' % (str(i).zfill(2)))
