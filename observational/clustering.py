import numpy as np
import pandas as pd
import networkx as nx
import string
import copy
import random
import itertools
import nltk
nltk.download('stopwords')
    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter



def clean(s):
    """
    Cleans keyword phrases
    
    0. Splits into key phrases
    1. Drops stopwords, punctuation, spaces
    2. Forces to lowercase
    
    """
    #words = word_tokenize(s)
    if '\n' in s:
        kws = s.split('\n')
    elif ';' in s:
        kws = s.split(';')
    else:
        kws = s
        
    #words = s.strip().replace('\t', '').replace(' ',"").split('\n')
    kws = [w.lower() for w in kws]
    
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    
    kws2 = []
    for kw in kws:
        words = kw.strip().replace('\t', ' ').split(' ')
        words = [w.translate(table) for w in words]  #remove punctuation
        words = [w for w in words if w.isalpha()]  #remove non-alphanumeric
        words = [w for w in words if not w in stop_words]
        phrase = ''.join(words)
        if len(phrase)>0:
            kws2.append(''.join(words))
        
    
    words = list(set(kws2))  # make unique
    return words


def shuffle_keywords(kw_df, n=100000):
    """Returns a new df with node locations shuffled"""
    #kw_df2 = kw_df.copy()
    kw_df2 = pd.DataFrame(columns=kw_df.columns, data=copy.deepcopy(kw_df.values))
    # shuffle keys
    for i in range(n):
        source, swap = np.random.choice(list(kw_df2.index), 2, replace=False)  # choose papers to swap
        if len(kw_df2.loc[source]['keys']) > 0 and len(kw_df2.loc[swap]['keys']) > 0:
            a = np.random.choice(kw_df2.loc[source]['keys'])
            b = np.random.choice(kw_df2.loc[swap]['keys'])
            if (a not in kw_df2.loc[swap]['keys']) and (b not in kw_df2.loc[source]['keys']):
                kw_df2.loc[swap]['keys'].remove(b)
                kw_df2.loc[source]['keys'].remove(a)
                kw_df2.loc[swap]['keys'].append(a)
                kw_df2.loc[source]['keys'].append(b)
            
    return kw_df2
            
def create_weighted_edgelist(kw_df, method="pairs"):
    """
    from each set of keywords, we could create edges either between *all* combinations,
    or just use each kw in a single *pair*. The second is more conservative, as we 
    can't create clusters from a single paper.
    
    This shouldn't be too important, as the popular edges are the ones that matter, 
    and they will show up in either strategy.
    """
    edges = Counter()
    for _, row in kw_df.iterrows():
        if len(row['keys']) >=2:
            
            if method == "all":
                for a in itertools.combinations(row['keys'], r=2):
                    sa = sorted(a)
                    edges[sa[0], sa[1]] +=1
                    
            elif method == "pairs":
                np.random.shuffle(row['keys'])
                for i in range(0,len(row['keys']),2):
                    sa = sorted(row['keys'][i:i+2])
                    if len(sa) == 2:
                        edges[sa[0], sa[1]] +=1
                    
    return edges
    
def shuffle_edgelist(edges):
    counts = list(edges.values())
    random.shuffle(counts)
    shuffle_edges = Counter({k:v for k,v in zip(edges.keys(), counts)})
    return shuffle_edges

def compute_clusterings_by_threshold(edges, thresholds=np.arange(0.01,1.01,.05)):
     # compute clustering by threshold
    clusterings = []
    for threshold in thresholds:
        selected_edges = [edge for (edge, count) in edges.most_common(int(threshold*len(edges)))]
        g = nx.from_edgelist(selected_edges)
        try:
            clusterings.append(nx.average_clustering(g))
        except:
            clusterings.append(0)
    return pd.Series(clusterings, index=thresholds)