import networkx as nx
import numpy as np
import itertools
import copy
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import random
from sklearn.decomposition import PCA
from scipy.stats import kurtosis


def setup(network = 'random',  # ['complete', 'random', 'Watts-Strogatz', 'connected caveman', 'Barabasi-Albert'] 
          n_agents=40,  # number of agents 
          deg=4,  # number of connections for each agent
          n_beliefs=25,  # number of knowledge graph elements each agent has
          n_concepts=30
    ):
    """
    Generates the initial conditions of the simulation

    Returns
    -------
    g: networkx graph
        primary graph represents the semantic network,
        each individual (node) has an attribute 'M' representing their semantic network

    all_beliefs: an array of tuples, which represents all edges anywhere in the semantic
        network network of any individual. Does not include edges that could be part of the
        semantic network (because they are present in a complete graph with `n_concepts`, 
        but were not selected).
    """
    np.random.seed()

    connected = False
    while not connected:
        if network == 'complete':
            g = nx.complete_graph(n=n_agents)
        elif network == 'random':
            g = nx.gnm_random_graph(n=n_agents, m=int(n_agents*deg/2))
        elif network == 'random regular':
            g = nx.random_regular_graph(d=deg, n=n_agents)
        elif network == 'Watts-Strogatz':
            g = nx.connected_watts_strogatz_graph(n=n_agents, k=deg, p=.02)  # valid for even deg
        elif network == 'connected caveman':
            g = nx.connected_caveman_graph(l=int(n_agents/deg), k=deg+1)
        elif network == 'Barabasi-Albert': 
            g = nx.barabasi_albert_graph(n=n_agents, m=int(deg/2))  # approximates deg for large n_agents, when deg is even.
        else:
            raise ValueError('%s is not a valid network name' % network)

        connected = nx.is_connected(g)

    # give starting information
    nx.set_node_attributes(
        g, 
        name='M',  # M for mind
        values={i: nx.gnm_random_graph(n_concepts, n_beliefs) for i in g})

    beliefs = np.unique([tuple(sorted(belief)) for agent in g 
                         for belief in g.nodes[agent]['M'].edges()], axis=0)
    return g, beliefs

def fast_adopt(g, ego, edge):
    """ Fast adoption function for triangle closing rule"""
    try:
        from_neighbors = set(g.nodes[ego]['M'][edge[0]])  # if concept 0 not in network, false
        if edge[1] in from_neighbors:  # edge already exists
            return False
        to_neighbors = set(g.nodes[ego]['M'][edge[1]])  # if concept 1 not in network, false
        if from_neighbors & to_neighbors:  # if susceptible
            for nb in g[ego]:  # check exposed
                if edge in g.nodes[nb]['M'].edges():
                    return True
        return False
    except:
        return False

def fast_susceptible(g, ego, edge):
    """ Fast check for 'susceptible or already adopted' under triangle closing rule"""
    try:
        from_neighbors = set(g.nodes[ego]['M'][edge[0]])  # if concept not in network, raise
        if edge[1] in from_neighbors:  # edge already exists
            return True
        to_neighbors = set(g.nodes[ego]['M'][edge[1]])
        if from_neighbors & to_neighbors:
            return True
        return False
    except:
        return False
    
def general_adopt(g, ego, edge, pl=2, th=1.1):
    """ 
    Expands adoption function to general case where semantic path lenth is 
    less than or equal to pl, or a fraction of neighbors greater than th believes
    """
    
    try:  # there may be no path between the nodes
        path_length = nx.shortest_path_length(g.nodes[ego]['M'], *edge)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        path_length = 1000  # assume that length is very large
    exposure = np.mean([edge in g.nodes[nb]['M'].edges() for nb in g[ego]])
    #exposure = np.sum([edge in g.nodes[nb]['M'].edges() for nb in g[ego]]) 
    return (1 < path_length <= pl and exposure > 0) or (exposure > th) 

def general_susceptible(g, ego, edge, pl=2):
    """ Expands susceptible check to general case for semantic path length <= pl """
    try:  # there may be no path between the nodes
        length = nx.shortest_path_length(g.nodes[ego]['M'], *edge)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        length = 1000  # assume that length is very large
    return 1 < length <= pl

def simulate_simultaneous(g, beliefs, n_steps=100, adopt=fast_adopt):
    for step in range(n_steps):
        changes = False
        for ego in np.random.permutation(g):  # select ego in random order
            for edge in np.random.permutation(beliefs):  # select a belief in random order to propagate
                if adopt(g, ego, edge):
                    changes = True
                    g.nodes[ego]['M'].add_edges_from([edge])
        if not changes:
            break

def simulate_individual(g, edge, n_steps=100, adopt=fast_adopt):
    # simulates the diffusion of a single edge in the given network
    for step in range(n_steps):
        changes = False
        for ego in np.random.permutation(g):  # select ego in random order            
            if adopt(g, ego, edge):
                g.nodes[ego]['M'].add_edges_from([edge])
                changes = True
        if not changes:
            return None

def simulate_independent_match_susceptibility(g, sus_target, n_steps, n_concepts, beliefs):
    # sus_target: average fraction of people to make susceptible to each belief (0-1)
    ex_sus = copy.deepcopy(g)  # exogenous susceptibility

    # give starting information
    nx.set_node_attributes(
        ex_sus,
        name='S',  # S for susceptibility
        values={i: nx.gnp_random_graph(n_concepts, sus_target) for i in g})

    s = {tuple(b): 0 for b in beliefs}
    for n in ex_sus:
        for b in beliefs:
            s[tuple(b)] += ex_sus.nodes[n]['S'].has_edge(*b) or g.nodes[n]['M'].has_edge(*b)

    for step in range(n_steps):
        yield step, g, s
        for ego in np.random.permutation(g):
            for edge in np.random.permutation(beliefs):
                if ex_sus.nodes[ego]['S'].has_edge(*edge):  # check susceptible
                    for nb in g[ego]:  # check exposed
                        if edge in g.nodes[nb]['M'].edges():
                            g.nodes[ego]['M'].add_edges_from([edge])

def simulate_independent(g, n_steps, beliefs):
    # sus_target: average fraction of people to make susceptible to each belief (0-1)
    ex_sus = copy.deepcopy(g)  # exogenous susceptibility

    # give starting information
    nx.set_node_attributes(
        ex_sus,
        name='S',  # S for susceptibility
        values={i: nx.from_edgelist([b for b in beliefs if fast_susceptible(g, i, b)]) for i in g})

    for step in range(n_steps):
        yield step, g
        for ego in np.random.permutation(g):
            for edge in np.random.permutation(beliefs):
                if ex_sus.nodes[ego]['S'].has_edge(*edge):  # check susceptible
                    for nb in g[ego]:  # check exposed
                        if edge in g.nodes[nb]['M'].edges():
                            g.nodes[ego]['M'].add_edges_from([edge])


def simulate_sequential(g, beliefs, n_steps=100, adopt=fast_adopt):
    for edge in np.random.permutation(beliefs):  # select a belief in random order to propagate
        for step in range(n_steps):
            changes = False
            for ego in np.random.permutation(g):  # select ego in random order            
                if adopt(g, ego, edge):
                    g.nodes[ego]['M'].add_edges_from([edge])
                    changes = True
            if not changes:
                break        


def measure_diffusion(g, beliefs):
    """what is the extent of diffusion of all belief in beliefs? 
    Returns a dictionary where keys are tuples representing beliefs
    """
    return {tuple(edge): np.sum([edge in g.nodes[nb]['M'].edges() for nb in g]) for edge in beliefs}  

def measure_susceptibility(g, beliefs, susceptible=fast_susceptible):
    susceptibility = {}
    for edge in beliefs:
        susceptibility[tuple(edge)] = np.sum([susceptible(g, agent, edge) for agent in g])
    return susceptibility

def randomize_beliefs(g, n_concepts=None, beliefs=None):
    """
    If `beliefs` is a list of all the beliefs in the network, draw from these in 
    equal measure to the number that an individual already has. The problem with this is
    that it will end up with fewer beliefs in the overall semantic network.
    
    If `n_concepts` is the number of concepts that are present in the semantic network,
    then randomly draw from the complete semantic network with this many concepts.
    The problem with this is that it will end up with more unique beliefs than exist
    in the network when diffusion has taken place.
    
    When neither are provided, shuffles the existing beliefs amongst individuals.
    The problem with this is that it's slow, and it's not truly random, as you bias 
    an agent against having the beliefs that they started with, so this
    still has some structure in it. However, it has the same number of beliefs, and each
    belief is held by the same number of people, so it's a conservative case.
    """
    np.random.seed()
    g2 = copy.deepcopy(g)
    if beliefs is not None:
        for n in g2:
            n_beliefs = g2.nodes[n]['M'].number_of_edges()
            new_beliefs = beliefs[np.random.choice(list(range(len(beliefs))), 
                                                   size=n_beliefs, replace=False)]
            g2.nodes[n]['M'] = nx.from_edgelist(new_beliefs)
    elif n_concepts is not None:
        for n in g2:
            n_beliefs = g2.nodes[n]['M'].number_of_edges()
            g2.nodes[n]['M'] = nx.gnm_random_graph(n_concepts, n_beliefs)     
    else:
        #Shuffle the beliefs between each individual. Guarantees taht there are the
        #same number of beliefs in the universe, each belief is diffused the same number
        #of times, and each agent has the same number of beliefs
        np.random.seed()
        g2 = copy.deepcopy(g)
        for agent in np.random.permutation(g2):
            agent_beliefs = g2.nodes[agent]['M'].edges()
            for belief in np.random.permutation(agent_beliefs):
                swap_belief = None
                for alter in np.random.permutation(g2): # look for a candidate
                    alter_beliefs = g2.nodes[alter]['M'].edges()
                    if tuple(belief) in set(alter_beliefs): # the alter cant have the belief we want to exchange
                        continue
                    candidates = set(alter_beliefs) - set(agent_beliefs)
                    if len(candidates) > 0: # the alter has to have at least one belief we have
                        swap_belief = list(candidates)[np.random.choice(list(range(len(candidates))))]
                        break
                if swap_belief is None: # no exchange is possible with this belief
                    continue
                g2.nodes[alter]['M'].remove_edges_from([swap_belief])
                g2.nodes[alter]['M'].add_edges_from([belief])
                g2.nodes[agent]['M'].remove_edges_from([belief])
                g2.nodes[agent]['M'].add_edges_from([swap_belief])
    return g2


def measure_belief_clustering_coefficient(diffusion, q=None, level=None, above=True):
    """Clustering coeff. of all beliefs above qth percentile"""
    if q is not None:
        # identify index representing the qth percentile belief
        thresh = int(np.ceil(len(diffusion)*(q/100)))
        # randomize the order of the diffusion dictionary so that 
        # the order doesn't suggest spurious clustering when
        # diffusion is uniform
        keys = list(diffusion.keys())
        random.shuffle(keys)
        shuff_diff = {key: diffusion[tuple(key)] for key in keys}
        # rank beliefs from least to most popular
        sortlist = sorted(shuff_diff, key=diffusion.get)
        # select the subset to keep
        edges = sortlist[thresh:] if above else sortlist[:thresh]
        
    elif level is not None:
        # select edges above a diffusion value
        edges = {k:v for (k,v) in diffusion.items() if v > level}
        if len(edges) == 0:
            return 0
        
    # create a subgraph with only the beliefs above 
    subgraph = nx.from_edgelist(edges)
    # measure clustering of the subgraph
    clustering = nx.average_clustering(subgraph)
    # measure modularity of the subgraph
    communities = nx.community.modularity_max.greedy_modularity_communities(subgraph)
    modularity = nx.community.modularity(subgraph, communities)
    return clustering, modularity

def measure_num_belief_clusters(diffusion):
    """Number of separable peaks in aggregate semantic network"""
    # identify unique values for extent of diffusion
    levels = set(diffusion.values())
    num_peaks = []
    for level in levels:
        # identify the edges that are above the level
        edges = [belief for (belief, adopters) in diffusion.items() 
                 if adopters >= level]
        # create a subgraph with only the beliefs above 
        subgraph = nx.from_edgelist(edges)
        # count the number of components in the subgraph
        num_peaks.append(nx.number_connected_components(subgraph))
    # return the maximum number of components discovered
    return np.max(num_peaks)

def measure_interpersonal_similarity(g):
    """Jaccard similarity between each pair of individuals"""
    jaccards = dict()
    # for each pair of agents in the simulation
    for a, b in itertools.combinations(g.nodes, r=2): 
        # identify the edges of each agent
        a_edges = set(g.nodes[a]['M'].edges())
        b_edges = set(g.nodes[b]['M'].edges())   
        # jaccard similarity is the intersection divided by the union
        intersect = len(a_edges.intersection(b_edges))
        union = len(a_edges.union(b_edges))
        jaccards[(a, b)] = intersect/union
    return jaccards

def measure_mean_interpersonal_similarity(jaccards, q, above=True):
    if above:
        # find out what index represents the qth percentile individual
        thresh = int(np.ceil(len(jaccards)*(q/100)))
        # average over all similarities above the qth percentile
        return np.mean(sorted(list(jaccards.values()))[thresh:])
    else:
        thresh = int(np.floor(len(jaccards)*(q/100)))
        return np.mean(sorted(list(jaccards.values()))[:thresh]) 

def measure_social_clusters_threshold(jaccards):
    """Number and size of social clusters"""
    # identify all unique values of similarity between pairings
    levels = set(jaccards.values())
    num_peaks = []
    gs = []
    for level in levels:
        # identify the pairings with similarity above the current level
        pairings = [pairing for (pairing, similarity) in jaccards.items() 
                    if similarity >= level]
        # create a subgraph with all pairings above the current level
        subgraph = nx.from_edgelist(pairings) 
        gs.append(subgraph)
        # count the number of separable components in the subgraph
        num_peaks.append(nx.number_connected_components(subgraph))

    # select the subgraph w/ max number separable components (factions)
    i = np.argmax(num_peaks)
    gq = gs[i]

    # measure the size of the average faction in the maximally separating subgraph
    mean_size = np.mean([len(faction) for faction in nx.connected_components(gq)])
    return num_peaks[i], mean_size

def measure_social_clusters_hierarchy(jaccards, method='average'):
    distances = 1-np.array(list(jaccards.values()))
    link = sch.linkage(distances, method=method)
    peaks = np.argwhere(link[:,3]==2).flatten()
    
    if len(peaks) > 1:
        sf = squareform(sch.cophenet(link))
        prominences = []
        for node, height in link[peaks,1:3]:
            distances = []
            for othernode in link[peaks,1]:
                if node == othernode:
                    continue
                distance = sf[int(node), int(othernode)]
                distances.append(distance)
            prominences.append(min(distances)-height)
        mean_prominence = np.mean(prominences)  
    else:
        mean_prominence = 0
    
    return len(peaks), mean_prominence

def point(args):
    
    network, n_agents, deg, n_beliefs, n_concepts, pathlength, threshold, clustering, randomization = args
    g, beliefs = setup(*args[:5])
    if pathlength == 2 and threshold > 1:
        adopt_func = fast_adopt
        susceptible_func = fast_susceptible
    else:
        adopt_func = lambda g, ego, edge: general_adopt(g, ego, edge, pl=pathlength, th=threshold)
        susceptible_func = lambda g, ego, edge: general_susceptible(g, ego, edge, pl=pathlength)
    
    res = dict()
    # initial conditions
    res['Initial susceptibility'] = np.mean(list(measure_susceptibility(g, beliefs, susceptible_func).values()))/n_agents*100
    res['Initial diffusion'] = np.mean(list(measure_diffusion(g, beliefs).values()))/n_agents*100

    # simultaneous diffusion
    g1 = copy.deepcopy(g)
    simulate_simultaneous(g1, beliefs, adopt=adopt_func)
    
    jaccards = measure_interpersonal_similarity(g1)
    res['RF top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
    res['RF num social clusters'], res['RF prominence of social clusters'] = \
        measure_social_clusters_hierarchy(jaccards, method=clustering)
    
    if randomization == 'concepts':
        g_random = randomize_beliefs(g1, n_concepts=n_concepts)
    elif randomization == 'beliefs':
        g_random = randomize_beliefs(g1, beliefs=beliefs)
    elif randomization == 'shuffle':
        g_random = randomize_beliefs(g1)
    jaccards = measure_interpersonal_similarity(g_random)
    res['Rand top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
    res['Rand num social clusters'], res['Rand prominence of social clusters'] = \
        measure_social_clusters_hierarchy(jaccards, method=clustering)
    
    res['RF susceptibility'] = np.mean(list(measure_susceptibility(g1, beliefs, susceptible_func).values()))/n_agents*100
    diffusion = measure_diffusion(g1, beliefs)
    res['RF diffusion'] = np.mean(list(diffusion.values()))/n_agents*100
    
    res['RF top decile clustering'] = measure_belief_clustering_coefficient(diffusion, 90)
    res['RF num semantic clusters'] = measure_num_belief_clusters(diffusion)
    
    
    # individual diffusion
    diffusion = dict()
    susceptibility = dict()
    for edge in beliefs: 
        g2 = copy.deepcopy(g)
        simulate_individual(g2, edge, adopt=adopt_func)
        diffusion.update(measure_diffusion(g2, [edge]))
        susceptibility.update(measure_susceptibility(g2, [edge], susceptible_func))
        
    res['NF susceptibility'] = np.mean(list(susceptibility.values()))/n_agents*100
    res['NF diffusion'] = np.mean(list(diffusion.values()))/n_agents*100
    
    res['NF top decile clustering'] = measure_belief_clustering_coefficient(diffusion, 90)
    res['NF num semantic clusters'] = measure_num_belief_clusters(diffusion)

    return res

def correlate_best(a_df):
    """ correlation of belief diffusion count with count of most diffused connected belief"""
    best_neighbors = []
    for c1 in a_df.columns:
        current_best = 0
        for c2 in a_df.columns:
            if c1 == c2:
                continue
            numnodes = len((set(c1) | set(c2)))
            if numnodes == 4:  # no shared nodes
                continue
            elif numnodes == 3:  # neighbors
                current_best = max(current_best, sum(a_df[c2]))
            else:
                print('same')
        best_neighbors.append((sum(a_df[c1]), current_best))

    return np.corrcoef([a for a, b in best_neighbors], [b for a, b in best_neighbors])[1,0]

def sim(args):
    network, n_agents, deg, n_beliefs, n_concepts, pathlength, threshold, clustering, randomization = args
    g, beliefs = setup(*args[:5])
    if pathlength == 2 and threshold > 1:
        adopt_func = fast_adopt
        susceptible_func = fast_susceptible
    else:
        adopt_func = lambda g, ego, edge: general_adopt(g, ego, edge, pl=pathlength, th=threshold)
        susceptible_func = lambda g, ego, edge: general_susceptible(g, ego, edge, pl=pathlength)
        
    res = pd.DataFrame(index=range(10))
    
    # simultaneous diffusion
    g1 = copy.deepcopy(g)
    s1 = measure_susceptibility(g1, beliefs, fast_susceptible)
    ai = measure_diffusion(g1, beliefs)

    for step in range(16):
        jaccards = measure_interpersonal_similarity(g1)
        #res.at[step, 'inter top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        #res.at[step, 'inter bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10, False)
        #res.at[step, 'inter 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        #res.at[step, 'inter 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        #res.at[step, 'inter num social clusters'], res.at[step, 'inter prominence of social clusters'] = \
        #    measure_social_clusters_hierarchy(jaccards, method=clustering)


        a_df = pd.DataFrame(index=g1.nodes(), columns=[tuple(b) for b in beliefs])
        for n in g1:
            for b in beliefs:
                a_df.at[n, tuple(b)] = g1.nodes[n]['M'].has_edge(*b)
        #res.at[step, 'inter interpersonal belief corr'] = a_df.astype(float).corr().abs().mean().mean()
        res.at[step, 'inter best connected belief corr'] = correlate_best(a_df)

        res.at[step, 'inter susceptibles'] = np.mean(
            list(measure_susceptibility(g1, beliefs, susceptible_func).values()))/n_agents*100
        diffusion = measure_diffusion(g1, beliefs)
        res.at[step, 'inter adopters'] = np.mean(list(diffusion.values()))/n_agents*100

        #res.at[step, 'inter initial suscep correlation'] = np.corrcoef(
        #    [s1[tuple(b)] for b in beliefs], [diffusion[tuple(b)] for b in beliefs])[1][0]
        res.at[step, 'inter net initial suscep correlation'] = np.corrcoef(
            [s1[tuple(b)] - ai[tuple(b)] for b in beliefs], [diffusion[tuple(b)] - ai[tuple(b)] for b in beliefs])[1][0]

        res.at[step, 'inter top decile semantic clustering'], res.at[step, 'inter top decile semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion, 90)
        #res.at[step, 'inter bottom decile semantic clustering'], res.at[step, 'inter bottom decile semantic modularity'] = \
        #    measure_belief_clustering_coefficient(diffusion, 10, above=False)
        #res.at[step, 'inter num semantic clusters'] = measure_num_belief_clusters(diffusion)

        corrs = a_df.astype(float).T.corr().mask(np.tri(n_agents, n_agents, 0, dtype='bool')).stack().sort_values()
        res.at[step, 'inter correlation similarity 95%'] = np.percentile(corrs, 95)
        #res.at[step, 'inter correlation similarity 90%'] = np.percentile(corrs, 90)
        #res.at[step, 'inter correlation similarity 50%'] = np.percentile(corrs, 50)
        #res.at[step, 'inter correlation similarity 10%'] = np.percentile(corrs, 10)
        res.at[step, 'inter correlation similarity 5%'] = np.percentile(corrs, 5)

        pca = PCA(n_components=n_agents)
        pca.fit(a_df)
        pca.fit(a_df)
        res.at[step, 'inter pca1 explained variance'] = pca.explained_variance_[0]
        res.at[step, 'inter pca2 explained variance'] = pca.explained_variance_[1]
        res.at[step, 'inter pca total variance'] = np.sum(pca.explained_variance_)
        pc1_pts = pca.transform(a_df)[:, 0]
        res.at[step, 'inter kurtosis of PC1'] = kurtosis(pc1_pts)


        # if randomization == 'concepts':
        #     g_random = randomize_beliefs(g1, n_concepts=n_concepts)
        # elif randomization == 'beliefs':
        #     g_random = randomize_beliefs(g1, beliefs=beliefs)
        # elif randomization == 'shuffle':
        #     g_random = randomize_beliefs(g1)
        # jaccards = measure_interpersonal_similarity(g_random)
        # res.at[step, 'inter rand top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        # res.at[step, 'inter rand 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        # res.at[step, 'inter rand bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10, False)
        # res.at[step, 'inter rand 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        # res.at[step, 'inter rand num social clusters'], res.at[step, 'inter rand prominence of social clusters'] = \
        #     measure_social_clusters_hierarchy(jaccards, method=clustering)
        #
        # a_df = pd.DataFrame(index=g_random.nodes(), columns=[tuple(b) for b in beliefs])
        # for n in g_random:
        #     for b in beliefs:
        #         a_df.at[n, tuple(b)] = g_random.node[n]['M'].has_edge(*b)
        # res.at[step, 'inter rand interpersonal belief corr'] = a_df.astype(float).corr().abs().mean().mean()
        # res.at[step, 'inter rand best connected belief corr'] = correlate_best(a_df)
        #
        # diffusion = measure_diffusion(g_random, beliefs)
        # res.at[step, 'inter rand top decile semantic clustering'], \
        # res.at[step, 'inter top decile semantic modularity'] = \
        #     measure_belief_clustering_coefficient(diffusion, 90)
        #
        # res.at[step, 'inter rand bottom decile semantic clustering'], \
        # res.at[step, 'inter bottom decile semantic modularity'] = \
        #     measure_belief_clustering_coefficient(diffusion, 10, above=False)
        #
        # res.at[step, 'inter num semantic clusters'] = measure_num_belief_clusters(diffusion)
        #
        # corrs = a_df.astype(float).T.corr().mask(np.tri(40, 40, 0, dtype='bool')).stack().sort_values()
        # res.at[step, 'inter rand correlation similarity 95%'] = np.percentile(corrs, 95)
        # res.at[step, 'inter rand correlation similarity 90%'] = np.percentile(corrs, 90)
        # res.at[step, 'inter rand correlation similarity 50%'] = np.percentile(corrs, 50)
        # res.at[step, 'inter rand correlation similarity 10%'] = np.percentile(corrs, 10)
        # res.at[step, 'inter rand correlation similarity 5%'] = np.percentile(corrs, 5)

        simulate_simultaneous(g1, n_steps=1, beliefs=beliefs, adopt=adopt_func)


    res['match susceptibles'] = res['inter susceptibles'].iloc[-1]
    # match final susceptibility
    for step, g4, s2 in simulate_independent_match_susceptibility(copy.deepcopy(g),
                                                        sus_target=res['inter susceptibles'].iloc[-1]/100,
                                                        n_steps=16,
                                                        n_concepts=n_concepts,
                                                        beliefs=beliefs):
        # jaccards = measure_interpersonal_similarity(g4)
        # res.at[step, 'match top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        # res.at[step, 'match bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10, False)
        # res.at[step, 'match 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        # res.at[step, 'match 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        # res.at[step, 'match num social clusters'], res.at[step, 'match prominence of social clusters'] = \
        #     measure_social_clusters_hierarchy(jaccards, method=clustering)

        a_df = pd.DataFrame(index=g4.nodes(), columns=[tuple(b) for b in beliefs])
        for n in g4:
            for b in beliefs:
                a_df.at[n, tuple(b)] = g4.nodes[n]['M'].has_edge(*b)
        # res.at[step, 'match interpersonal belief corr'] = a_df.astype(float).corr().abs().mean().mean()
        res.at[step, 'match best connected belief corr'] = correlate_best(a_df)

        diffusion = measure_diffusion(g4, beliefs)
        res.at[step, 'match top decile semantic clustering'], res.at[step, 'match top decile semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion, 90)
        # res.at[step, 'match bottom decile semantic clustering'], res.at[step, 'match bottom decile semantic modularity'] = \
        #     measure_belief_clustering_coefficient(diffusion, 10, above=False)
        # res.at[step, 'match num semantic clusters'] = measure_num_belief_clusters(diffusion)

        res.at[step, 'match adopters'] = np.mean(list(diffusion.values())) / n_agents * 100
        # res.at[step, 'match initial suscep correlation'] = np.corrcoef(
        #     [s2[tuple(b)] for b in beliefs], [diffusion[tuple(b)] for b in beliefs])[1][0]
        res.at[step, 'match net initial suscep correlation'] = np.corrcoef(
            [s2[tuple(b)] - ai[tuple(b)] for b in beliefs], [diffusion[tuple(b)] - ai[tuple(b)] for b in beliefs])[1][0]

        corrs = a_df.astype(float).T.corr().mask(np.tri(n_agents, n_agents, 0, dtype='bool')).stack().sort_values()
        res.at[step, 'match correlation similarity 95%'] = np.percentile(corrs, 95)
        # res.at[step, 'match correlation similarity 90%'] = np.percentile(corrs, 90)
        # res.at[step, 'match correlation similarity 50%'] = np.percentile(corrs, 50)
        # res.at[step, 'match correlation similarity 10%'] = np.percentile(corrs, 10)
        res.at[step, 'match correlation similarity 5%'] = np.percentile(corrs, 5)

        pca = PCA(n_components=n_agents)
        pca.fit(a_df)
        pca.fit(a_df)
        res.at[step, 'match pca1 explained variance'] = pca.explained_variance_[0]
        res.at[step, 'match pca2 explained variance'] = pca.explained_variance_[1]
        res.at[step, 'match pca total variance'] = np.sum(pca.explained_variance_)
        pc1_pts = pca.transform(a_df)[:, 0]
        res.at[step, 'match kurtosis of PC1'] = kurtosis(pc1_pts)

        # if randomization == 'concepts':
        #     g_random = randomize_beliefs(g4, n_concepts=n_concepts)
        # elif randomization == 'beliefs':
        #     g_random = randomize_beliefs(g4, beliefs=beliefs)
        # elif randomization == 'shuffle':
        #     g_random = randomize_beliefs(g4)
        # jaccards = measure_interpersonal_similarity(g_random)
        # res.at[step, 'match rand top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        # res.at[step, 'match rand 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        # res.at[step, 'match rand bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10, False)
        # res.at[step, 'match rand 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        # res.at[step, 'match rand num social clusters'], res.at[step, 'match rand prominence of social clusters'] = \
        #     measure_social_clusters_hierarchy(jaccards, method=clustering)
        #
        # a_df = pd.DataFrame(index=g_random.nodes(), columns=[tuple(b) for b in beliefs])
        # for n in g_random:
        #     for b in beliefs:
        #         a_df.at[n, tuple(b)] = g_random.node[n]['M'].has_edge(*b)
        # res.at[step, 'match rand interpersonal belief corr'] = a_df.astype(float).corr().abs().mean().mean()
        # res.at[step, 'match rand best connected belief corr'] = correlate_best(a_df)
        #
        # corrs = a_df.astype(float).T.corr().mask(np.tri(40, 40, 0, dtype='bool')).stack().sort_values()
        # res.at[step, 'match rand correlation similarity 95%'] = np.percentile(corrs, 95)
        # res.at[step, 'match rand correlation similarity 90%'] = np.percentile(corrs, 90)
        # res.at[step, 'match rand correlation similarity 50%'] = np.percentile(corrs, 50)
        # res.at[step, 'match rand correlation similarity 10%'] = np.percentile(corrs, 10)
        # res.at[step, 'match rand correlation similarity 5%'] = np.percentile(corrs, 5)

        # Independent
    for step, g5 in simulate_independent(copy.deepcopy(g),
                                                              n_steps=16,
                                                              beliefs=beliefs):
        # jaccards = measure_interpersonal_similarity(g5)
        # res.at[step, 'NF top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        # res.at[step, 'NF bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10, False)
        # res.at[step, 'NF 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        # res.at[step, 'NF 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        # res.at[step, 'NF num social clusters'], res.at[step, 'NF prominence of social clusters'] = \
        #     measure_social_clusters_hierarchy(jaccards, method=clustering)
        #
        # a_df = pd.DataFrame(index=g5.nodes(), columns=[tuple(b) for b in beliefs])
        # for n in g5:
        #     for b in beliefs:
        #         a_df.at[n, tuple(b)] = g5.node[n]['M'].has_edge(*b)
        # res.at[step, 'NF mean belief corr'] = a_df.astype(float).corr().abs().mean().mean()

        diffusion = measure_diffusion(g5, beliefs)
        res.at[step, 'NF diffusion'] = np.mean(list(diffusion.values())) / n_agents * 100
        res.at[step, 'NF susceptibles'] = res['inter susceptibles'].iloc[0]

        # res.at[step, 'NF top decile clustering'], res.at[step, 'RF top decile modularity'] = \
        #     measure_belief_clustering_coefficient(diffusion, 90)
        # res.at[step, 'NF bottom decile clustering'], res.at[step, 'RF bottom decile modularity'] = \
        #     measure_belief_clustering_coefficient(diffusion, 10, above=False)
        # res.at[step, 'NF num semantic clusters'] = measure_num_belief_clusters(diffusion)
        #
        # if randomization == 'concepts':
        #     g_random = randomize_beliefs(g5, n_concepts=n_concepts)
        # elif randomization == 'beliefs':
        #     g_random = randomize_beliefs(g5, beliefs=beliefs)
        # elif randomization == 'shuffle':
        #     g_random = randomize_beliefs(g5)
        # jaccards = measure_interpersonal_similarity(g_random)
        # res.at[step, 'NF Rand top decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 90)
        # res.at[step, 'NF Rand 90 percentile similarity'] = np.percentile(list(jaccards.values()), 90)
        # res.at[step, 'NF Rand bottom decile similarity'] = measure_mean_interpersonal_similarity(jaccards, 10,
        #                                                                                             False)
        # res.at[step, 'NF Rand 10 percentile similarity'] = np.percentile(list(jaccards.values()), 10)
        # res.at[step, 'NF Rand num social clusters'], res.at[step, 'Rand prominence of social clusters'] = \
        #     measure_social_clusters_hierarchy(jaccards, method=clustering)

        # sequential diffusion

    # seq_diff_df = pd.DataFrame(index=range(10), columns=[tuple(b) for b in beliefs])
    # seq_sus_df = pd.DataFrame(index=range(10), columns=[tuple(b) for b in beliefs])
    # g2 = copy.deepcopy(g)
    # for edge in np.random.permutation(beliefs):
    #     for step in range(10):
    #         seq_diff_df.at[step, tuple(edge)] = measure_diffusion(g2, [edge])[tuple(edge)]
    #         seq_sus_df.at[step, tuple(edge)] = measure_susceptibility(g2, [edge], susceptible_func)[tuple(edge)]
    #         simulate_individual(g2, edge, n_steps=1, adopt=adopt_func)
    # res['FF diffusion'] = seq_diff_df.mean(axis=1)/n_agents*100
    # res['FF susceptibility'] = seq_sus_df.mean(axis=1)/n_agents*100
    # res['FF top decile clustering'] = [measure_belief_clustering_coefficient(diffusion, 90)
    #                                    for i, diffusion in seq_diff_df.T.to_dict().items()]
    # res['FF num semantic clusters'] = [measure_num_belief_clusters(diffusion)
    #                                    for i, diffusion in seq_diff_df.T.to_dict().items()]
           
        
    # individual diffusion
    # ind_diff_df = pd.DataFrame(index=range(10), columns=[tuple(b) for b in beliefs])
    # ind_sus_df = pd.DataFrame(index=range(10), columns=[tuple(b) for b in beliefs])
    # adoptions = [pd.DataFrame(index=g.nodes(), columns=[tuple(b) for b in beliefs]) for i in range(10)]
    # for edge in np.random.permutation(beliefs):
    #     g3 = copy.deepcopy(g)
    #     for step in range(10):
    #         ind_diff_df.at[step, tuple(edge)] = measure_diffusion(g3, [edge])[tuple(edge)]
    #         ind_sus_df.at[step, tuple(edge)] = measure_susceptibility(g3, [edge], susceptible_func)[tuple(edge)]
    #         adoptions[step][tuple(edge)] = [g3.node[n]['M'].has_edge(*edge) for n in g3]
    #         simulate_individual(g3, edge, n_steps=1, adopt=adopt_func)
    #
    #
    # res['NF diffusion'] = ind_diff_df.mean(axis=1)/n_agents*100
    # res['NF susceptibility'] = ind_sus_df.mean(axis=1)/n_agents*100
    #
    # res['NF top decile clustering'], res['NF top decile modularity'] = zip(
    #     *[measure_belief_clustering_coefficient(diffusion, 90) for i, diffusion in ind_diff_df.T.to_dict().items()])
    # res['NF bottom decile clustering'], res['NF bottom decile modularity'] = zip(
    #     *[measure_belief_clustering_coefficient(diffusion, 10, above=False) for i, diffusion in ind_diff_df.T.to_dict().items()])
    #
    # res['NF num semantic clusters'] = [measure_num_belief_clusters(diffusion)
    #                                    for i, diffusion in ind_diff_df.T.to_dict().items()]
    #
    # res['NF mean belief corr'] = [a_df.astype(float).corr().abs().mean().mean() for a_df in adoptions]
           
   
    return res


def opt_point(args):


    network, n_agents, deg, n_beliefs, n_concepts, pathlength, threshold, clustering, randomization = args
    res = {
        'network': network,
        'n_agents': n_agents,
        'n_beliefs': n_beliefs,
        'n_concepts': n_concepts,
        'deg': deg,
    }

    try:
        g, beliefs = setup(*args[:5])
        adopt_func = fast_adopt
        susceptible_func = fast_susceptible



        ######## Interdependent ###########
        # initial conditions
        res['inter initial susceptibles'] = np.mean(
            list(measure_susceptibility(g, beliefs, susceptible_func).values())) / n_agents * 100
        res['inter initial adopters'] = np.mean(list(measure_diffusion(g, beliefs).values())) / n_agents * 100

        g1 = copy.deepcopy(g)
        s1 = measure_susceptibility(g1, beliefs, fast_susceptible)
        ai = measure_diffusion(g1, beliefs)

        simulate_simultaneous(g1, beliefs, n_steps=10, adopt=adopt_func)
        diffusion1 = measure_diffusion(g1, beliefs)

        res['inter final susceptibles'] = np.mean(
            list(measure_susceptibility(g1, beliefs, susceptible_func).values())) / n_agents * 100
        res['inter final adopters'] = np.mean(list(diffusion1.values())) / n_agents * 100

        res['inter net initial suscep correlation'] = np.corrcoef(
            [s1[tuple(b)] - ai[tuple(b)] for b in beliefs],
            [diffusion1[tuple(b)] - ai[tuple(b)] for b in beliefs]
        )[1][0]

        a_df1 = pd.DataFrame(index=g1.nodes(), columns=[tuple(b) for b in beliefs])
        for n in g1:
            for b in beliefs:
                a_df1.at[n, tuple(b)] = g1.node[n]['M'].has_edge(*b)
        res['inter best connected belief corr'] = correlate_best(a_df1)

        res['inter top decile semantic clustering'], res['inter top decile semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion1, 90)

        res['inter top 5% semantic clustering'], res['inter top 5% semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion1, 95)

        corrs1 = a_df1.astype(float).T.corr().mask(np.tri(n_agents, n_agents, 0, dtype='bool')).stack().sort_values()
        res['inter correlation similarity 95%'] = np.percentile(corrs1, 95)
        res['inter correlation similarity 5%'] = np.percentile(corrs1, 5)

        pca1 = PCA(n_components=n_agents)
        pca1.fit(a_df1)
        res['inter pca1 var'] = pca1.explained_variance_[0]
        res['inter pca2 var'] = pca1.explained_variance_[2]
        res['inter pca total variance'] = np.sum(pca1.explained_variance_)


        ####### Independent ###########


        for step, g2, s2 in simulate_independent_match_susceptibility(copy.deepcopy(g),
                                                                      sus_target=res['inter final susceptibles'] / 100,
                                                                      n_steps=10,
                                                                      n_concepts=n_concepts,
                                                                      beliefs=beliefs):

            pass

        res['match initial susceptibles'] = np.mean(list(s2.values())) / n_agents * 100
        res['match initial adopters'] = res['inter initial adopters']
        res['match final susceptibles'] = res['match initial susceptibles']
        res['match final adopters'] = np.mean(list(measure_diffusion(g2, beliefs).values())) / n_agents * 100

        diffusion2 = measure_diffusion(g2, beliefs)

        res['match net initial suscep correlation'] = np.corrcoef(
            [s2[tuple(b)] - ai[tuple(b)] for b in beliefs],
            [diffusion2[tuple(b)] - ai[tuple(b)] for b in beliefs]
        )[1][0]



        a_df2 = pd.DataFrame(index=g2.nodes(), columns=[tuple(b) for b in beliefs])
        for n in g2:
            for b in beliefs:
                a_df2.at[n, tuple(b)] = g2.nodes[n]['M'].has_edge(*b)
        res['match best connected belief corr'] = correlate_best(a_df2)

        res['match top decile semantic clustering'], res['match top decile semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion2, 90)

        res['match top 5% semantic clustering'], res['match top 5% semantic modularity'] = \
            measure_belief_clustering_coefficient(diffusion2, 95)

        corrs2 = a_df2.astype(float).T.corr().mask(np.tri(n_agents, n_agents, 0, dtype='bool')).stack().sort_values()
        res['match correlation similarity 95%'] = np.percentile(corrs2, 95)
        res['match correlation similarity 5%'] = np.percentile(corrs2, 5)

        pca2 = PCA(n_components=n_agents)
        pca2.fit(a_df2)
        res['match pca1 var'] = pca2.explained_variance_[0]
        res['match pca2 var'] = pca2.explained_variance_[1]
        res['match pca total variance'] = np.sum(pca2.explained_variance_)



    except:
        pass


    return res