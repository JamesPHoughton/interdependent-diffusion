"""
These functions help manipulate the experiment data, but are not something
that needs to be understood to understand how the analysis is performed.
"""

import numpy as np
import networkx as nx
from datetime import datetime


def retrace(game):
    """
    Uses the game log and starting conditions to recreate the state of the
    game at every change event.

    Returns a generator yielding (player_id, g, t) at each event in the game,

    *player_id* is the player logging the event
    *g* is the state of the game following the event
        - `g` is a networkx graph with players as nodes
        - each node in `g` has a semantic network `M` containing their 'leads'
        - each node in `g` has a semantic network `F` containing their 'dead ends'
    *t* timestamp in seconds since game start

    Does not return an action if the only change is a list reordering.
    """
    clues = game['data.clues']

    # create trace social network
    edge_list = []
    for player_id, player_data in game['players'].items():
        for alter_id in player_data['data.alterIDs']:
            edge_list.append([player_id, alter_id])
    g = nx.from_edgelist(edge_list)

    # give trace players starting information
    nx.set_node_attributes(
        g,
        name='pos',  # position in the social network
        values={a: game['players'][a]['data.position'] for a in g}
    )

    nx.set_node_attributes(
        g,
        name='M',  # M for mind/memory
        values={a: nx.from_edgelist([
            clues[bf]['nodes'] for bf in
            game['players'][a]['data.initialState']['promising_leads']['clueIDs']
        ]) for a in g}
    )

    nx.set_node_attributes(
        g,
        name='F',  # F for forgetory
        values={i: nx.Graph() for i in g}
    )

    # yield the initial state of the experiment
    yield (None, g, 0)


    # trace game
    #t_start = datetime.strptime(game['createdAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
    stage = [r for r in game['stages'] if r['name'] == 'response'][0]
    t_start = datetime.strptime(stage['startTimeAt'], '%Y-%m-%dT%H:%M:%S.%fZ')

    for event in game['log']:
        if event['name'] != 'drop': # only consider drop events
            continue

        player_id = event["playerId"]
        source = event['data']['source']
        dest = event['data']['dest']
        if 'clue' in event['data']:
            if event['data']['clue'] != None:
                edge = clues[event['data']['clue']]['nodes']
            else: # catch incomplete record
                print('Missing clueID for player %s from source %s at time %s' % (player_id, source, event['at']))
        else:
            print('player %s is missing a clue' % player_id)
            continue
        M = g.nodes()[player_id]['M']
        F = g.nodes()[player_id]['F']
        update = False

        if source == "promising_leads":
            assert g.nodes()[player_id]['M'].has_edge(*edge) # check that clue is still in memory
            if dest == "dead_ends":
                M.remove_edge(*edge)
                F.add_edge(*edge)
                update = True

        elif source == "dead_ends":
            assert g.nodes()[player_id]['F'].has_edge(*edge) # check that clue is still in forgettory
            if dest == "promising_leads":
                F.remove_edge(*edge)
                M.add_edge(*edge)
                update = True

        else:
            assert source in game['playerIds']  # check that source is another player
            if not g.nodes()[source]['M'].has_edge(*edge):  # check that clue is in source
                # this can fail if the exposer removes the clue while the exposed is dragging it.
                # turns out not to be a big deal
                print("%s no longer in source %s" % (str(edge), str(source)))
            if dest == "promising_leads":
                M.add_edge(*edge)
                if F.has_edge(*edge):
                    F.remove_edge(*edge)
                update = True
            elif dest == "dead_ends":
                F.add_edge(*edge)
                if M.has_edge(*edge):
                    M.remove_edge(*edge)
                update = True
            assert not (F.has_edge(*edge) and  # not in both memory and forgetery
                        M.has_edge(*edge))

        if update:
            t_current = datetime.strptime(event['createdAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
            t = (t_current - t_start).total_seconds()
            yield (player_id, g, t)

    # double check the final state at the end of the generator
    for player_id in g:
        leads = game['players'][player_id]['data.notebooks']['promising_leads']['clueIDs']
        should_have = set([tuple(sorted(clues[clue]['nodes'])) for clue in leads if clue != None])
        has = set([tuple(sorted(edge)) for edge in g.nodes()[player_id]['M'].edges()])
        assert should_have == has

        deads = game['players'][player_id]['data.notebooks']['dead_ends']['clueIDs']
        should_have = set([tuple(sorted(clues[clue]['nodes'])) for clue in deads if clue != None])
        has = set([tuple(sorted(edge)) for edge in g.nodes()[player_id]['F'].edges()])
        assert should_have == has



def flip1(m):
    """
    Chooses a single (i0, j0) location in the matrix to 'flip'
    Then randomly selects a different (i, j) location that creates
    a quad [(i0, j0), (i0, j), (i, j0), (i, j) in which flipping every
    element leaves the marginal distributions unaltered.
    Changes those elements, and returns 1.

    If such a quad cannot be completed from the original position,
    does nothing and returns 0.
    """
    i0 = np.random.randint(m.shape[0])
    j0 = np.random.randint(m.shape[1])

    level = m[i0, j0]
    flip = 0 if level == 1 else 1  # the opposite value

    for i in np.random.permutation(range(m.shape[0])):  # try in random order
        if (i != i0 and  # don't swap with self
            m[i, j0] != level):  # maybe swap with a cell that holds opposite value
            for j in np.random.permutation(range(m.shape[1])):
                if (j != j0 and  # don't swap with self
                    m[i, j] == level and  # check that other swaps work
                    m[i0, j] != level):
                    # make the swaps
                    m[i0, j0] = flip
                    m[i0, j] = level
                    m[i, j0] = level
                    m[i, j] = flip
                    return 1

    return 0

def shuffle(m1, n=100):
    """
    Randomizes a matrix leaving marginal distributions unaltered
    """
    m2 = m1.copy()
    f_success = np.mean([flip1(m2) for _ in range(n)])

    # f_success is the fraction of flip attempts that succeed, for diagnostics
    #print(f_success)

    # check the answer
    assert(all(m1.sum(axis=1) == m2.sum(axis=1)))
    assert(all(m1.sum(axis=0) == m2.sum(axis=0)))

    return m2
