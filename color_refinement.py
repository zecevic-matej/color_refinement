# libraries
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
# TODO: remove this if no proper way for the TODO's below is found
# import matplotlib as mpl
# mpl.rcParams['font.size'] = 20
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.unicode'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath,amssymb,amsfonts,amsthm}}\n\setcounter{MaxMatrixCols}{20}'

def create_dataframe_from_A_and_ids_and_labels(A, ids, labels):
    """

    :param A:
    :param labels:
    :return:
    """

    vnames, wnames = ids
    vlabels, wlabes = labels

    edges = {'from': [], 'to': []}
    for ind_x, row in enumerate(A):
        for ind_y, A_xy in enumerate(row):
            if A_xy != 0:
                edges['from'].append(vnames[ind_x])
                edges['to'].append(wnames[ind_y])

    edges = pd.DataFrame({'from': edges['from'], 'to': edges['to']})
    nodes = pd.DataFrame({'id': vnames + wnames, 'names': vnames + wnames, 'class': vlabels + wlabes, 'bipartite': [0 if x in vlabels else 1 for x in vlabels+wlabes]})

    G = (nodes, edges)

    return G


def draw_graph(G, bipartite=True, debug=False):
    """

    :param G:
    :return:
    """

    nodes, edges = G

    # Build your graph
    G = nx.from_pandas_edgelist(edges, 'from', 'to', create_using=nx.Graph())

    # The order of the node for networkX is the following order:
    G.nodes()
    # Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!

    # Here is the tricky part: I need to reorder carac to assign the good color to each node
    nodes = nodes.set_index('id')
    nodes = nodes.reindex(G.nodes())

    # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
    nodes['class'] = pd.Categorical(nodes['class'])

    if bipartite:
        for n in nodes.iterrows():
            G.nodes[n[0]]['bipartite'] = n[1]['bipartite']
        top_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite'] == 0)
        #bottom_nodes = set(G) - top_nodes
        #G = nx.bipartite.projected_graph(G, top_nodes)
        #G = nx.bipartite.project(G, [node[0] for node in nodes.iterrows() if node[1]['class']=='orangered'])
        pos = nx.bipartite_layout(G, nx.bipartite.sets(G, top_nodes)[0])

    if debug:
        import pdb; pdb.set_trace()

    # Custom the nodes:
    nx.draw(G, with_labels=True, node_color=nodes['class'], cmap=plt.cm.Set1, node_size=300, pos=pos)
    plt.show()

    return G


def merge(lsts):
    sets = [set(lst) for lst in lsts if lst]
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    import pdb; pdb.set_trace()
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def remove_sublists(ll):
    remove = []
    l = ll.copy()
    for ind, x in enumerate(l):
        for g in l[:ind] + l[ind + 1:]:
            if all(y in g for y in x):
                remove.append(ind)
    remove = list(set(remove))
    for i, ind in enumerate(sorted(remove)):
        ind = ind - i
        del l[ind]
    return l


def partition_with_ids(P, Q, ids):
    P_ids = []
    for ind_l, l in enumerate(P):
        l_id = []
        for ind_x, x in enumerate(l):
            l_id.append(ids[0][int(x)])
        P_ids.append(l_id)
    Q_ids = []
    for ind_l, l in enumerate(Q):
        l_id = []
        for ind_x, x in enumerate(l):
            l_id.append(ids[1][int(x)])
        Q_ids.append(l_id)
    return P_ids, Q_ids


def compute_partitions(A, ids=None, verbose=False):
    """
    Color refinement.

    :param A: graph matrix
    :return: stable partitions P and Q
    """

    P0 = [list(range(A.shape[0]))]
    P = P0
    P_prev = P
    Q0 = [list(range(A.shape[1]))]
    Q = Q0
    Q_prev = Q

    stable = False
    not_together = False
    dont0 = False
    dont1 = False
    colors = np.random.randint(500, 1000, 10)
    count = 0
    while not stable:

        # find all pairs belonging to same class, and join after rule
        new_classes_P = []
        for c_p in P:
            if verbose:
                print("P iter - Class " + str(c_p) + " of " + str(P))
            if count > 0 and len(c_p) == 1:
                new_classes_P.append(c_p)
            for p in itertools.combinations(c_p, 2):
                if verbose:
                    print("Pair " + str(p) + " of " + str([x for x in itertools.combinations(c_p, 2)]))
                for c_q in Q:
                    sum1 = sum([A[int(p[0]), int(x)] for x in c_q])
                    sum2 = sum([A[int(p[1]), int(x)] for x in c_q])
                    if not np.allclose(sum1, sum2):
                        #import pdb; pdb.set_trace()
                        for l in new_classes_P:
                            if p[0] in l:
                                dont0 = True
                            if p[1] in l:
                                dont1 = True
                        if not dont0:
                            new_classes_P.append([p[0]])
                        if not dont1:
                            new_classes_P.append([p[1]])
                        not_together = True
                        dont0 = False
                        dont1 = False
                        break
                if not not_together:
                    #import pdb; pdb.set_trace()
                    not_together = False
                    separate = False
                    for l in new_classes_P:
                        if p[0] in l or p[1] in l:
                            separate = True
                    if len(new_classes_P) == 0 or not separate:
                        new_classes_P.append([p[0],p[1]])
                    else:
                        for ind_l, l in enumerate(new_classes_P):
                            if p[0] in l:
                                new_classes_P[ind_l].append(p[1])
                            if p[1] in l:
                                new_classes_P[ind_l].append(p[1])
                not_together = False
                if verbose:
                    print("State is " + str(new_classes_P))
        new_classes_P = [list(set(s)) for s in new_classes_P]
        new_classes_P = remove_sublists(new_classes_P)

        # find all pairs belonging to same class, and join after rule
        new_classes_Q = []
        for c_q in Q:
            if verbose:
                print("Q iter - Class " + str(c_q) + " of " + str(Q))
            if count > 0 and len(c_q) == 1:
                new_classes_Q.append(c_q)
            for p in itertools.combinations(c_q, 2):
                if verbose:
                    print("Pair " + str(p) + " of " + str([x for x in itertools.combinations(c_q, 2)]))
                for c_p in P:
                    sum1 = sum([A[int(x), int(p[0])] for x in c_p])
                    sum2 = sum([A[int(x), int(p[1])] for x in c_p])
                    if not np.allclose(sum1, sum2):
                        #import pdb; pdb.set_trace()
                        for l in new_classes_Q:
                            if p[0] in l:
                                dont0 = True
                            if p[1] in l:
                                dont1 = True
                        if not dont0:
                            new_classes_Q.append([p[0]])
                        if not dont1:
                            new_classes_Q.append([p[1]])
                        not_together = True
                        dont0 = False
                        dont1 = False
                        break
                if not not_together:
                    #import pdb; pdb.set_trace()
                    not_together = False
                    separate = False
                    for l in new_classes_Q:
                        if p[0] in l or p[1] in l:
                            separate = True
                    if len(new_classes_Q) == 0 or not separate:
                        new_classes_Q.append([p[0],p[1]])
                    else:
                        for ind_l, l in enumerate(new_classes_Q):
                            if p[0] in l:
                                new_classes_Q[ind_l].append(p[1])
                            if p[1] in l:
                                new_classes_Q[ind_l].append(p[1])
                not_together = False
                if verbose:
                    print("State is " + str(new_classes_Q))
            #if count == 2:
            #    import pdb; pdb.set_trace()
        new_classes_Q = [list(set(s)) for s in new_classes_Q]
        new_classes_Q = remove_sublists(new_classes_Q)

        P = new_classes_P
        Q = new_classes_Q

        count += 1

        if P == P_prev and Q == Q_prev:

            stable = True

            if ids is not None:

                P_ids, Q_ids = partition_with_ids(P, Q, ids)

                print("***Converged to Stable Configuration after {} Iterations***\n"
                      "P_{} = P_{} {}\n"
                      "Q_{} = Q_{} {}\n"
                      "****************************".format(count, count, count-1, P_ids, count, count-1, Q_ids))

            else:

                print("***Converged to Stable Configuration after {} Iterations***\n"
                      "P_{} = P_{} {}\n"
                      "Q_{} = Q_{} {}\n"
                      "****************************".format(count, count, count-1, P, count, count-1, Q))

        else:

            print("***Completed Iteration {}***\n"
                  "P_{} {}\n"
                  "Q_{} {}\n"
                  "****************************".format(count, count, P, count, Q))

        P_prev = P
        Q_prev = Q

    return P, Q


def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)


def assume_bipartite(A):

    ids = ([x for x in char_range("A", [x for x in char_range('A','Z')][A.shape[0]-1])],[x for x in range(1,A.shape[1]+1)])

    labels = (['orangered' for x in range(len(ids[0]))], ['lightskyblue' for x in range(len(ids[1]))])

    return ids, labels


def show_partitions_colored(A, P, Q, ids):

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    ax1.imshow(A, cmap='Greens')
    ax1.set_title('Input Graph')
    ax2.imshow(permute_A_according_to_partitions(A, P, Q), cmap='Greens')
    P_ids, Q_ids = partition_with_ids(P, Q, ids)
    ax2.set_title('With Partitions Resorted\nP={}, Q={}'.format(P_ids, Q_ids))
    ax1.set_xticks([x for x in range(A.shape[1])])
    ax1.set_yticks([x for x in range(A.shape[0])])
    ax2.set_xticks([x for x in range(A.shape[1])])
    ax2.set_yticks([x for x in range(A.shape[0])])
    ax1.set_xticklabels(ids[1])
    ax1.set_yticklabels(ids[0])
    ax2.set_xticklabels([ids[1][x] for x in flatten(Q)])
    ax2.set_yticklabels([ids[0][x] for x in flatten(P)])
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    plt.suptitle("Color Refinement on Graph", fontweight='bold')
    plt.show()


flatten = lambda l: [item for sublist in l for item in sublist]


def permute_A_according_to_partitions(A, P, Q):

    Ap = A.copy()
    Ap = Ap[flatten(P),:]
    Ap = Ap[:,flatten(Q)]
    return Ap


def show_graph_and_partitions(A, P=None, Q=None):
    """
    Uses all above functions to compute the partitions,
    and visualize them.

    :param A: connectivity matrix of some Graph G
    :return:
    """

    ids, labels = assume_bipartite(A)

    G_raw = create_dataframe_from_A_and_ids_and_labels(A, ids, labels)
    G = draw_graph(G_raw)

    if not (P and Q):
        P, Q = compute_partitions(A, ids)

    show_partitions_colored(A, P, Q, ids)

    return {"ids": ids, "labels": labels, "G": G, "A": A, "partitions": (P, Q)}


def check_if_pair_is_fractional_automorphism(A, P, Q):
    """
    Check if P,Q are partitions that can act as a fractional automorphism on A.
    Corollary 6.1 XA = AY where X = CC and Y = DD

    :param A:
    :param P:
    :param Q:
    :return:
    """

    Pip, Piq = calculate_partition_matrices(P, Q)

    PipS = calculate_S_partition_matrix(Pip)
    PiqS = calculate_S_partition_matrix(Piq)

    CC = Pip @ PipS
    DD = Piq @ PiqS

    return np.allclose(CC @ A, A @ DD)


def calculate_partition_matrices(P, Q):
    """
    The partitions yield binary partitions matrices which yield which elements
    are in which partition for each of the available partitions.

    :param P:
    :param Q:
    :return:
    """

    Vp = sorted(flatten(P))
    Vq = sorted(flatten(Q))

    Pip = np.zeros((len(Vp), len(P)))
    Piq = np.zeros((len(Vq), len(Q)))

    for ind_p, p in enumerate(P):
        for v in Vp:
            if v in p:
                Pip[v, ind_p] = 1
    for ind_q, q in enumerate(Q):
        for v in Vq:
            if v in q:
                Piq[v, ind_q] = 1

    return Pip, Piq


def calculate_S_partition_matrix(Pi):
    """
    The scaled transpose is a transpose with the frequencies of
    each partition subset i.e., if there are 3 elements in a partition,
    then every 1 entry is converted to 1/3.
    It is used because it is a stochastic matrix.

    :param Pi:
    :return:
    """

    PiS = np.zeros((Pi.shape[1], Pi.shape[0]))

    for ind_y, r in enumerate(Pi):
        for ind_x, c in enumerate(r):
            PiS[ind_x, ind_y] = c / sum(Pi[:,ind_x])

    return PiS


def calculate_core_factor(A, P, Q):
    """
    The core factor is calculated according to what follows Corollary 6.1

    :param A:
    :param P:
    :param Q:
    :return:
    """

    if np.isinf(A).any() or np.isnan(A).any():

        raise Exception("Check A for Infinity or NaN.")

    Pip, Piq = calculate_partition_matrices(P, Q)

    PipS = calculate_S_partition_matrix(Pip)

    A_core = PipS @ A @ Piq # equivalent to np.dot(.,.)

    return A_core


def calculate_iterated_core_factor(A, show_visualizations=False):
    """
    Calculate the iterated core factor of the graph matrix A.
    Visualizations of the intermediate Graphs and permutated versions can be made.

    :param A: graph (weighted) connection matrix of form V x W
    :param show_visualizations: bool:
    :return: A_core: iterated core factor, a matrix smaller than the original A
    """

    print('\n++++++++++++++++++++ Initial Matrix of the Graph ++++++++++++++++++++\n\n\t A = {}\n\n'
          '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
          .format('\t' + str(A).replace('\n','\n\t\t')))

    iter = 1
    fin = False
    while True:

        ids, labels = assume_bipartite(A)

        P, Q = compute_partitions(A, ids)

        A_core = calculate_core_factor(A, P, Q)

        if A.shape == A_core.shape and np.allclose(A_core, A):
            fin = True

        old_A = A.copy()
        A = A_core

        if show_visualizations:

            show_graph_and_partitions(old_A, P, Q)

        if fin:
            break
        
        print('\n++++++++++++++++++++ Computed Core Factor {} ++++++++++++++++++++\n\n\t A = {}\n\n'
              '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'.format(iter, '\t' + str(A_core).replace('\n', '\n\t\t')))

        iter += 1

    return A_core


# TODO: make the below function more flexible
def matrix(a, l='p'):
    """Returns a LaTeX pmatrix

    :a: numpy array
    :returns: LaTeX pmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('matrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{pmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{pmatrix}']
    rv = ''.join(rv)
    rv = rv.replace('inf', '\\infty ')
    return '$' + rv + '$'

# TODO: find a way of doing this, because with this method there
#       is the bug with the matrix shape size restriction,
#       and other methods simply do not work
def create_matrix_comparison(A, B):

    # fig, ax = plt.subplots()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    #ax2.text(0.5, 0.5, matrix(np.matrix.round(np.random.rand(10,16), 2)))
    ax1.annotate(
        matrix(np.matrix.round(A, 2)),
        (0.25, 0.25),
        textcoords='axes fraction', size=20)

    plt.show()

"""
Function Section ^
Script Section   v
"""

# big A from Example 1.1
A = np.array([
    [3, -1, 1, 0.25, 0.25, 0.25, 0.25, 0, 0, 3, -2, 0.5, 0.5, 1],
    [-1, 1, 3, 0.25, 0.25, 0.25, 0.25, 0, 0, -2, 3, 0.5, 0.5, 1],
    [1, 3, -1, 0.25, 0.25, 0.25, 0.25, 0, 0, 0.5, 0.5, 0.5, 0.5, 1],

    [0, 1 / 3, 2 / 3, 0, 3 / 2, 0, 3 / 2, 2, 0, 1, 0, -1, 0, 1],
    [1/3, 1/3, 1/3, 3/2, 0, 3/2, 0, 2, 0, 0, 1, 0, -1, 1],
    [1/3, 1/3, 1/3, 0, 3/2, 0, 3/2, 0, 2, -1, 0, 1, 0, 1],
    [2/3, 1/3, 0, 3/2, 0, 3/2, 0, 0, 2, 0, -1, 0, 1, 1],

    [2, 2, 2, 3/2, 3/2, 3/2, 3/2, 1, 1, 0.5, 0.5, 0.5, 0.5, np.inf],
])
A[-1,-1] = 0
#d = show_graph_and_partitions(A)
A_itr_core = calculate_iterated_core_factor(A)

"""
More Examples Below v
"""

# example matrix
# A = np.array([
#     [1, 0, 1, 0],
#     [1, 1, 0, 0],
#     [1, 0, 1, 0],
#     [0, 0, 1, 1],
# ])
# d = show_graph_and_partitions(A)

# Examples 5.1
# A3 = np.array([
#     [1, 1, 0],
#     [1, 0, 1],
#     [0, 1, 1],
#
# ])
# A4 = np.array([
#     [1, 1, 0, 0],
#     [1, 0, 1, 0],
#     [0, 1, 0, 1],
#     [0, 0, 1, 1]
# ])
# A5 = np.array([
#     [1, 1, 0, 0, 0],
#     [1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [0, 0, 1, 0, 1],
#     [0, 0, 0, 1, 1],
# ])
# for A in [A3, A4, A5]:
#     d = show_graph_and_partitions(A)

# Direct Sum of Example 5.9
# A = np.array([
#     [1,0,0,0,   0,0,0,0],
#     [0,1,0,0,   0,0,0,0],
#     [0,0,1,1,   0,0,0,0],
#     [0,0,1,1,   0,0,0,0],
#     [0,0,0,0,   1,0,0,0],
#     [0,0,0,0,   0,1,1,0],
#     [0,0,0,0,   0,1,0,1],
#     [0,0,0,0,   0,0,1,1],
# ])
# d = show_graph_and_partitions(A)

# create multiple random binary graphs
# np.random.seed(0)
# N = 1
# for i in range(N):
#
#     A = np.random.randint(2, size=(np.random.randint(3,7), np.random.randint(3,7)))
#
#     d = show_graph_and_partitions(A)