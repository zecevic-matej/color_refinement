# libraries
import time
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True, linewidth=250)
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
from scipy.optimize import linprog
from scipy.special import comb
import functools
import pytest

def create_dataframe_from_A_and_ids_and_labels(A, ids, labels, bipartite=True):
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
                if bipartite:
                    edges['from'].append(vnames[ind_x])
                    edges['to'].append(wnames[ind_y])
                else:
                    edges['from'].append(vnames[ind_x])
                    edges['to'].append(vnames[ind_y])

    edges = pd.DataFrame({'from': edges['from'], 'to': edges['to']})
    nodes = pd.DataFrame({'id': vnames + wnames, 'names': vnames + wnames, 'class': vlabels + wlabes, 'bipartite': [0 if x in vlabels else 1 for x in vlabels+wlabes]})

    if not bipartite:
        nodes = nodes[:int(A.shape[0])]  # if V=W then simply cut off the duplication

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
    pos = None

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


def compute_partitions(A, ids=None, verbose=0):
    """
    Color refinement.

    :param A: graph matrix
    :param verbose: 1 detail only high level info, 2 detail anything
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
            if verbose==2:
                print("P iter - Class " + str(c_p) + " of " + str(P))
            if count > 0 and len(c_p) == 1:
                new_classes_P.append(c_p)
            for p in itertools.combinations(c_p, 2):
                if verbose==2:
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
                if verbose==2:
                    print("State is " + str(new_classes_P))
        new_classes_P = [list(set(s)) for s in new_classes_P]
        new_classes_P = remove_sublists(new_classes_P)

        # find all pairs belonging to same class, and join after rule
        new_classes_Q = []
        for c_q in Q:
            if verbose==2:
                print("Q iter - Class " + str(c_q) + " of " + str(Q))
            if count > 0 and len(c_q) == 1:
                new_classes_Q.append(c_q)
            for p in itertools.combinations(c_q, 2):
                if verbose==2:
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
                if verbose==2:
                    print("State is " + str(new_classes_Q))
            #if count == 2:
            #    import pdb; pdb.set_trace()
        new_classes_Q = [list(set(s)) for s in new_classes_Q]
        new_classes_Q = remove_sublists(new_classes_Q)

        P = sorted([sorted(x) for x in new_classes_P])
        Q = sorted([sorted(x) for x in new_classes_Q])

        count += 1

        if check_new_and_old_partition(P, P_prev) and check_new_and_old_partition(Q, Q_prev):

            stable = True

            if ids is not None:

                P_ids, Q_ids = partition_with_ids(P, Q, ids)

                if verbose:
                    print("***Converged to Stable Configuration after {} Iterations***\n"
                          "P_{} = P_{} {}\n"
                          "Q_{} = Q_{} {}\n"
                          "****************************".format(count, count, count-1, P_ids, count, count-1, Q_ids))

            else:
                if verbose:
                    print("***Converged to Stable Configuration after {} Iterations***\n"
                          "P_{} = P_{} {}\n"
                          "Q_{} = Q_{} {}\n"
                          "****************************".format(count, count, count-1, P, count, count-1, Q))

        else:
            if verbose:
                print("***Completed Iteration {}***\n"
                      "P_{} {}\n"
                      "Q_{} {}\n"
                      "****************************".format(count, count, P, count, Q))

        P_prev = P
        Q_prev = Q

    return P, Q


def check_new_and_old_partition(P, P_prev):

    return sorted([sorted(x) for x in P]) == sorted([sorted(x) for x in P_prev])

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


def show_graph_and_partitions(A, P=None, Q=None, bipartite=True):
    """
    Uses all above functions to compute the partitions,
    and visualize them.

    :param A: connectivity matrix of some Graph G
    :return:
    """

    if bipartite:
        ids, labels = assume_bipartite(A)
    else:
        ids = (list(np.arange(A.shape[0])), list(np.arange(A.shape[0])))
        labels = (list(np.ones(A.shape[0])), list(np.ones(A.shape[0])))

    G_raw = create_dataframe_from_A_and_ids_and_labels(A, ids, labels, bipartite)
    G = draw_graph(G_raw,bipartite=bipartite)

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

    # Vp = sorted(flatten(P))
    # Vq = sorted(flatten(Q))
    #
    # Pip = np.zeros((len(Vp), len(P)))
    # Piq = np.zeros((len(Vq), len(Q)))
    #
    # for ind_p, p in enumerate(P):
    #     for v in Vp:
    #         if v in p:
    #             Pip[v, ind_p] = 1
    # for ind_q, q in enumerate(Q):
    #     for v in Vq:
    #         if v in q:
    #             Piq[v, ind_q] = 1
    total_len = len(flatten(P))
    Pip = np.zeros((total_len, len(P)))
    prev_pre = 0
    for i, c in enumerate(P):
        if i == 0:
            pre = 0
        else:
            pre = prev_pre
        after = total_len - pre - len(c)
        d = np.hstack((np.zeros(pre), np.ones(len(c)), np.zeros(after)))
        Pip[:,i] = d
        prev_pre += len(c)
    total_len = len(flatten(Q))
    Piq = np.zeros((total_len, len(Q)))
    prev_pre = 0
    for i, c in enumerate(Q):
        if i == 0:
            pre = 0
        else:
            pre = prev_pre
        after = total_len - pre - len(c)
        d = np.hstack((np.zeros(pre), np.ones(len(c)), np.zeros(after)))
        Piq[:, i] = d
        prev_pre += len(c)

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


def calculate_iterated_core_factor(A, show_visualizations=False, verbose=0, alternate_cr_fun=None):
    """
    Calculate the iterated core factor of the graph matrix A.
    Visualizations of the intermediate Graphs and permutated versions can be made.

    :param A: graph (weighted) connection matrix of form V x W
    :param show_visualizations: bool:
    :return: A_core: iterated core factor, a matrix smaller than the original A,
             core_factors: list of all core factors including iterated one
    """

    if verbose:
        print('\n++++++++++++++++++++ Initial Matrix of the Graph ++++++++++++++++++++\n\n\t A = {}\n\n'
              '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'
              .format('\t' + str(A).replace('\n','\n\t\t')))

    iter = 1
    fin = False
    fin_due_loop = False
    core_factors = []
    loop_prevention = []
    partitions = []
    while True:

        if alternate_cr_fun:
            P, Q = alternate_cr_fun(A)
        else:
            P, Q = compute_partitions(A, verbose=verbose)
        partitions.append((P,Q))

        print('Extracted Partitions:\n\tP = {}\n\tQ = {}'.format(P,Q))

        A_core = calculate_core_factor(A, P, Q)

        if A.shape == A_core.shape and np.allclose(A_core, A):
            fin = True

        old_A = A.copy()
        A = A_core

        if show_visualizations:

            show_graph_and_partitions(old_A, P, Q)

        if fin:
            break

        if verbose:
            print('\n++++++++++++++++++++ Computed Core Factor {} ++++++++++++++++++++\n\n\t A = {}\n\n'
                  '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n'.format(iter, '\t' + str(A_core).replace('\n', '\n\t\t')))

        iter += 1

        core_factors.append(A_core)

        if len(loop_prevention) > 0:
            for x in loop_prevention:
                if x.shape == A_core.shape and np.allclose(A_core,x):
                    print('>> Loop detected during computation of Iterated Core Factor!')
                    fin_due_loop = True
        if fin_due_loop:
            break
        loop_prevention.append(A_core)

    return A_core, core_factors, partitions


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


def optimize_LP(A_LP, unbounded=False, method='revised simplex'):
    """
    Solve a given LP in matrix form.

    :param A_LP: matrix containing LP as in Example 1.1
    :param method: 'interior-point', 'revised simplex' or 'simplex'
    :return: optimization results
    """

    obj = A_LP[-1,:-1] # last row of matrix, up until last column

    lhs_eq = []
    rhs_eq = []
    for ind_row, y in enumerate(A_LP[:-1,:]): # all except last row

        lhs_eq.append(A_LP[ind_row,:-1]) # row up until last column
        rhs_eq.append(A_LP[ind_row, -1]) # last column element only

    if unbounded:
        bnd = [(-np.inf, np.inf) for x in range(A_LP.shape[1] - 1)] # default whole range for all variables
    else:
        bnd = [(0, np.inf) for x in range(A_LP.shape[1] - 1)]

    opt = linprog(c=obj, A_ub=None, b_ub=None,
                  A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
                  method=method)

    return opt


def check_if_x_solves_LP(A_LP, x):
    """
    Check whether the provided vector solves the LP i.e., satisfies constraints.
    :param A_LP:
    :param x:
    :return:
    """

    obj = A_LP[-1,:-1] # last row of matrix, up until last column
    lhs_eq = []
    rhs_eq = []
    for ind_row, y in enumerate(A_LP[:-1,:]): # all except last row

        lhs_eq.append(A_LP[ind_row,:-1]) # row up until last column
        rhs_eq.append(A_LP[ind_row, -1]) # last column element only

    no_solution = False
    for l,r in zip(lhs_eq, rhs_eq):

        if not np.allclose(l @ x, r):
            print('>> Following Mismatch was discovered: l@x {} != r {}'.format(l @ x, r))
            no_solution = True
            break

    if no_solution:
        print('The proposed vector does not satisfy the constraints.\nIt is NOT a solution to the provided LP!')
        return False
    else:
        print('\n*********************************SOLUTION FOUND******************************************************\n'
              'The proposed vector \n'
              '\t x = {}\n'
              'satisfies the constraints, and is thus a solution.\n'
              'The provided solution evaluates the Objective of the LP at: {}\n'
              '*****************************************************************************************************'.format(x, obj @ x))
        return True


def solve_LP_via_color_refinement(A_LP, verbose=0, alternate_cr_fun=None):
    """
    Solve a given LP in matrix form using the formalisms of the paper.
    Perform iterated color refinement to find out core factors and partitions matrices.
    Then solve the reduced LP.
    Finally, map back to the original space.

    :param A_LP: Linear Program as Matrix as in Example 1.1
    :return: x: solution to the LP that satisfies and optimizes the LP
    """

    A_LP_itr_core, core_factors, partitions = calculate_iterated_core_factor(A_LP, verbose=verbose, alternate_cr_fun=alternate_cr_fun)

    if A_LP_itr_core.shape == A_LP.shape:
        print(">> The Dimension Reduction has not worked! Terminating.")
        return None

    Piqs = []
    for ind, cf in enumerate([A_LP] + core_factors[:-1]):
        #t00 = time.time()

        #P, Q = compute_partitions(cf)
        P, Q = partitions[ind]
        #print('Time for Partitions {:.4f} seconds'.format(time.time() - t00))
        #t01 = time.time()
        _, Piq = calculate_partition_matrices(P, Q)
        Piqs.append(Piq[:-1,:-1]) # discard last row/column as it is only due to the np.inf/0 as right corner element
        #print('Time for Partition Matrix {:.4f} seconds'.format(time.time() - t01))

    #t0 = time.time()
    opt = optimize_LP(A_LP_itr_core)
    x_itr_core = opt.x
    #t1 = time.time()

    x = x_itr_core.copy()
    for p in reversed(Piqs):

        x = p @ x
    #t2 = time.time()

    assert check_if_x_solves_LP(A_LP, x)
    # t3 = time.time()
    # print('Time for Optimization {:.4f} seconds\n'
    #       'Time for Remapping    {:.4f} seconds\n'
    #       'Time for Asserting    {:.4f} seconds\n'.format(t1-t0, t2-t1, t3-t2))

    return x


def compare_speed_of_direct_and_cr_reduced_solving(A, verbose=0, alternate_cr_fun=None):
    """
    Perform both solving approaches on the matrix of the given LP.
    Compare performance speed.
    :param A:
    :return: dict: both acquired solutions, if
    """

    t0 = time.time()
    x = solve_LP_via_color_refinement(A_LP=A, verbose=verbose, alternate_cr_fun=alternate_cr_fun)
    if x is None:
        print("No solution found using Dimension Reduction.")
    t1 = (time.time() - t0)
    opt = optimize_LP(A)
    if opt.success:
        check_if_x_solves_LP(A, opt.x)
    else:
        print("No solution found using Standard LP Solver on Original Problem.")
    t2 = (time.time() - t0) - t1
    if x is not None and opt.success:
        print('\nSpeed Comparison --------------------------------------------------------------------\n'
              'Solving the Original LP directly took: {} seconds\n'
              'Performing CR, Solving the Reduced LP, and Mapping Back took: {} seconds'.format(t2, t1))
    else:
        print('>> No comparison available given that not both methods came up with a solution.')

    return {'cr': x, 'direct': opt.x}


def create_big_matrix_from_given(A, N=7):
    """
    Construct a big matrix by basically repeating the initial matrix in the
    Direct Sum fashion (i.e., diagonally concatenated) as defined in the paper.

    :param A:
    :param N:
    :return:
    """

    intermediate = [A]
    entries = 0
    for i in range(1,N+1):
        A1 = intermediate[-1].copy()
        M = np.hstack((np.vstack((A1, np.zeros(A1.shape))), np.vstack((np.zeros(A1.shape), A1))))
        intermediate.append(M)
        entries = np.power(2,i)
    l = list(range(100,(entries+1)*100,100))
    for i in range(0,entries):
        M[A.shape[0] - 1 + A.shape[0] * i, A.shape[1] - 1 + A.shape[1] * i] = l[i]

    print('Created Matrix of Shape {}\nWith {} times original Matrix A.'.format(M.shape, entries))

    return M


def inspect_LP_solving_speed(A, rounds=6, method='standard'):
    times = []
    for i in range(1,rounds+1):
        t0 = time.time()
        C = create_big_matrix_from_given(A, N=i)
        if method == 'standard':
            optimize_LP(C)
        elif method == 'cr':
            solve_LP_via_color_refinement(C,0,alternate_cr_fun=cr_efficient)
        t = time.time() - t0
        times.append(t)
        print('Round {}: Time for Matrix of Shape {} in Seconds: {:.4f}'.format(i, C.shape, t))
    return times

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

A[-1,-1] = 100

#d = show_graph_and_partitions(A) # uncomment this for visualization of the graph and matrix

x = solve_LP_via_color_refinement(A_LP=A)

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


"""
Prof. Holger Dell's Implementation of Color Refinement for Online Visualization
(https://github.com/holgerdell/color-refinement)

Runtime is O((N+M)*log(N))

ONLY SUPPORTS UNWEIGHTED EDGES SO FAR!

Originally written in JavaScript, below converted to Python and incorporated into
the above infrastructure.

Examples follow after the functional section.
"""

def hd_random_graph(n, m, seed, visualize=True):
    max_num_edges = int(n * (n-1) / 2) # n choose 2
    if n < 0 or m < 0 or m > max_num_edges:
        raise Exception('Please check again.')
    graph = {
        'vertices': [],
        'edges': [],
    }
    for i in range(n):
        graph['vertices'].append({
            'name': i,
            'nb': [],
            'crtree': []
        })
    np.random.seed(seed)

    state = np.ones(max_num_edges) * -1
    for i in range(m):
        j = np.random.randint(i, max_num_edges)
        if not (i in state):
            state[i] = i
        if not (j in state):
            state[j] = j
        tmp = state[j].copy()
        state[j] = state[i]
        state[i] = tmp

    def unpair(k):
        z = np.floor((-1 + np.sqrt(1+8*k)) / 2)
        return (int(k - z * (1+z) / 2), int(z * (3+z)/2 - k))

    for i in range(m):
        x,y = unpair(state[i])
        u = graph['vertices'][x]
        v = graph['vertices'][n - 1 - y]
        graph['edges'].append((u,v))
        u['nb'].append(v)
        v['nb'].append(u)

    print('Created Random Graph with {} Vertices, {} Edges (Seed {}).'.format(n,m,seed))
    for x in graph['edges']:
        print('{} -> {}'.format(x[0]['name'], x[1]['name']))

    graph_alt_repr = (pd.DataFrame({'id': [v['name'] for v in graph['vertices']],
                                    'class': np.ones(len(graph['vertices']))}),
                      pd.DataFrame({'from': [x[0]['name'] for x in graph['edges']],
                                    'to': [x[1]['name'] for x in graph['edges']]}))

    if visualize:
        draw_graph(graph_alt_repr, False)

    return graph, graph_alt_repr

def hd_cr(G,G_weights, debug=True, verbose=True):
    trees = []
    pNC = 0
    for i in range(99):
        trees.append([])
        lut = []
        for j in range(len(G['vertices'])):
            if i == 0:
                prev_trees = None
            else:
                prev_trees = []
                for ind_c, c in enumerate(trees[i-1]):
                    prev_trees.append([])
                    for n in c['class']:
                        prev_trees[ind_c].append(n['name'])
            v, treelist, lut = hd_refine_at_node(G['vertices'][j], i, trees[i], lut, G_weights, prev_trees, debug, verbose)
            G['vertices'][j] = v

        NC = len(trees[i])
        if pNC == NC:
            trees = trees[:-1]
            if verbose:
                for ind, t in enumerate(trees):
                    print('Round {} delivered {} Classes'.format(ind, len(t)))
            return trees
        else:
            pNC = NC
    return trees

def hd_refine_at_node(v, depth, treelist, lut, G_weights, prev_trees, debug=False, verbose=True):
    nb = []
    if verbose:
        print('\n>> Round {}:     '
              'Looking at Node {} with #NB = {},       '
              'Current #Trees {}\n'.format(depth, v['name'], len(v['nb']), len(treelist)))
    if debug:
        import pdb; pdb.set_trace()
    if depth > 0:
        for i in range(len(v['nb'])):
            nb.append(v['nb'][i]['crtree'][depth - 1])
        nb = sorted(nb, key=functools.cmp_to_key(hd_sort_trees))

    if debug:
        print('Collected NB. Searching Current Treelist...')
        import pdb; pdb.set_trace()
    if depth:
        ind = hd_find_tree(treelist, nb, v, lut, G_weights, prev_trees, debug)
    else:
        ind = hd_find_tree(treelist, nb, v, lut, None, prev_trees, debug)
    if ind >= 0:
        T = treelist[ind]
        # cond = list(filter(lambda a: a != -1, np.unique([ind if T['class'] == x[2] else -1 for ind, x in enumerate(lut)])))
        T['class'].append(v)
        # if cond:
        #     print('Update T[class] condition triggered.')
        #     import pdb; pdb.set_trace()
        #     lut[cond[0]] = (lut[cond[0]][0], lut[cond[0]][1], T['class'])
    else:
        T = {
            'rank': None,
            'size': 1,
            'children': nb,
            'class': []
        }
        # if len(nb) > 0:
        #     T['weight'] =  sum([x['weights'] for x in nb])
        # else:
        #     T['weight'] = 0
        for i in range(len(nb)):
            T['size'] += nb[i]['size']
        T['class'].append(v)
        treelist.append(T)
        if v['name'] in G_weights:
            lut.append((G_weights[v['name']], [x['name'] for x in v['nb']], T['class']))
    v['crtree'].append(T)

    return v, treelist, lut

def hd_find_tree(treelist, T, v, lut, G_weights, prev_trees, debug=False):
    for i in range(len(treelist)):
        if G_weights is None:
            condition1 = True
            condition2 = False
            condition3 = False
        else:
            condition1 = (len(treelist[i]['children']) == len(T))
            condition2 = (sum(lut[i][0]) == sum(G_weights[v['name']]))
            nbs = lut[i][1]
            vnbs = [x['name'] for x in v['nb']]
            condition3 = (sum([1 if x in nbs else 0 for x in vnbs]))

        # import pdb; pdb.set_trace()
        if (G_weights and condition1 and condition2) or (G_weights is None and condition1):
            couldbe = True
            for j in range(len(T)):
                if treelist[i]['children'][j] != T[j]:
                    couldbe = False
                    break
            if couldbe:
                return i
        if condition2 and condition3:
            pass
            #print('Triggered Condition 3')
            # import pdb; pdb.set_trace()
            # condition3 = False
            # n_in_current_tree = [x['name'] for x in treelist[i]['class']]
            # for t in prev_trees:
            #     if sum([1 if n in t else 0 for n in n_in_current_tree]) == len(n_in_current_tree):
            #         if v['name'] in t:
            #             condition3 = True
            # if condition3:
            #     return i

    return -1

def equal_dicts(d1, d2, ignore_keys):
    return {k: v for k, v in d1.items() if k not in ignore_keys} == {k: v for k, v in d2.items() if k not in ignore_keys}

# TODO: this works, but cannot be used for our use-case given that the structure itself is recursive
#       i.e., this function applied would never terminate
def without_keys(d, ignore_keys):
    d_fin = {}
    for k in d.keys():
        if k not in ignore_keys:
            if isinstance(d[k], list):
                d_updates = []
                for x in d[k]:
                    if not isinstance(x, dict):
                        d_updates.append(x)
                    else:
                        d_updates.append(without_keys(x, ignore_keys))
                d_fin.update({k: d_updates})
            elif isinstance(d[k], dict):
                d_fin.update({k: without_keys(d[k], ignore_keys)})
            else:
                d_fin.update({k: d[k]})
    return d_fin

def hd_sort_trees(T1, T2):

    #import pdb; pdb.set_trace()
    if equal_dicts(T1, T2, ['children_weights']): #T1 == T2:
        return 0
    elif len(T1['children']) != len(T2['children']):
        return len(T1['children']) - len(T2['children'])
    elif T1['size'] != T2['size']:
        return T1['size'] - T2['size']
    else:
        return 1
    # elif T1['children_weights'] != T2['children_weights']:
    #     return sum(T1['children_weights']) - sum(T2['children_weights'])
    # else:
    #     for ind, c in enumerate(T1['children']):
    #         res = hd_sort_trees(c, T2['children'][ind])
    #         if res != 0:
    #             return res

def extract_matrix_from_hd_G(G):
    """
    Convert a Graph structure to the corresponding connectivity matrix.

    :param G:
    :return:
    """

    A = np.zeros((len(G['vertices']), len(G['vertices'])))
    for e in G['edges']:
        y = int(e[0]['name'])
        x = int(e[1]['name'])
        A[y, x] = 1
        A[x, y] = 1  # both directions because assuming un-directedness

    return A


def create_hd_G_from_matrix_bipartite(A, with_weights=True, weights_dict=False):
    """
    A is assumed to be square given that V=W
    Simply translate a matrix to the corresponding graph data structure.
    Assumes bipartitness i.e., V = W
    :param A:
    :return:
    """

    G = {
        'vertices': [],
        'edges': [],
    }

    for i in range(max(A.shape)):
        if with_weights:
            G['vertices'].append({
                'name': i,
                'nb': [],
                'crtree': [],
                'weights': []
            })
        else:
            G['vertices'].append({
                'name': i,
                'nb': [],
                'crtree': []
            })
        if weights_dict:
            G_weights = {}

    A = np.tril(A)
    for y, row in enumerate(A):
        for x, val in enumerate(row):
            if val:
                u = G['vertices'][y]
                v = G['vertices'][x]
                G['edges'].append((u,v))
                u['nb'].append(v)
                v['nb'].append(u)
                if with_weights:
                    u['weights'].append(val)
                    v['weights'].append(val)
                if weights_dict:
                    if u['name'] in G_weights:
                        G_weights[u['name']].append(val)
                    else:
                        G_weights.update({u['name']: [val]})
                    if v['name'] in G_weights:
                        G_weights[v['name']].append(val)
                    else:
                        G_weights.update({v['name']: [val]})

    if weights_dict:
        return G, G_weights
    else:
        return G

def create_hd_G_from_matrix(A, with_weights=True, weights_dict=False):
    """
    A is assumed to be square given that V=W
    Simply translate a matrix to the corresponding graph data structure.

    :param A:
    :return:
    """

    G = {
        'vertices': [],
        'edges': [],
    }

    for i in range(sum(A.shape)):
        if with_weights:
            G['vertices'].append({
                'name': i,
                'nb': [],
                'crtree': [],
                'weights': []
            })
        else:
            G['vertices'].append({
                'name': i,
                'nb': [],
                'crtree': []
            })
        if weights_dict:
            G_weights = {}

    for y, row in enumerate(A):
        for x, val in enumerate(row):
            if val:
                u = G['vertices'][y]
                v = G['vertices'][x + A.shape[0]]
                G['edges'].append((u,v))
                u['nb'].append(v)
                v['nb'].append(u)
                if with_weights:
                    u['weights'].append(val)
                    v['weights'].append(val)
                if weights_dict:
                    if u['name'] in G_weights:
                        G_weights[u['name']].append(val)
                    else:
                        G_weights.update({u['name']: [val]})
                    if v['name'] in G_weights:
                        G_weights[v['name']].append(val)
                    else:
                        G_weights.update({v['name']: [val]})

    if weights_dict:
        return G, G_weights
    else:
        return G


def transform_hd_colors_to_partitions(list_classes):
    """
    Takes a 'round' from the Holger Dell algorithm i.e., a list of colors/classes
    and transforms it to a partition compatible with the previous implementation.
    P = Q is assumed given that V = W.

    :param list_classes:
    :return:
    """

    P = []
    for ind_c, c in enumerate(list_classes):
        P.append([])
        for v in c['class']:
            P[ind_c].append(v['name'])
    Q = P.copy()

    return P, Q


"""
More Efficient Implementation of Color Refinement Than the Original
I.e., instead of running in O(N*M) it runs in O(N*N*log(N))

It is based on the pseudo-code from Grohe et al. 'Color Refinement and its Applications'
and extended to support weighted edges.
"""


def create_G_from_A(A):

    nodes = np.arange(sum(A.shape))
    edges = []
    G = {}
    for n in nodes:
        G.update({n: {
            'nb': [],
            'weights': []
        }})
    for ind_r, row in enumerate(A):
        for ind_c, val in enumerate(row):
            if val:
                u = ind_r
                v = nodes[A.shape[0] + ind_c]
                edges.append([u,v])
                G[u]['nb'].append(v)
                G[v]['nb'].append(u)
                G[u]['weights'].append(val)
                G[v]['weights'].append(val)
    for v in G:
        assert len(G[v]['nb']) == len(G[v]['weights'])
    return G


def trunc(values, decimals=0):
    if values > round(values)-0.01:  # avoiding the 6.99 vs 7 case
        return round(values)
    else:
        return np.trunc(values*10**decimals)/(10**decimals)


def cr_efficient(A, debug=False):
    import queue

    t0 = time.time()
    if isinstance(A, np.ndarray):
        G = create_G_from_A(A)
    elif isinstance(A, dict):
        G = A
        raise Exception('Not yet implemented. A transform from G -> A is missing.')
    else:
        raise Exception('Unknown Input Type.')

    nodes = list(G.keys())

    # TODO: correct this generally formulated condition to hold on both graphs and matrices
    # if G[0]['weights'] ... or \
    #         (isinstance(A, np.ndarray) and np.max(A) != 1 or np.min(A) != 0 and \
    #         (not np.allclose(A, np.ones(A.shape)) or not np.allclose(A, np.zeros(A.shape)))):
    if True:
        weighted = True
        decimal_precision = 5#10
        #print('Precision on Sum of Weights for Unification is {} decimals.'.format(decimal_precision))
    else:
        weighted = False
        decimal_precision = None

    C = np.vstack((nodes, np.ones(len(nodes)))).T
    D = np.vstack((nodes, np.zeros(len(nodes)))).T
    P = np.vstack((np.ones(len(nodes)), nodes)).T
    c_min = c_max = 1
    Q = queue.Queue()
    Q.put(1)
    D = np.vstack((nodes, np.zeros(len(nodes)))).T
    if weighted:
        E = np.vstack((nodes, np.zeros(len(nodes)))).T  # edge sum per vertex for weighted CR
        for v in G:
            E[v,1] = sum(G[v]['weights'])
    iter = 1
    while not Q.empty():
        # print('Computing Iteration {}...'.format(iter))
        q = Q.get()
        for v in nodes:
            s = set(G[v]['nb']).intersection(set([int(P[ind,1]) for ind in np.where(P[:,0] == q)[0]]))
            D[v,1] = len(s)
            if weighted:
                E[v,1] = sum([G[v]['weights'][G[v]['nb'].index(w)] for w in list(s)]) # eine Perle der Codegeschichte ;)
                E[v,1] = trunc(E[v, 1], decimals=decimal_precision)
        if weighted:
            g = sorted(nodes, key=lambda i: (C[i, 1], E[i,1]))
            g = np.hstack((np.array(g)[:, np.newaxis], np.array([(C[i, 1], E[i, 1]) for i in g])))
        else:
            g = sorted(nodes, key = lambda i: (C[i,1], D[i,1]))
            g = np.hstack((np.array(g)[:, np.newaxis], np.array([(C[i, 1], D[i, 1]) for i in g])))
        unique_row_indices = np.unique(g[:,1:], return_index=True, axis=0)[1]
        B = []
        for i, ind in enumerate(unique_row_indices):
            if i+1 == len(unique_row_indices):
                start = unique_row_indices[i]
                end = len(g[:,0])
            else:
                start = unique_row_indices[i]
                end = unique_row_indices[i+1]
            partition = [int(x) for x in g[:,0][start:end]]
            B.append(partition)
        for j in range(c_min, c_max + 1):
            Pc = [int(P[ind,1]) for ind in np.where(P[:,0] == j)[0]]
            indices_B_considered = []
            for ind, b in enumerate(B):
                for v in Pc:
                    if v in b:
                        indices_B_considered.append(ind)
                        break
            k1 = indices_B_considered[0]
            k2 = indices_B_considered[-1]
            i_star = np.argmax([len(B[i]) for i in range(k1,k2+1)]) + k1
            new_colors = list(range(k1,k2+1))
            new_colors.remove(i_star)
            new_colors = [c_max + i + 1 for i in new_colors]
            for c in new_colors:
                Q.put(c)
        if debug:
            import pdb; pdb.set_trace()
        c_min = c_max + 1
        c_max = c_max + len(B)
        for b in range(c_min, c_max + 1):
            partition = np.array(B[b-c_min])
            new_color_stack = np.vstack((b*np.ones(partition.shape), partition)).T
            P = np.vstack((P, new_color_stack))
            for x in B[b-c_min]:
                C[x,1] = b

        if debug:
            print('Debug Information:\n'
                  '\tC(v) =  {}\n'
                  '\tq={} for D and E\n'
                  '\tD(v) =  {}\n'
                  '\tE(v) =  {}\n'
                  '\tP(c) =  {}\n'
                  '\tB =  {}\n'
                  '\tQ =  {}\n'.format(str(C).replace('\n','\n\t\t'),
                                       q,
                                       str(D).replace('\n','\n\t\t'),
                                       str(E).replace('\n','\n\t\t'),
                                       str(P).replace('\n','\n\t\t'),
                                       B,
                                       list(Q.queue)))
            print('Completed Iteration {}'.format(iter))
            import pdb; pdb.set_trace()
        iter += 1
    C = np.array(sorted(C, key=lambda i: i[1]))
    unique_row_indices = np.unique(C[:, 1], return_index=True, axis=0)[1]
    C_sub = []
    for i, ind in enumerate(unique_row_indices):
        if i + 1 == len(unique_row_indices):
            start = unique_row_indices[i]
            end = len(C[:, 0])
        else:
            start = unique_row_indices[i]
            end = unique_row_indices[i + 1]
        partition = [int(x) for x in C[:, 0][start:end]]
        C_sub.append(partition)
    P = []
    Q = []
    for ind, c in enumerate(C_sub):
        if c[0] < A.shape[0]:
            P.append([])
        else:
            Q.append([])
        for v in c:
            if v < A.shape[0]:
                P[-1].append(v)
            else:
                Q[-1].append(v)
    P = sorted(P)
    Q = sorted(Q)
    r = list(np.arange(A.shape[0], A.shape[0] + sum([len(x) for x in Q])))
    for ic, c in enumerate(Q):
        for ix, x in enumerate(c):
            Q[ic][ix] = r.index(Q[ic][ix])
    #print('Total Time {} seconds'.format(time.time() - t0))
    return P, Q

'''
Visualization of Four Different Formulas for possible Runtimes
of the Color Refinement Algorithm due to different Implementations
'''
# f1 = lambda N, M: N * M
# f2 = lambda N, M: np.power(N,2) * np.log10(N)
# f3 = lambda N, M: (N + M) * np.log10(N)
# f4 = lambda M: M * np.log10(N)
# n = 30
# N = np.arange(1,n+1)
# for ind, M in enumerate([[comb(x,2) for x in N], [x*3 for x in N]]):
#     plt.figure(figsize=(12,9))
#     plt.plot(np.arange(len(N)), f1(N, M), label='N * M')
#     plt.plot(np.arange(len(N)), f2(N, M), label='N * N * log(N)')
#     plt.plot(np.arange(len(N)), f3(N, M), label='(N + M) * log(N)')
#     plt.plot(np.arange(len(N)), f4(M), label='M * log(N)')
#     plt.title('Algorithmic Runtimes for Implemented Color Refinement Routines')
#     plt.ylabel('Calculated Runtime')
#     if ind:
#         plt.xlabel('Indexes of Tuples (N,M) where N in {}...{} and M = N * 3'.format(N[0], N[-1]))
#     else:
#         plt.xlabel('Indexes of Tuples (N,M) where N in {}...{} and M = N choose 2'.format(N[0], N[-1]))
#     plt.legend()
#     plt.show()

'''
Speed comparison of solving high-dimensional or reducing first
or the LP from Example 1.1, solving directly is still clearly faster
however scaling to 'really high dimensional' matrices should turn the sides
Artifically creating bigger Matrix that contains symmetry information via Direct Sum
'''
# M = create_big_matrix_from_given(A, N=2)
# d1 = compare_speed_of_direct_and_cr_reduced_solving(M, alternate_cr_fun=None)
# d2 = compare_speed_of_direct_and_cr_reduced_solving(M, alternate_cr_fun=cr_efficient)

def test_color_refinement_partitions():
    """
    Use some confirmed results of color refinement for the CEP on
    different matrices, with all different implementations to be tested.

    IMPORTANT:
    Change the list of methods for this test.

    """

    A1 = np.array([
            [3, -1, 1, 0.25, 0.25, 0.25, 0.25, 0, 0, 3, -2, 0.5, 0.5, 1],
            [-1, 1, 3, 0.25, 0.25, 0.25, 0.25, 0, 0, -2, 3, 0.5, 0.5, 1],
            [1, 3, -1, 0.25, 0.25, 0.25, 0.25, 0, 0, 0.5, 0.5, 0.5, 0.5, 1],

            [0, 1 / 3, 2 / 3, 0, 3 / 2, 0, 3 / 2, 2, 0, 1, 0, -1, 0, 1],
            [1/3, 1/3, 1/3, 3/2, 0, 3/2, 0, 2, 0, 0, 1, 0, -1, 1],
            [1/3, 1/3, 1/3, 0, 3/2, 0, 3/2, 0, 2, -1, 0, 1, 0, 1],
            [2/3, 1/3, 0, 3/2, 0, 3/2, 0, 0, 2, 0, -1, 0, 1, 1],

            [2, 2, 2, 3/2, 3/2, 3/2, 3/2, 1, 1, 0.5, 0.5, 0.5, 0.5, 100],
    ])

    A2 = np.array([
        [  3.,   1.,   0.,   2.,   1.],
        [  1.,   3.,   2.,   0.,   1.],
        [  6.,   6.,   2.,   2., 100]
    ])

    A3 = np.array([
        [  4.,   2.,   1.],
        [ 12.,   4., 100.]
    ])

    A4 = np.array([
        [   1,  1,  0],
        [   0,  1,  1]
    ])

    A5 = np.array([
        [   1,  2,  0],
        [   0,  1,  1]
    ])

    A6 = np.array([
        [   2,  1,  0],
        [   0,  1,  2]
    ])

    test_matrices = [A1, A2, A3, A4, A5, A6]

    solution_A1 = ([[0, 1, 2], [3, 4, 5, 6], [7]], [[0, 1, 2], [3, 4, 5, 6], [7, 8], [9, 10, 11, 12], [13]])
    solution_A2 = ([[0, 1], [2]], [[0, 1], [2, 3], [4]])
    solution_A3 = ([[0], [1]], [[0], [1], [2]])
    solution_A4 = ([[0, 1]], [[0, 2], [1]])
    solution_A5 = ([[0], [1]], [[0], [1], [2]])
    solution_A6 = ([[0, 1]], [[0, 1, 2]])
    solutions = [solution_A1, solution_A2, solution_A3, solution_A4, solution_A5, solution_A6]

    for ind_m, method in enumerate([compute_partitions, cr_efficient]):

        for ind_mat, M in enumerate(test_matrices):

            pred = method(M)

            assert pred == solutions[ind_mat]

            print('     Method {} computed {}/{} CEPs correctly.                       '
                  .format(ind_m+1, ind_mat+1, len(test_matrices)), end='\r', flush=True)
        print()


def test_color_refinement_classes():

    # 5 node graph, like X with one end being connected
    A1 = np.array([
        [0., 1., 0., 0., 0.],
        [1., 0., 1., 1., 1.],
        [0., 1., 0., 1., 0.],
        [0., 1., 1., 0., 0.],
        [0., 1., 0., 0., 0.],
    ])
    A1mod = np.array([
        [0., 2., 0., 0., 0.],
        [2., 0., 1., 1., 1.],
        [0., 1., 0., 1., 0.],
        [0., 1., 1., 0., 0.],
        [0., 1., 0., 0., 0.],
    ])
    # 5 node graph, like Square with Tail
    A2 = np.array([
        [0., 1., 0., 0., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
    ])
    A2mod = np.array([
        [0., 1., 0., 0., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 2., 0.],
        [0., 0., 2., 0., 1.],
        [0., 1., 0., 1., 0.],
    ])
    # 5 node graph, like W (or chain simply)
    A3 = np.array([
        [0., 1., 0., 0., 0.],
        [1., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1.],
        [0., 0., 0., 1., 0.],
    ])
    A3mod = np.array([
        [0., 2., 0., 0., 0.],
        [2., 0., 1., 0., 0.],
        [0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 2.],
        [0., 0., 0., 2., 0.],
    ])
    A3mod2 = np.array([
        [0., 1., 0., 0., 0.],
        [1., 0., 2., 0., 0.],
        [0., 2., 0., 1., 0.],
        [0., 0., 1., 0., 1.],
        [0., 0., 0., 1., 0.],
    ])
    # 3 node graph, chain
    A4 = np.array([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ])
    A4mod = np.array([
        [0., 1., 0.],
        [1., 0., 2.],
        [0., 2., 0.]
    ])

    #test_matrices = [A4, A4mod, A3, A3mod, A3mod2, A2, A2mod, A1, A1mod]
    #solutions = [2, 3, 3, 2, 5, 4, 5, 3, 4]

    test_matrices = [A4,A3,A2,A1] + [A4mod, A3mod, A3mod2, A2mod, A1mod]
    solutions = [2, 3, 4, 3] + [3, 2, 5, 5, 4]
    titles = ['3 V chain', '5 V chain', '5 V square-tail', '5 V x'] + \
             ['3 V chain with one outer 2',
              '5 V chain with two outer 2',
              '5 V chain with one inner 2',
              '5 V square-tail with one 2',
              '5 V x with one outer 2']

    assert [len(compute_partitions(x)[0]) for x in test_matrices] == solutions # with initial method

    hit = []
    for ind, mat in enumerate(test_matrices):

        G, G_weights = create_hd_G_from_matrix_bipartite(mat, with_weights=False, weights_dict=True)

        trees = hd_cr(G, G_weights, debug=False, verbose=False)

        try:
            assert solutions[ind] == len(trees[-1])

            print('     '
                  'Computed Matrix #{} Color Class Cardinalities correctly.'
                  '\t\tDescription: {}                       \n'
                  .format(ind + 1, titles[ind]), end='\r', flush=True)
            hit.append(1)

        except Exception as e:

            print('\nFailed at Matrix {}:\n{}\n\n'
                  'Which has {} color classes, but the Algorithm gave back {}'.format(
                titles[ind], mat, solutions[ind], len(trees[-1])
            ))
            hit.append(0)
            #break

    print()
    print('Correct results for {}/{} matrices.'.format(sum(hit), len(test_matrices)))