# Color Refinement

Based on [Grohe et al. "Dimension Reduction via Color Refinement"](https://arxiv.org/pdf/1307.5697.pdf). 

A graph can be viewed as a matrix and vice versa:
![graph_and_matrix_representations.pdf](imgs/graph_and_matrix_representations.pdf)

Given for instance following matrix, and running the subsequent function,

```python
A = np.array([
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
])
d = show_graph_and_partitions(A)
```

generates the corresponding bipartite graph:
![small_graph.png](imgs/small_graph.png)

The following plots show the Original Graph as a Matrix (color means there exists an edge for the given pair), and on the right is the *permutated* Matrix according to the partitions calculated by Color Refinement (also known as 1-dimensional Weisfeiler-Lehman):

![partitions.png](imgs/partitions.png)

Color Refinement finds the coarsest stable coloring. The plots show a kind of symmetry information.

The equitable partitions also allow for calculating the iterated core factor of a possibly big matrix:

![iterated_core_factor.pdf](imgs/iterated_core_factor.pdf)

Simply run

```python
A_itr_core = calculate_iterated_core_factor(A) 
```

