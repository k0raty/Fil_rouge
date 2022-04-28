""" Import librairies """
from tqdm import tqdm
from math import sqrt


"""
Measure the length of a solution to the TSP, which has the same shape as a delivery of the VRP

Parameters
----------
tsp_solution: list - a solution to the TSP
graph: ? - the graph of the problem
----------

Returns
-------
distance: float - the length of the solution
-------
"""


def compute_delivery_length(tsp_solution, graph):
    length = 0

    depot = graph.nodes[0]
    x_from, y_from = depot['pos']

    for index_customer in tsp_solution:
        customer = graph.nodes[index_customer]
        x_to, y_to = customer['pos']

        length += sqrt((x_to - x_from) ** 2 + (y_to - y_from) ** 2)

        x_from, y_from = x_to, y_to

    x_to, y_to = depot['pos']
    length += sqrt((x_to - x_from) ** 2 + (y_to - y_from) ** 2)

    return length


def permutation(tsp_solution, graph):
    length = compute_delivery_length(tsp_solution, graph)
    d0 = length + 1

    while length < d0:
        d0 = length

        for i in tqdm(range(len(tsp_solution))):
            for j in range(i + 2, len(tsp_solution)):
                shuffle = tsp_solution[i:j].copy()
                shuffle.reverse()

                new_tsp_solution = tsp_solution[:i] + shuffle + tsp_solution[j:]
                new_length = compute_delivery_length(new_tsp_solution, graph)

                if new_length < length:
                    length = new_length
                    tsp_solution = new_tsp_solution

    return tsp_solution


"""
Generate a random order to deliver all customers, and then improve it by minimizing the length of the delivery

Parameters
----------
graph: ? - the graph of the problem
----------

Returns
-------
tsp_solution: list - a solution to the TSP
-------
"""


def solve_tsp(graph):
    tsp_solution = list(range(1, len(graph)))

    length = compute_delivery_length(tsp_solution, graph)
    print('initial length of the TSP solution {}'.format(length))

    tsp_solution = permutation(tsp_solution, graph)

    length = compute_delivery_length(tsp_solution, graph)
    print('final length of the TSP solution {}'.format(length))

    tsp_solution = [0, *tsp_solution, 0]

    return tsp_solution
