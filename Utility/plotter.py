""" Import librairies """
import matplotlib.pyplot as plt
from seaborn import color_palette


"""
Draw the Graph showing all the customers summits and the depots, with the road taken to go through them

Parameters
----------
solution: list - an individual of the population with the lowest fitness score
filepath: string - a path where to save the plotted image
----------
"""


def plot_solution(solution, graph, title='Solution to the VRP', filepath=None):
    plt.figure(figsize=[10, 7])

    colors = color_palette(n_colors=len(solution))

    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]

        list_x = [graph.nodes[index_customer]['pos'][0] for index_customer in delivery]
        list_y = [graph.nodes[index_customer]['pos'][1] for index_customer in delivery]

        vehicle_cost = graph.nodes[0]['Vehicles']["VEHICLE_VARIABLE_COST_KM"][index_delivery]
        label = 'Vehicle n°{} cost : {}'.format(index_delivery, vehicle_cost)

        plt.plot(list_x, list_y, label=label, marker='o', markerfacecolor='blue',
                 markeredgecolor='blue', markersize=1, linestyle='solid', linewidth=0.8,
                 color=colors[index_delivery],
                 )

    depot = graph.nodes[0]
    plt.plot(depot['pos'][0], depot['pos'][1], 'rs', markersize=1.2)

    plt.xlabel('coordinate x (in km)')
    plt.ylabel('coordinate y (in km)')
    plt.title(title)

    plt.legend(loc='upper left')
    plt.show()

    if filepath is not None:
        fig = plt.gcf()
        fig.savefig(filepath, format='png')


"""
Draw the Graph of the sum of the fitness values of the population at every iteration

:param iteration: int - the number of the iteration in the algorithm shown
:param history: list - the list of the fitness values for each previous iteration
"""


def plot_fitness(iteration, history, title='Fitness evolution', filepath=None):
    plt.figure()

    plt.plot(range(iteration), history)

    plt.xlabel('iteration')
    plt.ylabel('Mean fitness')
    plt.title(title)

    plt.show()

    if filepath is not None:
        fig = plt.gcf()
        fig.savefig(filepath, format='png')
