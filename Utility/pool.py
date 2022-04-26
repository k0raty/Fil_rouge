""" Import librairies """
import matplotlib.pyplot as plt
import numpy as np

""" Import utilities """
from Utility.common import compute_fitness


class Pool:
    pool = []

    def __init__(self, cost_matrix, vehicles, pool_max_size=10, pr=10):
        self.COST_MATRIX = cost_matrix
        self.Vehicles = vehicles
        self.PR = pr
        self.POOL_MAX_SIZE = pool_max_size

    """
    Compute the lambda parameter of 2 solutions
    Which gives the number of arcs not shared by the 2 solutions
    
    Parameters
    ----------
    solution_1 :list - one solution to the problem
    solution_2 :list - another solution to the problem
    ----------
    
    Returns
    -------
    counter :int - the number of arcs not shared by the 2 solutions
    -------
    """

    def count_non_shared_arc(self, solution_1, solution_2):
        list_arc_1 = []
        list_arc_2 = []

        for sub_road in solution_1:
            nbr_of_site = len(sub_road)
            list_arc_1 += [(sub_road[i], sub_road[i + 1]) for i in range(nbr_of_site)]

        for sub_road in solution_2:
            nbr_of_site = len(sub_road)
            list_arc_2 += [(sub_road[i], sub_road[i + 1]) for i in range(nbr_of_site)]

        counter = 0

        for arc in list_arc_1:
            if arc not in list_arc_2:
                counter += 1

        for arc in list_arc_2:
            if arc not in list_arc_1:
                counter += 1

        return counter

    """
    Compute the distance between 2 solutions thanks to their lambda parameter
    
    Parameters
    ----------
    solution_1 :list - one solution to the problem
    solution_2 :list - another solution to the problem
    ----------
    
    Returns
    -------
    distance :float - the distance between the 2 solutions
    -------
    """

    def distance_between_solution(self, solution_1, solution_2):
        distance = 0
        coeff_lambda = self.count_non_shared_arc(solution_1, solution_2)

        if coeff_lambda <= self.PR:
            distance = 1 - (coeff_lambda / self.PR)

        return distance

    """ 
    Évalue la proximité de solutions S d'un pool avec la solution L.
    Plus la somme est grande, plus les solutions sont proches, et donc il faudra en supprimer
    
    Parameters
    ----------
    new_solution: list - a solution given by a metaheuristic
    ----------
    """

    def inject_in_pool(self, new_solution, new_fitness):
        pool_occupation = len(self.pool)

        if pool_occupation < self.POOL_MAX_SIZE:
            self.pool.append(new_solution)
            return

        index_candidate = -1
        fitness_candidate = np.inf

        for index_solution in range(pool_occupation):
            solution = self.pool[index_solution]

            fitness = compute_fitness(solution, self.COST_MATRIX, self.Vehicles)

            if fitness > new_fitness:
                continue

            distance = self.distance_between_solution(new_solution, solution)

            if distance <= self.PR:
                continue

            if fitness < fitness_candidate:
                index_candidate = index_solution
                fitness_candidate = fitness

        if index_candidate > -1:
            self.pool[index_candidate] = new_solution
            self.pool = sorted(self.pool, key=lambda x: compute_fitness(x, self.COST_MATRIX, self.Vehicles))

    """ Trace le graphe représentant la distance des solutions contenues dans S à la solution L """

    def graph(self, new_solution):
        pool_occupation = len(self.pool)

        X = [1 for i in range(pool_occupation)]
        Y = [self.count_non_shared_arc(new_solution, self.pool[i]) for i in range(pool_occupation)]

        plt.scatter(X, Y, s=100, alpha=0.5)
        plt.scatter(1, 0, s=150, c='red')
        plt.axhline(self.PR, color='black', linestyle='dashdot')

        plt.title("Graphe de similarité entre la solution L et le pool de solutions S")
        plt.ylabel('$\lambda_{new solution - solutions in the pool}$')
        plt.legend(['pr'], loc='best')

        plt.show()
