""" Import librairies """
import random as rd
from copy import deepcopy

""" Import utilities """
from Utility.common import generate_initial_solution, set_root_dir, compute_fitness
from Utility.validator import is_solution_valid, pick_valid_solution
from Utility.plotter import plot_solution
from Utility.database import Database

set_root_dir()


class Tabou:
    MAX_ITERATION = 10
    MAX_NEIGHBORS = 20

    VEHICLE_CAPACITY = 5000

    fitness: float = 0
    solution: list = []
    tabou_list: list = []

    """
    Initialize the genetic algorithm with proper parameters and problem's data

    Parameters
    ----------
    customers: list - the list of customers to be served
    vehicles: list - the list of vehicles that can be used to deliver
    depot: Utility.depot - the unique depot of the delivering company
    cost_matrix: numpy.ndarray - the cost of the travel from one summit to another
    ----------
    """

    def __init__(self, graph=None, vehicle_speed=40):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph

        self.Graph = graph

        self.NBR_OF_CUSTOMER = len(graph) - 1
        self.NBR_OF_VEHICLE = len(graph.nodes[0]['Vehicles'])

    """
    Apply the tabou algorithm to improve an initial solution over a maximum number of iterations 

    Parameters
    ----------
    initial_solution: list - a given solution that the algorithm must improve
    -------
    
    Returns
    -------
    solution: list - new solution more optimized than the initial one
    fitness: float - the fitness score of the new solution
    -------
    """

    def main(self, initial_solution=None, speedy=False):
        if initial_solution is None:
            iteration = 0
            initial_solution = self.initialisation()

            while not is_solution_valid(initial_solution, self.Graph) and iteration < 5:
                initial_solution = self.initialisation()
                iteration += 1

            if iteration == 5:
                initial_solution = pick_valid_solution()

        plot_solution(initial_solution, self.Graph, title='Initial solution to the VRP')

        solution, fitness = self.find_best_neighbor(initial_solution)

        for iteration in range(self.MAX_ITERATION):
            solution, fitness = self.find_best_neighbor(solution)

            print('Iteration {}, fitness {}'.format(iteration, fitness))

        self.solution = solution
        self.fitness = fitness

        plot_solution(self.solution, self.Graph)

    """
    Find the best solution in the close neighborhood of the given solution
    
    Parameters
    ----------
    initial_solution: list - a given solution
    ----------
    
    Returns
    -------
    solution: list - the best neighbor of the given solution
    fitness: float - the fitness score of the best neighbor of the given solution
    """

    def find_best_neighbor(self, initial_solution: list):
        solution = initial_solution
        fitness = compute_fitness(initial_solution, self.Graph)

        for iteration in range(self.MAX_NEIGHBORS):
            neighbor, inversion_couple = self.find_neighbor(initial_solution)
            neighbor_fitness = compute_fitness(neighbor, self.Graph)

            is_fitness_better = neighbor_fitness < fitness
            is_neighbor_valid = is_solution_valid(neighbor, self.Graph)
            is_couple_new = inversion_couple not in self.tabou_list

            if is_fitness_better and is_neighbor_valid and is_couple_new:
                solution = neighbor
                fitness = neighbor_fitness

        return solution, fitness

    """
    Switch 2 summits of the given solution to create a new solution really close to the given one
    
    Parameters
    ----------
    solution: list - a given solution
    ----------
    
    Returns
    -------
    neighbor: list - a new solution close to the given one
    inversion_couple: tuple - the 2 summits that were inverted to add to the tabou list
    -------
    """

    @staticmethod
    def find_neighbor(solution: list):
        neighbor = deepcopy(solution)

        nbr_of_delivery = len(solution)

        index_delivery_i = rd.randint(0, nbr_of_delivery - 1)
        index_delivery_j = rd.randint(0, nbr_of_delivery - 1)

        delivery_i = solution[index_delivery_i]
        delivery_j = solution[index_delivery_j]

        length_delivery_i = len(delivery_i)
        length_delivery_j = len(delivery_j)

        if length_delivery_i > 2 and length_delivery_j > 2:
            index_summit_i = rd.randint(1, length_delivery_i - 1)
            index_summit_j = rd.randint(1, length_delivery_j - 1)

            summit_i = delivery_i[index_summit_i]
            summit_j = delivery_j[index_summit_j]

            neighbor[index_delivery_i][index_summit_i] = summit_j
            neighbor[index_delivery_j][index_summit_j] = summit_i

            inversion_couple = (index_delivery_i, index_delivery_j, summit_i, summit_j)

        return neighbor, inversion_couple

    """
    Generate an initial valid solution taking into account the time and weigh constraints 
    
    Returns
    -------
    solution: list - a solution of the problem containing all the deliveries to be done
    -------
    """

    def initialisation(self):
        solution = generate_initial_solution(self.Graph)

        return solution
