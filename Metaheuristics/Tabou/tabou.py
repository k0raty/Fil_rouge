""" Import librairies """
from copy import deepcopy
import random as rd
from os.path import join

""" Import utilities """
from Utility.database import Database
from Utility.common import *
from Utility.validator import *

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

    def __init__(self, cost_matrix=None, graph=None, vehicle_speed=40):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph
            cost_matrix = compute_cost_matrix(graph)

        self.COST_MATRIX = cost_matrix
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

        solution, fitness = self.find_best_neighbor(initial_solution)

        for iteration in range(self.MAX_ITERATION):
            solution, fitness = self.find_best_neighbor(solution)

            print('Iteration {}, fitness {}'.format(iteration, fitness))

        self.solution = solution
        self.fitness = fitness

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
        fitness = compute_fitness(initial_solution, self.COST_MATRIX, self.Graph)

        for iteration in range(self.MAX_NEIGHBORS):
            neighbor, inversion_couple = self.find_neighbor(initial_solution)
            neighbor_fitness = compute_fitness(neighbor, self.COST_MATRIX, self.Graph)

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
    Generate a delivery for a portion of the solution, accordingly to the time and weigh constraints

    Parameters
    ----------
    delivered_customers: list - the list of the customers indexes who have been delivered so far in the solution
    -------
    
    Returns
    -------
    delivered_customers: list - the list of the customers indexes who have been delivered so far in the solution
    delivery: list - a portion of the solution
    """

    def generate_delivery(self, delivered_customers):
        weight = 0
        delivery = [0]

        current_time = 481

        seed = [index for index in range(1, len(self.Graph))]

        while len(seed) > 0:
            index_customer = rd.sample(seed, 1)[0]
            index_value = seed.index(index_customer)
            seed.pop(index_value)

            customer = self.Graph.nodes[index_customer]

            if customer['INDEX'] in delivered_customers:
                continue

            if weight + customer['TOTAL_WEIGHT_KG'] > self.VEHICLE_CAPACITY:
                continue

            potential_current_time = current_time

            if len(delivery) > 1:
                previous_customer = self.Graph.nodes[delivery[-1]]

                dist = compute_spherical_distance(
                    previous_customer['CUSTOMER_LATITUDE'],
                    previous_customer['CUSTOMER_LONGITUDE'],
                    customer['CUSTOMER_LATITUDE'],
                    customer['CUSTOMER_LONGITUDE'],
                )

                time_to_new_customer = dist / self.Graph.nodes[0]['VEHICLE_SPEED'] / 60

                potential_current_time += time_to_new_customer

            if customer['CUSTOMER_TIME_WINDOW_FROM_MIN'] > potential_current_time \
                    or potential_current_time > customer['CUSTOMER_TIME_WINDOW_TO_MIN']:
                print('customer out of time', customer)
                continue

            current_time = potential_current_time
            weight += customer['TOTAL_WEIGHT_KG']
            delivery.append(customer['INDEX'])
            delivered_customers.append(customer['INDEX'])

            if weight >= self.VEHICLE_CAPACITY:
                break

        delivery.append(0)

        return delivered_customers, delivery

    """
    Generate an initial valid solution taking into account the time and weigh constraints 
    
    Parameters
    -------
    
    -------
    
    Returns
    -------
    solution: list - a solution of the problem containing all the deliveries to be done
    -------
    """

    def initialisation(self):
        solution = []
        delivered_customers = []

        while len(delivered_customers) < self.NBR_OF_CUSTOMER:
            delivered_customers, delivery = self.generate_delivery(delivered_customers)
            solution.append(delivery)

            # post-processing to remove duplicated customers
            delivered_customers = list(set(delivered_customers))

        return solution
