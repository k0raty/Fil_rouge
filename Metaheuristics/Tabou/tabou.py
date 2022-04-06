""" Import librairies """
from copy import deepcopy
import os
import random as rd

""" Import utilities """
from Utility.database import Database
from Utility.common import compute_cost_matrix, compute_fitness, compute_distance

os.chdir(os.path.join('', '..'))


class Tabou:
    MAX_ITERATION = 10
    MAX_NEIGHBORS = 20

    VEHICLE_SPEED = 50
    VEHICLE_CAPACITY = 5000

    def __init__(self, customers=None, depots=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depots = database.Depots
            cost_matrix = compute_cost_matrix(customers, depots[0])

        self.solution = None

        nbr_of_customers = len(customers)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_CUSTOMERS = nbr_of_customers
        self.PROBA_MUTATION = 1 / nbr_of_customers

        self.Customers = customers
        self.Depot = depots[0]
        self.Vehicles = vehicles[0]

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

    def main(self, initial_solution=None):
        if initial_solution is None:
            initial_solution = self.initialisation()

        solution, fitness = self.find_best_neighbor(initial_solution)

        for iteration in range(self.MAX_ITERATION):
            solution, fitness = self.find_best_neighbor(solution)

            print('Iteration {}, fitness {}'.format(iteration, fitness))

        self.solution = solution

        return self.solution, fitness

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

    def find_best_neighbor(self, initial_solution):
        solution = initial_solution
        fitness = compute_fitness(initial_solution, self.COST_MATRIX)

        for iteration in range(self.MAX_NEIGHBORS):
            neighbor = self.find_neighbor(initial_solution)
            neighbor_fitness = compute_fitness(neighbor, self.COST_MATRIX)

            if neighbor_fitness >= fitness and self.is_solution_valid(neighbor):
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
    -------
    """

    @staticmethod
    def find_neighbor(solution):
        neighbor = deepcopy(solution)

        nbr_of_delivery = len(solution)

        index_delivery_i = rd.randint(0, nbr_of_delivery - 1)
        index_delivery_j = rd.randint(0, nbr_of_delivery - 1)

        delivery_i = solution[index_delivery_i]
        delivery_j = solution[index_delivery_j]

        length_delivery_i = len(delivery_i)
        length_delivery_j = len(delivery_j)

        if length_delivery_i >= 3 and length_delivery_j >= 3:
            index_summit_i = rd.randint(1, length_delivery_i - 1)
            index_summit_j = rd.randint(1, length_delivery_j - 1)

            summit_i = delivery_i[index_summit_i]
            summit_j = delivery_j[index_summit_j]

            neighbor[index_delivery_i][index_summit_i] = summit_j
            neighbor[index_delivery_j][index_summit_j] = summit_i

        return neighbor

    """
    Check if a solution match the validity criteria 

    Parameters
    -------
    solution: list - a given solution
    -------
    
    Returns
    -------
    :bool - the validity of the given solution
    -------
    """

    def is_solution_valid(self, solution):
        nbr_of_delivery = len(solution)

        for index_delivery in range(nbr_of_delivery):
            delivery = solution[index_delivery]

            if not self.is_delivery_valid(delivery):
                return False

        return True

    """
    Check if a delivery match the validity criteria (for instance time constraints and vehicle's capacity)

    Parameters
    -------
    delivery: list - a portion of a solution
    -------
    
    Returns
    -------
    :bool - the validity of the given delivery
    -------
    """

    def is_delivery_valid(self, delivery):
        weight = 0
        time = 481

        nbr_of_summit = len(delivery)

        # we don't look at the arc with the depot
        for index_summit in range(1, nbr_of_summit - 1):
            summit = delivery[index_summit]
            customer = self.Customers[summit - 1]

            weight += customer.TOTAL_WEIGHT_KG

            if nbr_of_summit > 1:
                previous_summit = delivery[index_summit - 1]
                previous_customer = self.Customers[previous_summit - 1]

                dist = compute_distance(
                    previous_customer.LATITUDE,
                    previous_customer.LONGITUDE,
                    customer.LATITUDE,
                    customer.LONGITUDE,
                )

                time += dist / self.VEHICLE_SPEED / 60

            weight_criteria = weight > self.VEHICLE_CAPACITY
            time_criteria = customer.CUSTOMER_TIME_WINDOW_FROM_MIN > time \
                            or time > customer.CUSTOMER_TIME_WINDOW_TO_MIN

            if weight_criteria or time_criteria:
                return False

        return True

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

        for index_customer in range(self.NBR_OF_CUSTOMERS):
            customer = self.Customers[index_customer]

            if customer.INDEX in delivered_customers:
                continue

            if weight + customer.TOTAL_WEIGHT_KG > self.VEHICLE_CAPACITY:
                continue

            potential_current_time = current_time

            if len(delivery) > 1:
                previous_customer = self.Customers[delivery[-1]]

                dist = compute_distance(previous_customer.LATITUDE, previous_customer.LONGITUDE, customer.LATITUDE,
                                        customer.LONGITUDE)

                time_to_new_customer = dist / self.VEHICLE_SPEED / 60

                potential_current_time += time_to_new_customer

            if customer.CUSTOMER_TIME_WINDOW_FROM_MIN > potential_current_time \
                    or potential_current_time > customer.CUSTOMER_TIME_WINDOW_TO_MIN:
                continue

            current_time = potential_current_time
            weight += customer.TOTAL_WEIGHT_KG
            delivery.append(customer.INDEX)
            delivered_customers.append(customer.INDEX)

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

        while len(delivered_customers) < self.NBR_OF_CUSTOMERS:
            delivered_customers, delivery = self.generate_delivery(delivered_customers)

            solution.append(delivery)

            # post-processing to remove duplicated customers
            delivered_customers = list(set(delivered_customers))

        return solution
