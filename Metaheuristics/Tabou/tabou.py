""" Import librairies """
from copy import deepcopy
import os
import random as rd

""" Import utilities """
from Utility.database import Database
from Utility.common import compute_cost_matrix, compute_fitness, distance

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
            cost_matrix = compute_cost_matrix(customers)

        self.solution = None

        nbr_of_customers = len(customers)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_CUSTOMERS = nbr_of_customers
        self.PROBA_MUTATION = 1 / nbr_of_customers

        self.Customers = customers
        self.Depots = depots
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
        fitness = compute_fitness(initial_solution)

        for iteration in range(self.MAX_NEIGHBORS):
            neighbor = self.find_neighbor(initial_solution)
            neighbor_fitness = compute_fitness(neighbor)

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

        for index_summit in range(1, nbr_of_summit - 1):
            summit = self.Customers[delivery[index_summit]]

            package_weight = summit.TOTAL_WEIGHT_KG

            if nbr_of_summit >= 2:
                summit_i = self.Customers[delivery[nbr_of_summit - 1]]
                summit_j = self.Customers[delivery[nbr_of_summit - 2]]

                dist = distance(summit_i.LATITUDE, summit_i.LONGITUDE, summit_j.LATITUDE, summit_j.LONGITUDE)

                time += dist / self.VEHICLE_SPEED / 60

            time_min = summit.CUSTOMER_TIME_WINDOW_FROM_MIN
            time_max = summit.CUSTOMER_TIME_WINDOW_TO_MIN

            weight_criteria = package_weight > self.VEHICLE_CAPACITY
            time_criteria = time_min > time or time > time_max

            if weight_criteria or time_criteria:
                return False

            else:
                weight += package_weight

        return True

    """
    Crée une livraison pour un camion donné selon les paramètres définis:
    horaires de livraison respectées
    masse maximale des camions non dépassées

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

            weight += customer.TOTAL_WEIGHT_KG
            delivery.append(customer.INDEX)
            delivered_customers.append(customer.INDEX)

            if not (customer.INDEX in delivered_customers):
                package_weight = customer.TOTAL_WEIGHT_KG

                m = len(delivery)

                if m >= 2:
                    customer_i = self.Customers[index_customer_i]
                    customer_j = self.Customers[index_customer_j]

                    dist = distance(customer_i.LATITUDE, customer_i.LONGITUDE, customer_j.LATITUDE, customer_j.LONGITUDE)

                    time_from_i_to_j = dist / self.VEHICLE_SPEED / 60

                    total_time = current_time + time_from_i_to_j

                time_min = customer.CUSTOMER_TIME_WINDOW_FROM_MIN
                time_max = customer.CUSTOMER_TIME_WINDOW_TO_MIN

                if weight + package_weight < self.VEHICLE_CAPACITY and time_min <= total_time <= time_max:


                current_time += time_from_i_to_j

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
        all_treated_lines = []

        while len(all_treated_lines) < self.NBR_OF_CUSTOMERS - 1:
            all_treated_lines, delivery = self.generate_delivery(all_treated_lines)

            solution.append(delivery)

            # post-processing to remove duplicated customers
            all_treated_lines = list(set(all_treated_lines))

        return solution
