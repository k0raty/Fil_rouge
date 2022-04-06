""" Import librairies """
import os
from math import pi, cos, sqrt, asin
import numpy

"""
Compute the distance in km between 2 coordinates around the earth

Parameters
----------
lat :float - the summits' latitude
lon :float - the summits' longitude
----------

Returns
-------
distance: float - the distance between the 2 given summits
-------
"""


def compute_distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    deg_2_rad = pi / 180
    a = 0.5 - cos((lat_2 - lat_1) * deg_2_rad) / 2
    b = cos(lat_1 * deg_2_rad) * cos(lat_2 * deg_2_rad) * (1 - cos((lon_2 - lon_1) * deg_2_rad)) / 2
    radius_earth = 6371

    distance = 2 * radius_earth * asin(sqrt(a + b))

    return distance


"""
Evaluate the cost of visiting sites with this configuration, depending on the number of cars
and the cost from one site to the next one

Parameters
----------
solution: list - list of all the visited sites from the first to the last visited
cost_matrix: numpy.ndarray - the cost of each travel to use to compute the fitness
nbr_of_vehicle: int - the number of vehicle used in the given solution
----------

Returns
-------
fitness_score:float - value of the cost of this configuration
-------
"""


def compute_fitness(solution: list, cost_matrix: numpy.ndarray, nbr_of_vehicle: int = 8) -> float:
    travel_cost = 0

    for delivery in solution:
        nbr_of_summit = len(delivery)

        for index_summit in range(nbr_of_summit - 1):
            summit_from = delivery[index_summit]
            summit_to = delivery[index_summit + 1]

            travel_cost += cost_matrix[summit_from][summit_to]

    penalty = 5
    vehicle_cost = penalty * nbr_of_vehicle

    return travel_cost + vehicle_cost


"""
Fill a matrix storing the cost of the travel between every customers

Parameters
----------
customers: list - the list of customers with their coordinates
depot: Utility.Depot - the unique depot of the problem
----------

Returns
-------
cost_matrix: numpy.ndarray - a matrix containing in a cell (i, j) the distance of the travel between
site i and j
-------
"""


def compute_cost_matrix(customers: list, depot) -> numpy.ndarray:
    nbr_of_customer = len(customers)
    cost_matrix = numpy.zeros((nbr_of_customer + 1, nbr_of_customer + 1))

    for index_i in range(nbr_of_customer):
        customer_i = customers[index_i]

        distance_to_depot = compute_distance(
            depot.LATITUDE,
            depot.LONGITUDE,
            customer_i.LATITUDE,
            customer_i.LONGITUDE,
        )

        cost_matrix[0, index_i] = distance_to_depot
        cost_matrix[index_i, 0] = distance_to_depot

        for index_j in range(nbr_of_customer):
            customer_j = customers[index_j]

            distance_from_i_to_j = compute_distance(
                customer_i.LATITUDE,
                customer_i.LONGITUDE,
                customer_j.LATITUDE,
                customer_j.LONGITUDE,
            )

            cost_matrix[index_i + 1, index_j + 1] = distance_from_i_to_j
            cost_matrix[index_j + 1, index_i + 1] = distance_from_i_to_j

    return cost_matrix


def set_root_dir():
    current_dir = os.getcwd()

    head, tail = os.path.split(current_dir)

    while tail != 'Fil_rouge' and len(head) > 0:
        head, tail = os.path.split(head)

    root_dir = os.path.join(head, tail)
    os.chdir(root_dir)


"""
Shortcut to get the class of an object
"""


def classe(instance):
    return type(instance).__name__
