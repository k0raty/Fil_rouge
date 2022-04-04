""" Import librairies """
from math import pi, cos, sqrt, asin
import numpy as np

"""
Compute the distance in km between 2 coordinates around the earth

:param lat_1: float - from customer's latitude
:param lon_1: float - from customer's longitude
:param lat_2: float - to customer's latitude
:param lon_2: float - to customer's longitude
"""


def distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    deg_2_rad = pi / 180
    a = 0.5 - cos((lat_2 - lat_1) * deg_2_rad) / 2
    b = cos(lat_1 * deg_2_rad) * cos(lat_2 * deg_2_rad) * (1 - cos((lon_2 - lon_1) * deg_2_rad)) / 2
    radius_earth = 6371

    return 2 * radius_earth * asin(sqrt(a + b))


"""
Evaluate the cost of visiting sites with this configuration, depending on the number of cars
and the cost from one site to the next one

@param{list} individual - list of all the visited sites from the first to the last visited
@return{int} value of the cost of this configuration
"""


def fitness(solution: list, cost_matrix) -> float:
    travel_cost = 0

    nbr_of_stop = len(solution)

    for index in range(nbr_of_stop - 1):
        site_from = solution[index]
        site_to = solution[index + 1]

        class_name_from = type(site_from).__name__
        class_name_to = type(site_to).__name__

        if class_name_from != 'Depot' and class_name_to != 'Depot':
            travel_cost += cost_matrix[int(site_from.CUSTOMER_ID), int(site_to.CUSTOMER_ID)]

        elif class_name_from == 'Depot':
            travel_cost += distance(
                float(site_to.CUSTOMER_LATITUDE),
                float(site_to.CUSTOMER_LONGITUDE),
                float(site_from.DEPOT_LATITUDE),
                float(site_from.DEPOT_LONGITUDE),
            )

        elif class_name_to != 'Depot':
            travel_cost += distance(
                float(site_from.CUSTOMER_LATITUDE),
                float(site_from.CUSTOMER_LONGITUDE),
                float(site_to.DEPOT_LATITUDE),
                float(site_to.DEPOT_LONGITUDE),
            )

    return travel_cost


"""
Fill a matrix storing the cost of the travel between every customers

:param customers: list - the list of customers with their coordinates
"""


def compute_cost_matrix(customers: list):
    nbr_of_customer = len(customers)
    cost_matrix = np.zeros((nbr_of_customer, nbr_of_customer))

    for i in range(nbr_of_customer):
        customer_i = customers[i]

        for j in range(nbr_of_customer):
            customer_j = customers[j]
            lat_i = float(customer_i.CUSTOMER_LATITUDE)
            lon_i = float(customer_i.CUSTOMER_LONGITUDE)
            lat_j = float(customer_j.CUSTOMER_LATITUDE)
            lon_j = float(customer_j.CUSTOMER_LONGITUDE)

            cost_matrix[i, j] = distance(lat_i, lon_i, lat_j, lon_j)

    return cost_matrix