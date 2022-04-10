""" Import librairies """
import os
from math import pi, cos, sqrt, asin
import numpy as np
import utm
import networkx as nx
import pandas as pd

v=50 #Vitesse des véhicules
df_customers= pd.read_excel("table_2_customers_features.xls")
df_vehicles=pd.read_excel("table_3_cars_features.xls")


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


def compute_fitness(solution: list, cost_matrix: numpy.ndarray, vehicles: list) -> float:
    penalty = 5
    nbr_of_vehicle = len(solution)

    solution_cost = nbr_of_vehicle * penalty

    for index_vehicle in range(nbr_of_vehicle):
        vehicle = vehicles[index_vehicle]
        cost_by_distance = vehicle.VEHICLE_VARIABLE_COST_KM

        delivery_distance = 0

        delivery = solution[index_vehicle]
        nbr_of_summit = len(delivery)

        for index_summit in range(nbr_of_summit - 1):
            summit_from = delivery[index_summit]
            summit_to = delivery[index_summit + 1]

            delivery_distance += cost_matrix[summit_from][summit_to]

        solution_cost += delivery_distance * cost_by_distance

    return solution_cost


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


def create_G(df_customers, df_vehicles, v):
    """
    Parameters
    ----------
    df_customers : Dataframe contenant les informations sur les clients
    df_vehicles : Dataframe contenant les informations sur les camions livreurs
    Returns
    -------
    G : Graph plein généré
    """
    n_max = len(df_vehicles)  # Le nombre maximum de voitures qu'on mettrai à disposition.
    n_sommet = len(df_customers)
    G = nx.empty_graph(n_sommet)
    (x_0, y_0) = utm.from_latlon(43.37391833, 17.60171712)[:2]
    dict_0 = {'CUSTOMER_CODE': 0, 'CUSTOMER_LATITUDE': 43.37391833, 'CUSTOMER_LONGITUDE': 17.60171712,
              'CUSTOMER_TIME_WINDOW_FROM_MIN': 360, 'CUSTOMER_TIME_WINDOW_TO_MIN': 1080, 'TOTAL_WEIGHT_KG': 0,
              'pos': (x_0, y_0), "CUSTOMER_DELIVERY_SERVICE_TIME_MIN": 0}
    G.nodes[0].update(dict_0)
    G.nodes[0]['n_max'] = n_max  # Nombre de voiture maximal
    dict_vehicles = df_vehicles.to_dict()
    G.nodes[0]['Camion'] = dict_vehicles
    for i in range(1, len(G.nodes)):
        dict = df_customers.iloc[i].to_dict()
        dict['pos'] = utm.from_latlon(dict['CUSTOMER_LATITUDE'], dict['CUSTOMER_LONGITUDE'])[:2]
        G.nodes[i].update(dict)

        ###On rajoute les routes###
    for i in range(0, len(G.nodes)):
        for j in range(0, len(G.nodes)):
            if i != j:
                z_1 = G.nodes[i]['pos']
                z_2 = G.nodes[j]['pos']
                G.add_edge(i, j, weight=get_distance(z_1, z_2))
                G[i][j]['time'] = (G[i][j]['weight'] / v) * 60

    G.nodes[0]['n_max'] = n_max  # Nombre de voiture maximal
    p = [2, -100, n_sommet]  # Equation pour trouver n_min
    roots = np.roots(p)
    n_min = max(1,
                int(roots.min()) + 1)  # Nombre de voiture minimal possible , solution d'une équation de second degrès.
    G.nodes[0]['n_min'] = n_min

    return G

def get_distance(z_1,z_2):
    """
    Distance entre deux points sur plan z_1 et z_2
    """
    x_1,x_2,y_1,y_2=z_1[0],z_2[0],z_1[1],z_2[1]
    d=math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
    return d/1000 #en km