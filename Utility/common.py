""" Import librairies """
import os
from math import pi, cos, sqrt, asin
import numpy as np
import utm
import networkx as nx
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

from Utility.tsp_solver import solve_tsp
from Utility.validator import is_delivery_time_valid, is_solution_valid
from Utility.plotter import plot_solution


def set_root_dir():
    current_dir = os.getcwd()

    head, tail = os.path.split(current_dir)

    while tail != 'Fil_rouge' and len(head) > 0:
        head, tail = os.path.split(head)

    root_dir = os.path.join(head, tail)
    os.chdir(root_dir)


set_root_dir()

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


def compute_spherical_distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    deg_2_rad = pi / 180
    a = 0.5 - cos((lat_2 - lat_1) * deg_2_rad) / 2
    b = cos(lat_1 * deg_2_rad) * cos(lat_2 * deg_2_rad) * (1 - cos((lon_2 - lon_1) * deg_2_rad)) / 2
    radius_earth = 6371

    distance = 2 * radius_earth * asin(sqrt(a + b))

    return distance


"""
Compute the distance in km between 2 coordinates on a plan

Parameters
----------
pos_1 :tuple - first summit coordinates
pos_2 :tuple - second summit coordinates 
----------

Returns
-------
distance :float - the distance between the 2 points
-------
"""


def compute_plan_distance(pos_1: tuple, pos_2: tuple) -> float:
    x_1, x_2, y_1, y_2 = pos_1[0], pos_2[0], pos_1[1], pos_2[1]
    distance = sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

    return distance


"""
Evaluate the cost of visiting sites with this configuration, depending on the number of cars
and the cost from one site to the next one

Parameters
----------
solution: list - list of all the visited sites from the first to the last visited
nbr_of_vehicle: int - the number of vehicle used in the given solution
----------

Returns
-------
fitness_score:float - value of the cost of this configuration
-------
"""


def compute_fitness(solution: list, graph) -> float:
    fitness = 0

    for index_delivery in range(len(solution)):
        fitness += compute_delivery_fitness(solution, graph, index_delivery)

    return fitness


"""
Fonction coût partielle pour le recuit qui calcule uniquement le coût d'un trajet uniquement

Parameters
----------
solution: list - a solution to the VRP 
graph: networkx.Graph - graph of the problem
index_delivery: int - the index of the delivery of interest
----------

Returns
-------
delivery_fitness: float - the cost of the delivery
-------
"""


def compute_delivery_fitness(solution, graph, index_delivery):
    delivery = solution[index_delivery]

    cost_by_distance = graph.nodes[0]['Vehicles']['VEHICLE_VARIABLE_COST_KM'][index_delivery]
    distance = 0

    # TODO: take into account the cost of going from and to the depot
    for index_summit in range(1, len(delivery) - 2):
        summit_from = delivery[index_summit]
        summit_to = delivery[index_summit + 1]

        if summit_from == summit_to:
            print('Error : two consecutive nodes are equal')

        distance += graph[summit_from][summit_to]['weight']

    delivery_fitness = cost_by_distance * distance

    return delivery_fitness


"""
Define Gini

Parameters
----------
model: ModelSma - the model gathering the agents containing the metaheuristics
----------

Returns 
-------
gini: float - the gini score
-------
"""


def compute_gini(model) -> float:
    agents_fitness = sorted([agent.fitness for agent in model.schedule.agents])

    total_fitness = sum(agents_fitness)

    part_a = model.nbr_of_agent * total_fitness
    part_b = sum(fitness * (model.nbr_of_agent - index) for index, fitness in enumerate(agents_fitness))

    gini = 1 + (1 / model.nbr_of_agent) - 2 * part_a / part_b

    return gini


"""
Parameters
----------
df_customers : Dataframe contenant les informations sur les clients
df_vehicles : Dataframe contenant les informations sur les camions livreurs
----------

Returns
-------
graph : Graph plein généré
-------
"""


def create_graph(df_customers, df_vehicles, vehicle_speed=50):
    nbr_of_vehicle = len(df_vehicles)
    nbr_of_summit = len(df_customers)

    graph = nx.empty_graph(nbr_of_summit + 1)  # we add one for the depot

    pos_depot = utm.from_latlon(43.37391833, 17.60171712)[:2]
    depot_x = pos_depot[0] / (10 ** 3)
    depot_y = pos_depot[1] / (10 ** 3)

    depot_dict = {
        'CUSTOMER_CODE': 0,
        'CUSTOMER_LATITUDE': 43.37391833,
        'CUSTOMER_LONGITUDE': 17.60171712,
        'CUSTOMER_TIME_WINDOW_FROM': 360,
        'CUSTOMER_TIME_WINDOW_TO': 1080,
        'TOTAL_WEIGHT_KG': 0,
        'pos': (depot_x, depot_y),
        'CUSTOMER_DELIVERY_SERVICE_TIME': 0,
        'INDEX': 0,
    }

    graph.nodes[0].update(depot_dict)

    graph.nodes[0]['VEHICLE_SPEED'] = vehicle_speed
    graph.nodes[0]['NBR_OF_VEHICLE'] = nbr_of_vehicle

    dict_vehicles = df_vehicles.to_dict()

    graph.nodes[0]['Vehicles'] = dict_vehicles

    for index_customer in range(nbr_of_summit):
        customer = df_customers.iloc[index_customer]
        customer_dict = customer.to_dict()

        latitude = customer_dict['CUSTOMER_LATITUDE']
        longitude = customer_dict['CUSTOMER_LONGITUDE']

        pos = utm.from_latlon(latitude, longitude)[:2]
        pos_x = pos[0] / (10 ** 3)
        pos_y = pos[1] / (10 ** 3)
        customer_dict['pos'] = (pos_x, pos_y)

        graph.nodes[customer_dict['INDEX']].update(customer_dict)

    for index_summit_i in range(len(graph.nodes)):
        summit_i = graph.nodes[index_summit_i]

        for index_summit_j in range(len(graph.nodes)):
            summit_j = graph.nodes[index_summit_j]

            if index_summit_i != index_summit_j:
                distance = compute_plan_distance(summit_i['pos'], summit_j['pos'])
                graph.add_edge(index_summit_i, index_summit_j, weight=distance)

                duration = graph[index_summit_i][index_summit_j]['weight'] / vehicle_speed * 60
                graph[index_summit_i][index_summit_j]['time'] = duration

    p = [2, -100, nbr_of_summit]  # Equation pour trouver n_min
    roots = np.roots(p)
    n_min = max(1, int(roots.min()) + 1)  # Nombre de voitures minimal possible
    graph.nodes[0]['n_min'] = n_min

    return graph


"""
Fonction d'initialisation du solution possible à n camions. 
Il y a beaucoup d'assertions car en effet, certains graph généré peuvent ne pas présenter de solution: 
    -Pas assez de voiture pour finir la livraison dans le temps imparti
    -Les ressources demandées peuvent être trop conséquentes 
    -Ect...

Parameters
----------
graph : Graph du problème
----------

Returns
-------
solution: list - a first valid solution .
-------
"""


def generate_initial_solution(graph, initial_order=None):
    total_vehicles_capacity = sum(graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"].values())
    total_customers_capacity = sum([graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(graph.nodes))])

    max_vehicle_capacity = max(graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"].values())
    max_customer_capacity = max([graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(graph.nodes))])

    message = 'Some customers have packages heavier than vehicles capacity'
    assert (max_customer_capacity < max_vehicle_capacity), message

    message = 'There is not enough vehicles to achieve the deliveries in time, regardless the configuration'
    assert (graph.nodes[0]['NBR_OF_VEHICLE'] > graph.nodes[0]['n_min']), message

    customers = [node for node in graph.nodes]
    customers.pop(0)

    if initial_order is None:
        initial_order = solve_tsp(graph)

        plot_solution([initial_order], graph, title='TSP solution')

    message = 'The initial order should start with 0 (the depot)'
    assert (initial_order[0] == 0), message

    message = 'All packages to deliver are heavier than total vehicles capacity'
    assert (total_customers_capacity <= total_vehicles_capacity), message

    # On remplit la solution de la majorité des sommets
    df_camion = pd.DataFrame()  # Dataframe renseignant sur les routes, important pour la seconde phase de remplissage
    df_camion.index = range(graph.nodes[0]['NBR_OF_VEHICLE'])
    vehicles_capacity = graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"]
    ressources = [vehicles_capacity[i] for i in range(graph.nodes[0]['NBR_OF_VEHICLE'])]

    df_camion['Ressources'] = ressources
    columns = [
        'Vehicles',
        'Ressource_to_add',
        'Id',
        'CUSTOMER_TIME_WINDOW_FROM',
        'CUSTOMER_TIME_WINDOW_TO',
        'Ressource_camion',
    ]
    df_initial_order = pd.DataFrame(columns=columns)

    solution = [[0] for index_delivery in range(graph.nodes[0]['NBR_OF_VEHICLE'])]

    index_delivery = 0

    for index_customer in tqdm(initial_order):
        message = 'should not require more vehicles than the given ones'
        assert (index_delivery < graph.nodes[0]['NBR_OF_VEHICLE']), message

        customer_capacity = graph.nodes[index_customer]['TOTAL_WEIGHT_KG']
        int_min = graph.nodes[index_customer]["CUSTOMER_TIME_WINDOW_FROM"]
        int_max = graph.nodes[index_customer]["CUSTOMER_TIME_WINDOW_TO"]

        is_vehicle_big_enough = df_camion.loc[index_delivery]['Ressources'] >= customer_capacity
        is_time_valid = is_delivery_time_valid(solution[index_delivery], graph)

        if is_vehicle_big_enough and is_time_valid:
            message = 'should not have customer order heavier than vehicle capacity'
            assert (customer_capacity <= vehicles_capacity[index_delivery]), message

            solution[index_delivery].append(index_customer)

            df_camion['Ressources'].loc[index_delivery] -= customer_capacity

        else:
            index_delivery += 1

        dict_camion = {
            'Vehicles': index_delivery,
            "Ressource_to_add": customer_capacity,
            "Id": index_customer,
            "CUSTOMER_TIME_WINDOW_FROM": int_min,
            "CUSTOMER_TIME_WINDOW_TO": int_max,
            "Ressource_camion": df_camion.loc[index_delivery]['Ressources'],
        }
        df_initial_order = pd.concat([df_initial_order, pd.DataFrame.from_dict([dict_camion])])

    for delivery in solution:
        delivery.append(0)

    assert (is_solution_valid(solution, graph) is True), 'initial solution should be valid'

    return solution
