""" Import librairies """
import pandas as pd
import random as rd
from os.path import join

"""
Fonction de vérification de contrainte concernant les intervalles de temps.
    -Chaque camion part au même moment, cependant leurs temps de trajets sont pris en compte
    seulement lorsque ceux-ci sont arrivés chez le premier client.
    
Parameters
----------
solution :list - solution to the problem
graph :? - graph of the problem
----------

Returns
-------
:bool - is the solution respecting the time constraint
-------
"""


def is_solution_time_valid(solution: list, graph) -> bool:
    return True

    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]

        if not is_delivery_time_valid(delivery, graph):
            return False

    return True


"""
Fonction de vérification de contrainte concernant les intervalles de temps.
    -On considère une itération de temps à chaque trajet , peu importe sa longueur.
    -Chaque camion part au même moment.
    
Parameters
----------
delivery: list - a portion of the solution to the VSP
graph: ? - graph of the problem
----------

Returns
-------
:bool - is the delivery respecting the time constraint
-------
"""


def is_delivery_time_valid(delivery, graph):
    df_delivery_time = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])

    depot = graph.nodes[0]
    current_time = depot['CUSTOMER_TIME_WINDOW_FROM']

    for index_customer in range(1, len(delivery) - 1):

        customer_from = delivery[index_customer]
        customer_to = delivery[index_customer + 1]

        current_time += graph[customer_from][customer_to]['time']

        while current_time < graph.nodes[customer_to]['CUSTOMER_TIME_WINDOW_FROM']:
            current_time += 1

        new_arc = {
            'temps': current_time,
            'route': (customer_from, customer_to),
            'temps_de_parcours': graph[customer_from][customer_to]['time'],
            'limite_inf': graph.nodes[customer_to]['CUSTOMER_TIME_WINDOW_FROM'],
            'limite_sup': graph.nodes[customer_to]['CUSTOMER_TIME_WINDOW_TO'],
        }

        df_delivery_time = pd.concat([df_delivery_time, pd.DataFrame.from_dict([new_arc])])

        if current_time > graph.nodes[customer_to]['CUSTOMER_TIME_WINDOW_TO']:
            return False

        current_time += graph.nodes[customer_to]["CUSTOMER_DELIVERY_SERVICE_TIME"]

    if current_time > depot['CUSTOMER_TIME_WINDOW_TO']:
        return False

    return True


"""
Check that a delivery match vehicle capacity constraint

Parameters
----------
delivery :list - list of customers to deliver for this delivery
vehicle_capacity :float - capacity of the delivery's vehicle
graph :? - graph of the problem
----------

Returns
-------
:bool - is the delivery capacity constraint respected
-------
"""


def is_delivery_capacity_valid(graph, delivery: list, vehicle_capacity: float) -> bool:
    for customer in delivery:
        if vehicle_capacity - graph.nodes[customer]['TOTAL_WEIGHT_KG'] < 0:
            return False

    return True


"""
Check that each delivery in the given solution match capacity constraint

Parameters
----------
----------

Returns
-------
-------
"""


def is_solution_capacity_valid(solution: list, graph) -> bool:
    vehicles_capacity = graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG']

    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]
        flag = is_delivery_capacity_valid(graph, delivery, vehicles_capacity[index_delivery])

        if not flag:
            return False

    return True


"""
Check the shape of the solution

Parameters
----------
x : solution
G : Graphe du problème
----------

Returns
-------
Assertions.
-------
"""


def is_solution_shape_valid(solution, graph):
    visited_customers = pd.DataFrame(columns=['client', 'passage'])

    for delivery in solution:
        for customer in delivery:
            if customer not in list(visited_customers["client"]):
                new_customer = {
                    'client': customer,
                    'passage': 1,
                }

                visited_customers = pd.concat([visited_customers, pd.DataFrame.from_dict([new_customer])])

            else:
                visited_customers['passage'][visited_customers['client'] == customer] += 1

    if len(visited_customers) != len(graph.nodes):
        return False

    visite_2 = visited_customers[visited_customers['client'] != 0]

    if len(visite_2[visite_2['passage'] > 1]) != 0:
        return False

    for delivery in solution:
        if (delivery[0], delivery[-1]) != (0, 0):
            return False

        if 0 in delivery[1: -1]:
            return False

    return True


"""
Check that all constraints are matched by a given solution

Parameters
----------
solution :list - solution to the problem
graph :? - graph of the problem
----------

Returns
-------
:bool - is the solution matching all constraints
-------
"""


def is_solution_valid(solution, graph):
    match_shape_constraint = is_solution_shape_valid(solution, graph)
    match_capacity_constraint = is_solution_capacity_valid(solution, graph)
    match_time_constraint = is_solution_time_valid(solution, graph)

    return match_time_constraint and match_capacity_constraint and match_shape_constraint


def pick_valid_solution():
    solution_df_path = join('Dataset', 'Initialized', 'valid_initial_solution.pkl')
    solution_df = pd.read_pickle(solution_df_path)
    solution_set = list(solution_df.iloc[0])

    index_solution = rd.randint(0, len(solution_set))
    solution = solution_set[index_solution]

    return solution
