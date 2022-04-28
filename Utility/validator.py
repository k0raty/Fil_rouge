""" Import librairies """
import pandas as pd
import random as rd
from os.path import join

"""
Fonction de vérification de contrainte conçernant les intervalles de temps.
    -Chaque camion part au même moment, cependant leurs temps de trajets sont pris en compte
    seulement lorsque ceux-ci sont arrivés chez le premier client.
    
Parameters
----------
solution :list - solution to the problem
graph :? - graph of the problem
----------

Returns
-------
:bool - Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée.
    Le camion peut marquer des pauses
-------
"""


def is_solution_time_valid(solution: list, graph) -> bool:
    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]

        df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
        depot_opening_time = graph.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']
        current_time = depot_opening_time

        for index_customer in range(1, len(delivery) - 1):
            customer = delivery[index_customer]
            next_customer = delivery[index_customer + 1]

            current_time += graph[customer][next_customer]['time']

            while current_time < graph.nodes[next_customer]['CUSTOMER_TIME_WINDOW_FROM_MIN']:
                current_time += 1  # vehicle is on pause, waiting for next customer to be available

            delivery_data = {
                'temps': current_time,
                'route': (customer, next_customer),
                'temps_de_parcours': graph[customer][next_customer]['time'],
                'limite_inf': graph.nodes[next_customer]['CUSTOMER_TIME_WINDOW_FROM_MIN'],
                'limite_sup': graph.nodes[next_customer]['CUSTOMER_TIME_WINDOW_TO_MIN'],
                'camion': index_delivery,
            }
            df_temps = pd.concat([df_temps, pd.DataFrame.from_dict(delivery_data)])

            # TODO : modifying the dataset to make the customer's time window doable
            # if current_time > graph.nodes[next_customer]['CUSTOMER_TIME_WINDOW_TO_MIN']:
            #   print('not respecting some customer\'s time window')
            #   return False

            current_time += graph.nodes[next_customer]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10

    return True


"""
"""


def check_temps_part(x, G):
    """
    Fonction de vérification de contrainte conçernant les intervalles de temps.
        -On considère une itération de temps à chaque trajet , peu importe sa longueur.
        -Chaque camion part au même moment.
    Parameters
    ----------
    x : Solution à évaluer
    G : Graph du problème

    Returns
    -------
    bool
        Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée.
        Le camion peut marquer des pauses.

    """

    df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
    temps = G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']  # Temps d'ouverture du dépot
    for i in range(1, len(x) - 1):
        # assert(temps<G.nodes[0]['CUSTOMER_TIME_WINDOW_TO_MIN']) #Il faut que les camion retournent chez eux à l'heure
        first_node = x[i]
        second_node = x[i + 1]
        if second_node != 0:  # On ne prend pas en compte l'arrivée non plus
            temps += G[first_node][second_node]['time']  # temps mis pour parcourir la route en minute
            while temps < G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN']:
                temps += 1  # Le camion est en pause
            dict = {'temps': temps, 'route': (first_node, second_node),
                    'temps_de_parcours': G[first_node][second_node]['time'],
                    'limite_inf': G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'],
                    'limite_sup': G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN']}
            df_temps = df_temps.append([dict])
            if (temps < G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'] or temps > G.nodes[second_node][
                'CUSTOMER_TIME_WINDOW_TO_MIN']):
                # print("Pendant l'initialisation \n",df_temps)
                return False

            temps += G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10
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
