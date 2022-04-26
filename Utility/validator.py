import pandas as pd

"""
Fonction de vérification de contrainte conçernant les intervalles de temps.
    -Chaque camion part au même moment, cependant leurs temps de trajets sont pris en compte
    seulement lorsque ceux-ci sont arrivés chez le premier client.
    
Parameters
----------
solution :list - solution to the problem
graph :? - graph of the problem

Returns
-------
:bool - Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée.
    Le camion peut marquer des pauses
-------
"""


def check_time(solution, graph):
    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]

        df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
        depot_opening_time = graph.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']
        current_time = depot_opening_time

        for index in range(len(delivery) - 1):
            customer = delivery[index]
            next_customer = delivery[index + 1]

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
            df_temps = df_temps.append([delivery_data])

            if current_time > graph.nodes[next_customer]['CUSTOMER_TIME_WINDOW_TO_MIN']:
                return False

            current_time += graph.nodes[next_customer]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10

    return True


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
delivery : list of customers to deliver for this delivery
vehicle_capacity : capacity of the delivery's vehicle
graph : graph of the problem
----------

Returns
-------
:bool - is the delivery capacity constraint respected
-------
"""


def check_delivery_capacity(graph, delivery, vehicle_capacity):
    for customer in delivery:
        message = 'Delivery packages capacity is heavier than vehicle capacity'
        assert (vehicle_capacity - graph.nodes[customer]['TOTAL_WEIGHT_KG'] >= 0), message


"""
Check that each delivery in the given solution match capacity constraint
"""


def check_solution_capacity(solution: list, graph) -> bool:
    vehicles_capacity = graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG']

    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]
        check_delivery_capacity(graph, delivery, vehicles_capacity[index_delivery])


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


def check_solution_shape(solution, graph):
    visited_customers = pd.DataFrame(columns=["Client", "passage"])

    for delivery in solution:
        for customer in delivery:
            if customer not in list(visited_customers["Client"]):
                visited_customers = visited_customers.append([{
                    'Client': customer,
                    'passage': 1,
                }])

            else:
                visited_customers['passage'][visited_customers['Client'] == customer] += 1

    message = 'Some customers are not visited'
    assert (len(visited_customers) == len(graph.nodes)), message

    visite_2 = visited_customers[visited_customers['Client'] != 0]
    message = 'Some customers are delivered more than once'
    assert (len(visite_2[visite_2['passage'] > 1]) == 0), message

    for delivery in solution:
        assert ((delivery[0], delivery[-1]) == (0, 0)), "Vehicle should start and end with depot"
        assert (0 not in delivery[1: -1]), "Vehicle should not go back to depot in the middle of delivery"


def check_all_constraints(solution, graph):
    check_time(solution, graph)
    check_solution_capacity(solution, graph)
    check_solution_shape(solution, graph)
