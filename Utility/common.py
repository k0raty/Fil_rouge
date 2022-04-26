""" Import librairies """
import os
from math import pi, cos, sqrt, asin
import numpy as np
import utm
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def compute_plan_distance(pos_i: tuple, pos_j: tuple) -> float:
    x_i, x_j, y_i, y_j = pos_i[0], pos_j[0], pos_i[1], pos_j[1]
    distance = sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2) / 1000

    return distance


"""
Evaluate the cost of visiting sites with this configuration, depending on the number of cars
and the cost from one site to the next one

Parameters
----------
solution: list - list of all the visited sites from the first to the last visited
cost_matrix: np.ndarray - the cost of each travel to use to compute the fitness
nbr_of_vehicle: int - the number of vehicle used in the given solution
----------

Returns
-------
fitness_score:float - value of the cost of this configuration
-------
"""


def compute_fitness(solution: list, cost_matrix: np.ndarray, vehicles: list) -> float:
    nbr_of_vehicle = len(solution)

    solution_cost=0
    for index_vehicle in range(nbr_of_vehicle):
        cost_by_distance = vehicles['VEHICLE_VARIABLE_COST_KM'][index_vehicle]

        delivery_distance = 0

        delivery = solution[index_vehicle]
        nbr_of_summit = len(delivery)

        for index_summit in range(nbr_of_summit - 1):
            # remove 1 as the customer's index start at 1 (because depot is 0)
            summit_from = delivery[index_summit] - 1
            summit_to = delivery[index_summit + 1] - 1

            delivery_distance += cost_matrix[summit_from][summit_to]

        solution_cost += delivery_distance * cost_by_distance 

    return solution_cost


"""
Fill a matrix storing the cost of the travel between every customers

Parameters
----------
customers: list - the list of customers with their coordinates
----------

Returns
-------
cost_matrix: np.ndarray - a matrix containing in a cell (i, j) the distance of the travel between
site i and j
-------
"""


def compute_cost_matrix(graph) -> np.ndarray:
    nbr_of_customer = len(graph)
    cost_matrix = np.zeros((nbr_of_customer + 1, nbr_of_customer + 1))

    depot = graph.nodes[0]

    for index_i in range(nbr_of_customer):
        customer_i = graph.nodes[index_i]

        distance_to_depot = compute_spherical_distance(
            depot['CUSTOMER_LATITUDE'],
            depot['CUSTOMER_LONGITUDE'],
            customer_i['CUSTOMER_LATITUDE'],
            customer_i['CUSTOMER_LONGITUDE'],
        )

        cost_matrix[0, index_i] = distance_to_depot
        cost_matrix[index_i, 0] = distance_to_depot

        for index_j in range(nbr_of_customer):
            customer_j = graph.nodes[index_j]

            distance_from_i_to_j = compute_spherical_distance(
                customer_i['CUSTOMER_LATITUDE'],
                customer_i['CUSTOMER_LONGITUDE'],
                customer_j['CUSTOMER_LATITUDE'],
                customer_j['CUSTOMER_LONGITUDE'],
            )

            cost_matrix[index_i + 1, index_j + 1] = distance_from_i_to_j
            cost_matrix[index_j + 1, index_i + 1] = distance_from_i_to_j

    return cost_matrix


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

    (x_0, y_0) = utm.from_latlon(43.37391833, 17.60171712)[:2]
    depot_dict = {
        'CUSTOMER_CODE': 0,
        'CUSTOMER_LATITUDE': 43.37391833,
        'CUSTOMER_LONGITUDE': 17.60171712,
        'CUSTOMER_TIME_WINDOW_FROM_MIN': 360,
        'CUSTOMER_TIME_WINDOW_TO_MIN': 1080,
        'TOTAL_WEIGHT_KG': 0,
        'pos': (x_0, y_0),
        'CUSTOMER_DELIVERY_SERVICE_TIME_MIN': 0,
        'INDEX': 0,
    }

    graph.nodes[0].update(depot_dict)

    graph.nodes[0]['n_max'] = nbr_of_vehicle

    dict_vehicles = df_vehicles.to_dict()

    graph.nodes[0]['Camion'] = dict_vehicles

    for index_customer in range(nbr_of_summit):
        customer = df_customers.iloc[index_customer]
        customer_dict = customer.to_dict()
        customer_dict['pos'] = utm.from_latlon(customer['CUSTOMER_LATITUDE'], customer_dict['CUSTOMER_LONGITUDE'])[:2]

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


# Fonction pour le recuit_simulé

def check_temps(x, G):
    """
    Fonction de vérification de contrainte conçernant les intervalles de temps.
        -Chaque camion part au même moment, cependant leurs temps de trajets sont pris en compte
        seulement lorsque ceux-ci sont arrivés chez le premier client.
    Parameters
    ----------
    x : Solution à évaluer
    G : Graphe du problème
    Returns
    -------
    bool
        Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée.
        Le camion peut marquer des pauses.
    """
    K = len(x)
    for route in range(0, K):
        df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
        temps = G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']  # Temps d'ouverture du dépot
        for i in range(1, len(x[route]) - 1):  # On ne prend pas en compte l'aller dans l'intervalle de temps
            first_node = x[route][i]
            second_node = x[route][i + 1]
            if second_node != 0:
                temps += G[first_node][second_node]['time']  # temps mis pour parcourir la route en minute
                while temps < G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN']:
                    temps += 1  # Le camion est en pause
                dict = {'temps': temps, 'route': (first_node, second_node),
                        'temps_de_parcours': G[first_node][second_node]['time'],
                        'limite_inf': G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'],
                        'limite_sup': G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN'], "camion": route}
                df_temps = df_temps.append([dict])
                if (temps < G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'] or temps > G.nodes[second_node][
                    'CUSTOMER_TIME_WINDOW_TO_MIN']):
                    return False
                temps += G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10
    return True


# A ADAPTER DANS LA CLASSE MAIS JE PENSE QU'IL EST DANS validator
def check_temps_part(x, G):
    """
    Fonction de vérification de contrainte conçernant les intervalles de temps.
        -Chaque camion part au même moment, cependant leurs temps de trajets sont pris en compte
        seulement lorsque celui-ci est arrivé chez le premier client.
    Parameters
    ----------
    x : Trajectoire de camion à évaluer
    G : Graphe du problème
    Returns
    -------
    bool
        Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée pour le camion en question .
        Le camion peut marquer des pauses.
    """

    df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
    temps = G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']  # Temps d'ouverture du dépot
    for i in range(1, len(x) - 1):
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
                return False

            temps += G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10
    return True


# A ADAPTER DANS LA CLASSE
def check_ressource(route, Q, G):
    """
    Fonction de vérification de contrainte des ressources.
    Parameters
    ----------
    route : x[route], correspond à la route que va parcourir notre camion.
    Q : Ressource du camion considéré
    G : Graph du problème
    Returns
    -------
    bool
        Si oui ou non, le camion peut en effet desservir toute les villes en fonction de ses ressources.
    """
    ressource = Q
    for nodes in route:
        ressource = ressource - G.nodes[nodes]['TOTAL_WEIGHT_KG']
        if ressource < 0:
            return False
    return True


# A ADAPTER DANS LA CLASSE
def plotting(x, G):
    """
    Affiche les trajectoires des différents camion entre les clients.
    Chaque trajectoire a une couleur différente.
    Parameters
    ----------
    x : Routage solution
    G : Graphe en question
    Returns
    -------
    None.
    """
    plt.clf()
    X = [G.nodes[i]['pos'][0] for i in range(0, len(G))]
    Y = [G.nodes[i]['pos'][1] for i in range(0, len(G))]
    plt.plot(X, Y, "o")
    plt.text(X[0], Y[0], "0", color="r", weight="bold", size="x-large")
    plt.title("Trajectoire de chaque camion")
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    couleur = 0
    print(x)
    for camion in range(0, len(x)):
        assert (camion < len(colors)), "Trop de camion, on ne peut pas afficher"
        if len(x) > 2:
            xo = [X[o] for o in x[camion]]
            yo = [Y[o] for o in x[camion]]
            plt.plot(xo, yo, colors[couleur], label=G.nodes[0]["Camion"]["VEHICLE_VARIABLE_COST_KM"][camion])
            couleur += 1
    plt.legend(loc='upper left')
    plt.show()


def energie(x, G):
    """
    Fonction coût pour le recuit

    Parameters
    ----------
    x : solution
    G : Graph du problème

    Returns
    -------
    somme : Le coût de la solution

    """
    K = len(x)
    somme = 0
    for route in range(0, K):
        if len(x[route]) > 2:  # si la route n'est pas vide
            w = G.nodes[0]['Camion']['VEHICLE_VARIABLE_COST_KM'][
                route]  # On fonction du coût d'utilisation du camion
            weight_road = sum(
                [G[x[route][sommet]][x[route][sommet + 1]]['weight'] for sommet in range(0, len(x[route]) - 1)])
            somme += w * weight_road
    return somme


# A ADAPTER DANS LA CLASSE
def energie_part(x, G, camion):
    """
    Fonction coût partielle pour le recuit qui calcule uniquement le coût d'un trajet uniquement

    Parameters
    ----------
    x : solution
    G : Graph du problème
    camion : camion à évaluer
    Returns
    -------
    somme : Le coût de la solution partielle

    """
    if len(x) > 2:  # si la route n'est pas vide
        w = G.nodes[0]['Camion']['VEHICLE_VARIABLE_COST_KM'][camion]  # On fonction du coût d'utilisation du camion
        somme = sum([G[x[sommet]][x[sommet + 1]]['weight'] for sommet in range(0, len(x) - 1)])
        somme += w * somme  # facteur véhicule
        return somme
    else:
        return 0


# A ADAPTER DANS LA CLASSE MAIS JE PENSE QU'IL EST DANS validator
def check_constraint(x, G):
    """
    Vérifie que les contraintes principales sont vérifiée:
        -Les ressources demandée par chaque client sur un trajet ne sont pas supérieure au
        disponibilités du camion
        -Les villes sont livrées en temps et en heure.
    """

    Q = [G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0, len(x))]
    if (check_temps(x, G) == True):
        for i in range(0, len(x)):
            if (check_ressource(x[i], Q[i], G) != True):
                return False
        else:
            return True
    else:
        return False


# A ADAPTER DANS LA CLASSE
def perturbation_intra(x, G):
    """
    Deuxième phase de perturbation , on n'échange plus des clients entre chaque trajectoire de camion  mais
    seulement l'initial_order des client pour chaque route

    Parameters
    ----------
    x : solution après la première phase
    G : Graphe du problème

    Returns
    -------
    x : solution finale.

    """
    d = energie(x, G)
    d0 = d + 1
    it = 1
    list_E = [d]
    while d < d0:
        it += 1
        print("iteration", it, "d=", d)
        d0 = d
        for camion in tqdm(range(0, len(x))):
            route = x[camion]
            for i in range(1, len(route) - 1):
                for j in range(i + 2, len(route)):
                    d_part = energie_part(route, G, camion)
                    r = route[i:j].copy()
                    r.reverse()
                    route2 = route[:i] + r + route[j:]
                    t = energie_part(route2, G, camion)
                    if (t < d_part):
                        if check_temps_part(route2, G) == True:
                            x[camion] = route2
        d = energie(x, G)
        list_E.append(d)
        assert (check_temps(x, G) == True)
        plotting(x, G)
    plt.clf()
    plt.plot(list_E, 'o-')
    plt.title("Evoluation de l'énergie lors de la seconde phase")
    plt.show()

    ###Assertions de fin###

    check_forme(x, G)
    assert (check_constraint(x, G) == True), "Mauvaise initialisation au niveau du temps"
    return x


# A ADAPTER DANS LA CLASSE MAIS JE PENSE QU'IL EST DANS validator
def check_forme(x, G):
    """
    Vérifie que la forme de la solution est correcte

    Parameters
    ----------
    x : solution
    G : Graphe du problème

    Returns
    -------
    Assertions.

    """
    visite = pd.DataFrame(columns=["Client", "passage"])
    for l in x:
        for m in l:
            if m not in list(visite["Client"]):
                dict = {"Client": m, "passage": 1}
                visite = visite.append([dict])
            else:
                visite['passage'][visite['Client'] == m] += 1
    assert (len(visite) == len(
        G.nodes)), "Tout les sommets ne sont pas pris en compte"  # On vérifie que tout les sommets sont pris en compte
    visite_2 = visite[visite['Client'] != 0]
    assert (len(visite_2[visite_2['passage'] > 1]) == 0), "Certains sommets sont plusieurs fois déservis"
    for i in range(0, len(x)):
        assert ((x[i][0], x[i][-1]) == (0, 0)), "Ne départ pas ou ne revient pas au dépot"
        assert (0 not in x[i][1:-1]), "Un camion repasse par 0"
# A ADAPTER DANS LA CLASSE