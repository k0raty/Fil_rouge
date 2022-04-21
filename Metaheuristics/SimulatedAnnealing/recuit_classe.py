# -*- coding: utf-8 -*-

"""
Recuit-simulé sur la base de donnée du fil rouge, plusieurs notions à prendre en compte, l'algo se déroule en 3 étapes:
    -Initialisation -> on fait un premier recuit qui réalise l'algorithme du problème du voyageur, à partir de cela on initialise une première solution vérifiant les contraintes
    -Recuit -> Par un recuit , on perturbe notre solution en mélangeant les clients des différents camions uniquement, on récupère une solution x lorsque cela n'évolue plus.
    -Affinement -> On perturbe l'ordre de desservissement des clients pour un camion en question puis on retourne la meilleure solution.
Au niveau des contraintes:
    - On respecte les intervalles de temps pour desservir en temps et en heure chaque client
    -On respecte les ressources à livrer au niveau du poids en kg
    -Chaque camion a son propre coefficient et sa propre capacité de livraison
    -Le temps de trajets d'un camion pour aller du dépot à la première ville et de la dernière ville au dépot n'est pas pris en compte
    -Le temps de livraison est pris en compte mais divisé par 10 car pafois trop long (+ d'1h pour délivrer...)
    En effet, certaines villes sont a plus de 500 minutes du dépot lorsque le camion roule à 50 km/h !
Concernant la solution retournée :
    De la forme x=[[0,client1,client2,...,0],[0,client_n,client_n+1..,0],[clients_du_camion_3],[client_du_camion_4]...]
    Les informations du camion numéro i sont accessibles via : G.nodes[0]['Camion'][camion]
    Chaque client i a un identifiant client accessible via G.nodes[i]['CUSTOMER_CODE'], G.nodes[i] donne d'ailleurs d'autres informations sur le client i.
    
"""

""" Import librairies """
import copy
import numpy as np
import random as rd
import networkx as nx
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings
import utm

""" Import utilities """
from Utility.database import Database
from Utility.common import *
from Metaheuristics.SimulatedAnnealing.simulated_annealing_initialization import main

warnings.simplefilter(action='ignore', category=FutureWarning)


class Annealing:
    INITIAL_TEMPERATURE = 1500
    VEHICLE_SPEED = 50

    fitness: float = 0
    solution: list = []

    def __init__(self, customers=None, depot=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depot = database.Depots[0]
            cost_matrix = compute_cost_matrix(customers, depot)

        self.COST_MATRIX = cost_matrix

        self.Customers = customers
        self.Depot = depot
        self.Vehicles = vehicles

        self.NBR_OF_CUSTOMER = len(customers)
        self.NBR_OF_VEHICLE = len(vehicles)

    """
    Main function , réalise le recuit. 

    Parameters
    ----------
    df_customers : tableur excel renseignant sur les clients
    df_vehicles : tableur excel renseignant sur les camions à disposition 
    v : Vitesse des véhicules
    T : Température de départ lors de la phase de recuit.
    ----------
    
    Returns
    -------
    x : Solution proposée 
    -------
    """

    def main(self, initial_solution=None):
        graph = self.create_graph()
        print("Initialisation de la solution \n")

        solution = self.init(graph)
        self.plotting(solution, graph)
        print("Solution initialisée , début de la phase de recuit \n")

        solution = self.simulated_annealing(solution, graph, self.INITIAL_TEMPERATURE)
        print("Début de la phase de perfectionnement de la solution \n")

        solution = self.perturbation_intra(solution, graph)

        return solution

    """
    Generate a graph
    
    Returns
    -------
    graph : fully generated and hydrated graph 
    -------
    """

    def create_graph(self):
        df_vehicles = self.Vehicles

        graph = nx.empty_graph(self.NBR_OF_CUSTOMER)
        dict_0 = {
            'CUSTOMER_CODE': 0,
            'CUSTOMER_LATITUDE': self.Depot.LATITUDE,
            'CUSTOMER_LONGITUDE': self.Depot.LONGITUDE,
            'CUSTOMER_TIME_WINDOW_FROM_MIN': self.Depot.DEPOT_AVAILABLE_TIME_FROM_MIN,
            'CUSTOMER_TIME_WINDOW_TO_MIN': self.Depot.DEPOT_AVAILABLE_TIME_TO_MIN,
            'TOTAL_WEIGHT_KG': 0,
            'CUSTOMER_DELIVERY_SERVICE_TIME_MIN': 0,
        }

        graph.nodes[0].update(dict_0)
        graph.nodes[0]['n_max'] = self.NBR_OF_VEHICLE
        dict_vehicles = df_vehicles.to_dict()
        graph.nodes[0]['Camion'] = dict_vehicles

        for index_customer in range(0, len(graph.nodes)):
            dict = self.Customers[index_customer].to_dict()
            graph.nodes[index_customer].update(dict)

        for index_i in range(len(graph.nodes)):
            customer_i = self.Customers[index_i]

            for index_j in range(len(graph.nodes)):
                customer_j = self.Customers[index_j]

                if index_i != index_j:
                    distance = compute_distance(
                        customer_i.LATITUDE,
                        customer_i.LONGITUDE,
                        customer_j.LATITUDE,
                        customer_j.LONGITUDE,
                    )
                    graph.add_edge(index_i, index_j, weight=distance)

                    graph[index_i][index_j]['time'] = (graph[index_i][index_j]['weight'] / self.VEHICLE_SPEED) * 60

        p = [2, -100, self.NBR_OF_CUSTOMER]  # Equation pour trouver n_min
        roots = np.roots(p)

        # Nombre de voiture minimal possible , solution d'une équation de second degrès.
        n_min = max(1, int(roots.min()) + 1)

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

    def init(self, graph):
        # Assertions du début, afin de vérifier que l'on ne demande pas d'initialiser l'impossible

        max_Q = max(graph.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())
        max_ressources = sum(graph.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())

        # On vérifie que les ressources de chaque sommet sont au moins <= Q
        max_nodes_ressources = max([graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(graph.nodes))])

        message = "Les ressources de certaines villes sont plus grandes que les ressources des voitures !"
        assert (max_nodes_ressources < max_Q), message

        message = "Peu-importe la configuration , il n'y a pas assez de camion pour terminer le trajet dans le temps imparti (<100)"
        assert (self.NBR_OF_VEHICLE > graph.nodes[0]['n_min']), message
        # En effet, le temps de livraison des derniers sommets peuvent ne pas être atteint...

        message = "On demande trop de voiture , <= à %s normalement " % graph.nodes[0]['n_max']
        assert (self.NBR_OF_VEHICLE <= graph.nodes[0]['n_max']), message

        # Construction de la solution qui fonctionne

        solution = []  # Notre première solution

        for index_delivery in range(self.NBR_OF_VEHICLE):
            solution.append([0])

        nodes = [node for node in graph.nodes]
        nodes.pop(0)

        # Initialisation du dataframe renseignant sur les sommets et leurs contraintes
        initial_order = main(graph)

        message = "L'ordre initial n'est pas bon ,il ne commence pas par 0"
        assert (initial_order[0] == 0), message

        # Nos camions peuvent-ils livrer tout le monde ?
        sum_ressources = sum([graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(graph.nodes))])

        if sum_ressources > max_ressources:
            print("Les ressources demandées par les villes sont trop importantes")
            return False

        # On remplit la solution de la majorité des sommets
        df_camion = pd.DataFrame()  # Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
        df_camion.index = range(self.NBR_OF_VEHICLE)

        ressources = [graph.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"][i] for i in
                      range(0, n)]  # On commence par les camions aux ressources les plus importantes
        df_camion['Ressources'] = ressources
        df_ordre = pd.DataFrame(
            columns=["Camion", "Ressource_to_add", "Id", "CUSTOMER_TIME_WINDOW_FROM_MIN", "CUSTOMER_TIME_WINDOW_TO_MIN",
                     "Ressource_camion"])
        camion = 0

        i = 1  # On ne prend pas en compte le zéros du début
        with tqdm(total=len(Ordre)) as pbar:
            while i < len(Ordre):
                assert camion < n, "Impossible d'initialiser , les camions ne sont pas assez nombreux"
                nodes_to_add = Ordre[i]
                assert (nodes_to_add != 0), "Le chemin proposé repasse par le dépot !"

                q_nodes = G.nodes[nodes_to_add]['TOTAL_WEIGHT_KG']
                int_min = G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_FROM_MIN"]
                int_max = G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_TO_MIN"]
                dict = [{"Camion": camion, "Ressource_to_add": q_nodes, "Id": nodes_to_add,
                         "CUSTOMER_TIME_WINDOW_FROM_MIN": int_min, "CUSTOMER_TIME_WINDOW_TO_MIN": int_max,
                         "Ressource_camion": df_camion.loc[camion]['Ressources']}]
                temp = copy.deepcopy(solution[camion])
                temp.append(nodes_to_add)
                if (df_camion.loc[camion]['Ressources'] >= q_nodes and check_temps_part(temp, G) == True):
                    Q = G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                    assert (
                                q_nodes <= Q), "Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
                    solution[camion].append(nodes_to_add)
                    df_camion['Ressources'].loc[camion] += -q_nodes
                    i += 1
                    pbar.update(1)
                    assert (solution[camion] == temp)
                else:
                    print(nodes_to_add)
                    assert (solution[camion] != temp)
                    camion += 1

                df_ordre = df_ordre.append(dict)

        for i in solution:
            i.append(0)
        ###Assertion pour vérifier que tout fonctionne bien###

        assert (check_constraint(solution, G) == True), "Mauvaise initialisation au niveau du temps"
        check_forme(solution, G)

        ###Affichage de la première solution###
        plotting(solution, G)
        return solution

    # A ADAPTER DANS LA CLASSE
    def temperature(E, E0):
        """
        Fonction température
        """
        return (1 / math.log(E0 - E)) * 1500

    # A ADAPTER DANS LA CLASSE MAIS JE PENSE QU'IL EST DANS validator
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
        for camion in range(0, len(x)):
            assert (camion < len(colors)), "Trop de camion, on ne peut pas afficher"
            if len(x) > 2:
                xo = [X[o] for o in x[camion]]
                yo = [Y[o] for o in x[camion]]
                plt.plot(xo, yo, colors[couleur], label=G.nodes[0]["Camion"]["VEHICLE_VARIABLE_COST_KM"][camion])
                couleur += 1
        plt.legend(loc='upper left')
        plt.show()

    # A ADAPTER DANS LA CLASSE
    def recuit_simulé(x, G, T):

        """
        Fonction de recuit qui mélanges les clients de chaque camion mais qui ne modifie pas l'ordre de deservissement pour un camion en question. 
        Il y a beaucoup d'assertions afin de vérifier que la perturbation engendrée ne crée pas de problème de contraintes.  
           -On prend un sommet d'une route parcourue par un camion pour l'ajouter à une autre route
           -Il est possible qu'à chaque étape ,il n'y ait pas de modification. 
           -La baisse de température n'est pas graduelle mais dépend de l'écart entre deux solutions proposées
           (+ l'écart est petit, + la température augmente).
            
        Parameters
        ----------
        x : Solution à perturber
        G : Graph du problème 
        
        Returns
        -------
        Une solution x améliorée .
        
        """
        ###Assertions de début###
        check_forme(x, G)  # On vérifie que la solution fournie en initialisation est viable
        assert (check_constraint(x, G) == True), "Mauvaise initialisation"  # Que celle-ci vérifie les contraintes

        it = 0
        k = 10e-5  # arbitraire
        Q = [G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0, len(x))]
        nb_sommet = len(G.nodes)
        E = energie(x, G)
        E0 = E + 2
        best_x = copy.deepcopy(x)  # La meilleure solution trouvée lors de toute les perturbation
        T_list = [T]
        E_list = [E]
        E_min = E
        fig, axs = plt.subplots(1, 2)
        while E0 - E >= 0.5:  # Tant que la solution retournée diffère d'un km de coût , on continue de chercher

            it += 1
            print("iteration", it, "E=", E, "Température =", T)

            E0 = E
            x = copy.deepcopy(best_x)

            ###Phase de perturbation de x ###

            for i in tqdm(range(0,
                                nb_sommet - 1)):  # i correspond au client qui verra le client j être rajouté avant lui dans le desservissement.
                for j in range(i + 1, nb_sommet):
                    very_old_x = copy.deepcopy(
                        x)  # Solution du début enregistrée , on la garde en mémoire si après perturbation ,x n'est pas meilleur.

                    ###On ajoute un sommet à une route , on en retire à une autre###
                    if i != 0:
                        flatten_x = [item for sublist in x for item in sublist]
                        ajout, retire = flatten_x.index(i), flatten_x.index(j)
                        num_zero_ajout, num_zero_retire = flatten_x[:ajout].count(0) - 1, flatten_x[:retire].count(
                            0) - 1
                        camion_ajout, camion_retire = int(num_zero_ajout / 2 - 1) + 1, int(num_zero_retire / 2 - 1) + 1
                        if camion_ajout != camion_retire:  # Pour des raisons de compléxité , on ne s'occupe pas des cas ou i et j sont dans le routage d'un même camion ,cela vient après !
                            nouvelle_place = x[camion_ajout].index(i)
                            x[camion_ajout].insert(nouvelle_place, j)
                            x[camion_retire].remove(j)


                    ### On ajoute le client j a une trajectoire "vide" dans laquelle aucun camion n'est deservie ###
                    else:
                        flatten_x = [item for sublist in x for item in sublist]
                        retire = flatten_x.index(j)
                        num_zero_retire = flatten_x[:retire].count(0) - 1
                        camion_retire = int(num_zero_retire / 2 - 1) + 1
                        assert (j in x[camion_retire])
                        unfilled_road = [i for i in range(0, len(x)) if len(x[i]) == 2]
                        if len(unfilled_road) != 0:
                            nouvelle_place = 1
                            camion_ajout = unfilled_road[0]
                            x[camion_ajout].insert(nouvelle_place, j)
                            x[camion_retire].remove(j)

                    ###On compare l'énergie (seulement une partie) de notre solution perturbé avec la précédente solution###
                    if camion_ajout != camion_retire:
                        E1, E2, E3, E4 = energie_part(x[camion_ajout], G, camion_ajout), energie_part(x[camion_retire],
                                                                                                      G,
                                                                                                      camion_retire), energie_part(
                            very_old_x[camion_ajout], G, camion_ajout), energie_part(very_old_x[camion_retire], G,
                                                                                     camion_retire)

                        ###Principe du recuit###
                        if (E1 + E2 >= E3 + E4):
                            p = np.exp(-(E1 + E2 - (E3 + E4)) / (k * T))
                            r = rd.random()
                            if r <= p and p != 1:  # Si p fait 1 , cela ralentit énormément.
                                if (check_constraint(x, G) != True):
                                    x = copy.deepcopy(very_old_x)  # sinon on conserve x tel qu'il est
                                else:
                                    ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###                           
                                    assert (len(x[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                    assert (len(x[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                            else:
                                x = copy.deepcopy(very_old_x)

                        else:
                            if (check_ressource(x[camion_ajout], Q[camion_ajout], G) == False or check_temps_part(
                                    x[camion_ajout], G) == False):
                                x = copy.deepcopy(very_old_x)
                            else:
                                E = energie(x, G)
                                if E < E_min:
                                    ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###
                                    assert (len(x[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                    assert (len(x[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                                    best_x = copy.deepcopy(
                                        x)  # On garde en mémoire le meilleur x trouvé jusqu'à présent
                                    E_min = E

            ###Assertions de fin###
            plotting(best_x, G)
            assert (check_constraint(best_x, G) == True)
            check_forme(best_x, G)
            num_very_old_x = sum([len(i) for i in very_old_x])
            num_x = sum([len(i) for i in x])  # On vérifie qu'aucun sommet n'a été oublié
            assert (num_very_old_x == num_x)
            E = energie(best_x, G)

            ###Modification de la température###
            if E0 > E:
                T = temperature(E, E0)
                T_list.append(T)
                E_list.append(E)

        plt.clf()
        axs[0].plot(E_list, 'o-')
        axs[0].set_title("Energie de la meilleure solution en fonction des itérations")
        axs[1].plot(T_list, 'o-')
        axs[1].set_title("Température en fonction des itérations")
        fig.suptitle("Profil de la première partie")
        fig.savefig("Profil de la première partie")
        fig.show()

        ###Assertions de fin###

        check_forme(best_x, G)
        assert (check_constraint(best_x, G) == True)
        plotting(x, G)

        return best_x

    # A ADAPTER DANS LA CLASSE
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
                somme += weight_road
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
        seulement l'ordre des client pour chaque route

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


recuit = Recuit()
x
