# -*- coding: utf-8 -*-
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
from simulated_annealing_initialization import main

""" Import utilities """
from Utility.database import Database
from Utility.common import distance, compute_fitness, compute_cost_matrix
from Utility.validator import check_constraint, check_temps, check_temps_2, check_ressource

warnings.simplefilter(action='ignore', category=FutureWarning)


class Annealing:
    alpha = 0.01  # Pas de la descente de la température

    temperature_initial = 1500
    temperature_minimal = 200

    penalty = 5
    vehicle_speed = 50

    df_ordre_init = pd.read_pickle("df_ordre_init.pkl")

    def __init__(self, customers=None, depots=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles[0]
            depots = database.Depots
            cost_matrix = compute_cost_matrix(customers)

        self.solution = None

        self.COST_MATRIX = cost_matrix

        self.CUSTOMERS = customers
        self.DEPOTS = depots
        self.VEHICLES = vehicles

        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_CUSTOMERS = len(customers)

    """
    Run the GeneticAlgorithm
    """

    def main(self, initial_solution=None):
        graph = self.create_graph(self.CUSTOMERS, self.VEHICLES)

        if initial_solution is None:
            initial_solution = self.init(graph, self.NBR_OF_VEHICLES)

        df_x = pd.read_pickle("test_avec_pause.pkl")

        solution = list(df_x['ordre'])

        assert (check_constraint(solution, graph)), "Mauvaise initialisation"

        self.plotting(solution, graph)

        solution = self.perturbation(solution, graph, self.temperature_initial)

        print("Début de la seconde phase \n")

        solution = self.perturbation_2(solution, graph)

        return solution

    """
    Compute the energy of a given solution (different from the fitness score)

    Parameters
    ----------
    solution: list - solution
    graph: ? - problem's graph

    Returns
    -------
    energy: float - the energy (cost) of the solution
    """

    def energy_of_solution(self, solution, graph):
        nbr_of_delivery = len(solution)
        energy = 0

        for index in range(nbr_of_delivery):
            delivery = solution[index]
            energy += self.energy_of_delivery(delivery, graph, index)

        return energy

    """
    Compute the energy of the delivery of a given solution

    Parameters
    ----------
    delivery: list - portion of a solution
    graph: ? - problem's graph
    index_vehicle: int - the index of the vehicle on this delivery

    Returns
    -------
    energy: float - the energy of the delivery
    """

    def energy_of_delivery(self, delivery, graph, index_vehicle):
        energy = 0
        nbr_of_summit = len(delivery)

        # if delivery is not empty
        if nbr_of_summit > 2:
            w = graph.nodes[0]['Camion']['VEHICLE_VARIABLE_COST_KM'][index_vehicle]
            weights = [graph[delivery[summit]][delivery[summit + 1]]['weight'] for summit in range(nbr_of_summit - 1)]
            weight_road = sum(weights)

            energy = w * weight_road

        return energy

    """
    Fonction de descente de la température
    """

    def temperature(self, temperature, alpha):
        return (1 - alpha) * temperature

    """
    Create a graph

    Parameters
    ----------
    df_customers : Dataframe contenant les informations sur les clients
    df_vehicles : Dataframe contenant les informations sur les camions livreurs

    Returns
    -------
    graph: filled graph
    """

    def create_graph(self, df_customers, df_vehicles):
        graph = nx.empty_graph(self.NBR_OF_CUSTOMERS)
        (x_0, y_0) = utm.from_latlon(43.37391833, 17.60171712)[:2]
        dict_0 = {'CUSTOMER_CODE': 0, 'CUSTOMER_LATITUDE': 43.37391833, 'CUSTOMER_LONGITUDE': 17.60171712,
                  'CUSTOMER_TIME_WINDOW_FROM_MIN': 360, 'CUSTOMER_TIME_WINDOW_TO_MIN': 1080, 'TOTAL_WEIGHT_KG': 0,
                  'pos': (x_0, y_0), "CUSTOMER_DELIVERY_SERVICE_TIME_MIN": 0}
        graph.nodes[0].update(dict_0)

        graph.nodes[0]['n_max'] = self.NBR_OF_VEHICLES
        dict_vehicles = df_vehicles.to_dict()
        graph.nodes[0]['Camion'] = dict_vehicles

        for i in range(1, len(graph.nodes)):
            dict = df_customers.iloc[i].to_dict()
            dict['pos'] = utm.from_latlon(dict['CUSTOMER_LATITUDE'], dict['CUSTOMER_LONGITUDE'])[:2]
            graph.nodes[i].update(dict)

        # On rajoute les routes
        for i in range(len(graph.nodes)):
            for j in range(len(graph.nodes)):
                if i != j:
                    graph.add_edge(i, j, weight=self.COST_MATRIX[i][j])
                    graph[i][j]['time'] = (graph[i][j]['weight'] / self.vehicle_speed) * 60

        # On colorie les routes et noeuds
        colors = [0]
        colors += [graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(1, len(graph.nodes))]
        pos = nx.get_node_attributes(graph, 'pos')
        nx.draw_networkx_nodes(graph, pos, node_color=colors)
        p = [2, -100, self.NBR_OF_CUSTOMERS]  # Equation pour trouver n_min
        roots = np.roots(p)

        # Nombre de voiture minimal possible , solution d'une équation de second degrès.
        graph.nodes[0]['n_min'] = max(1, int(roots.min()) + 1)

        plt.title("graphe initial")
        plt.show()
        plt.clf()

        return graph

    # G.nodes[i] avec i un entier de 1 à 540 environ pour acceder au données d'un client.
    # G[i][j] pour accéder au données de la route entre i et j.
    # G.nodes[0] est le dépots, il contient également les informations relatives aux camions.

    def init(self, G, n):
        """
        Fonction d'initialisation du solution possible à n camions.
        Il y a beaucoup d'assertions car en effet, certains graph généré peuvent ne pas présenter de solution:
            -Pas assez de voiture pour finir la livraison dans le temps imparti
            -Les ressources demandées peuvent être trop conséquentes
            -Ect...

        Parameters
        ----------
        G : Graph du problème
        n : Nombre de camions à utiliser dans notre solution

        Returns
        -------
        Une prémière solution x fonctionnelle .

        """
        ###Positions de chaque points###
        X = [G.nodes[i]['pos'][0] for i in range(0, len(G))]
        Y = [G.nodes[i]['pos'][1] for i in range(0, len(G))]

        ###Assertions du début , afin de vérifier que on ne demande pas d'initialiser l'impossible ###
        max_Q = max(G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())
        max_ressources = sum(G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())
        max_nodes_ressources = max([G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(0,
                                                                                 len(G.nodes))])  # On vérifie que les ressources de chaque summits sont au moins <= Q
        assert (
                max_nodes_ressources < max_Q), "Les ressouces de certaines villes sont plus grandes que les ressources des voitures !"
        assert (n > G.nodes[0][
            'n_min']), "Peu-importe la configuration , il n'y a pas assez de camion pour terminer le trajet dans le temps imparti (<100)"  # En effet, le temps de livraison des derniers summits peuvent ne pas être atteint...
        assert (n <= G.nodes[0]['n_max']), "On demande trop de voiture , <= à %s normalement " % G.nodes[0]['n_max']

        #####Construction de la solution qui fonctionne#####

        x = []  # Notre première solution
        for i in range(n):
            x.append([0])  # On initialise chaque route, celles-ci commencent par 0 à chaque fois
        nodes = [i for i in G.nodes]
        nodes.pop(0)

        ###Initialisation du dataframe renseignant sur les summits et leurs contraintes###
        Ordre = main(G)  # Ordre initial
        # Ordre=list(df_ordre_init['Ordre'])
        assert (Ordre[0] == 0), "L'ordre initial n'est pas bon ,il ne commence pas par 0"
        ##Nos camions peuvent-ils livrer tout le monde ?##
        sum_ressources = sum([G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(0, len(G.nodes))])
        if sum_ressources > max_ressources:
            print("Les ressources demandées par les villes sont trop importantes")
            return False

        ###On remplit la solution de la majorité des summits###
        df_camion = pd.DataFrame()  # Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
        df_camion.index = [i for i in range(0, n)]
        ressources = [G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"][i] for i in
                      range(0, n)]  # On commence par les camions aux ressources les plus importantes
        df_camion['Ressources'] = ressources
        df_ordre = pd.DataFrame(
            columns=["Camion", "Ressource_to_add", "Id", "CUSTOMER_TIME_WINDOW_FROM_MIN", "CUSTOMER_TIME_WINDOW_TO_MIN",
                     "Ressource_camion"])
        camion = 0

        ###1ere phase###
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
                temp = copy.deepcopy(x[camion])
                temp.append(nodes_to_add)
                if (df_camion.loc[camion]['Ressources'] >= q_nodes and check_temps_2(temp, G) == True):
                    Q = G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                    assert (
                            q_nodes <= Q), "Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
                    x[camion].append(nodes_to_add)
                    df_camion['Ressources'].loc[camion] += -q_nodes
                    i += 1
                    pbar.update(1)
                    assert (x[camion] == temp)
                else:
                    print(nodes_to_add)
                    assert (x[camion] != temp)
                    camion += 1

                df_ordre = df_ordre.append(dict)
        ##Seconde phase, on remplit la solution des summits qui n'ont pu être affecté lors de la première phase##
        ###Assertion pour vérifier que tout fonctionne bien###
        visummit = pd.DataFrame(columns=["Client", "passage"])
        for i in x:
            for j in i:
                if j not in list(visummit["Client"]):
                    dict = {"Client": j, "passage": 1}
                    visummit = visummit.append([dict])
                else:
                    visummit['passage'][visummit['Client'] == j] += 1
        assert (len(visummit) == len(
            G.nodes)), "Tout les summits ne sont pas pris en compte"  # On vérifie que tout les summits sont pris en compte
        visummit_2 = visummit[visummit['Client'] != 0]
        assert (len(visummit_2[visummit_2['passage'] > 1]) == 0), "Certains summits sont plusieurs fois déservis"
        for i in x:
            i.append(0)
        assert (check_temps(x, G) == True), "Mauvaise initialisation au niveau du temps"

        for i in range(0, len(x)):
            Q = G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i]  # Ressource du camion utilisé
            assert (check_ressource(x[i], Q, G) == True), "Mauvaise initialisation au niveau des ressources"
            assert (0 not in x[i][1:-1]), "Un camion repasse par 0"
        self.plotting(x, G)
        return x

    def plotting(self, x, G):
        plt.clf()
        X = [G.nodes[i]['pos'][0] for i in range(0, len(G))]
        Y = [G.nodes[i]['pos'][1] for i in range(0, len(G))]
        plt.plot(X, Y, "o")
        plt.text(X[0], Y[0], "0", color="r", weight="bold", size="x-large")

        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        couleur = 0
        for camion in range(0, len(x)):
            assert (camion < len(colors)), "Trop de camion, on ne peut pas afficher"
            if len(x) > 2:
                xo = [X[o] for o in x[camion]]
                yo = [Y[o] for o in x[camion]]
                plt.plot(xo, yo, colors[couleur])
                couleur += 1
        plt.show()

    def perturbation(self, x, G, T):
        """
        Fonction de perturbation.
        Il y a beaucoup d'assertions afin de vérifier que la perturbation ne crée pas de problème de contraintes.
           -On prend un summit d'une route parcourue par un camion pour l'ajouter à une autre route
           -Pour chaque route, on permute deux summits
           -Il est possible qu'à chaque étape ,il n'y ait pas de modification.
        Parameters
        ----------
        x : Solution à perturber
        G : Graph du problème

        Returns
        -------
        Une solution x' perturbée de x et  fonctionnelle .

        """
        it = 0
        k = 10e-5  # arbitraire
        Q = [G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0, len(x))]
        nb_summit = len(G.nodes)
        E = self.energy_of_solution(x, G)
        E0 = E + 2
        best_x = copy.deepcopy(x)
        T_list = [T]
        E_list = [E]
        E_min = E
        fig, axs = plt.subplots(1, 2)
        while E0 - E >= 1:

            it += 1
            print("iteration", it, "E=", E, "Température =", T)

            E0 = E
            x = copy.deepcopy(best_x)
            for i in tqdm(range(0, nb_summit - 1)):
                for j in range(i + 1, nb_summit):
                    very_old_x = copy.deepcopy(
                        x)  # Solution du début enregistrée , sert aux assertions de fin. Utiliser deepcopy ! et non copy
                    if i != 0:
                        flatten_x = [item for sublist in x for item in sublist]
                        ajout, retire = flatten_x.index(i), flatten_x.index(j)
                        num_zero_ajout, num_zero_retire = flatten_x[:ajout].count(0) - 1, flatten_x[:retire].count(
                            0) - 1
                        camion_ajout, camion_retire = int(num_zero_ajout / 2 - 1) + 1, int(num_zero_retire / 2 - 1) + 1
                        nouvelle_place = x[camion_ajout].index(i)
                        x[camion_ajout].insert(nouvelle_place, j)
                        x[camion_retire].remove(j)
                    ###On ajoute un summit à une route , on en retire à une autre###
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

                    E1 = self.energy_of_delivery(x[camion_ajout], G, camion_ajout)
                    E2 = self.energy_of_delivery(x[camion_retire], G, camion_retire)
                    E3 = self.energy_of_delivery(very_old_x[camion_ajout], G, camion_ajout)
                    E4 = self.energy_of_delivery(very_old_x[camion_retire], G, camion_retire)

                    if (E1 + E2 >= E3 + E4):
                        p = np.exp(-(E1 + E2 - (E3 + E4)) / (k * T))
                        # print(p)
                        r = rd.random()  # ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier !
                        if r <= p and p != 1:
                            if (check_constraint(x, G) != True):
                                x = copy.deepcopy(very_old_x)  # sinon on conserve x tel qu'il est
                        else:
                            x = copy.deepcopy(very_old_x)

                    else:
                        if (check_ressource(x[camion_ajout], Q[camion_ajout], G) == False or check_temps_2(
                                x[camion_ajout],
                                G) == False):
                            x = copy.deepcopy(very_old_x)
                        else:
                            E = self.energy_of_solution(x, G)
                            if E < E_min:
                                best_x = copy.deepcopy(x)  # On garde en mémoire le meilleur x trouvé jusqu'à présent
                                E_min = E

            ###Assertions de fin###
            self.plotting(best_x, G)
            assert (check_constraint(best_x, G) == True)
            num_very_old_x = sum([len(i) for i in very_old_x])
            num_x = sum([len(i) for i in x])  # On vérifie qu'aucun summit n'a été oublié
            assert (num_very_old_x == num_x)
            E = self.energy_of_solution(best_x, G)
            if E0 > E:
                T = (1 / math.log(E0 - E)) * 1500
                T_list.append(T)
                E_list.append(E)

        plt.clf()
        axs[0].plot(E_list, 'o-')
        axs[0].set_title("Energie de la meilleure solution en fonction des itérations")
        axs[1].plot(T_list, 'o-')
        axs[1].set_title("Température en fonction des itérations")
        fig.suptitle("Profil de la première partie")
        plt.savefig("Profil de la première partie")
        plt.show()

        ###Assertions de fin###

        visummit = pd.DataFrame(columns=["Client", "passage"])
        for i in best_x:
            for j in i:
                if j not in list(visummit["Client"]):
                    dict = {"Client": j, "passage": 1}
                    visummit = visummit.append([dict])
                else:
                    visummit['passage'][visummit['Client'] == j] += 1
        assert (len(visummit) == len(
            G.nodes)), "Tout les summits ne sont pas pris en compte"  # On vérifie que tout les summits sont pris en compte
        visummit_2 = visummit[visummit['Client'] != 0]
        assert (len(visummit_2[visummit_2['passage'] > 1]) == 0), "Certains summits sont plusieurs fois déservis"
        assert (check_constraint(best_x, G) == True)
        self.plotting(x, G)

        return best_x

    """
    Parameters:
    ----------
    solution: list - a given solution
    graph: ? - the problem's graph
     
    Returns
    --------
    
    """

    def perturbation_2(self, solution, graph):
        energy = self.energy_of_solution(solution, graph)
        d0 = energy + 1
        list_E = [energy]

        while energy < d0:
            d0 = energy

            for camion in tqdm(range(0, len(solution))):
                route = solution[camion]

                for i in range(1, len(route) - 1):

                    for j in range(i + 2, len(route)):
                        d_part = self.energy_of_delivery(route, graph, camion)
                        r = route[i:j].copy()
                        r.reverse()
                        route2 = route[:i] + r + route[j:]
                        t = self.energy_of_delivery(route2, graph, camion)

                        if (t < d_part):
                            if check_temps_2(route2, graph) == True:
                                solution[camion] = route2

            energy = self.energy_of_solution(solution, graph)
            list_E.append(energy)

            assert (check_temps(solution, graph) == True)

            self.plotting(solution, graph)

        plt.clf()
        plt.plot(list_E, 'o-')
        plt.title("Evoluation de l'énergie lors de la seconde phase")
        plt.show()

        # Assertions de fin###

        visummit = pd.DataFrame(columns=["Client", "passage"])

        for i in solution:

            for j in i:
                if j not in list(visummit["Client"]):
                    dict = {"Client": j, "passage": 1}
                    visummit = visummit.append([dict])

                else:
                    visummit['passage'][visummit['Client'] == j] += 1

        assert (len(visummit) == len(
            graph.nodes)), "Tout les summits ne sont pas pris en compte"  # On vérifie que tout les summits sont pris en compte
        visummit_2 = visummit[visummit['Client'] != 0]

        assert (len(visummit_2[visummit_2['passage'] > 1]) == 0), "Certains summits sont plusieurs fois déservis"

        assert (check_constraint(solution, graph) == True), "Mauvaise initialisation au niveau du temps"

        return solution
