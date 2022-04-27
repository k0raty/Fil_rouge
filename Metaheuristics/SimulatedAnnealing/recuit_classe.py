# -*- coding: utf-8 -*-

"""
Recuit-simule sur la base de donnée du fil rouge, plusieurs notions à prendre en compte, l'algo se déroule en 3 étapes:
    -Initialisation -> on fait un premier recuit qui réalise l'algorithme du problème du voyageur, à partir de cela on initialise une première solution vérifiant les contraintes
    -Recuit -> Par un recuit , on perturbe notre solution en mélangeant les clients des différents camions uniquement, on récupère une solution x lorsque cela n'évolue plus.
    -Affinement -> On perturbe l'initial_order de desservissement des clients pour un camion en question puis on retourne la meilleure solution.
Au niveau des contraintes:
    - On respecte les intervalles de temps pour desservir en temps et en heure chaque client
    -On respecte les ressources à livrer au niveau du poids en kg
    -Chaque camion a son propre coefficient et sa propre capacité de livraison
    -Le temps de trajets d'un camion pour aller du dépot à la première ville et de la dernière ville au dépot n'est pas pris en compte
    -Le temps de livraison est pris en compte mais divisé par 10 car pafois trop long (+ d'1h pour délivrer...)
    En effet, certaines villes sont a plus de 500 minutes du dépot lorsque le camion roule à 50 km/h !
Concernant la solution retournée :
    De la forme x=[[0,client1,client2,...,0],[0,client_n,client_n+1..,0],[clients_du_camion_3],[client_du_camion_4]...]
    Les informations du camion numéro i sont accessibles via : G.nodes[0]['Vehicles'][camion]
    Chaque client i a un identifiant client accessible via G.nodes[i]['CUSTOMER_CODE'], G.nodes[i] donne d'ailleurs d'autres informations sur le client i.
    
"""

""" Import librairies """
import copy
import math
import warnings

""" Import utilities """
from Utility.database import Database
from Utility.common import *
from Utility.validator import *
from Metaheuristics.SimulatedAnnealing.simulated_annealing_initialization import generate_order

set_root_dir()

warnings.simplefilter(action='ignore', category=FutureWarning)


class Annealing:
    fitness: float = 0
    solution: list = []

    speedy = True

    def __init__(self, graph=None, initial_temperature=1500, vehicle_speed=50):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph

        self.graph = graph

        self.NBR_OF_CUSTOMER = len(graph) - 1
        self.T = initial_temperature
        self.speed = vehicle_speed

    """
    Main function , réalise le recuit. 

    Parameters
    ----------
    df_customers : tableur excel renseignant sur les clients
    df_vehicles : tableur excel renseignant sur les camions à disposition 
    v : Vitesse des véhicules
    T : Température de départ lors de la phase de recuit.
    speedy: Si c'est en phase rapide ou non (très peu d'itérations)
    ----------

    Returns
    -------
    x : Solution proposée 
    -------
    """

    def main(self, initial_solution=None, speedy=False):
        ordre_init_path = os.path.join('Metaheuristics', 'SimulatedAnnealing', 'df_ordre_init.pkl')
        df = pd.read_pickle(ordre_init_path)

        initial_solution = list(df['Ordre'])

        self.speedy = speedy
        print("Initialisation de la solution \n")

        if initial_solution is None:
            solution = self.generate_initial_solution()

        else:
            solution = initial_solution

        plotting(solution, self.graph)
        print("Solution initialisée , début de la phase de recuit \n")

        solution = self.recuit_simule(solution, self.graph, self.T, self.speedy)
        self.fitness = energie(solution, self.graph)
        print("Pour l'instant l'énergie est de :%d" % self.fitness)
        print("Début de la phase de perfectionnement de la solution \n")

        solution = self.perturbation_intra(solution, self.graph, speedy)
        self.fitness = energie(solution, self.graph)
        print("Finalement l'énergie est de :%d" % self.fitness)

        return solution

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

    def generate_initial_solution(self, initial_order=None):
        total_vehicles_capacity = sum(self.graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"].values())
        total_customers_capacity = sum([self.graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(self.graph.nodes))])

        max_vehicle_capacity = max(self.graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"].values())
        max_customer_capacity = max([self.graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(self.graph.nodes))])

        message = 'Some customers have packages heavier than vehicles capacity'
        assert (max_customer_capacity < max_vehicle_capacity), message

        message = 'There is not enough vehicles to achieve the deliveries in time, regardless the configuration'
        assert (self.graph.nodes[0]['NBR_OF_VEHICLE'] > self.graph.nodes[0]['n_min']), message

        solution = []  # Notre première solution

        for index_delivery in range(self.graph.nodes[0]['NBR_OF_VEHICLE']):
            solution.append([0])

        customers = [node for node in self.graph.nodes]
        customers.pop(0)
        print('step 1')
        # Initialisation du dataframe renseignant sur les sommets et leurs contraintes
        if initial_order is None:
            initial_order = generate_order(self.graph)
        print('step 2')
        message = 'The initial order should start with 0 (the depot)'
        assert (initial_order[0] == 0), message

        message = 'All packages to deliver are heavier than total vehicles capacity'
        assert (total_customers_capacity <= total_vehicles_capacity), message

        # On remplit la solution de la majorité des sommets
        df_camion = pd.DataFrame()  # Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
        df_camion.index = range(self.graph.nodes[0]['NBR_OF_VEHICLE'])
        vehicles_capacity = self.graph.nodes[0]['Vehicles']["VEHICLE_TOTAL_WEIGHT_KG"]
        ressources = [vehicles_capacity[i] for i in range(self.graph.nodes[0]['NBR_OF_VEHICLE'])]

        df_camion['Ressources'] = ressources
        columns = [
            'Vehicles',
            'Ressource_to_add',
            'Id',
            'CUSTOMER_TIME_WINDOW_FROM_MIN',
            'CUSTOMER_TIME_WINDOW_TO_MIN',
            'Ressource_camion',
        ]
        df_initial_order = pd.DataFrame(columns=columns)
        camion = 0

        i = 1  # On ne prend pas en compte le zéros du début
        with tqdm(total=len(initial_order)) as pbar:
            while i < len(initial_order):
                message = 'Not enough vehicles'
                assert (camion < self.graph.nodes[0]['NBR_OF_VEHICLE']), message

                nodes_to_add = initial_order[i]
                message = 'Given delivery goes back to depot'
                assert (nodes_to_add != 0), message

                q_nodes = self.graph.nodes[nodes_to_add]['TOTAL_WEIGHT_KG']
                int_min = self.graph.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_FROM_MIN"]
                int_max = self.graph.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_TO_MIN"]

                temp = copy.deepcopy(solution[camion])
                temp.append(nodes_to_add)

                if df_camion.loc[camion]['Ressources'] >= q_nodes and check_temps_part(temp, self.graph):
                    Q = self.graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                    message = 'Some customers have packages heavier than vehicles capacity'
                    assert (q_nodes <= Q), message

                    solution[camion].append(nodes_to_add)
                    df_camion['Ressources'].loc[camion] += -q_nodes
                    i += 1
                    pbar.update(1)
                    assert (solution[camion] == temp)
                else:
                    print(nodes_to_add)
                    assert (solution[camion] != temp)
                    camion += 1

                df_initial_order = df_initial_order.append([{
                    'Vehicles': camion,
                    "Ressource_to_add": q_nodes,
                    "Id": nodes_to_add,
                    "CUSTOMER_TIME_WINDOW_FROM_MIN": int_min,
                    "CUSTOMER_TIME_WINDOW_TO_MIN": int_max,
                    "Ressource_camion": df_camion.loc[camion]['Ressources'],
                }])

        for delivery in solution:
            delivery.append(0)

        is_solution_time_valid(solution, self.graph)
        is_solution_capacity_valid(solution, self.graph)
        is_solution_shape_valid(solution, self.graph)

        plotting(solution, self.graph)

        return solution

    """
    Compute a temperature thanks to the energy of the solution
    """

    def compute_temperature(self, energy, initial_energy):
        temperature = 1500 / math.log(initial_energy - energy)
        return temperature

    """
    Fonction de recuit qui mélanges les clients de chaque camion mais qui ne modifie pas l'initial_order de deservissement pour un camion en question. 
    Il y a beaucoup d'assertions afin de vérifier que la perturbation engendrée ne crée pas de problème de contraintes.  
       -On prend un sommet d'une route parcourue par un camion pour l'ajouter à une autre route
       -Il est possible qu'à chaque étape ,il n'y ait pas de modification. 
       -La baisse de température n'est pas graduelle mais dépend de l'écart entre deux solutions proposées
       (+ l'écart est petit, + la température augmente).

    Parameters
    ----------
    x : Solution à perturber
    G : Graph du problème 
    speedy : Si oui ou non on effectue une unique itération
    ----------
    
    Returns
    -------
    Une solution x améliorée .
    -------
    """

    def recuit_simule(self, solution, graph, temperature, speedy):
        is_solution_shape_valid(solution, graph)
        is_solution_time_valid(solution, graph)
        is_solution_capacity_valid(solution, graph)

        it = 0
        k = 10e-5  # arbitraire
        Q = [graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0, len(solution))]
        nb_sommet = len(graph.nodes)
        E = energie(solution, graph)
        E0 = E + 2
        best_x = copy.deepcopy(solution)  # La meilleure solution trouvée lors de toute les perturbation
        T_list = [temperature]
        E_list = [E]
        E_min = E
        fig, axs = plt.subplots(1, 2)

        while E0 - E >= 0.5:  # Tant que la solution retournée diffère d'un km de coût , on continue de chercher
            it += 1
            print("iteration", it, "E=", E, "Température =", temperature)

            E0 = E
            solution = copy.deepcopy(best_x)

            ###Phase de perturbation de x ###

            for i in tqdm(range(0,
                                nb_sommet - 1)):  # i correspond au client qui verra le client j être rajouté avant lui dans le desservissement.
                for j in range(i + 1, nb_sommet):
                    very_old_x = copy.deepcopy(
                        solution)  # Solution du début enregistrée , on la garde en mémoire si après perturbation ,x n'est pas meilleur.

                    ###On ajoute un sommet à une route , on en retire à une autre###
                    if i != 0:
                        flatten_x = [item for sublist in solution for item in sublist]
                        ajout, retire = flatten_x.index(i), flatten_x.index(j)
                        num_zero_ajout, num_zero_retire = flatten_x[:ajout].count(0) - 1, flatten_x[:retire].count(
                            0) - 1
                        camion_ajout, camion_retire = int(num_zero_ajout / 2 - 1) + 1, int(num_zero_retire / 2 - 1) + 1
                        if camion_ajout != camion_retire:  # Pour des raisons de compléxité , on ne s'occupe pas des cas ou i et j sont dans le routage d'un même camion ,cela vient après !
                            nouvelle_place = solution[camion_ajout].index(i)
                            solution[camion_ajout].insert(nouvelle_place, j)
                            solution[camion_retire].remove(j)


                    ### On ajoute le client j a une trajectoire "vide" dans laquelle aucun camion n'est deservie ###
                    else:
                        flatten_x = [item for sublist in solution for item in sublist]
                        retire = flatten_x.index(j)
                        num_zero_retire = flatten_x[:retire].count(0) - 1
                        camion_retire = int(num_zero_retire / 2 - 1) + 1
                        assert (j in solution[camion_retire])
                        unfilled_road = [i for i in range(0, len(solution)) if len(solution[i]) == 2]
                        if len(unfilled_road) != 0:
                            nouvelle_place = 1
                            camion_ajout = unfilled_road[0]
                            solution[camion_ajout].insert(nouvelle_place, j)
                            solution[camion_retire].remove(j)

                    ###On compare l'énergie (seulement une partie) de notre solution perturbé avec la précédente solution###
                    if camion_ajout != camion_retire:
                        E1, E2, E3, E4 = energie_part(solution[camion_ajout], graph, camion_ajout), energie_part(solution[camion_retire],
                                                                                                                 graph,
                                                                                                                 camion_retire), energie_part(
                            very_old_x[camion_ajout], graph, camion_ajout), energie_part(very_old_x[camion_retire], graph,
                                                                                         camion_retire)

                        ###Principe du recuit###
                        if (E1 + E2 >= E3 + E4):
                            p = np.exp(-(E1 + E2 - (E3 + E4)) / (k * temperature))
                            r = rd.random()

                            if r <= p and p != 1:  # Si p fait 1 , cela ralentit énormément.
                                if not is_solution_valid(solution, graph) :
                                    solution = copy.deepcopy(very_old_x)  # sinon on conserve x tel qu'il est
                                #else :
                                    ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###
                                   # assert (len(solution[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                   # assert (len(solution[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                            else :
                                solution = copy.deepcopy(very_old_x)

                        else:
                            #if (is_delivery_capacity_valid(graph, solution[camion_ajout], Q[camion_ajout]) == False or check_temps_part(
                                    #solution[camion_ajout], graph) == False):
                               #solution = copy.deepcopy(very_old_x)
                            #else:
                            E = energie(solution, graph)
                            if E < E_min:
                                ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###
                                #assert (len(solution[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                #assert (len(solution[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                                best_x = copy.deepcopy(
                                    solution)  # On garde en mémoire le meilleur x trouvé jusqu'à présent
                                E_min = E

            ###Assertions de fin###
            plotting(best_x, graph)

            is_solution_valid(best_x, graph)

            num_very_old_x = sum([len(i) for i in very_old_x])
            num_x = sum([len(i) for i in solution])  # On vérifie qu'aucun sommet n'a été oublié
            assert (num_very_old_x == num_x)
            E = energie(best_x, graph)

            if speedy:
                print("Mode speed_run \n")
                return best_x

            ###Modification de la température###
            if E0 > E:
                self.T = self.compute_temperature(E, E0)
                T_list.append(temperature)
                E_list.append(E)

        plt.clf()
        axs[0].plot(E_list, 'o-')
        axs[0].set_title("Energie de la meilleure solution en fonction des itérations")
        axs[1].plot(T_list, 'o-')
        axs[1].set_title("Température en fonction des itérations")
        fig.suptitle("Profil de la première partie")
        fig.savefig("Profil de la première partie")
        fig.show()

        is_solution_valid(best_x, graph)

        plotting(solution, graph)

        return best_x

    """
    Deuxième phase de perturbation , on n'échange plus des clients entre chaque trajectoire de camion  mais 
    seulement l'ordre des client pour chaque route

    Parameters
    ----------
    solution : solution après la première phase
    graph : Graphe du problème
    ----------

    Returns
    -------
    x : solution finale
    -------
    """

    @staticmethod
    def perturbation_intra(solution, graph, speedy):
        d = energie(solution, graph)
        d0 = d + 1
        iteration = 1
        list_E = [d]

        while d < d0:
            iteration += 1
            print("iteration", iteration, "d=", d)
            d0 = d

            for camion in tqdm(range(0, len(solution))):
                route = solution[camion]

                for i in range(1, len(route) - 1):
                    for j in range(i + 2, len(route)):
                        d_part = energie_part(route, graph, camion)
                        r = route[i:j].copy()
                        r.reverse()
                        route2 = route[:i] + r + route[j:]
                        t = energie_part(route2, graph, camion)

                        if (t < d_part):
                            #if check_temps_part(route2, graph) == True:
                            solution[camion] = route2

            d = energie(solution, graph)
            list_E.append(d)

            is_solution_time_valid(solution, graph)

            plotting(solution, graph)
            if speedy:
                break

        plt.clf()
        plt.plot(list_E, 'o-')
        plt.title("Evolution de l'énergie lors de la seconde phase")
        plt.show()

        is_solution_valid(solution, graph)

        return solution
