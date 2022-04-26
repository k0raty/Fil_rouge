# -*- coding: utf-8 -*-

"""
Recuit-simulé sur la base de donnée du fil rouge, plusieurs notions à prendre en compte, l'algo se déroule en 3 étapes:
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
    Les informations du camion numéro i sont accessibles via : G.nodes[0]['Camion'][camion]
    Chaque client i a un identifiant client accessible via G.nodes[i]['CUSTOMER_CODE'], G.nodes[i] donne d'ailleurs d'autres informations sur le client i.
    
"""

""" Import librairies """
import os 
import copy
import random as rd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings


""" Import utilities """
from Utility.database import Database
from Utility.common import *
from Metaheuristics.SimulatedAnnealing.simulated_annealing_initialization import main


set_root_dir()

warnings.simplefilter(action='ignore', category=FutureWarning)


class Annealing:
    fitness: float = 0
    solution: list = []

    def __init__(self, customers=None, depot=None, vehicles=None, graph=None, initial_temperature=1500, vehicle_speed=50):
        if customers is None:
            database = Database(vehicle_speed)
            customers = database.Customers
            vehicles = database.Vehicles
            graph = database.graph
            depot = database.Depots[0]

        self.graph = graph
        self.Customers = customers
        self.Depot = depot
        self.Vehicles = vehicles

        self.NBR_OF_CUSTOMER = len(customers)
        self.NBR_OF_VEHICLE = len(self.Vehicles['VEHICLE_CODE'])
        self.T = initial_temperature
        self.speed = vehicle_speed

    def main(self, initial_solution=None, speedy=True):
        
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
        ordre_init_path = os.path.join('Metaheuristics', 'SimulatedAnnealing', 'df_ordre_init.pkl')
        df = pd.read_pickle(ordre_init_path)

        initial_solution = list(df['Ordre'])

        graph = self.graph
        self.speedy=speedy
        print("Initialisation de la solution \n")
        if initial_solution==None :
            solution = self.init(graph)
        else: solution=initial_solution
        plotting(solution, graph)
        print("Solution initialisée , début de la phase de recuit \n")
        
        solution = self.recuit_simulé(solution,graph,self.T,self.speedy)
        self.fitness= energie(solution,graph)
        print("Pour l'instant l'énergie est de :%d" %self.fitness)
        print("Début de la phase de perfectionnement de la solution \n")

        solution = self.perturbation_intra(solution,graph,speedy)
        self.fitness= energie(solution,graph)
        print("Finalement l'énergie est de :%d" %self.fitness)

        return solution

  

    def init(self, graph,initial_order=None):
        
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
        if initial_order == None : 
            initial_order = main(graph)
        
        message = "L'initial_order initial n'est pas bon ,il ne commence pas par 0"
        assert (initial_order[0] == 0), message

        # Nos camions peuvent-ils livrer tout le monde ?
        sum_ressources = sum([graph.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(len(graph.nodes))])

        if sum_ressources > max_ressources:
            print("Les ressources demandées par les villes sont trop importantes")
            return False

        # On remplit la solution de la majorité des sommets
        df_camion = pd.DataFrame()  # Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
        df_camion.index = range(self.NBR_OF_VEHICLE)
        n=self.NBR_OF_VEHICLE
        ressources = [graph.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"][i] for i in
                      range(0, n)]  # On commence par les camions aux ressources les plus importantes
        df_camion['Ressources'] = ressources
        df_initial_order = pd.DataFrame(
            columns=["Camion", "Ressource_to_add", "Id", "CUSTOMER_TIME_WINDOW_FROM_MIN", "CUSTOMER_TIME_WINDOW_TO_MIN",
                     "Ressource_camion"])
        camion = 0

        i = 1  # On ne prend pas en compte le zéros du début
        with tqdm(total=len(initial_order)) as pbar:
            while i < len(initial_order):
                assert camion < n, "Impossible d'initialiser , les camions ne sont pas assez nombreux"
                nodes_to_add = initial_order[i]
                assert (nodes_to_add != 0), "Le chemin proposé repasse par le dépot !"

                q_nodes = graph.nodes[nodes_to_add]['TOTAL_WEIGHT_KG']
                int_min = graph.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_FROM_MIN"]
                int_max = graph.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_TO_MIN"]
                dict = [{"Camion": camion, "Ressource_to_add": q_nodes, "Id": nodes_to_add,
                         "CUSTOMER_TIME_WINDOW_FROM_MIN": int_min, "CUSTOMER_TIME_WINDOW_TO_MIN": int_max,
                         "Ressource_camion": df_camion.loc[camion]['Ressources']}]
                temp = copy.deepcopy(solution[camion])
                temp.append(nodes_to_add)
                if (df_camion.loc[camion]['Ressources'] >= q_nodes and check_temps_part(temp, graph) == True):
                    Q = graph.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                    assert (q_nodes <= Q), "Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
                    solution[camion].append(nodes_to_add)
                    df_camion['Ressources'].loc[camion] += -q_nodes
                    i += 1
                    pbar.update(1)
                    assert (solution[camion] == temp)
                else:
                    print(nodes_to_add)
                    assert (solution[camion] != temp)
                    camion += 1

                df_initial_order = df_initial_order.append(dict)

        for i in solution:
            i.append(0)
        ###Assertion pour vérifier que tout fonctionne bien###

        assert (check_constraint(solution, graph) == True), "Mauvaise initialisation au niveau du temps"
        check_forme(solution, graph)

        ###Affichage de la première solution###
        plotting(solution, graph)
        return solution

    # A ADAPTER DANS LA CLASSE
    def temperature(self,E, E0):
        """
        Fonction température
        """
        return (1 / math.log(E0 - E)) * 1500

    # A ADAPTER DANS LA CLASSE MAIS JE PENSE QU'IL EST DANS validator
   
    def recuit_simulé(self,x, G, T,speedy):

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
            if speedy==True :
                print("Mode speed_run \n")
                return best_x
            ###Modification de la température###
            if E0 > E:
                self.T = self.temperature(E, E0)
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
        
    def perturbation_intra(self,x,G,speedy):
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
        d  = energie(x,G)
        d0 = d+1
        it = 1
        list_E=[d]
        while d < d0 :
            it += 1
            print("iteration",it, "d=",d)
            d0 = d
            for camion in tqdm(range(0,len(x))):
                route=x[camion]
                for i in range(1,len(route)-1) :
                    for j in range(i+2,len(route)):
                        d_part=energie_part(route,G,camion)
                        r = route[i:j].copy()
                        r.reverse()
                        route2 = route[:i] + r + route[j:]
                        t = energie_part(route2,G,camion)
                        if (t < d_part): 
                            if check_temps_part(route2,G)==True: 
                                x[camion] = route2   
            d=energie(x,G)
            list_E.append(d)
            assert(check_temps(x,G)==True)  
            plotting(x,G)
            if speedy == True :
                break
        plt.clf()
        plt.plot(list_E,'o-')
        plt.title("Evoluation de l'énergie lors de la seconde phase")
        plt.show()
        
        ###Assertions de fin###
        
        check_forme(x,G)
        assert(check_constraint(x,G)==True),"Mauvaise initialisation au niveau du temps"
        return x
    # A ADAPTER DANS LA CLASSE
   
   