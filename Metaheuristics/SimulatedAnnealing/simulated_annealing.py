""" Import librairies """
import copy
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random as rd

from Utility.database import Database
from Utility.common import set_root_dir, generate_initial_solution, compute_fitness, compute_delivery_fitness
from Utility.validator import is_solution_valid, pick_valid_solution
from Utility.plotter import plot_solution

set_root_dir()

warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Recuit-simule sur la base de donnée du fil rouge, plusieurs notions à prendre en compte, l'algo se déroule en 3 étapes:
    -Initialisation -> on fait un premier recuit qui réalise l'algorithme du problème du voyageur, à partir de cela on
     initialise une première solution vérifiant les contraintes
    -Recuit -> Par un recuit , on perturbe notre solution en mélangeant les clients des différents camions uniquement,
     on récupère une solution x lorsque cela n'évolue plus.
    -Affinement -> On perturbe l'initial_order de desservissement des clients pour un camion en question puis on 
    retourne la meilleure solution.
Au niveau des contraintes:
    - On respecte les intervalles de temps pour desservir en temps et en heure chaque client
    -On respecte les ressources à livrer au niveau du poids en kg
    -Chaque camion a son propre coefficient et sa propre capacité de livraison
    -Le temps de trajets d'un camion pour aller du dépot à la première ville et de la dernière ville au dépot n'est pas
     pris en compte
    -Le temps de livraison est pris en compte mais divisé par 10 car pafois trop long (+ d'1h pour délivrer...)
    En effet, certaines villes sont a plus de 500 minutes du dépot lorsque le camion roule à 50 km/h !
Concernant la solution retournée :
    De la forme x=[[0,client1,client2,...,0],[0,client_n,client_n+1..,0],[clients_du_camion_3],[client_du_camion_4]...]
    Les informations du camion numéro i sont accessibles via : G.nodes[0]['Vehicles'][camion]
    Chaque client i a un identifiant client accessible via G.nodes[i]['CUSTOMER_CODE'], G.nodes[i] donne d'ailleurs 
    d'autres informations sur le client i.
    
"""


class Annealing:
    fitness: float = 0
    solution: list = []

    speedy = True

    def __init__(self, graph=None, initial_temperature=1500, vehicle_speed=50, speedy=False):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph

        self.Graph = graph

        self.NBR_OF_CUSTOMER = len(graph) - 1
        self.T = initial_temperature
        self.speed = vehicle_speed
        self.Speedy = speedy

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

    def main(self, initial_solution=None):
        if initial_solution is None:
            solution = generate_initial_solution(self.Graph)

            if not is_solution_valid(solution, self.Graph):
                initial_solution = pick_valid_solution()

        solution = initial_solution

        plot_solution(solution, graph=self.Graph, title='initial solution')

        print("Solution initialisée , début de la phase de recuit \n")

        solution = self.recuit_simule(solution, self.Graph, self.T, self.speedy)

        self.solution = solution
        self.fitness = compute_fitness(self.solution, self.Graph)

        plot_solution(self.solution, graph=self.Graph)

        print("Pour l'instant l'énergie est de :%d" % self.fitness)
        print("Début de la phase de perfectionnement de la solution \n")

        solution = self.perturbation_intra(solution, self.Graph, self.Speedy)

        self.solution = solution
        self.fitness = compute_fitness(self.solution, self.Graph)

        plot_solution(self.solution, graph=self.Graph)

        print("Finalement l'énergie est de :%d" % self.fitness)

    """
    Compute a temperature thanks to the energy of the solution
    """

    @staticmethod
    def compute_temperature(energy, initial_energy):
        temperature = 1500 / np.log(initial_energy - energy)
        return temperature

    """
    Fonction de recuit qui mélanges les clients de chaque camion mais qui ne modifie pas l'initial_order de 
    livraison pour un camion en question. 
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
        assert (is_solution_valid(solution, graph) is True), 'solution should be valid'

        it = 0
        k = 10e-5  # arbitraire
        nb_sommet = len(graph.nodes)
        E = compute_fitness(solution, graph)
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
                        E1, E2, E3, E4 = compute_delivery_fitness(solution[camion_ajout], graph,
                                                                  camion_ajout), compute_delivery_fitness(
                            solution[camion_retire], graph, camion_retire), compute_delivery_fitness(
                            very_old_x[camion_ajout], graph,
                            camion_ajout), compute_delivery_fitness(
                            very_old_x[camion_retire], graph, camion_retire)

                        ###Principe du recuit###
                        if (E1 + E2 >= E3 + E4):
                            p = np.exp(-(E1 + E2 - (E3 + E4)) / (k * temperature))
                            r = rd.random()

                            if r <= p and p != 1:  # Si p fait 1 , cela ralentit énormément.
                                if not is_solution_valid(solution, graph):
                                    solution = copy.deepcopy(very_old_x)  # sinon on conserve x tel qu'il est
                                # else :
                                ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###
                                # assert (len(solution[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                # assert (len(solution[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                            else:
                                solution = copy.deepcopy(very_old_x)

                        else:
                            # if (is_delivery_capacity_valid(graph, solution[camion_ajout], Q[camion_ajout]) == False or check_temps_part(
                            # solution[camion_ajout], graph) == False):
                            # solution = copy.deepcopy(very_old_x)
                            # else:
                            E = compute_fitness(solution, graph)
                            if E < E_min:
                                ###Assertion non trop coûteuse mais efficace pour vérifier que notre solution est modifiée comme voulue###
                                # assert (len(solution[camion_ajout]) == len(very_old_x[camion_ajout]) + 1)
                                # assert (len(solution[camion_retire]) == len(very_old_x[camion_retire]) - 1)
                                best_x = copy.deepcopy(
                                    solution)  # On garde en mémoire le meilleur x trouvé jusqu'à présent
                                E_min = E

            ###Assertions de fin###

            is_solution_valid(best_x, graph)

            num_very_old_x = sum([len(i) for i in very_old_x])
            num_x = sum([len(i) for i in solution])  # On vérifie qu'aucun sommet n'a été oublié
            assert (num_very_old_x == num_x)
            E = compute_fitness(best_x, graph)

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
        d = compute_fitness(solution, graph)
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
                        d_part = compute_delivery_fitness(route, graph, camion)
                        r = route[i:j].copy()
                        r.reverse()
                        route2 = route[:i] + r + route[j:]
                        t = compute_delivery_fitness(route2, graph, camion)

                        if (t < d_part):
                            # if check_temps_part(route2, graph) == True:
                            solution[camion] = route2

            d = compute_fitness(solution, graph)
            list_E.append(d)

            if speedy:
                break

        plt.clf()
        plt.plot(list_E, 'o-')
        plt.title("Evolution de l'énergie lors de la seconde phase")
        plt.show()

        is_solution_valid(solution, graph)

        return solution
