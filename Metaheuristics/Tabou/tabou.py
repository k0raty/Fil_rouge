""" Import librairies """
import copy
import os
import pandas as pd
import random as rd

""" Import utilities """
from Utility.database import Database
from Utility.common import compute_cost_matrix, compute_fitness, distance

os.chdir(os.path.join('', '..'))


class Tabou:
    MAX_ITERATION = 10
    MAX_NEIGHBORS = 20

    VEHICLE_SPEED = 50
    VEHICLE_CAPACITY = 5000

    def __init__(self, customers=None, depots=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depots = database.Depots
            cost_matrix = compute_cost_matrix(customers)

        self.solution = None

        nbr_of_sites = len(customers)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_SITES = nbr_of_sites
        self.PROBA_MUTATION = 1 / nbr_of_sites

        self.Customers = customers
        self.Depots = depots
        self.Vehicles = vehicles[0]

    """
    Apply the tabou algorithm to improve an initial solution over a maximum number of iterations 

    Parameters
    ----------
    initial_solution: list - a given solution that the algorithm must improve
    -------
    Returns
    -------
    solution: list - new solution more optimized than the initial one
    fitness: float - the fitness score of the new solution
    -------
    """

    def main(self, initial_solution=None):
        solution, fitness = self.find_best_neighbor(initial_solution)

        for iteration in range(self.MAX_ITERATION):
            solution, fitness = self.find_best_neighbor(solution)

        self.solution = solution

        return self.solution, fitness

    """
    Find the best solution in the close neighborhood of the given solution
    
    Parameters
    ----------
    initial_solution: list - a given solution
    ----------
    Returns
    -------
    solution: list - the best neighbor of the given solution
    fitness: float - the fitness score of the best neighbor of the given solution
    """

    def find_best_neighbor(self, initial_solution):
        solution = initial_solution
        fitness = compute_fitness(initial_solution)

        for iteration in range(self.MAX_NEIGHBORS):
            neighbor = self.find_neighbor(initial_solution)
            neighbor_fitness = compute_fitness(neighbor)

            if neighbor_fitness >= fitness and self.is_solution_valid(neighbor):
                solution = neighbor
                fitness = neighbor_fitness

        return solution, fitness

    """
    Switch 2 summits of the given solution to create a new solution really close to the given one
    
    Parameters
    ----------
    solution: list - a given solution
    ----------
    Returns
    -------
    neighbor: list - a new solution close to the given one
    -------
    """

    def find_neighbor(self, solution):
        neighbor = copy.deepcopy(solution)

        nbr_of_sub_road = len(solution)

        index_sub_road_i = rd.randint(0, nbr_of_sub_road - 1)
        index_sub_road_j = rd.randint(0, nbr_of_sub_road - 1)

        sub_road_i = solution[index_sub_road_i]
        sub_road_j = solution[index_sub_road_j]

        length_sub_road_i = len(sub_road_i)
        length_sub_road_j = len(sub_road_j)

        if length_sub_road_i >= 3 and length_sub_road_j >= 3:
            index_summit_i = rd.randint(1, length_sub_road_i - 1)
            index_summit_j = rd.randint(1, length_sub_road_j - 1)

            summit_i = sub_road_i[index_summit_i]
            summit_j = sub_road_j[index_summit_j]

            neighbor[index_sub_road_i][index_summit_i] = summit_j
            neighbor[index_sub_road_j][index_summit_j] = summit_i

        return neighbor

    """
    Vérifie si une solution vérifie les critères définis

    Parameters
    -------
    solution: list - a given solution
    -------
    Returns
    -------
    :bool - the validity of the given solution
    -------
    """

    def is_solution_valid(self, solution):
        nbr_of_sub_road = len(solution)

        for index_sub_road in range(nbr_of_sub_road):
            sub_road = solution[index_sub_road]

            if not self.is_sub_road_valid(sub_road):
                return False

        return True

    """
    Vérifie si une livraison vérifie les critères définis
    en loccurence les horaires de passage et la masse de chaque camion

    Parameters
    -------
    sub_road: list - a portion of a solution
    -------
    Returns:
    -------
    :bool - the validity of the given sub road
    -------
    """

    def is_sub_road_valid(self, sub_road):
        weight = 0  # en kg
        current_t = 481  # en min

        nbr_of_summit = len(sub_road)

        for index_summit in range(1, nbr_of_summit - 1):
            line = df.iloc[sub_road[index_summit]]
            package_weight = line['TOTAL_WEIGHT_KG']

            t_1to2 = 0

            if nbr_of_summit >= 2:
                line_1 = df.iloc[sub_road[nbr_of_summit - 1]]
                line_2 = df.iloc[sub_road[nbr_of_summit - 2]]
                lat_1, lon_1 = line_1['CUSTOMER_LATITUDE'], line_1['CUSTOMER_LONGITUDE']
                lat_2, lon_2 = line_2['CUSTOMER_LATITUDE'], line_2['CUSTOMER_LONGITUDE']
                dist = distance(lat_1, lon_1, lat_2, lon_2)
                t_1to2 = (dist / self.VEHICLE_SPEED) / 60

            current_t + t_1to2

            t_min, t_max = line['CUSTOMER_TIME_WINDOW_FROM_MIN'], line['CUSTOMER_TIME_WINDOW_TO_MIN']

            if t_min == 780: t_min = 480

            if package_weight > self.VEHICLE_CAPACITY or (t_min > current_t or current_t > t_max):
                return False

            else:
                weight += package_weight

        return True


    def livraison_un_camion(self, df, all_treated_lines):
                                                       """
                                                       Crée une livraison pour un camion donné selon les paramètres définis:
                                                       horaires de livraison respectées
                                                       masse maximale des camions non dépassées

                                                       Parameters
                                                       ----------
                                                       df : dataframe
                                                       dataframe de travail.
                                                       all_treated_lines : list of intagers
                                                       clients déjà traités.
                                                       -------
                                                       Returns
                                                       -------
                                                       all_treated_lines: la liste des commandes déjà traitées avec celles de cette commande en plus
                                                       L_parcours: la livraison du camion en question ex: [1,4,5,7,8,9,11,552,665,..]
                                                       """
                                                       Q = 5000  # en kg
                                                       weight = 0
                                                       # la liste tabou contient toutes les lignes déjà traitées
                                                       L_lines = []  # liste des lignes traitées lors de cette tournée
                                                       L_customers = []  # c'est une liste informative pour savoir le chemin qu'a pris le camion sans avoir à reparcourir chaque ligne pour comprendre quel client correspondait à quelle ligne

                                                       N = df.shape[0]

                                                       current_t = 481
                                                       v = 50  # m/s
                                                       for i in range(1, N):
                                                       if (i in all_treated_lines) == False:
                                                       line = df.iloc[i]
                                                       package_weight = line['TOTAL_WEIGHT_KG']

                                                       L_parallel = deepcopy(L_lines)
                                                       L_parallel.append(i)

                                                       m = len(L_lines)

                                                       abs_t = current_t
                                                       t_1to2 = 0

                                                       if m >= 2:
                                                       line_1 = df.iloc[L_lines[m - 1]]
                                                       line_2 = df.iloc[L_lines[m - 2]]
                                                       lat_1, lon_1 = line_1['CUSTOMER_LATITUDE'], line_1['CUSTOMER_LONGITUDE']
                                                       lat_2, lon_2 = line_2['CUSTOMER_LATITUDE'], line_2['CUSTOMER_LONGITUDE']
                                                       dist = dt.distance(lat_1, lon_1, lat_2, lon_2)
                                                       t_1to2 = (dist / v) / 60

                                                       abs_t = current_t + t_1to2

                                                       t_min, t_max = line['CUSTOMER_TIME_WINDOW_FROM_MIN'], line['CUSTOMER_TIME_WINDOW_TO_MIN']
                                                       if t_min == 780:
                                                       t_min = 480

                                                       if weight + package_weight < Q and t_min <= abs_t <= t_max:
                                                       weight += package_weight

                                                       current_t += t_1to2

                                                       L_lines.append(i)

                                                       for i in range(len(L_lines)):
                                                       all_treated_lines.append(L_lines[i])

                                                       L_parcours = [0]
                                                       for i in range(len(L_lines)):
                                                       L_parcours.append(L_lines[i])
                                                       L_parcours.append(0)

                                                       return (all_treated_lines, L_parcours)


    def initialisation(self):
                             """
                             Cette fonction crée une solution initiale non aléatoire, non optimisée, mais valable
                             Elle prend en compte:
                             le temps dans lequel le client veut être livré
                             le poids maximal dans le camion
                             Parameters
                             -------
                             -------
                             Returns:
                             -------
                             SOLUTION : [[0,1,2,3,5,6,9,..],[0,45,48,45,...],...] chaque sous liste correspond à un camion
                             -------
                             """
                             df = pd.read_excel(os.getcwd() + '\\data\\table_2_customers_features.xls')

                             SOLUTION = []

                             all_treated_lines = []

                             N = df.shape[0]

                             nb_camions = 0
                             while len(all_treated_lines) < N - 1:
                             nb_camions += 1

                             all_treated_lines, L_parcours = livraison_un_camion(df, all_treated_lines)

                             SOLUTION.append(L_parcours)

                             all_treated_lines = list(set(all_treated_lines))  # On supprime les doublons de la liste des clients livrés
                             # print(SOLUTION)
                             return (SOLUTION)


    def fitness(self, df, S):
                           """
                           Prend une solution en argument et donne son score fitness (propre au tabou, séparé du score fitness du SMA)
                           PARAMETERS
                           ------
                           df:dataframe de travail
                           S: Solution
                           ------
                           RETURNS
                           ------
                           fitness_score: plus il est haut plus la solution est attractive
                           ------
                           """
                           fitness_score = 0
                           for i in range(len(S)):
                           for j in range(len(S[i]) - 1):
                           line_1 = df.iloc[S[i][j]]
                           line_2 = df.iloc[S[i][j + 1]]
                           lat_1, lon_1 = line_1['CUSTOMER_LATITUDE'], line_1['CUSTOMER_LONGITUDE']
                           lat_2, lon_2 = line_2['CUSTOMER_LATITUDE'], line_2['CUSTOMER_LONGITUDE']
                           dist = dt.distance(lat_1, lon_1, lat_2, lon_2)
                           fitness_score += dist
                           fitness_score = fitness_score / 4600000
                           fitness_score += (fitness_score / 100) * len(S)  # prise en compte relative du nombre de camions
                           return (1 / fitness_score)


    def total_time(self, df, L_customers):
                                        """
                                        Fonction utilisée pour créer la solution initiale: permet le calcul du temps déjà traversé par un camion
                                        """
                                        current_time = 481
                                        N = len(L_customers)
                                        v = 50  # m/s
                                        if N >= 2:
                                        for i in range(N - 1):
                                        line_1 = df.iloc[i]
                                        line_2 = df.iloc[i + 1]
                                        lat_1, lon_1 = line_1['CUSTOMER_LATITUDE'], line_1['CUSTOMER_LONGITUDE']
                                        lat_2, lon_2 = line_2['CUSTOMER_LATITUDE'], line_2['CUSTOMER_LONGITUDE']
                                        dist = dt.distance(lat_1, lon_1, lat_2, lon_2)
                                        t_1to2 = (dist / v) / 60
                                        current_time += t_1to2
                                        return (current_time)
