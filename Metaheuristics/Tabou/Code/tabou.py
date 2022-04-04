""" Import librairies """
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

""" Import utilities """
from Utility.common import distance, fitness


class Tabou:
    def __init__(self, customers, depots, vehicles, cost_matrix):
        self.solution = None

        self.COST_MATRIX = cost_matrix

        self.CUSTOMERS = customers
        self.DEPOTS = depots
        self.VEHICLES = vehicles

        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_CUSTOMERS = len(customers)

    def one_vehicle_delivery(self, initial_solution=None):
        vehicle_weight = float(self.VEHICLES.VEHICLE_TOTAL_WEIGHT_KG)
        vehicle_volume = float(self.VEHICLES.VEHICLE_TOTAL_VOLUME_M3)

        total_weight = 0
        total_volume = 0

        # road starts from depot
        solution = [self.DEPOTS]

        # select first customer
        index_neighbor = self.find_closest_neighbors(solution)
        customer = self.CUSTOMERS[index_neighbor]

        for iteration in range(self.NBR_OF_CUSTOMERS):
            index_neighbor = self.find_closest_neighbors(solution)
            customer = self.CUSTOMERS[index_neighbor]
            total_weight += customer.TOTAL_WEIGHT_KG

            # if vehicle is full it goes back to the depot
            if total_weight >= vehicle_weight:
                solution.append(self.DEPOTS)

        # road ends with return to depot
        solution.append(self.DEPOTS)

        return solution, fitness(solution)

    """
    Find the closest customer to the current position, which has not been already visited
    """

    def find_closest_neighbor(self, solution: list) -> int:
        dist_min = np.inf
        index_neighbor = 0
        current_customer = solution[-1]

        for index_customer in range(self.NBR_OF_CUSTOMERS):
            customer = self.CUSTOMERS[index_customer]

            if customer not in solution:
                dist = distance(
                    current_customer.CUSTOMER_LATITUDE,
                    current_customer.CUSTOMER_LONGITUDE,
                    customer.CUSTOMER_LATITUDE,
                    customer.CUSTOMER_LONGITUDE,
                )

                if dist <= dist_min:
                    dist_min = dist
                    index_neighbor = index_customer

        return index_neighbor

    def livraison_plusieurs_camions(self):
        df = pd.read_excel(os.getcwd() + '\\Dataset\\2_detail_table_customers.xls')
        # df=df.drop(df.loc[df['TOTAL_WEIGHT_KG']>=2000].index)

        all_treated_lines = []
        camions_positions = [[] for k in range(8)]  # la liste contient les positions par lesquelles passe le camion
        camions_clients = [[] for k in range(8)]
        camions_lines = [[] for k in range(8)]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'b']
        n = 0
        while len(all_treated_lines) < df.shape[0] - 6:  # df.shape[0]-6
            n += 1
            for i in range(8):
                # print('Itération',n,'camion',i,'.\nIl y a',len(all_treated_lines),len(list(set(all_treated_lines))),'commandes servies.')
                all_treated_lines, L_customers, L_lines, positions = livraison_un_camion(df, all_treated_lines)
                # print(positions)
                camions_positions[i] += positions
                for k in range(len(L_lines)):
                    camions_lines[i].append(L_lines[k])
                camions_clients[i] += L_customers
                if len(all_treated_lines) > df.shape[0] - 6:  # df.shape[0]-6
                    break
            # print(camions_positions)

        for i in range(8):
            p = 0
            for j in range(len(camions_positions[i]) - 1):
                plt.scatter(43.37391833, 17.60171712, marker='o', c='r', linewidths=2)
                plt.text(43.37391833 + 0.001, 17.60171712 + 0.001, 'Depôt', color='w', backgroundcolor='r', fontsize=9)
                (xj, yj) = camions_positions[i][j]
                (xjj, yjj) = camions_positions[i][j + 1]
                plt.plot([xj, xjj], [yj, yjj], linewidth='0.3', color=colors[i])
                if (xj, yj) != (43.37391833, 17.60171712):
                    p += 1
                    plt.scatter(xj, yj, marker='+', c=colors[i], linewidths=1)
                    # plt.text(xj,yj, str(p), fontsize=7)
            """        
            print('Trajectoires du camion '+str(i+1)+' en couleur '+colors[i]+'.')
            print(camions_positions[i])
            print('Il passe par '+str(len(camions_clients[i]))+' clients.\n')
            print(camions_clients[i])
            """
        # camions[i]
        plt.xlabel('Lattitude')
        plt.ylabel('Longitude')
        plt.title('Itinéraires des camions pour la livraison des clients')
        plt.show()
        return (camions_clients, camions_lines)

    def livraison_voisine(self, camions_lines, k, p, i, j):
        new_line_k = camions_lines[k][:i]
        new_line_k += camions_lines[p][j:]
        new_line_p = camions_lines[p][:j]
        new_line_p += camions_lines[k][i:]

        camions_lines[k] = new_line_k
        camions_lines[p] = new_line_p

        # print(camions_lines[k],'\n\n',camions_lines[p],'\n\n',new_line_k,'\n\n',new_line_p)

        return (camions_lines)

    def calcul_positions(self, camions_lines):
        camions_unseparated_lines = [[] for k in range(8)]
        for k in range(8):
            for j in range(len(camions_lines[k])):
                for i in range(len(camions_lines[k][j])):
                    camions_unseparated_lines[k].append(camions_lines[k][j][i])

        camions_positions = [[] for k in range(8)]

        depot_table = np.array([[5000, 43.37391833, 17.60171712]])
        depot_df = pd.DataFrame(depot_table, index=['DEPOT DATAFRAME'],
                                columns=['CUSTOMER_CODE', 'CUSTOMER_LATITUDE', 'CUSTOMER_LONGITUDE'])
        depot_line = depot_df.iloc[0]

        depot_lat = depot_line['CUSTOMER_LATITUDE']
        depot_lon = depot_line['CUSTOMER_LONGITUDE']
        for k in range(8):
            j = 0
            Q = 5000
            weight = 0
            while j < len(camions_unseparated_lines[k]):
                if weight == 0:
                    camions_positions[k].append((depot_lat, depot_lon))
                    weight += 0.000000000000001
                else:
                    line = df.iloc[camions_unseparated_lines[k][
                        j]]  # on regarde chaque ligne et si il nous reste assez de poids pour livrer la commande correspondante
                    package_weight = line['TOTAL_WEIGHT_KG']
                    if weight + package_weight <= Q:
                        weight += package_weight
                        lat = line['CUSTOMER_LATITUDE']
                        lon = line['CUSTOMER_LONGITUDE']
                        camions_positions[k].append((lat, lon))
                        j += 1
                    else:
                        weight = 0
                        camions_positions[k].append((depot_lat, depot_lon))
        return (camions_positions)

    def cout_livraison(self, camions_lines):
        camions_positions = calcul_positions(df, camions_lines)
        D = 0

        K = 8
        w = 0

        for i in range(8):
            dist_camion_i = 0
            for j in range(len(camions_positions[i]) - 1):
                (lat1, lon1) = camions_positions[i][j]
                (lat2, lon2) = camions_positions[i][j + 1]
                dist_camion_i += dt.distance(lat1, lon1, lat2, lon2)
            print('Camion ' + str(i) + ':', dist_camion_i)
            D += dist_camion_i
        return (w * K + round(D))

    def main(self, camions_lines):
        solutions_voisines_k = np.zeros((1, self.NBR_OF_VEHICLES))
        cout_solutions = np.zeros((1, self.NBR_OF_VEHICLES))

        cout_solutions[0] = self.cout_livraison(camions_lines)

        solutions_voisines_k[0] = camions_lines

        for p in range(1, self.NBR_OF_VEHICLES):
            nb_clients_camion_k = len(camions_lines[0])
            nb_clients_camion_p = len(camions_lines[p])

            m = min(nb_clients_camion_k, nb_clients_camion_p)

            n = m // 2
            i = rd.randint(n, 2 * n - 1)
            j = rd.randint(i, 2 * n - 1)

            solution_p_lines = self.livraison_voisine(camions_lines, 0, p, i, j)

            solutions_voisines_k[p] = solution_p_lines

            cout_solutions[p] = self.cout_livraison(solution_p_lines)

        indice_min = np.argmin(cout_solutions)

        return solutions_voisines_k[indice_min]


k_dt = 1.1515247508025006

# condition sur les plages horaires
# df[(df['CUSTOMER_TIME_WINDOW_TO_MIN']>=480) & (df['CUSTOMER_TIME_WINDOW_TO_MIN']<=840)]
