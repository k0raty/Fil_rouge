# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:20:12 2022

@author: anton
"""
import utm
import random
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import networkx as nx
import random as rd
import sys

df_customers = pd.read_excel("table_2_customers_features.xls")
df_vehicles = pd.read_excel("table_3_cars_features.xls")
df_vehicles = df_vehicles.drop(['Unnamed: 0'], axis=1)
df_customers = df_customers.drop(['Unnamed: 0'], axis=1)


def get_distance(z_1, z_2):
    x_1, x_2, y_1, y_2 = z_1[0], z_2[0], z_1[1], z_2[1]
    d = math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    return d / 1000  # en km


def create_G(df_customers, df_vehicles, v):
    """

    Parameters
    ----------
    df_customers : Dataframe contenant les informations sur les clients
    df_vehicles : Dataframe contenant les informations sur les camions livreurs

    Returns
    -------
    G : Graph plein généré
    """
    n_max = len(df_vehicles)  # Le nombre maximum de voitures qu'on mettrai à disposition.
    n_sommet = len(df_customers)
    G = nx.empty_graph(n_sommet)
    (x_0, y_0) = utm.from_latlon(43.37391833, 17.60171712)[:2]
    dict_0 = {'CUSTOMER_CODE': 0, 'CUSTOMER_LATITUDE': 43.37391833, 'CUSTOMER_LONGITUDE': 17.60171712,
              'CUSTOMER_TIME_WINDOW_FROM_MIN': 360, 'CUSTOMER_TIME_WINDOW_TO_MIN': 1080, 'TOTAL_WEIGHT_KG': 0,
              'pos': (x_0, y_0), "CUSTOMER_DELIVERY_SERVICE_TIME_MIN": 0}
    G.nodes[0].update(dict_0)
    G.nodes[0]['n_max'] = n_max  # Nombre de voiture maximal
    dict_vehicles = df_vehicles.to_dict()
    G.nodes[0]['Camion'] = dict_vehicles
    for i in range(0, len(G.nodes)):
        dict = df_customers.iloc[i].to_dict()
        dict['pos'] = utm.from_latlon(dict['CUSTOMER_LATITUDE'], dict['CUSTOMER_LONGITUDE'])[:2]
        G.nodes[i].update(dict)

        ###On rajoute les routes###
    for i in range(0, len(G.nodes)):
        for j in range(0, len(G.nodes)):
            if i != j:
                z_1 = G.nodes[i]['pos']
                z_2 = G.nodes[j]['pos']
                G.add_edge(i, j, weight=get_distance(z_1, z_2))
                G[i][j]['time'] = (G[i][j]['weight'] / v) * 60
    ###On colorie les routes et noeuds###

    colors = [0]
    colors += [G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(1, len(G.nodes))]
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_color=colors)
    G.nodes[0]['n_max'] = n_max  # Nombre de voiture maximal
    p = [2, -100, n_sommet]  # Equation pour trouver n_min
    roots = np.roots(p)
    n_min = max(1,
                int(roots.min()) + 1)  # Nombre de voiture minimal possible , solution d'une équation de second degrès.
    G.nodes[0]['n_min'] = n_min
    plt.title("graphe initial")
    plt.show()
    plt.clf()
    return G


def longueur(x, y, ordre):
    i = ordre[-1]
    x0, y0 = x[i], y[i]
    d = 0
    for o in ordre:
        x1, y1 = x[o], y[o]
        d += (x0 - x1) ** 2 + (y0 - y1) ** 2
        x0, y0 = x1, y1
    return d / 1000


def permutation(x, y, ordre):
    d = longueur(x, y, ordre)
    d0 = d + 1
    it = 1
    while d < d0:
        if ordre[0] != 0:
            sys.exit()
        it += 1
        print("iteration", it, "d=", d)
        d0 = d
        for i in tqdm(range(1, len(ordre) - 1)):
            for j in range(i + 2, len(ordre)):
                r = ordre[i:j].copy()
                r.reverse()
                ordre2 = ordre[:i] + r + ordre[j:]
                t = longueur(x, y, ordre2)
                if t < d:
                    d = t
                    ordre = ordre2

        plt.clf()
        xo = [x[o] for o in ordre + [ordre[0]]]
        yo = [y[o] for o in ordre + [ordre[0]]]
        plt.plot(xo, yo, "o-")
        plt.plot(xo[0], y[0], "r")
        plt.show()

    return ordre


def main(G):
    x = [G.nodes[i]['pos'][0] for i in range(0, len(G))]
    y = [G.nodes[i]['pos'][1] for i in range(0, len(G))]
    ordre = list(range(len(x)))
    ordre = permutation(x, y, ordre)
    print("longueur initiale", longueur(x, y, ordre))
    plt.plot(x, y, "o")
    print("longueur min", longueur(x, y, ordre))
    xo = [x[o] for o in ordre + [ordre[0]]]
    yo = [y[o] for o in ordre + [ordre[0]]]
    plt.plot(xo, yo, "o-")
    plt.text(xo[0], yo[0], "0", color="r", weight="bold", size="x-large")
    plt.text(xo[-2], yo[-2], "N-1", color="r", weight="bold", size="x-large")

    return ordre
# G=create_G(df_customers,df_vehicles,v) #En jaune le centre.
# df_ordre_init=pd.DataFrame(ordre,columns=['Ordre'])
# df_ordre_init.to_pickle("df_ordre_init.pkl")
# ordre=main(G)
