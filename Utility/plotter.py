# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:04:21 2022

@author: Floriane
"""
import networkx as nx
import matplotlib.pyplot as plt
import utm


class Graphe:
    def __init__(self, df):
        df['pos']=[utm.from_latlon(df['CUSTOMER_LATITUDE'].iloc[i],df['CUSTOMER_LONGITUDE'].iloc[i])[:2] for i in range(0,len(df))]
        self.df = df 
    
    def create_dico_position(self):
        """crée le dictionnaire des positions de chacun des points"""
        #dico_position = {'0':(43.37391833, 17.60171712)}
        df = self.df
        dico_position = {'0':(710775.4574066155, 4805626.797484203)}
        nb_lines = df.shape[0]
        for i in range(nb_lines):
            dico_position[str(df["CUSTOMER_CODE"][i])] = df["pos"][i]
        self.dico_position = dico_position
    
    def create_dico_colors(self):
        """crée la liste des couleurs des noeuds pour l'affichage graphique"""
        df = self.df
        nb_lines = df.shape[0]
        dico_demande = {'0': 0}
        for i in range(0,nb_lines):
            dico_demande[str(df["CUSTOMER_CODE"][i])] = df["TOTAL_WEIGHT_KG"][i]
        self.dico_demande = dico_demande

    def plot_graph(self, x, spring_layout = True):
        graph_route = nx.DiGraph() #création du nouveau graphe des routes
        colors = [0]
        liste_temp = [0]
        demande = self.dico_demande
        position = self.dico_position
        for i in range(0,len(x)):
            route = x[i]
            for j in range(0, len(route)-1):
                graph_route.add_edges_from([(str(route[j]), str(route[j+1]))]) #pour chacune des routes, on ajoute les sommets 
                if route[j] not in liste_temp:
                    colors.append(demande[str(route[j])])
                    liste_temp.append(route[j])
        if spring_layout:
            new_position = nx.spring_layout(graph_route, pos = position)
            nx.draw_networkx(graph_route,arrows=True, with_labels=True, node_color =colors, arrowstyle = '-|>', arrowsize = 12, pos = new_position)
            graph_label = "Solution finale, version éclatée"
        else:
            nx.draw_networkx(graph_route,arrows=True, with_labels=True, node_color =colors, arrowstyle = '-|>', arrowsize = 12, pos = position)
            graph_label = "Solution finale, avec les vraies coordonnées"
        plt.title(graph_label)
        plt.show()
        plt.clf()


"""
##### TEST DE LA CLASSE #####
x = [[0,138087,138157,26,478,0], [0,921127,15076,0],[0,1408,0], [0,922636,141727,15027,0], [0,137381,1998,0]]
df = pd.read_excel("test.xlsx")
new_graphe = Graphe(df)
position = new_graphe.create_dico_position()
demande = new_graphe.create_dico_colors()
new_graphe.plot_graph(x)
##### FIN TEST #####
"""