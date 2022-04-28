""" Import librairies """
import networkx as nx
import matplotlib.pyplot as plt
import utm
from seaborn import color_palette


class Plotter:
    def __init__(self, graph, df=None):
        self.Graph = graph

        if df is not None:
            df['pos'] = [utm.from_latlon(df['CUSTOMER_LATITUDE'].iloc[i], df['CUSTOMER_LONGITUDE'].iloc[i])[:2] for i in
                         range(0, len(df))]
            self.df = df

    def create_dico_position(self):
        """crée le dictionnaire des positions de chacun des points"""
        # dico_position = {'0':(43.37391833, 17.60171712)}
        df = self.df
        dico_position = {'0': (710775.4574066155, 4805626.797484203)}
        nb_lines = df.shape[0]
        for i in range(nb_lines):
            dico_position[str(df["CUSTOMER_CODE"][i])] = df["pos"][i]
        self.dico_position = dico_position

    def create_dico_colors(self):
        """crée la liste des couleurs des noeuds pour l'affichage graphique"""
        df = self.df
        nb_lines = df.shape[0]
        dico_demande = {'0': 0}
        for i in range(0, nb_lines):
            dico_demande[str(df["CUSTOMER_CODE"][i])] = df["TOTAL_WEIGHT_KG"][i]
        self.dico_demande = dico_demande

    def plot_graph(self, x, spring_layout=True):
        graph_route = nx.DiGraph()  # création du nouveau graphe des routes
        colors = [0]
        liste_temp = [0]
        demande = self.dico_demande
        position = self.dico_position
        for i in range(0, len(x)):
            route = x[i]
            for j in range(0, len(route) - 1):
                graph_route.add_edges_from(
                    [(str(route[j]), str(route[j + 1]))])  # pour chacune des routes, on ajoute les sommets
                if route[j] not in liste_temp:
                    colors.append(demande[str(route[j])])
                    liste_temp.append(route[j])
        if spring_layout:
            new_position = nx.spring_layout(graph_route, pos=position)
            nx.draw_networkx(graph_route, arrows=True, with_labels=True, node_color=colors, arrowstyle='-|>',
                             arrowsize=12, pos=new_position)
            graph_label = "Solution finale, version éclatée"
        else:
            nx.draw_networkx(graph_route, arrows=True, with_labels=True, node_color=colors, arrowstyle='-|>',
                             arrowsize=12, pos=position)
            graph_label = "Solution finale, avec les vraies coordonnées"
        plt.title(graph_label)
        plt.show()
        plt.clf()


"""
Draw the Graph showing all the customers summits and the depots, with the road taken to go through them

Parameters
----------
solution: list - an individual of the population with the lowest fitness score
filepath: string - a path where to save the plotted image
----------
"""


def plot_solution(solution, graph, title='Solution to the VRP', filepath=None):
    colors = color_palette(n_colors=len(solution))

    for index_delivery in range(len(solution)):
        delivery = solution[index_delivery]

        list_x = []
        list_y = []

        for index in range(len(delivery)):
            index_customer = delivery[index]
            customer = graph.nodes[index_customer]

            x, y = utm.from_latlon(customer['CUSTOMER_LATITUDE'], customer['CUSTOMER_LONGITUDE'])[:2]
            list_x.append(x)
            list_y.append(y)

        label = graph.nodes[0]['Vehicles']["VEHICLE_VARIABLE_COST_KM"][index_delivery]

        plt.plot(list_x, list_y, label=label, marker='o', markerfacecolor='blue',
                 markeredgecolor='blue', markersize=0.8, linestyle='solid', linewidth=0.7,
                 color=colors[index_delivery],
                 )

    depot = graph.nodes[0]
    x, y = utm.from_latlon(depot['CUSTOMER_LATITUDE'], depot['CUSTOMER_LONGITUDE'])[:2]
    plt.plot(x, y, 'rs', markersize=0.9)

    plt.xlabel('coordinate x (in km)')
    plt.ylabel('coordinate y (in km)')
    plt.title(title)

    plt.legend(loc='upper left')
    plt.show()

    if filepath is not None:
        fig = plt.gcf()
        fig.savefig(filepath, format='png')
