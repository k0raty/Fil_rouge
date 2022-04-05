import pandas as pd


def check_temps(x, G):
    """
    Fonction de vérification de contrainte conçernant les intervalles de temps.
        -On considère une itération de temps à chaque trajet , peu importe sa longueur.
        -Chaque camion part au même moment.
    Parameters
    ----------
    x : Solution à évaluer
    G : Graph du problème

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
            # assert(temps<G.nodes[0]['CUSTOMER_TIME_WINDOW_TO_MIN']) #Il faut que les camion retournent chez eux à l'heure
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
                    # print(df_temps)
                    return False
                temps += G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10
    return True


def check_temps_2(x, G):
    """
    Fonction de vérification de contrainte conçernant les intervalles de temps.
        -On considère une itération de temps à chaque trajet , peu importe sa longueur.
        -Chaque camion part au même moment.
    Parameters
    ----------
    x : Solution à évaluer
    G : Graph du problème

    Returns
    -------
    bool
        Si oui ou non, les intervalles de temps sont bien respectés dans le routage crée.
        Le camion peut marquer des pauses.

    """

    df_temps = pd.DataFrame(columns=['temps', 'route', 'temps_de_parcours', 'limite_inf', 'limite_sup'])
    temps = G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN']  # Temps d'ouverture du dépot
    for i in range(1, len(x) - 1):
        # assert(temps<G.nodes[0]['CUSTOMER_TIME_WINDOW_TO_MIN']) #Il faut que les camion retournent chez eux à l'heure
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
                # print("Pendant l'initialisation \n",df_temps)
                return False

            temps += G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"] / 10
    return True


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


def check_constraint(x, G):
    Q = [G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0, len(x))]
    if (check_temps(x, G) == True):
        for i in range(0, len(x)):
            if (check_ressource(x[i], Q[i], G) != True):
                return False
        else:
            return True
    else:
        return False
