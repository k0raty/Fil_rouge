""" Import librairies """
import random as rd


"""
Switch 2 random customers on the same random delivery

Parameters
----------
solution: list - a solution to the given problem
----------

Returns
-------
solution: list - the modified solution
-------
"""


def intra_route_swap(solution):
    index_delivery = rd.randint(0, len(solution) - 1)
    delivery = solution[index_delivery]

    index_customer_1 = rd.randint(1, len(delivery) - 1)
    index_customer_2 = rd.randint(1, len(delivery) - 1)

    while index_customer_1 == index_customer_2:
        index_customer_2 = rd.randint(1, len(delivery) - 1)
    
    customer_1 = solution[index_delivery][index_customer_1]
    customer_2 = solution[index_delivery][index_customer_2]

    solution[index_delivery][index_customer_1] = customer_2
    solution[index_delivery][index_customer_1] = customer_1

    return solution

def InterRouteSwap(solution):
    """ Echange de deux clients de routes différentes de façon aléatoire """

    # On récupère deux routes au hasard
    nb_road1 = rd.randrange(0,len(solution),1)
    road1 = solution[nb_road1]
    nb_road2 = rd.randrange(0,len(solution),1)
    while nb_road1 == nb_road2 :
        nb_road2 = rd.randrange(0,len(solution),1)
    road2 = solution[nb_road2]

    # On récupère deux clients différents au hasard
    nb_client1 = rd.randrange(1,len(road1)-1,1)  # on ne prend pas en compte les dépôts !
    nb_client2 = rd.randrange(1,len(road2)-1,1)
    
    # On procède à l'échange des clients
    aux = solution[nb_road1][nb_client1]
    solution[nb_road1][nb_client1] = solution[nb_road2][nb_client2]
    solution[nb_road2][nb_client2] = aux

    return solution

def IntraRouteShift(solution):
    """ Déplacement d'un client vers une autre position sur la même route """
    # On récupère une route au hasard
    nb_road = rd.randrange(0,len(solution),1)
    road = solution[nb_road].copy()

    # On récupère un client au hasard et son nouvel indice
    nb_client = rd.randrange(1,len(road)-1,1)
    new_index = rd.randrange(1,len(road)-1,1)
    while nb_client == new_index :
        new_index = rd.randrange(1,len(road)-1,1)
    
    # On procède au déplacement du client
    new_road = []
    road.pop(nb_client)  # on crée une route auxiliaire sans le client à déplacer
    i = 0
    while i != new_index :
        new_road.append(road[0])
        road.pop(0)
        i += 1
    client = solution[nb_road][nb_client]
    new_road.append(client)   # on ajoute le client déplacé
    while road != []:
        new_road.append(road[0])
        road.pop(0)
    solution[nb_road] = new_road

    return solution

def InterRouteShift(solution):
    """ Déplacement d'un client vers une autre position sur une route différente """

    # On récupère deux routes au hasard
    nb_road1 = rd.randrange(0,len(solution),1)
    road1 = solution[nb_road1]
    nb_road2 = rd.randrange(0,len(solution),1)
    while nb_road1 == nb_road2 :
        nb_road2 = rd.randrange(0,len(solution),1)
    road2 = solution[nb_road2].copy()

    # On récupère un client au hasard et son nouvel indice
    nb_client = rd.randrange(1,len(road1)-1,1)
    new_index = rd.randrange(1,len(road2)-1,1)
    while nb_client == new_index :
        new_index = rd.randrange(1,len(road2)-1,1)

    # On supprime le client de la liste initiale
    client = road1.pop(nb_client)
    solution[nb_road1] = road1
    
    # On procède au déplacement du client
    new_road = []
    i = 0
    while i != new_index :
        new_road.append(road2[0])
        road2.pop(0)
        i += 1
    new_road.append(client)   # on ajoute le client déplacé
    while road2 != []:
        new_road.append(road2[0])
        road2.pop(0)
    solution[nb_road2] = new_road

    return solution

def TwoIntraRouteSwap(solution):
    """ Echange de deux clients consécutifs d'une même route avec deux autres clients consécutifs de façon aléatoire """

    # On récupère une route au hasard
    nb_road = rd.randrange(0,len(solution),1)
    road = solution[nb_road]

    # On récupère quatre clients (deux consécutifs à chaque fois) au hasard
    nb_client1 = rd.randrange(1,len(road)-2,1)    # on ne prend pas en compte les dépôts !
    nb_client2 = nb_client1 + 1
    nb_client3 = rd.randrange(1,len(road)-2,1)
    while abs(nb_client1 - nb_client3) < 2 :
        nb_client3 = rd.randrange(1,len(road)-2,1)
    nb_client4 = nb_client3 + 1
    
    # On procède à l'échange des clients
    aux1 = solution[nb_road][nb_client1]
    solution[nb_road][nb_client1] = solution[nb_road][nb_client3]
    solution[nb_road][nb_client3] = aux1
    aux2 = solution[nb_road][nb_client2]
    solution[nb_road][nb_client2] = solution[nb_road][nb_client4]
    solution[nb_road][nb_client4] = aux2

    return solution

def TwoIntraRouteShift(solution):
    """ Déplacement de deux clients consécutifs vers une autre position sur la même route """
    # On récupère une route au hasard
    nb_road = rd.randrange(0,len(solution),1)
    road = solution[nb_road].copy()

    # On récupère deux clients au hasard et leurs nouveaux indices
    nb_client1 = rd.randrange(1,len(road)-2,1)
    new_index = rd.randrange(1,len(road)-2,1)
    while abs(new_index - nb_client1) < 2 :
        new_index = rd.randrange(1,len(road)-2,1)
    
    # On procède au déplacement des clients
    new_road = []
    road.pop(nb_client1)  # on crée une route auxiliaire sans les clients à déplacer
    road.pop(nb_client1)  # on a déjà décalé d'un indice
    i = 0
    while i != new_index :
        new_road.append(road[0])
        road.pop(0)
        i += 1
    client1 = solution[nb_road][nb_client1]
    new_road.append(client1)   # on ajoute le client déplacé
    client2 = solution[nb_road][nb_client1 + 1]
    new_road.append(client2)   # on ajoute le client déplacé
    while road != []:
        new_road.append(road[0])
        road.pop(0)
    solution[nb_road] = new_road

    return solution


"""
Remove the shortest delivery
"""


def remove_smallest_road(solution):
    if len(solution) == 1:
        return solution

    smallest_length = 10000
    smallest_index = 0

    for index_delivery in range(len(solution)):
        delivery_length = len(solution[index_delivery])

        if delivery_length < smallest_length:
            smallest_index = index_delivery
            smallest_length = delivery_length
    
    smallest_delivery = solution.pop(smallest_index)
    smallest_delivery = smallest_delivery[1: smallest_length - 1]

    while len(smallest_delivery) > 0:
        index_delivery = rd.randint(0, len(solution) - 1)

        new_index_customer = rd.randint(1, len(solution[index_delivery]) - 1)

        customer = smallest_delivery.pop(0)

        solution[index_delivery].insert(new_index_customer, customer)

    return solution


"""
Remove a random delivery
"""


def remove_random_road(solution):
    if len(solution) == 1:
        return solution

    index_delivery_to_remove = rd.randint(0, len(solution) - 1)
    
    delivery_to_remove = solution.pop(index_delivery_to_remove)
    delivery_to_remove = delivery_to_remove[1: len(delivery_to_remove) - 1]
    
    while len(delivery_to_remove) > 0:
        index_delivery = rd.randint(0, len(solution) - 1)

        new_index_customer = rd.randint(1, len(solution[index_delivery]) - 1)

        customer = delivery_to_remove.pop(0)

        solution[index_delivery].insert(new_index_customer, customer)

    return solution


"""
Map the action's index to the modification of the solution to use

Parameters
----------
action: int - the index of the action to perform on the solution
solution: list - a solution to the problem
----------

Returns
-------
solution: list - the modified solution by the chosen action
-------
"""


def perform_action(index_action, solution):
    if index_action == 0:
        return intra_route_swap(solution)

    elif index_action == 1:
        return InterRouteSwap(solution)

    elif index_action == 2:
        return IntraRouteShift(solution)

    elif index_action == 3:
        return InterRouteShift(solution)

    elif index_action == 4:
        return TwoIntraRouteSwap(solution)

    elif index_action == 5:
        return TwoIntraRouteShift(solution)

    elif index_action == 6:
        return remove_smallest_road(solution)

    elif index_action == 7:
        return remove_random_road(solution)

    return solution
