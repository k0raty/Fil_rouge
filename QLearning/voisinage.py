import random as rd
import numpy as np

def IntraRouteSwap(solution):
    """ Echange de deux clients d'une même route de façon aléatoire """

    # On récupère une route au hasard
    nb_road = rd.randrange(0,len(solution),1)
    road = solution[nb_road]

    # On récupère deux clients différents au hasard
    nb_client1 = rd.randrange(1,len(road)-1,1)    # on ne prend pas en compte les dépôts !
    nb_client2 = rd.randrange(1,len(road)-1,1)
    while nb_client1 == nb_client2 :
        nb_client2 = rd.randrange(1,len(road)-1,1)
    
    # On procède à l'échange des clients
    aux = solution[nb_road][nb_client1]
    solution[nb_road][nb_client1] = solution[nb_road][nb_client2]
    solution[nb_road][nb_client2] = aux

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

def RemoveSmallestRoad(solution):
    """ Enlève la route de plus petite taille """
    
    # Cherche la route de plus petite taille
    small_length = 10000
    small_index = 0

    for i in range(len(solution)):
        length = len(solution[i])
        if length < small_length :
            small_index = i
            small_length = length
    
    # Tranfère tous les clients de la plus petite route vers les autres routes
    small_road = solution.pop(small_index)
    small_road.pop(0)     # on enlève les dépôts
    small_road.pop()
    
    while small_road != [] :
        # On récupère une route au hasard
        nb_road = rd.randrange(0,len(solution),1)
        road = solution[nb_road].copy()

        # On trouve un nouvel indice au client déplacé
        new_index = rd.randrange(1,len(road)-1,1)

        # On supprime le client de la liste initiale
        client = small_road.pop(0)
        
        # On procède au déplacement du client
        new_road = []
        i = 0
        while i != new_index :
            new_road.append(road[0])
            road.pop(0)
            i += 1
        new_road.append(client)   # on ajoute le client déplacé
        while road != []:
            new_road.append(road[0])
            road.pop(0)
        solution[nb_road] = new_road
    
    return solution

def RemoveRandomRoad(solution):
    """ Enlève une route au hasard """
    
    # Choisit une route au hasard
    nb_road_to_remove = rd.randrang(0,len(solution),1)
    
    # Tranfère tous les clients de la route à supprimer vers les autres routes
    road_to_remove = solution.pop(nb_road_to_remove)
    road_to_remove.pop(0)    # on enlève les dépôts
    road_to_remove.pop()
    
    while road_to_remove != [] :
        # On récupère une route au hasard
        nb_road = rd.randrange(0,len(solution),1)
        road = solution[nb_road].copy()

        # On trouve un nouvel indice au client déplacé
        new_index = rd.randrange(1,len(road)-1,1)

        # On supprime le client de la liste initiale
        client = road_to_remove.pop(0)
        
        # On procède au déplacement du client
        new_road = []
        i = 0
        while i != new_index :
            new_road.append(road[0])
            road.pop(0)
            i += 1
        new_road.append(client)   # on ajoute le client déplacé
        while road != []:
            new_road.append(road[0])
            road.pop(0)
        solution[nb_road] = new_road
    
    return solution

def action_state(a, solution):                  #Elle est horriblement écrite je sais :)
    if a == 1 :
        solution = IntraRouteSwap(solution)
    elif a == 2 :
        solution = InterRouteSwap(solution)
    elif a == 3 :
        solution = IntraRouteShift(solution)
    elif a == 4 :
        solution = InterRouteShift(solution)
    elif a == 5 :
        solution = TwoIntraRouteSwap(solution)
    elif a == 6 :
        solution = TwoIntraRouteShift(solution)
    elif a == 7 :
        solution = RemoveSmallestRoad(solution)
    elif a == 8 :
        solution = RemoveRandomRoad(solution)
    return solution


#TESTS

"""
solution = [[0,1,2,3,0],[0,4,5,6,0]]
print(solution)
print(IntraRouteSwap(solution))
print(InterRouteSwap(solution))
print(IntraRouteShift(solution))
print(InterRouteShift(solution))
print()
solution1 = [[0,1,2,3,4,5,6,7,0],[0,8,9,10,11,12,13,0]]
print(solution1)
print(TwoIntraRouteSwap(solution1))
print(TwoIntraRouteShift(solution1))
print()
solution2 = [[0,1,2,3,4,0],[0,5,6,7,0],[0,8,9,10,11,0]]
print(solution2)
print(RemoveSmallestRoad(solution2))
print()
solution3 = [[0,1,2,3,4,0],[0,5,6,7,0],[0,8,9,10,11,0]]
print(solution3)
print(RemoveRandomRoad(solution3))
"""