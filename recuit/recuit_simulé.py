# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import copy
import numpy as np
import random as rd 
import networkx as nx

adj_matrix = np.array([[0,3,10],[1,0,4],[5,2,0]])
dict_g={}
for i in range(0,len(adj_matrix)):
    dict_g[i]={'demande':rd.randint(0,10),'intervalle':sorted([rd.randint(0,100),rd.randint(0,100)])}
G = nx.from_numpy_matrix(adj_matrix)
for i in range(0,len(G.nodes)):
    G.nodes[i].update(dict_g[i])
labels = {n: n for n in G.nodes}
colors = [G.nodes[i]['demande'] for i in G.nodes]
nx.draw(G, with_labels=True, labels=labels, node_color=colors)

def temperature(T,alpha):
    return (1-alpha)*T 
def check_temps(x):
    K=len(x)
    temps=0
    for route in range(0,K):
        for nodes in x[route]:
            if  (temps<G.nodes[nodes]['intervalle'][0] or temps>G.nodes[nodes]['intervalle'][1]):
                return False
            temps+=1
    return True
def check_ressource(L,Q):
    ressource=Q
    for nodes in L:
        ressource=ressource-G.nodes[nodes]['demande']
        if  ressource<0:
            return False
    return True
def perturbation_2(x,Q):
    K=len(x)
    ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
    ##On ajoute un sommet à une route , on en retire à une autre##
    if ajout != retire :
        ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
        nouveau_sommet=rd.randint(0,len(x[retire])-1)
        nouvelle_place=rd.randint(0,len(x[ajout])-1)
        x[ajout].insert(x[retire][nouveau_sommet],nouvelle_place)
        x[retire].pop(nouveau_sommet)

        while (check_ressource(x[ajout],Q)== False or check_temps(x)==False):
            ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
            nouveau_sommet=rd.randint(0,len(x[retire])-1)
            nouvelle_place=rd.randint(0,len(x[ajout])-1)
            x[ajout].insert(x[retire][nouveau_sommet],nouvelle_place)
    ##On permute pour chaque route , des sommets##
    for route in range(0,K):
        permutation=[rd.randint(0,len(x[route])-1),rd.randint(0,len(x[route])-1)]

        if(permutation[0] != permutation[1]):
            x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]

            while(check_temps(x)==False):
                permutation=[rd.randint(0,len(x[route])-1),rd.randint(0,len(x[route])-1)]
                x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]
def energie(x,w):
    K= len(x)
    somme=w*K
    for route in range(0,K):
        for sommet in range(0,len(x[route])):
            somme+=G[x[route][sommet]][x[route][sommet+1]]
        somme+=G[x[route][-1]][0]
    return somme
def recuitsimule(t,x,alpha,Q):
    global T
    global chemin
    global k
    global x
    while T >=t-10:
        new_x=perturbation(x)
        E=energie(new_x)
        E0=energie(x)
        if E<E0: 
            x = copy.deepcopy(new_x)
        else:
            p=np.exp(-(E-E0)/(k*T))
            r=rd.random() #ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier bordel !
            if r<= p:
                x=copy.deepcopy(new_x)
        T=temperature(T,x) 