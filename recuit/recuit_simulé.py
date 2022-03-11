# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import copy
import numpy as np
import random as rd 
import networkx as nx
import pandas as pd 

adj_matrix = np.array([[0,3,10,5],[1,0,4,2],[5,2,0,7],[4,3,10,0]])
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
def init(G,n):
    """
    Graphs supposés pleins
    """
    
    x=[]
    nodes=[i for i in G.nodes]
    nodes.pop(0)
    num_nodes=len(nodes)
    assert n<=num_nodes, "Le nombre de voitures doit être inférieur au nombre de noeuds !"
    ###On vérifie si les intervalles permettent au moins de créer une solution###
    Ordre=[]
    df=pd.DataFrame()
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(0,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(0,len(G.nodes))]
    df=df.sort_values(by='min')
    assert(df['min'].iloc[-1]<len(G.nodes)-1+n),"Les intervalles de temps sont en dehors du nombre d'itération possible"
    for i in range(0,len(df)-1):
        assert (df['min'].iloc[i+1]<=df['max'].iloc[i]+1),"Les intervalles de temps sont trop espacés, on ne considère pas de trou entre les intervalles"
    Ordre=list(df.index)
    #pas de ressource > Q
    num_edges=[]
    nodes_left=len(G.nodes)-1
    for i in range(0,n):
        ressource=Q
        route=[0]
        while(nodes_left>n-i):
            nodes_to_add=Ordre[0]
            q_nodes=G.nodes[nodes_to_add]['demande']
            while ressource>q_nodes:
                route.append(nodes_to_add)
                ressource=ressource-q_nodes
                
    i= round(num_nodes/n)
    h=i
    while h<num_nodes:
        num_edges.append(i)
        h+=i
    num_edges.append(num_nodes-h+i)
    for i in range(0,n):
        route=[0]
        route+=nodes[:num_edges[i]]
        nodes=nodes[num_edges[i]:]
        route.append(0)
        x.append(route)
    return x
def perturbation(x,Q):
    """
    Q: Les ressources de chaque voiture
    """
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
def recuitsimule(t,x,alpha,Q,N):
    global T
    global k
    T_old=T
    for i in range(0,N):
        T=T_old
        x=init(G,i)
        while T >=t-10:
            new_x=perturbation(x)
            E=energie(new_x)
            E0=energie(x)
            if E<E0: 
                x = copy.deepcopy(new_x)
                #plot(x) +Temp
            else:
                p=np.exp(-(E-E0)/(k*T))
                r=rd.random() #ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier bordel !
                if r<= p:
                    x=copy.deepcopy(new_x)
                    #plot(x) +Temp
            T=temperature(T,alpha) 