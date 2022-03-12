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

Q=100
adj_matrix = np.random.randint(1, 20, size=(25, 25))
np.fill_diagonal(adj_matrix,np.zeros(len(adj_matrix)))
def create_G(adj_matrix,Q):
    n_max=int((len(adj_matrix)-1)/5) #Le nombre maximum de voitures qu'on mettrai à disposition. 
    m=len(adj_matrix)-1 #Car il y a le 0
    assert(m>=12),"Matrice trop petite, on souhaite au moins 12 sommets afin de répartir des temps toute les 2h sur une journée"
    ###Répartition de temps cohérents###
    n=m//11
    
    dict_g={}
    dict_g[0]={'demande':0,'intervalle':0}
    heure_debut=0
    for i in range(1,len(adj_matrix)):
        if heure_debut>22:
            a=rd.randint(0,20)
            b=rd.randint(a+4,min(a+6,24))
            dict_g[i]={'demande':rd.randint(0,int(Q*n_max/(len(adj_matrix)-1))),'intervalle':[a,b].sort()}
        print(heure_debut)
        print("i:",i)
        dict_g[i]={'demande':rd.randint(1,int(Q*n_max/(len(adj_matrix)-1))),'intervalle':[heure_debut,rd.randint(min(heure_debut+4,24),min(heure_debut+6,24))]}
        if i%n==0: #i toujours différent de 0 , donc le cas i = 0 ne se pose pas
            heure_debut+=2 #On rajoute deux heures dès que i est un multiple de n.
    G = nx.from_numpy_matrix(adj_matrix)
    for i in range(0,len(G.nodes)):
        G.nodes[i].update(dict_g[i])
    labels = {n: n for n in G.nodes}
    colors = [G.nodes[i]['demande'] for i in G.nodes]
    nx.draw(G, with_labels=True, labels=labels, node_color=colors)
    return G,n_max
G,n_max=create_G(adj_matrix,Q)
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
def init(G,n,Q):
    
    """
    Graphs supposés pleins
    
    """
    
    x=[]
    nodes=[i for i in G.nodes]
    nodes.pop(0)
    num_nodes=len(nodes)
    if( n>num_nodes):
        print("Le nombre de voitures doit être inférieur ou égal au nombre de noeuds !")
        return False
    ###On vérifie si les intervalles permettent au moins de créer une solution###
    Ordre=[]
    df=pd.DataFrame()
    df['nodes']=[i for i in range(1,len(G.nodes))]
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(1,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(1,len(G.nodes))]
    df=df.sort_values(by='min')
    if(df['min'].iloc[-1]>len(G.nodes)-1+n):
        print("Les intervalles de temps sont en dehors du nombre d'itération possible")
        return False
    for i in range(0,len(df)-1):
        if (df['min'].iloc[i+1]>df['max'].iloc[i]+2):
            print(df['min'].iloc[i+1],df['max'].iloc[i]+1)
            print("Les intervalles de temps sont trop espacés, on ne considère pas de trou entre les intervalles")
            return False
    Ordre=list(df['nodes']) #Ordre dans lequel les noeuds seront ajoutés.    
    #La sommes des ressources <nQ:#
    sum_ressources=sum([G.nodes[i]['demande'] for i in range(0,len(G.nodes))])
    if sum_ressources>n*Q: 
        print("Les ressources demandées par les villes sont trop importantes")
        return False
    #temps trop petit.#   
    
    ###On remplit le graphe###
    for i in range(0,n):
        ressource=Q
        route=[0]
        nodes_to_add=Ordre[0]
        q_nodes=G.nodes[nodes_to_add]['demande']
        route.append(nodes_to_add)
        ressource=ressource-q_nodes
        Ordre.pop(0)
        nodes_left=len(Ordre)
        while (ressource>=q_nodes and nodes_left>n-i):
            nodes_to_add=Ordre[0]
            q_nodes=G.nodes[nodes_to_add]['demande']
            route.append(nodes_to_add)
            ressource=ressource-q_nodes
            Ordre.pop(0)
            nodes_left=len(Ordre)
            print(nodes_left,n-i,ressource,q_nodes)
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
def recuitsimule(t,x,alpha,Q,n_max):
    global T
    global k
    ###Assertions principales###
    max_nodes_ressources=max([G.nodes[i]['demande'] for i in range(0,len(G.nodes))])
    assert(max_nodes_ressources<Q), "Les ressouces de certaines villes sont plus grandes que les ressources des voitures !"
    for (x,y) in np.ndindex(len(G)):
        if x!=y:
            assert(G.get_edge_data(x,y)['weight'] !=0) , "Certaines routes sont nulles , le graphe n'est pas plein , il faut revoir la matrice d'adjacence !"
    T_old=T

    for i in range(0,n_max):
        T=T_old
        x=init(G,i)
        if x!=False:
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