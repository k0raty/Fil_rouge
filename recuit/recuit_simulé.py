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
import sys
Q=100
n=10
alpha=0.1
adj_matrix = np.random.randint(1, 20, size=(100, 100))
np.fill_diagonal(adj_matrix,np.zeros(len(adj_matrix)))

def create_G(adj_matrix,Q):
    
    n_max=int((len(adj_matrix)-1)/5) #Le nombre maximum de voitures qu'on mettrai à disposition. 
    nb_sommet=len(adj_matrix)-1 #Car il y a le 0
    tps_max=2+nb_sommet
    ###Répartition de temps cohérents###
    n=nb_sommet//10
    dict_g={}
    dict_g[0]={'demande':0,'intervalle':0}
    heure_debut=0
    k=1
    for i in range(1,len(adj_matrix)):
        print(heure_debut)
        print("i:",i)
        if heure_debut>80: #Intervalles larges
            a=rd.randint(0,40)
            b=rd.randint(a+1,a+60)
            dict_g[i]={'demande':rd.randint(0,int(0.5*Q*n_max/(len(adj_matrix)-1))),'intervalle':[a,b]}
       
        else:
            print("k is",k)
            dict_g[i]={'demande':rd.randint(1,int(0.5*Q*n_max/(len(adj_matrix)-1))),'intervalle':[heure_debut,heure_debut+20]} #+2 car il faut au moins 2 d'écart entre deux classes de temps
            if i%n==0: #i toujours différent de 0 , donc le cas i = 0 ne se pose pas
                heure_debut+=10 #On rajoute 10 unités dès que i est un multiple de n.
                k+=1
    G = nx.from_numpy_matrix(adj_matrix)
    for i in range(0,len(G.nodes)):
        G.nodes[i].update(dict_g[i])  

    df=pd.DataFrame()
    df['nodes']=[i for i in range(1,len(adj_matrix))]
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(1,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(1,len(G.nodes))]
    df=df.sort_values(by='min')
    #for i in range(0,int(df['min'].max()/5)):    #Il faut qu'il y est assez de ville dans chaque classe de temps pour que une voiture y passe assez longtemps afin de ne pas arriver trop tot à la prochaine classe
       # assert(len(df[df['min']==i*5])>=5)
    labels = {n: n for n in G.nodes}
    colors = [G.nodes[i]['demande'] for i in G.nodes]
    nx.draw(G, with_labels=True, labels=labels, node_color=colors)
    G.nodes[0]['n_max']=n_max #Nombre de voiture maximal
    G.nodes[0]['tps_max']=tps_max
    p=[2,-100,nb_sommet]
    roots=np.roots(p)
    n_min=max(1,int(roots.min())+1)
    G.nodes[0]['n_min']=n_min
    return G
def temperature(T,alpha):
    return (1-alpha)*T 
def check_temps(x,G):
    K=len(x)
    for route in range(0,K):
        temps=0
        for nodes in x[route]:
            if nodes !=0:
                while  temps<G.nodes[nodes]['intervalle'][0]:
                    temps+=1 #Le camion est en pause
                if (temps<G.nodes[nodes]['intervalle'][0] or temps>G.nodes[nodes]['intervalle'][1]):
                    return False
    return True
def check_ressource(L,Q,G):
    ressource=Q
    #print("route est",L)
    for nodes in L:
        print(nodes)
        ressource=ressource-G.nodes[nodes]['demande']
        if  ressource<0:
            return False
    return True
def init(G,n,Q):
    
    """
    Graphs supposés pleins
    
    """
    assert(n>G.nodes[0]['n_min']),"Peut-importe la configuration , il n'y a pas assez de camion pour terminer le trajet dans le temps imparti (<100)"    
    assert(n<=G.nodes[0]['n_max']),"On demande trop de voiture , <= à %s normalement " %G.nodes[0]['n_max']
    x=[]
    for i in range(n):
        x.append([0])
    nodes=[i for i in G.nodes]
    nodes.pop(0)
    
    ###On vérifie si les intervalles permettent au moins de créer une solution###
    Ordre=[]
    df=pd.DataFrame()
    df['nodes']=[i for i in range(1,len(G.nodes))]
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(1,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(1,len(G.nodes))]
    df=df.sort_values(by='min')
    Ordre=list(df['nodes']) #Ordre dans lequel les noeuds seront ajoutés.    
    #La sommes des ressources <nQ:#
    sum_ressources=sum([G.nodes[i]['demande'] for i in range(0,len(G.nodes))])
    if sum_ressources>n*Q: 
        print("Les ressources demandées par les villes sont trop importantes")
        return False
    #temps trop petit.#   
    
    ###On remplit le graphe###
    for i in range(0,len(Ordre)):
        ressources=n*[Q]
        q_nodes=0
        nodes_left=len(Ordre)
        route=i%n
        if (ressources[route]>=q_nodes and nodes_left>=n-i):
            nodes_to_add=Ordre[i]
            q_nodes=G.nodes[nodes_to_add]['demande']
            x[route].append(nodes_to_add)
            ressources[route]=ressources[route]-q_nodes
            nodes_left+=-1
            print(nodes_left,n-i,ressources,q_nodes)
    for i in x:
        i.append(0)
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources"
    return x
    
def perturbation(x,Q,G):
    
    """
    Q: Les ressources de chaque voiture
    
    """
    very_old_x=copy.deepcopy(x) #Utiliser deepcopy ! et non copy
    K=len(x)
    ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
    print("ajout et retire sont ", ajout,retire)
    ##On ajoute un sommet à une route , on en retire à une autre##
    if ajout != retire :
        if len(x[retire])>3:
            print("x fut",x)
            nouveau_sommet=rd.randint(1,len(x[retire])-2) #On évite les 0 dans chaque routes
            nouvelle_place=rd.randint(1,len(x[ajout])-2)
            print("les sommets et places sont ",nouveau_sommet,nouvelle_place)
            x[ajout].insert(nouvelle_place,x[retire][nouveau_sommet])
            x[retire].pop(nouveau_sommet)
            print("x est",x)
            #sys.exit()
            while (check_ressource(x[ajout],Q,G)== False or check_temps(x,G)==False):
                ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
                if ajout !=retire :
                    x=copy.deepcopy(very_old_x)
                    print ("x est" ,x)
                    print("ajout et retire 1 sont", ajout,retire)
                    if len(x[retire])>3:
                        nouveau_sommet=rd.randint(1,len(x[retire])-2)
                        nouvelle_place=rd.randint(1,len(x[ajout])-2)
                        x[ajout].insert(nouvelle_place,x[retire][nouveau_sommet])
                        x[retire].pop(nouveau_sommet)
                else:
                    x=copy.deepcopy(very_old_x)
                    break
        #else:sys.exit()
    #else: sys.exit()        
    ##On permute pour chaque route , des sommets##
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps pendant l'insertion"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources pendant l'insertion"    
    for route in range(0,K):
        assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps pendant la perturbation"
        for i in range(0,len(x)):
            assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources pendant la perturbation"    
        permutation=[rd.randint(1,len(x[route])-2),rd.randint(1,len(x[route])-2)]
        
        if(permutation[0] != permutation[1]): # Si c'est pareil : Passe
            old_x=copy.deepcopy(x)
            print("permutation 0 : ",x,permutation, route)

            x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]
            print(check_temps(x,G))
            while(check_temps(x,G)==False):
                x=copy.deepcopy(old_x)
                permutation=[rd.randint(1,len(x[route])-2),rd.randint(1,len(x[route])-2)]
                print("permutation : \n",x,permutation)
                print("route :\n",route)
                print("les permutations :\n",x[route][permutation[0]],x[route][permutation[1]])

                x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]
                print(check_temps(x,G))
                print("old_x is: ",old_x)
            print("x après permutation",x,x[route][permutation[0]],x[route][permutation[1]])
            print(x==old_x)
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps à la fin de la permutation"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources à la fin de la permutation"    
    num_very_old_x=sum([len(i) for i in very_old_x])
    num_x=sum([len(i) for i in x])
    assert(num_very_old_x==num_x)
    print("x a-il-changé ? :", x!=very_old_x)
    return very_old_x,x
def energie(x,w,G):
    K= len(x)
    somme=w*K
    for route in range(0,K):
        for sommet in range(0,len(x[route])):
            somme+=G[x[route][sommet]][x[route][sommet+1]]
        somme+=G[x[route][-1]][0]
    return somme
def recuitsimule(t,x,alpha,Q,G):
    global T
    global k
    ###Assertions principales###
    max_nodes_ressources=max([G.nodes[i]['demande'] for i in range(0,len(G.nodes))])
    assert(max_nodes_ressources<Q), "Les ressouces de certaines villes sont plus grandes que les ressources des voitures !"
    for (x,y) in np.ndindex(len(G)):
        if x!=y:
            assert(G.get_edge_data(x,y)['weight'] !=0) , "Certaines routes sont nulles , le graphe n'est pas plein , il faut revoir la matrice d'adjacence !"
    T_old=T
    n_min=G.nodes[0]['n_min']
    n_max=G.nodes[0]['n_max']
    for i in range(n_min,n_max+1):
        T=T_old
        x=init(G,i,Q)
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
G=create_G(adj_matrix,Q)
x=init(G,n,Q)
v,x=perturbation(x,Q,G)