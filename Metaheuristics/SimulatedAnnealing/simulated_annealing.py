# -*- coding: utf-8 -*-
"""
Antony davi et Floriane Ronzon.
"""
import copy
import numpy as np
import random as rd 
import networkx as nx
import pandas as pd 
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
 
Q=100 #Ressources par Camion
alpha=0.001 #Pas de descente de la température
w=5 #Pénalité dûe au rajout d'un camion dans la solution
adj_matrix = np.random.randint(1, 20, size=(25, 25)) #Matrice d'adjacence générérant notre graphique
np.fill_diagonal(adj_matrix,np.zeros(len(adj_matrix))) 
T=1500 #Température de départ
t=200 #Température minimale


class Annealing:
    def __init__(self, database):
        self.solution = None
        self.Database = database

    def main(self, initial_solution=None):
        solution = []
        return solution


def create_G(adj_matrix,Q):
    """

    Parameters
    ----------
    adj_matrix : matrice d'adjacence crée aléatoirement'
    Q : Ressources par camions qui influent sur le choix des ressources demandées par chaque sommet

    Returns
    -------
    G : Graph plein généré 
    """
    n_max=int((len(adj_matrix)-1)/2) #Le nombre maximum de voitures qu'on mettrai à disposition. 
    nb_sommet=len(adj_matrix)-1 #Car il y a le 0
    tps_max=2+nb_sommet
    
    ###Répartition uniforme d'intervalles cohérents###
    n=nb_sommet//10
    dict_g={}
    dict_g[0]={'demande':0,'intervalle':0}
    heure_debut=0
    k=1
    for i in range(1,len(adj_matrix)):
        if heure_debut>80: #Intervalles larges
            a=rd.randint(0,40)
            b=rd.randint(a+30,a+60)
            dict_g[i]={'demande':rd.randint(0,int(0.5*Q*n_max/(len(adj_matrix)-1))),'intervalle':[a,b]}
            #n'hésitez pas à monter la valeur 0.5 aux lignes 46 et 49 pour augmenter les quantités demandées par les clients
        else:
            dict_g[i]={'demande':rd.randint(1,int(0.5*Q*n_max/(len(adj_matrix)-1))),'intervalle':[heure_debut,heure_debut+20]} #+2 car il faut au moins 2 d'écart entre deux classes de temps
            if i%n==0: #i toujours différent de 0 , donc le cas i = 0 ne se pose pas
                heure_debut+=10 #On rajoute 10 unités dès que i est un multiple de n.
                k+=1
    
    ###Le dictionnaire est crée , on rempli maintenant le graph###
    
    G = nx.from_numpy_matrix(adj_matrix)
    for i in range(0,len(G.nodes)):
        G.nodes[i].update(dict_g[i])  

    df=pd.DataFrame()
    df['nodes']=[i for i in range(1,len(adj_matrix))]
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(1,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(1,len(G.nodes))]
    df=df.sort_values(by='min')
    
    ###On colorie les routes et noeuds###
    
    labels = {n: n for n in G.nodes}
    colors = [G.nodes[i]['demande'] for i in G.nodes]
    nx.draw(G, with_labels=True, labels=labels, node_color=colors)
    G.nodes[0]['n_max']=n_max #Nombre de voiture maximal
    G.nodes[0]['tps_max']=tps_max
    p=[2,-100,nb_sommet] #Equation pour trouver n_min
    roots=np.roots(p)
    n_min=max(1,int(roots.min())+1) # Nombre de voiture minimal possible , solution d'une équation de second degrès. 
    G.nodes[0]['n_min']=n_min
    plt.title("graphe initial")
    plt.show()
    plt.clf()
    return G

def temperature(T,alpha):
    """
    Fonction de descente de la température
    """
    
    return (1-alpha)*T 

def check_temps(x,G):
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

def check_ressource(route,Q,G):
    """
    Fonction de vérification de contrainte des ressources. 

    Parameters
    ----------
    route : x[route], correspond à la route que va parcourir notre camion.
    Q : Ressource du camion
    G : Graph du problème

    Returns
    -------
    bool
        Si oui ou non, le camion peut en effet desservir toute les villes en fonction de ses ressources. 

    """
    ressource=Q
    for nodes in route:
        ressource=ressource-G.nodes[nodes]['demande']
        if  ressource<0:
            return False
    return True

def init(G,n,Q):
    """
    Fonction d'initialisation du solution possible à n camions. 
    Il y a beaucoup d'assertions car en effet, certains graph généré peuvent ne pas présenter de solution: 
        -Pas assez de voiture pour finir la livraison dans le temps imparti
        -Les ressources demandées peuvent être trop conséquentes 
        -Ect...

    Parameters
    ----------
    G : Graph du problème
    n : Nombre de camions à utiliser dans notre solution
    Q : Ressources disponibles par camions.

    Returns
    -------
    Une prémière solution x fonctionnelle .

    """
    ###Assertions du début , afin de vérifier que on ne demande pas d'initialiser l'impossible ###
    
    max_nodes_ressources=max([G.nodes[i]['demande'] for i in range(0,len(G.nodes))]) #On vérifie que les ressources de chaque sommets sont au moins <= Q
    assert(max_nodes_ressources<Q), "Les ressouces de certaines villes sont plus grandes que les ressources des voitures !"
    assert(n>G.nodes[0]['n_min']),"Peut-importe la configuration , il n'y a pas assez de camion pour terminer le trajet dans le temps imparti (<100)" #En effet, le temps de livraison des derniers sommets peuvent ne pas être atteint...   
    assert(n<=G.nodes[0]['n_max']),"On demande trop de voiture , <= à %s normalement " %G.nodes[0]['n_max']
    
    #####Construction de la solution qui fonctionne#####
    x=[]#Notre première solution
    for i in range(n):
        x.append([0]) #On initialise chaque route, celles-ci commencent par 0 à chaque fois
    nodes=[i for i in G.nodes]
    nodes.pop(0)
    
    ###Initialisation du dataframe renseignant sur les sommets et leurs contraintes###
    Ordre=[]
    df=pd.DataFrame()
    df['nodes']=[i for i in range(1,len(G.nodes))]
    df['min']=[G.nodes[i]['intervalle'][0] for i in range(1,len(G.nodes))]
    df['max']=[G.nodes[i]['intervalle'][1] for i in range(1,len(G.nodes))]
    df=df.sort_values(by='min') #On tri ce dataframe selon le début des intervalles de temps de chaque sommets dans l'ordre croissant. 
    Ordre=list(df['nodes']) #Ordre dans lequel les sommets seront parcouru, + un sommet demande d'être livré tard, + son ordre de passage sera tard... 
    
    ##Nos camions peuvent-ils livrer tout le monde ?##
    sum_ressources=sum([G.nodes[i]['demande'] for i in range(0,len(G.nodes))])
    if sum_ressources>n*Q: 
        print("Les ressources demandées par les villes sont trop importantes")
        return False
       
    
    ###On remplit la solution de la majorité des sommets###
    stock=[] #Liste des sommets qui n'ont pas pu être rajouté dans la solution durant cette première phase
    df_route=pd.DataFrame() #Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
    df_route.index=[i for i in range(0,n)]
    ressources=n*[Q]
    df_route['Ressources']=ressources
    
    ##1ere phase##
    for i in range(0,len(Ordre)):
        
        nodes_to_add=Ordre[i]
        q_nodes=G.nodes[nodes_to_add]['demande']
        route=i%n #On remplit parallèlement les chemins parcourut par chaque camions.
        if (df_route.loc[route]['Ressources']>=q_nodes):
            assert(q_nodes<=Q),"Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
            x[route].append(nodes_to_add)
            df_route.loc[route]['Ressources']+=-q_nodes
        else:
            stock.append(Ordre[i]) #Si les ressources du camion ne sont pas suffisante, on stock ce sommets dans une liste, on le rajoutera plus tard. 
            
    ##Seconde phase, on remplit la solution des sommets qui n'ont pu être affecté lors de la première phase##
    x=vide_stock(x,stock,df_route) #2nd phase
    
    ###Assertion pour vérifier que tout fonctionne bien###
    visite=[]
    for i in x:
        for j in i:
            if j not in visite:
                visite.append(j)
    assert(len(visite)==len(G.nodes)) #On vérifie que tout les sommets sont pris en compte
    for i in x:
        i.append(0)
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources"    
    return x
    
def perturbation(x,Q,G):
    
    """
    Fonction de perturbation. 
    Il y a beaucoup d'assertions afin de vérifier que la perturbation ne crée pas de problème de contraintes.  
       -On prend un sommet d'une route parcourue par un camion pour l'ajouter à une autre route
       -Pour chaque route, on permute deux sommets
       -Il est possible qu'à chaque étape ,il n'y ait pas de modification. 
    Parameters
    ----------
    x : Solution à perturber
    Q : Ressources disponibles par camions.
    G : Graph du problème 
    
    Returns
    -------
    Une solution x' perturbée de x et  fonctionnelle .
    
    """
    ###Assertion de début afin d'être sur que la solution d'entrée en est bien une ###
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps dès le début de la perturbation , pas la peine de continuer !"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources dès le début de la perturbation , pas la peine de continuer !"    
     
    very_old_x=copy.deepcopy(x) #Solution du début enregistrée , sert aux assertions de fin. Utiliser deepcopy ! et non copy
    K=len(x)
    ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
    
    ###On ajoute un sommet à une route , on en retire à une autre###
    if ajout != retire :
        if len(x[retire])>3:
            nouveau_sommet=rd.randint(1,len(x[retire])-2) #On évite les 0 dans chaque routes
            nouvelle_place=rd.randint(1,len(x[ajout])-2)
            x[ajout].insert(nouvelle_place,x[retire][nouveau_sommet])
            x[retire].pop(nouveau_sommet)
            while (check_ressource(x[ajout],Q,G)== False or check_temps(x,G)==False):
                ajout,retire=rd.randint(0,K-1),rd.randint(0,K-1)
                if ajout !=retire :
                    x=copy.deepcopy(very_old_x)
                    if len(x[retire])>3:
                        nouveau_sommet=rd.randint(1,len(x[retire])-2)
                        nouvelle_place=rd.randint(1,len(x[ajout])-2)
                        x[ajout].insert(nouvelle_place,x[retire][nouveau_sommet])
                        x[retire].pop(nouveau_sommet)
                else:
                    x=copy.deepcopy(very_old_x)
                    break
           
    ###On permute pour chaque route , des sommets###
    
    ##Assertions intermédiaires##
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps pendant l'insertion"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources pendant l'insertion" 
    ##Début des permutations##
    for route in range(0,K):
        assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps pendant la perturbation"
        for i in range(0,len(x)):
            assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources pendant la perturbation"    
        permutation=[rd.randint(1,len(x[route])-2),rd.randint(1,len(x[route])-2)]
        
        if(permutation[0] != permutation[1]): # Si c'est pareil : Passe
            old_x=copy.deepcopy(x)

            x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]
            while(check_temps(x,G)==False):
                x=copy.deepcopy(old_x)
                permutation=[rd.randint(1,len(x[route])-2),rd.randint(1,len(x[route])-2)]
                
                x[route][permutation[0]], x[route][permutation[1]]= x[route][permutation[1]], x[route][permutation[0]]
    
    ###Assertions de fin###
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps à la fin de la permutation"
    for i in range(0,len(x)):
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources à la fin de la permutation"    
    num_very_old_x=sum([len(i) for i in very_old_x])
    num_x=sum([len(i) for i in x]) #On vérifie qu'aucun sommet n'a été oublié
    assert(num_very_old_x==num_x)
    return x

def energie(x,w,G):
    """
    Fonction coût pour le SimulatedAnnealing

    Parameters
    ----------
    x : solution
    w : Pénalité dûe au rajout d'une voiture
    G : Graph du problème

    Returns
    -------
    somme : Le coût de la solution

    """
    K= len(x)
    for route in range(0,K):
        somme=sum([G[x[route][sommet]][x[route][sommet+1]]['weight'] for sommet in range(0,len(x[route])-1)])
    somme+=w*K
    return somme

def plot_graph(x,E,G, final = False):
    graph_route = nx.DiGraph() #création du nouveau graphe des routes
    for i in range(0,len(x)):
        route = x[i]
        for j in range(0, len(route)-1):
            graph_route.add_edges_from([(str(route[j]), str(route[j+1]))]) #pour chacune des routes, on ajoute les sommets 
    colors = [G.nodes[i]['demande'] for i in G.nodes]
    nx.draw_networkx(graph_route,arrows=True, with_labels=True, node_color =colors, arrowstyle = '-|>', arrowsize = 12) #edge_color = ['red', 'black', 'green']
    if final == False:
        graph_label = "Energie du test : " + str(E) + ", nombre de route : " + str(len(x))
    else:
        graph_label = "Energie finale : " + str(E) + ", nombre de route optimal : " + str(len(x))
    plt.title(graph_label)
    plt.show()
    plt.clf()

def recuitsimule(alpha,Q,G,T,t):
    """
    

    Parameters
    ----------
    alpha :Descente de température.
    Q : Ressources par camion
    G : Graph du problème
    T : Température initiale
    t : Temprérature finale

    Returns
    -------
    best_roads : Dataframe des meilleurs solutions trouvée en fonction du nombre de route
    best_X :Meilleure solution

    """
    
    
    X=[] # Meilleures solutions à un nombre n de camion fixé
    k=1.38*10e-23
    for (x,y) in np.ndindex(len(G),len(G)):
        if x!=y:
            assert(G.get_edge_data(x,y)['weight'] !=0) , "Certaines routes sont nulles , le graphe n'est pas plein , il faut revoir la matrice d'adjacence !"
    T_old=T
    n_min=G.nodes[0]['n_min']
    n_max=G.nodes[0]['n_max']
    E_min_X=math.inf
    best_X=nx.empty_graph()
    
    ###Boucle itérative sur le nombre de voiture###
    for i in tqdm(range(n_min+1,n_max+1)): 
        E_min=math.inf
        best_x=nx.empty_graph()
        T=T_old
        T_list = []
        E_list = []
        x=init(G,i,Q)
        if(x!=False):
            while T >=t:
                old_x=copy.deepcopy(x)
                x=perturbation(x,Q,G)
                E=energie(x,w,G)
                E0=energie(old_x,w,G)
                if E<E0: 
                    old_x = copy.deepcopy(x)
                    if E<E_min:
                        best_x=copy.deepcopy(x)
                        E_min=E
                        #plot_graph(x,E,G)
                        T_list.append(T)
                        E_list.append(E)
                        print(x)
                    if E<E_min_X:
                        best_X=copy.deepcopy(x)
                        E_min_X=E
                else:
                    p=np.exp(-(E-E0)/(k*T))
                    r=rd.random() #ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier !
                    if r<= p:
                        old_x=copy.deepcopy(x)
                        #plot_graph(x,E,G)
                        T_list.append(T)
                        E_list.append(E)
                T=temperature(T,alpha) 
            X.append(best_x)
            #Tracé du graphe de l'énergie en fonction de la température
            plt.plot(T_list, E_list, linestyle='', linewidth = 3,
                     marker='x', markerfacecolor='black', markersize=8)
            plt.title("Energie en fonction de la température pour " + str(i) + " routes")
            plt.xlabel('Température')
            plt.ylabel('Energie')
            plt.show()
            plt.clf()
    best_roads=pd.DataFrame()
    best_roads['X']=X
    best_roads['Energie']=[energie(x,w,G) for x in X]
    best_roads=best_roads.sort_values(by='Energie',ascending=True, ignore_index=True)
    best_roads['n']=[len(best_roads['X'].iloc[i]) for i in range(0,len(best_roads))]
    assert(energie(best_X,w,G)==min(list(best_roads['Energie']))) # On vérifie qu'on prend bien la meilleure solution
    plot_graph(best_roads['X'][0], best_roads['Energie'][0], G, True)
    return best_roads,best_X

def recuitsimule_2(alpha,Q,G,T,t,n):
    """
    Recuitsimulé à route fixé
    """
    
    ###Assertions principales###
    k=1.38*10e-23
    scores=[]
    E_min=math.inf
    for (x,y) in np.ndindex(len(G),len(G)):
        if x!=y:
            assert(G.get_edge_data(x,y)['weight'] !=0) , "Certaines routes sont nulles , le graphe n'est pas plein , il faut revoir la matrice d'adjacence !"
    T_old=T
    n_min=G.nodes[0]['n_min']
    n_max=G.nodes[0]['n_max']
    best_x=nx.empty_graph()

    T=T_old
    x=init(G,n,Q)
    if(x!=False):
        while T >=t:
            old_x=copy.deepcopy(x)
            x=perturbation(x,Q,G)
            E=energie(x,w,G)
            E0=energie(old_x,w,G)
            if E<E0: 
                old_x = copy.deepcopy(x)
                scores.append(E)
                if E<E_min:
                    best_x=copy.deepcopy(x)
                    E_min=E
                #plot(x) +Temp
            else:
                p=np.exp(-(E-E0)/(k*T))
                r=rd.random() #ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier !
                if r<= p:
                    old_x=copy.deepcopy(x)
                    #plot(x) +Temp
            T=temperature(T,alpha) 
    assert(energie(best_x,w,G)==min(scores))
    return best_x,scores

def vide_stock(x,stock,df_route):
    """
    Fonction pour remplir x des sommets qui n'ont pu être ajouté lors de la première phase d'initialisation. 

    Parameters
    ----------
    x : Solution initialisée. 
    stock : Sommet à rajouter
    df_route : Dataframe renseignant sur les routes

    Returns
    -------
    x : Solution initialisée

    """
    df_route['Chemin']=[x[route] for route in range(0,len(x))]
    df_route=df_route.sort_values(by='Ressources',ascending=False)
    
    extra_node=0
    for i in range(0,len(stock)):
        node=stock[i]
        q_nodes=G.nodes[node]['demande']
        int_min=G.nodes[node]['intervalle'][0]
        route_a_remplir=0
        while route_a_remplir <len(df_route):
            if (df_route.iloc[route_a_remplir]['Ressources']>= q_nodes):
                chemin_route=list(df_route.iloc[route_a_remplir]['Chemin'])
                good=0
                j=1
                while (good!=1 and j < len(chemin_route)-1):
                    if j ==1:
                        if (int_min<=G.nodes[chemin_route[j]]['intervalle'][0]):
                            place_node=j
                            
                            x[df_route.index[route_a_remplir]].insert(place_node,node)
                            temp=df_route.iloc[route_a_remplir]['Ressources']
                            df_route['Ressources'].iloc[route_a_remplir]+=-q_nodes
                            assert(temp!=df_route.iloc[route_a_remplir]['Ressources']),'%s'%q_nodes
                            good=1
                            extra_node+=1
                        elif (int_min>=G.nodes[chemin_route[-1]]['intervalle'][0]):
                            temp=df_route.iloc[route_a_remplir]['Ressources']
                            
                            x[df_route.index[route_a_remplir]].append(node)

                            df_route['Ressources'].iloc[route_a_remplir]+=-q_nodes
                            good=1
                            extra_node+=1
                            assert(temp!=df_route.iloc[route_a_remplir]['Ressources']),'%s'%q_nodes
    
                        elif (G.nodes[chemin_route[j]]['intervalle'][0]<int_min<G.nodes[chemin_route[j+1]]['intervalle'][0]):
                            temp=df_route.iloc[route_a_remplir]['Ressources']
                            place_node=j+1
                            x[df_route.index[route_a_remplir]].insert(place_node,node)

                            df_route['Ressources'].iloc[route_a_remplir]+=-q_nodes
                            good=1
                            extra_node+=1
                            assert(temp!=df_route.iloc[route_a_remplir]['Ressources']),'%s'%q_nodes
    
                        else:
                            j+=1
                    
                    else:
                        
                        if (G.nodes[chemin_route[j]]['intervalle'][0]<int_min<G.nodes[chemin_route[j+1]]['intervalle'][0]):
                            place_node=j+1
                            x[df_route.index[route_a_remplir]].insert(place_node,node)
                            temp=df_route.iloc[route_a_remplir]['Ressources']
    
                            df_route['Ressources'].iloc[route_a_remplir]+=-q_nodes
                            good=1
                            extra_node+=1
                            assert(temp!=df_route.iloc[route_a_remplir]['Ressources']),'%s'%q_nodes
    
                        else:
                            j+=1
                
                if j==len(chemin_route)-1:  
                     route_a_remplir +=1
                     assert(route_a_remplir<len(df_route)) ,"Impossible d'initialiser le graph, il faut revoir les ressources et intervalles"
                else:
                    break
            else:
                route_a_remplir+=1
                assert(route_a_remplir<len(df_route)) ,"Impossible d'initialiser le graph, il faut revoir les ressources et intervalles"
    assert(extra_node==len(stock)),"Impossible d'initialiser le graph,on ne peut pas vider le stock, il faut revoir les ressources et intervalles"
    return x

G=create_G(adj_matrix,Q)
best_roads,best_X=recuitsimule(alpha,Q,G,T,t)

#ıdict(G.nodes)
