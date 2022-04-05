# -*- coding: utf-8 -*-

import copy
import numpy as np
import random as rd 
import networkx as nx
import pandas as pd 
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import warnings
import sys
import utm
from recuit_init2_0 import main
warnings.simplefilter(action='ignore', category=FutureWarning)
alpha=0.01 #Pas de la descente de la température
w=5 #Pénalité dûe au rajout d'un camion dans la solution
T =1500 #Température de départ
global t
t=200 #Température minimale
v=50
df_customers= pd.read_excel("table_2_customers_features.xls")
df_vehicles=pd.read_excel("table_3_cars_features.xls")
df_vehicles=df_vehicles.drop(['Unnamed: 0'],axis=1)
df_customers=df_customers.drop(['Unnamed: 0'],axis=1)
df_ordre_init=pd.read_pickle("df_ordre_init.pkl")
def create_G(df_customers,df_vehicles,v):
    """

    Parameters
    ----------
    df_customers : Dataframe contenant les informations sur les clients
    df_vehicles : Dataframe contenant les informations sur les camions livreurs

    Returns
    -------
    G : Graph plein généré 
    """
    n_max= len(df_vehicles) #Le nombre maximum de voitures qu'on mettrai à disposition. 
    n_sommet=len(df_customers)
    G=nx.empty_graph(n_sommet)
    (x_0,y_0)=utm.from_latlon(43.37391833,17.60171712)[:2]
    dict_0={'CUSTOMER_CODE':0, 'CUSTOMER_LATITUDE': 43.37391833,'CUSTOMER_LONGITUDE':17.60171712,'CUSTOMER_TIME_WINDOW_FROM_MIN':360,'CUSTOMER_TIME_WINDOW_TO_MIN':1080,'TOTAL_WEIGHT_KG':0, 'pos':(x_0,y_0),"CUSTOMER_DELIVERY_SERVICE_TIME_MIN":0}
    G.nodes[0].update(dict_0)
    G.nodes[0]['n_max']=n_max #Nombre de voiture maximal
    dict_vehicles=df_vehicles.to_dict()
    G.nodes[0]['Camion']=dict_vehicles
    for i in range(1,len(G.nodes)):
        dict=df_customers.iloc[i].to_dict()
        dict['pos']=utm.from_latlon(dict['CUSTOMER_LATITUDE'],dict['CUSTOMER_LONGITUDE'])[:2]
        G.nodes[i].update(dict)  
    
    ###On rajoute les routes###
    for i in range(0,len(G.nodes)):
        for j in range(0,len(G.nodes)):
            if i!=j:
                z_1=G.nodes[i]['pos']
                z_2=G.nodes[j]['pos']
                G.add_edge(i,j,weight=get_distance(z_1,z_2))
                G[i][j]['time']=(G[i][j]['weight']/v)*60
    ###On colorie les routes et noeuds###
    
    colors=[0]
    colors+=[G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(1,len(G.nodes))]
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_nodes(G,pos, node_color=colors)
    G.nodes[0]['n_max']=n_max #Nombre de voiture maximal
    p=[2,-100,n_sommet] #Equation pour trouver n_min
    roots=np.roots(p)
    n_min=max(1,int(roots.min())+1) # Nombre de voiture minimal possible , solution d'une équation de second degrès. 
    G.nodes[0]['n_min']=n_min
    plt.title("graphe initial")
    plt.show()
    plt.clf()
    return G

def get_distance(z_1,z_2):
    x_1,x_2,y_1,y_2=z_1[0],z_2[0],z_1[1],z_2[1]
    d=math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
    return d/1000 #en km


#G.nodes[i] avec i un entier de 1 à 540 environ pour acceder au données d'un client.
#G[i][j] pour accéder au données de la route entre i et j. 
#G.nodes[0] est le dépots, il contient également les informations relatives aux camions. 

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
        df_temps=pd.DataFrame(columns=['temps','route','temps_de_parcours','limite_inf','limite_sup'])
        temps=G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN'] #Temps d'ouverture du dépot
        for i in range(1,len(x[route])-1): #On ne prend pas en compte l'aller dans l'intervalle de temps
            #assert(temps<G.nodes[0]['CUSTOMER_TIME_WINDOW_TO_MIN']) #Il faut que les camion retournent chez eux à l'heure
            first_node=x[route][i]
            second_node=x[route][i+1]
            if second_node !=0:
                temps+=G[first_node][second_node]['time'] #temps mis pour parcourir la route en minute
                while  temps<G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN']:
                    temps+=1 #Le camion est en pause
                dict={'temps':temps,'route':(first_node,second_node),'temps_de_parcours':G[first_node][second_node]['time'],'limite_inf':G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'],'limite_sup':G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN'],"camion":route}
                df_temps=df_temps.append([dict])
                if (temps<G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'] or temps>G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN']):
                    #print(df_temps)
                    return False
                temps+=G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]/10
    return True

def check_temps_2(x,G):
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
    
    
    df_temps=pd.DataFrame(columns=['temps','route','temps_de_parcours','limite_inf','limite_sup'])
    temps=G.nodes[0]['CUSTOMER_TIME_WINDOW_FROM_MIN'] #Temps d'ouverture du dépot
    for i in range(1,len(x)-1):
        #assert(temps<G.nodes[0]['CUSTOMER_TIME_WINDOW_TO_MIN']) #Il faut que les camion retournent chez eux à l'heure
        first_node=x[i]
        second_node=x[i+1]
        if second_node != 0: #On ne prend pas en compte l'arrivée non plus
            temps+=G[first_node][second_node]['time'] #temps mis pour parcourir la route en minute
            while  temps<G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN']:
                temps+=1 #Le camion est en pause
            dict={'temps':temps,'route':(first_node,second_node),'temps_de_parcours':G[first_node][second_node]['time'],'limite_inf':G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'],'limite_sup':G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN']}
            df_temps=df_temps.append([dict])
            if (temps<G.nodes[second_node]['CUSTOMER_TIME_WINDOW_FROM_MIN'] or temps>G.nodes[second_node]['CUSTOMER_TIME_WINDOW_TO_MIN']):
                #print("Pendant l'initialisation \n",df_temps)
                return False
        
            temps+=G.nodes[second_node]["CUSTOMER_DELIVERY_SERVICE_TIME_MIN"]/10
    return True
   

def check_ressource(route,Q,G):
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
    ressource=Q
    for nodes in route:
        ressource=ressource-G.nodes[nodes]['TOTAL_WEIGHT_KG']
        if  ressource<0:
            return False
    return True

def init(G,n):
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

    Returns
    -------
    Une prémière solution x fonctionnelle .

    """
    ###Positions de chaque points###
    X = [ G.nodes[i]['pos'][0] for i in range(0,len(G))]
    Y = [ G.nodes[i]['pos'][1] for i in range(0,len(G))]
    
    ###Assertions du début , afin de vérifier que on ne demande pas d'initialiser l'impossible ###
    max_Q=max(G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())
    max_ressources=sum(G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"].values())
    max_nodes_ressources=max([G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(0,len(G.nodes))]) #On vérifie que les ressources de chaque sommets sont au moins <= Q
    assert(max_nodes_ressources<max_Q), "Les ressouces de certaines villes sont plus grandes que les ressources des voitures !"
    assert(n>G.nodes[0]['n_min']),"Peu-importe la configuration , il n'y a pas assez de camion pour terminer le trajet dans le temps imparti (<100)" #En effet, le temps de livraison des derniers sommets peuvent ne pas être atteint...   
    assert(n<=G.nodes[0]['n_max']),"On demande trop de voiture , <= à %s normalement " %G.nodes[0]['n_max']
    
    #####Construction de la solution qui fonctionne#####
    
    
    x=[]#Notre première solution
    for i in range(n):
        x.append([0]) #On initialise chaque route, celles-ci commencent par 0 à chaque fois
    nodes=[i for i in G.nodes]
    nodes.pop(0)
    
    ###Initialisation du dataframe renseignant sur les sommets et leurs contraintes###
    Ordre=main(G) #Ordre initial
    #Ordre=list(df_ordre_init['Ordre'])
    assert(Ordre[0]==0),"L'ordre initial n'est pas bon ,il ne commence pas par 0"
    ##Nos camions peuvent-ils livrer tout le monde ?##
    sum_ressources=sum([G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(0,len(G.nodes))])
    if sum_ressources>max_ressources: 
        print("Les ressources demandées par les villes sont trop importantes")
        return False
       
    
    ###On remplit la solution de la majorité des sommets###
    df_camion=pd.DataFrame() #Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
    df_camion.index=[i for i in range(0,n)]
    ressources=[G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"][i] for i in range(0,n) ] #On commence par les camions aux ressources les plus importantes
    df_camion['Ressources']=ressources
    df_ordre=pd.DataFrame(columns=["Camion","Ressource_to_add","Id","CUSTOMER_TIME_WINDOW_FROM_MIN","CUSTOMER_TIME_WINDOW_TO_MIN","Ressource_camion"])
    camion=0
    
    ###1ere phase###
    i=1 #On ne prend pas en compte le zéros du début
    with tqdm(total=len(Ordre)) as pbar:
        while i < len(Ordre):
            assert camion<n,"Impossible d'initialiser , les camions ne sont pas assez nombreux"
            nodes_to_add=Ordre[i]
            assert(nodes_to_add != 0),"Le chemin proposé repasse par le dépot !"
                
            q_nodes=G.nodes[nodes_to_add]['TOTAL_WEIGHT_KG']
            int_min=G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_FROM_MIN"]
            int_max=G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_TO_MIN"]
            dict=[{"Camion":camion,"Ressource_to_add":q_nodes,"Id":nodes_to_add,"CUSTOMER_TIME_WINDOW_FROM_MIN":int_min,"CUSTOMER_TIME_WINDOW_TO_MIN":int_max , "Ressource_camion":df_camion.loc[camion]['Ressources']}]
            temp=copy.deepcopy(x[camion])
            temp.append(nodes_to_add)
            if (df_camion.loc[camion]['Ressources']>=q_nodes and check_temps_2(temp,G)==True):
                Q=G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                assert(q_nodes<=Q),"Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
                x[camion].append(nodes_to_add)
                df_camion['Ressources'].loc[camion]+=-q_nodes
                i+=1
                pbar.update(1)
                assert(x[camion]==temp)
            else:
                print(nodes_to_add)
                assert(x[camion]!=temp)
                camion+=1
               
            df_ordre=df_ordre.append(dict)
    ##Seconde phase, on remplit la solution des sommets qui n'ont pu être affecté lors de la première phase##
    ###Assertion pour vérifier que tout fonctionne bien###
    visite=pd.DataFrame(columns=["Client" , "passage"])
    for i in x:
        for j in i:
            if j not in list(visite["Client"]):
                dict={"Client":j,"passage":1}
                visite=visite.append([dict])
            else:
                visite['passage'][visite['Client']==j]+=1
    assert(len(visite)==len(G.nodes)),"Tout les sommets ne sont pas pris en compte" #On vérifie que tout les sommets sont pris en compte
    visite_2=visite[visite['Client']!=0]
    assert(len(visite_2[visite_2['passage']>1])==0),"Certains sommets sont plusieurs fois déservis"
    for i in x:
        i.append(0)
    assert(check_temps(x,G)==True),"Mauvaise initialisation au niveau du temps"

    for i in range(0,len(x)):
        Q=G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] #Ressource du camion utilisé
        assert(check_ressource(x[i],Q,G)==True),"Mauvaise initialisation au niveau des ressources" 
        assert(0 not in x[i][1:-1]),"Un camion repasse par 0"
    plotting(x,G)
    return x

def plotting(x,G):
    plt.clf()
    X = [ G.nodes[i]['pos'][0] for i in range(0,len(G))]
    Y = [ G.nodes[i]['pos'][1] for i in range(0,len(G))]
    plt.plot(X,Y, "o")
    plt.text(X[0],Y[0],"0",color="r",weight="bold",size="x-large")

    colors= ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    couleur=0
    for camion in range(0,len(x)):
        assert(camion<len(colors)),"Trop de camion, on ne peut pas afficher"
        if len(x)>2:
            xo = [ X[o] for o in x[camion] ]
            yo = [ Y[o] for o in x[camion] ]
            plt.plot(xo,yo, colors[couleur])
            couleur+=1
    plt.show()
def perturbation(x,G,T):
    
    """
    Fonction de perturbation. 
    Il y a beaucoup d'assertions afin de vérifier que la perturbation ne crée pas de problème de contraintes.  
       -On prend un sommet d'une route parcourue par un camion pour l'ajouter à une autre route
       -Pour chaque route, on permute deux sommets
       -Il est possible qu'à chaque étape ,il n'y ait pas de modification. 
    Parameters
    ----------
    x : Solution à perturber
    G : Graph du problème 
    
    Returns
    -------
    Une solution x' perturbée de x et  fonctionnelle .
    
    """
    it=0
    k=10e-5 #arbitraire
    Q=[G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0,len(x))]
    nb_sommet=len(G.nodes)
    E=energie(x,G)
    E0=E+2
    best_x=copy.deepcopy(x)
    T_list = [T]
    E_list = [E]
    E_min=E
    fig,axs=plt.subplots(1,2)
    while E0-E >= 1  :
        
        it += 1
        print("iteration",it, "E=",E,"Température =",T)
        
        E0 = E
        x=copy.deepcopy(best_x)
        for i in tqdm(range(0,nb_sommet-1)) :
            for j in range(i+1,nb_sommet):
                very_old_x=copy.deepcopy(x) #Solution du début enregistrée , sert aux assertions de fin. Utiliser deepcopy ! et non copy
                if i!=0:
                    flatten_x=[item for sublist in x for item in sublist]
                    ajout,retire=flatten_x.index(i),flatten_x.index(j)
                    num_zero_ajout,num_zero_retire=flatten_x[:ajout].count(0)-1,flatten_x[:retire].count(0)-1
                    camion_ajout,camion_retire=int(num_zero_ajout/2-1)+1,int(num_zero_retire/2-1)+1
                    nouvelle_place=x[camion_ajout].index(i)
                    x[camion_ajout].insert(nouvelle_place,j)
                    x[camion_retire].remove(j)
                ###On ajoute un sommet à une route , on en retire à une autre###
                else: 
                    flatten_x=[item for sublist in x for item in sublist]
                    retire=flatten_x.index(j)
                    num_zero_retire=flatten_x[:retire].count(0)-1
                    camion_retire=int(num_zero_retire/2-1)+1
                    assert(j in  x[camion_retire])
                    unfilled_road=[i for i in range(0,len(x)) if len(x[i])==2]
                    if len(unfilled_road) !=0:
                        nouvelle_place=1
                        camion_ajout=unfilled_road[0]
                        x[camion_ajout].insert(nouvelle_place,j)
                        x[camion_retire].remove(j)
                E1,E2,E3,E4=energie_part(x[camion_ajout],G,camion_ajout),energie_part(x[camion_retire],G,camion_retire),energie_part(very_old_x[camion_ajout],G,camion_ajout),energie_part(very_old_x[camion_retire],G,camion_retire)
                if (E1+E2>=E3+E4):
                   p=np.exp(-(E1+E2-(E3+E4))/(k*T))
                   #print(p)
                   r=rd.random() #ici c'est rd.random et non rd.randint(0,1) qui renvoit un entier !
                   if r<= p and p!=1:
                       if(check_constraint(x,G) != True):
                           x= copy.deepcopy(very_old_x) #sinon on conserve x tel qu'il est
                   else:  x=copy.deepcopy(very_old_x)
                
                else:
                    if (check_ressource(x[camion_ajout],Q[camion_ajout],G) == False or check_temps_2(x[camion_ajout],G)==False):
                        x=copy.deepcopy(very_old_x)
                    else:
                        E=energie(x,G)
                        if E<E_min:
                            best_x=copy.deepcopy(x) #On garde en mémoire le meilleur x trouvé jusqu'à présent
                            E_min=E
                            
       
        ###Assertions de fin###
        plotting(best_x,G)
        assert(check_constraint(best_x,G)==True)
        num_very_old_x=sum([len(i) for i in very_old_x])
        num_x=sum([len(i) for i in x]) #On vérifie qu'aucun sommet n'a été oublié
        assert(num_very_old_x==num_x)
        E=energie(best_x,G)
        if E0 > E:
            T=(1/math.log(E0-E))*1500
            T_list.append(T)
            E_list.append(E)
        
    plt.clf()
    axs[0].plot(E_list,'o-')
    axs[0].set_title("Energie de la meilleure solution en fonction des itérations")
    axs[1].plot(T_list,'o-')
    axs[1].set_title("Température en fonction des itérations")
    fig.suptitle("Profil de la première partie")
    plt.savefig("Profil de la première partie")
    plt.show()
    
    ###Assertions de fin###
    
    visite=pd.DataFrame(columns=["Client" , "passage"])
    for i in best_x:
        for j in i:
            if j not in list(visite["Client"]):
                dict={"Client":j,"passage":1}
                visite=visite.append([dict])
            else:
                visite['passage'][visite['Client']==j]+=1
    assert(len(visite)==len(G.nodes)),"Tout les sommets ne sont pas pris en compte" #On vérifie que tout les sommets sont pris en compte
    visite_2=visite[visite['Client']!=0]
    assert(len(visite_2[visite_2['passage']>1])==0),"Certains sommets sont plusieurs fois déservis"
    assert(check_constraint(best_x,G)==True)
    plotting(x,G)
   
    return best_x

def energie(x,G):
    """
    Fonction coût pour le recuit

    Parameters
    ----------
    x : solution
    G : Graph du problème

    Returns
    -------
    somme : Le coût de la solution

    """
    K= len(x)
    somme=0
    for route in range(0,K):
        if len(x[route])>2: #si la route n'est pas vide
            w=G.nodes[0]['Camion']['VEHICLE_VARIABLE_COST_KM'][route] #On fonction du coût d'utilisation du camion 
            weight_road=sum([G[x[route][sommet]][x[route][sommet+1]]['weight'] for sommet in range(0,len(x[route])-1)])
            somme+=weight_road
            somme+=w*weight_road
    return somme

def energie_part(x,G,camion):
    """
    Fonction coût pour le recuit

    Parameters
    ----------
    x : solution
    G : Graph du problème

    Returns
    -------
    somme : Le coût de la solution

    """
    if len(x)>2: #si la route n'est pas vide
        w=G.nodes[0]['Camion']['VEHICLE_VARIABLE_COST_KM'][camion] #On fonction du coût d'utilisation du camion 
        somme=sum([G[x[sommet]][x[sommet+1]]['weight'] for sommet in range(0,len(x)-1)])
        somme+=w*somme #facteur véhicule
        return somme
    else:
        return 0



def check_constraint(x,G):
    Q=[G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][i] for i in range(0,len(x))]
    if(check_temps(x,G)==True):
        for i in range(0,len(x)):
            if(check_ressource(x[i],Q[i],G)!=True):
                return False
        else: return True
    else: return False

   



def perturbation_2(x,G):
    d  = energie(x,G)
    d0 = d+1
    it = 1
    list_E=[d]
    while d < d0 :
        it += 1
        print("iteration",it, "d=",d)
        d0 = d
        for camion in tqdm(range(0,len(x))):
            route=x[camion]
            for i in range(1,len(route)-1) :
                for j in range(i+2,len(route)):
                    d_part=energie_part(route,G,camion)
                    r = route[i:j].copy()
                    r.reverse()
                    route2 = route[:i] + r + route[j:]
                    t = energie_part(route2,G,camion)
                    if (t < d_part): 
                        if check_temps_2(route2,G)==True: 
                            x[camion] = route2   
        d=energie(x,G)
        list_E.append(d)
        assert(check_temps(x,G)==True)  
        plotting(x,G)
    plt.clf()
    plt.plot(list_E,'o-')
    plt.title("Evoluation de l'énergie lors de la seconde phase")
    plt.show()
    ###Assertions de fin###
    
    visite=pd.DataFrame(columns=["Client" , "passage"])
    for i in x:
        for j in i:
            if j not in list(visite["Client"]):
                dict={"Client":j,"passage":1}
                visite=visite.append([dict])
            else:
                visite['passage'][visite['Client']==j]+=1
    assert(len(visite)==len(G.nodes)),"Tout les sommets ne sont pas pris en compte" #On vérifie que tout les sommets sont pris en compte
    visite_2=visite[visite['Client']!=0]
    assert(len(visite_2[visite_2['passage']>1])==0),"Certains sommets sont plusieurs fois déservis"
    assert(check_constraint(x,G)==True),"Mauvaise initialisation au niveau du temps"
    return x

def main_2(df_customers,df_vehicles,v,T):
    G=create_G(df_customers,df_vehicles,v) #En jaune le centre. 
    n=len(df_vehicles)
    x=init(G,n)
    df_x=pd.read_pickle("test_avec_pause.pkl")
    x=list(df_x['ordre'])
    assert(check_constraint(x,G)),"Mauvaise initialisation"
    plotting(x,G)
    x=perturbation(x,G,T)
    print("Début de la seconde phase \n")
    x=perturbation_2(x,G)
main_2(df_customers,df_vehicles,v,T)
