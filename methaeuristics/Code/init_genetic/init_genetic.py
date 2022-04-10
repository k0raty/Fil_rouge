# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 18:08:26 2022
Programme qui initialise un nombre n_sol de solution , voir la fonction init_gen.
De plus df_ordre_x est optionnel mais permet degagner du temps, c'est la solution au voyageur de commerce.
Pour 50 solutions initiales, c'est un peu long , je te conseille de conserver la solution X_INIT par la suite.
@author: antony
"""

import pandas as pd
import networkx as nx
import numpy as np
import utm
import matplotlib.pyplot as plt
import math
from prob_voyageur import main
from function_init import check_temps,check_temps_part,check_forme,check_constraint
from tqdm import tqdm
import copy
import random as rd
T =1500 #Température de départ
v=50 #Vitesse des véhicules
df_customers= pd.read_excel("table_2_customers_features.xls")
df_vehicles=pd.read_excel("table_3_cars_features.xls")
df_vehicles=df_vehicles.drop(['Unnamed: 0'],axis=1)
df_customers=df_customers.drop(['Unnamed: 0'],axis=1)
df_ordre_x=pd.read_pickle("for_init_genetic.pkl")
ordre_x=list(df_ordre_x['Ordre'])
#ordre_x =main(G)

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
    
    G.nodes[0]['n_max']=n_max #Nombre de voiture maximal
    p=[2,-100,n_sommet] #Equation pour trouver n_min
    roots=np.roots(p)
    n_min=max(1,int(roots.min())+1) # Nombre de voiture minimal possible , solution d'une équation de second degrès. 
    G.nodes[0]['n_min']=n_min

    return G

def get_distance(z_1,z_2):
    x_1,x_2,y_1,y_2=z_1[0],z_2[0],z_1[1],z_2[1]
    d=math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
    return d/1000 #en km

def plotting(x,G):
    """
    Affiche les trajectoires des différents camion entre les clients.
    Chaque trajectoire a une couleur différente. 

    Parameters
    ----------
    x : Routage solution
    G : Graphe en question 

    Returns
    -------
    None.

    """
    plt.clf()
    X = [ G.nodes[i]['pos'][0] for i in range(0,len(G))]
    Y = [ G.nodes[i]['pos'][1] for i in range(0,len(G))]
    plt.plot(X,Y, "o")
    plt.text(X[0],Y[0],"0",color="r",weight="bold",size="x-large")
    plt.title("Trajectoire de chaque camion")
    colors= ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    couleur=0
    for camion in range(0,len(x)):
        assert(camion<len(colors)),"Trop de camion, on ne peut pas afficher"
        if len(x)>2:
            xo = [ X[o] for o in x[camion] ]
            yo = [ Y[o] for o in x[camion] ]
            plt.plot(xo,yo, colors[couleur],label=G.nodes[0]["Camion"]["VEHICLE_VARIABLE_COST_KM"][camion])
            couleur+=1
    plt.legend(loc='upper left')
    plt.show()

def init(G,n,ordre_camion,ordre_x):
    
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
    ordre_camion:liste d'indices de 0 à 8 correspondant à l'ordre de remplissage des camions :[1,5,8,6..],[8,4,1,2...]
    ordre_x: Résultat du problème du voyageur de commerce 

    Returns
    -------
    Une prémière solution x fonctionnelle .

    """
    assert(n==len(ordre_camion))
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
    Ordre=ordre_x
    assert(Ordre[0]==0),"L'ordre initial n'est pas bon ,il ne commence pas par 0"
    
    ##Nos camions peuvent-ils livrer tout le monde ?##
    sum_ressources=sum([G.nodes[i]['TOTAL_WEIGHT_KG'] for i in range(0,len(G.nodes))])
    if sum_ressources>max_ressources: 
        print("Les ressources demandées par les villes sont trop importantes")
        return False
       
    
    ###On remplit la solution de la majorité des sommets###
    df_camion=pd.DataFrame() #Dataframe renseignant sur les routes, important pour la seconde phase de  remplissage
    df_camion.index=[i for i in ordre_camion]
    ressources=[G.nodes[0]["Camion"]["VEHICLE_TOTAL_WEIGHT_KG"][i] for i in ordre_camion ] #On commence par les camions aux ressources les plus importantes
    df_camion['Ressources']=ressources
    df_ordre=pd.DataFrame(columns=["Camion","Ressource_to_add","Id","CUSTOMER_TIME_WINDOW_FROM_MIN","CUSTOMER_TIME_WINDOW_TO_MIN","Ressource_camion"])
    indice_camion=0
    
    i=1 #On ne prend pas en compte le zéros du début
    with tqdm(total=len(Ordre),position=0, leave=True) as pbar:
        while i < len(Ordre):
            
            if (indice_camion >=n):
                print("Impossible d'initialiser , les camions ne sont pas assez nombreux\n")
                return False

            camion=ordre_camion[indice_camion]
            nodes_to_add=Ordre[i]
            assert(nodes_to_add != 0),"Le chemin proposé repasse par le dépot !"
                
            q_nodes=G.nodes[nodes_to_add]['TOTAL_WEIGHT_KG']
            int_min=G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_FROM_MIN"]
            int_max=G.nodes[nodes_to_add]["CUSTOMER_TIME_WINDOW_TO_MIN"]
            dict=[{"Camion":camion,"Ressource_to_add":q_nodes,"Id":nodes_to_add,"CUSTOMER_TIME_WINDOW_FROM_MIN":int_min,"CUSTOMER_TIME_WINDOW_TO_MIN":int_max , "Ressource_camion":df_camion.loc[camion]['Ressources']}]
            temp=copy.deepcopy(x[camion])
            temp.append(nodes_to_add)
            if (df_camion.loc[camion]['Ressources']>=q_nodes and check_temps_part(temp,G)==True):
                Q=G.nodes[0]['Camion']['VEHICLE_TOTAL_WEIGHT_KG'][camion]
                assert(q_nodes<=Q),"Certaines ville ont des ressources plus élevés que la capacité de stockage du camion"
                x[camion].append(nodes_to_add)
                df_camion['Ressources'].loc[camion]+=-q_nodes
                i+=1
                pbar.update(1)
                assert(x[camion]==temp)
            else:
                assert(x[camion]!=temp)
                indice_camion+=1
               
            df_ordre=df_ordre.append(dict)
    
       
    for i in x:
        i.append(0)
    ###Assertion pour vérifier que tout fonctionne bien###

    assert(check_constraint(x,G)==True),"Mauvaise initialisation au niveau du temps"
    check_forme(x, G)
    
    ###Affichage de la première solution###
    plotting(x,G)
    return x

def init_gen(G,n_sol,ordre_x,n_car=9):
    """
    

    Parameters
    ----------
    G : Graphe du problème
    n_sol :nombre de solutions souhaitées
    ordre_x :solution du voyageur de commerce.
    n_car : Nombre de camion à prendre en compte dans la solution
        DESCRIPTION. The default is 9.

    Returns
    -------
    X_INIT : Liste de solution x, sachant que x=[x[0],x[1],x[2]...]
    avec x[i] la trajectoire du camion i alias G.nodes[0]['Camion'][i] ,l'ordre étant du camion le plus au moins imposant (donc coûteux).

    """
    ordre_camion=[i for i in range(0,n_car)]
    list_ordre_camion=[ordre_camion.copy()]
    X_INIT=[]
    print("Création des solutions")
    with tqdm(total=n_sol-1) as pbar2:
        while len(X_INIT) < n_sol:
            rd.shuffle(ordre_camion)
            if ordre_camion.copy() not in list_ordre_camion:
                list_ordre_camion.append(ordre_camion.copy())
                x=init(G,n_car,ordre_camion.copy(),ordre_x)
                if x!= False:
                    X_INIT.append(x)
                    pbar2.update(1)
    
    return X_INIT

G=create_G(df_customers,df_vehicles,v)

n_sol=50
X_INIT=init_gen(G,n_sol,ordre_x)
        