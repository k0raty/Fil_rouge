import numpy as np
import os
import pandas as pd
import distances as dt
import matplotlib.pyplot as plt
import random as rd

k_dt=1.1515247508025006

#IMPORTATION DES DONNEES CLIENT
df=pd.read_excel(os.getcwd()+'\\data_PTV_Fil_rouge\\2_detail_table_customers.xls')

#condition sur les plages horaires
#df[(df['CUSTOMER_TIME_WINDOW_TO_MIN']>=480) & (df['CUSTOMER_TIME_WINDOW_TO_MIN']<=840)]

def livraison_un_camion(df,all_treated_lines):
    Q=5000 #en kg
    weight=0
    positions=[]
    total_weight_obtained=False

    #la liste tabou contient toutes les lignes déjà traitées
    L_lines=[]#liste des lignes traitées lors de cette tournée
    L_customers=[]#c'est une liste informative pour savoir le chemin qu'a pris le camion sans avoir à reparcourir chaque ligne pour comprendre quel client correspondait à quelle ligne
    #choix_premier_client
    depot_table = np.array([[5000, 43.37391833, 17.60171712]])
    depot_df = pd.DataFrame(depot_table, index = ['DEPOT DATAFRAME'], columns = ['CUSTOMER_CODE', 'CUSTOMER_LATITUDE', 'CUSTOMER_LONGITUDE'])
    depot_line=depot_df.iloc[0]
    
    depot_lat=depot_line['CUSTOMER_LATITUDE']
    depot_lon=depot_line['CUSTOMER_LONGITUDE']
    
    act_lat=depot_lat
    act_lon=depot_lon
    act_line=depot_line
    
    positions.append((act_lat,act_lon))
    #on est au client client_actuel
    while not total_weight_obtained:
        customer_code_min=dt.client_le_plus_proche(act_line,df,all_treated_lines)   #Client le plus proche du dépot qui n'a pas encore été livré
        customer_lines_numbers=df.loc[df['CUSTOMER_CODE']==customer_code_min].index #Toutes les commandes du client le plus proche
        N=len(customer_lines_numbers)   #nb de commandes du client
        
        for i,number in enumerate(customer_lines_numbers):
            line=df.iloc[number] #on regarde chaque ligne et si il nous reste assez de poids pour livrer la commande correspondante
            customer_code=line['CUSTOMER_CODE']
            package_weight=line['TOTAL_WEIGHT_KG']
            if weight+package_weight<Q:                                         #et on supprime la commande du dataframe général df
                weight+=package_weight
                
                all_treated_lines.append(number)     #On ajoute la ligne à la liste des lignes 
                if len(L_lines)>=1:
                    last_customer_code=df.iloc[L_lines[len(L_lines)-1][0]]['CUSTOMER_CODE']
                    if last_customer_code==customer_code:
                        L_lines[len(L_lines)-1].append(number)
                    else:
                        L_lines.append([number])
                else:
                    L_lines.append([number])
                lat=line['CUSTOMER_LATITUDE']
                lon=line['CUSTOMER_LONGITUDE']
                positions.append((lat,lon))
                act_lat,act_lon=lat,lon
                
                if customer_code not in L_customers:    #On évite les doublons dans la liste des clients livrés
                    L_customers.append(customer_code)
                act_line=df.iloc[number]    #On crée un dataframe avec seulement la ligne de la commande en question

                
            else:
                if i==N-1:
                    total_weight_obtained=True 
                        #si la plus légère commande de ce client ne peut pas être livré on considère que le camion est plein
        if all_treated_lines!=[]:
            act_line=df.iloc[all_treated_lines[len(all_treated_lines)-1]]
            
    positions.append((depot_lat,depot_lon))
    
    return(all_treated_lines,L_customers,L_lines,positions) #L_customers,weight

def livraison_plusieurs_camions():
    df=pd.read_excel(os.getcwd()+'\\data_PTV_Fil_rouge\\2_detail_table_customers.xls')
    #df=df.drop(df.loc[df['TOTAL_WEIGHT_KG']>=2000].index)
    
    all_treated_lines=[]
    camions_positions=[[] for k in range(8)] #la liste contient les positions par lesquelles passe le camion
    camions_clients=[[] for k in range(8)]
    camions_lines=[[] for k in range(8)]
    colors=['b','g','r','c','m','y','k','b','b']
    n=0
    while len(all_treated_lines)<df.shape[0]-6:   #df.shape[0]-6
        n+=1
        for i in range(8):
            #print('Itération',n,'camion',i,'.\nIl y a',len(all_treated_lines),len(list(set(all_treated_lines))),'commandes servies.')
            all_treated_lines,L_customers,L_lines,positions=livraison_un_camion(df,all_treated_lines)
            #print(positions)
            camions_positions[i]+=positions
            for k in range(len(L_lines)):
                camions_lines[i].append(L_lines[k])
            camions_clients[i]+=L_customers
            if len(all_treated_lines)>df.shape[0]-6: #df.shape[0]-6
                break
        #print(camions_positions)
        
    for i in range(8):
        p=0
        for j in range(len(camions_positions[i])-1):
            plt.scatter(43.37391833,17.60171712,marker='o',c='r',linewidths=2)
            plt.text(43.37391833+0.001,17.60171712+0.001, 'Depôt',color='w',backgroundcolor='r', fontsize=9)
            (xj,yj)=camions_positions[i][j]
            (xjj,yjj)=camions_positions[i][j+1]
            plt.plot([xj,xjj],[yj,yjj],linewidth='0.3',color=colors[i])
            if (xj,yj)!=(43.37391833,17.60171712):
                p+=1
                plt.scatter(xj,yj,marker='+',c=colors[i],linewidths=1)
                #plt.text(xj,yj, str(p), fontsize=7)
        """        
        print('Trajectoires du camion '+str(i+1)+' en couleur '+colors[i]+'.')
        print(camions_positions[i])
        print('Il passe par '+str(len(camions_clients[i]))+' clients.\n')
        print(camions_clients[i])
        """
    #camions[i]
    plt.xlabel('Lattitude')
    plt.ylabel('Longitude')
    plt.title('Itinéraires des camions pour la livraison des clients')
    plt.show()
    return(camions_clients,camions_lines)

def livraison_voisine(df,camions_lines,k,p,i,j):
    
    new_line_k=camions_lines[k][:i]
    new_line_k+=camions_lines[p][j:]
    new_line_p=camions_lines[p][:j]
    new_line_p+=camions_lines[k][i:]
    
    camions_lines[k]=new_line_k
    camions_lines[p]=new_line_p
    
    #print(camions_lines[k],'\n\n',camions_lines[p],'\n\n',new_line_k,'\n\n',new_line_p)
    
    return(camions_lines)
                
def calcul_positions(df,camions_lines):
    camions_unseparated_lines=[[] for k in range(8)]
    for k in range(8):
        for j in range(len(camions_lines[k])):
            for i in range(len(camions_lines[k][j])):
                camions_unseparated_lines[k].append(camions_lines[k][j][i])
    
    camions_positions=[[] for k in range(8)]

    depot_table = np.array([[5000, 43.37391833, 17.60171712]])
    depot_df = pd.DataFrame(depot_table, index = ['DEPOT DATAFRAME'], columns = ['CUSTOMER_CODE', 'CUSTOMER_LATITUDE', 'CUSTOMER_LONGITUDE'])
    depot_line=depot_df.iloc[0]
    
    depot_lat=depot_line['CUSTOMER_LATITUDE']
    depot_lon=depot_line['CUSTOMER_LONGITUDE']
    for k in range(8):
        j=0
        Q=5000
        weight=0
        while j<len(camions_unseparated_lines[k]):
            if weight==0:
                camions_positions[k].append((depot_lat,depot_lon))
                weight+=0.000000000000001
            else:
                line=df.iloc[camions_unseparated_lines[k][j]] #on regarde chaque ligne et si il nous reste assez de poids pour livrer la commande correspondante
                package_weight=line['TOTAL_WEIGHT_KG']
                if weight+package_weight<=Q:
                    weight+=package_weight
                    lat=line['CUSTOMER_LATITUDE']
                    lon=line['CUSTOMER_LONGITUDE']
                    camions_positions[k].append((lat,lon))
                    j+=1
                else:
                    weight=0
                    camions_positions[k].append((depot_lat,depot_lon))
    return(camions_positions)

def cout_livraison(df,camions_lines):
    camions_positions=calcul_positions(df,camions_lines)
    D=0
    
    K=8
    w=0
    
    for i in range(8):
        dist_camion_i=0
        for j in range(len(camions_positions[i])-1):
            (lat1,lon1)=camions_positions[i][j]
            (lat2,lon2)=camions_positions[i][j+1]
            dist_camion_i+=dt.distance(lat1, lon1, lat2, lon2)
        print('Camion '+str(i)+':',dist_camion_i)
        D+=dist_camion_i
    return(w*K+round(D))

def itération_tabou(df,camions_lines):
    M=[0,1,2,3,4,5,6,7]
    k=0
    lines=camions_lines
    solutions_voisines_k=[0 for i in range(8)]
    cout_solutions=[0 for i in range(8)]
    cout_solutions[k]=cout_livraison(df,camions_lines)
    
    solutions_voisines_k[k]=camions_lines
    
    
    for p in M:
        if p!=k:
            nb_clients_camion_k=len(camions_lines[k])
            nb_clients_camion_p=len(camions_lines[p])
            
            m=min(nb_clients_camion_k,nb_clients_camion_p)
            
            n=int(m/2)
            i=rd.randint(n,2*n-1)
            j=rd.randint(i,2*n-1)
            
            solution_p_lines=livraison_voisine(df,camions_lines,k,p,i,j)
            
            solutions_voisines_k[p]=solution_p_lines
            
            cout_solutions[p]=cout_livraison(df,solution_p_lines)
    #recherche du minimum:
    indice_min=0
    cout_min=cout_solutions[0]
    for i in range(len(M)):
        if cout_solutions[i]<=cout_min:
            indice_min=i
    return(solutions_voisines_k[i])

def display_positions(camions_positions):  
    None