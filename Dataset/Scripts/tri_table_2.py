# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:55:59 2022

@author: anton
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import math
import utm
v=50
df= pd.read_excel("2_detail_table_customers.xls")
df=df[df.columns[2:]]
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def check_customer_intervals(df):
    """
    

    Parameters
    ----------
    df : Dataframe customers

    Returns
    -------
    Default : List of customers where intervalls of delivery aren't of the same features

    """
    Default=[]
    customers=pd.unique(df["CUSTOMER_CODE"])
    for custom in customers : 
        m=df[df["CUSTOMER_CODE"]==custom]
        int_min=pd.unique(m["CUSTOMER_TIME_WINDOW_FROM_MIN"])
        int_max=pd.unique(m["CUSTOMER_TIME_WINDOW_TO_MIN"])
        if( len((int_max))!=1 or len(int_min)!=1):
            Default.append(custom)
    return Default

def sum_customer_commands(df):
    new_df=pd.DataFrame(columns=df.columns)
    customers=pd.unique(df["CUSTOMER_CODE"])
    for i in tqdm(range(0,len(customers))) : 
        custom=customers[i]
        m=df[df["CUSTOMER_CODE"]==custom]
        dict=m[m.columns[:7]].head(1).to_dict('records')
        dict_2=m[m.columns[7:]].sum(axis=0).to_dict()
        dict[0].update(dict_2)
        new_df=new_df.append(dict)
    return new_df
        
def smooth_customers_commands(df):
    q75,q25 = np.percentile(df["TOTAL_WEIGHT_KG"],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    filter = (df["TOTAL_WEIGHT_KG"] >= min) & (df["TOTAL_WEIGHT_KG"]<=max) #ce filtre garde l'information sur les index
    df=df[filter]
    return df

def get_distance(z_1,z_2):
    x_1,x_2,y_1,y_2=z_1[0],z_2[0],z_1[1],z_2[1]
    d=math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
    return d/1000

fig, axs = plt.subplots(2, 2)

Default=check_customer_intervals(df)
odd=df[df["CUSTOMER_CODE"].isin(Default)] #Récupère les élèments aux index souhaités.
pd.set_option('display.max_columns', None)
print("Voici les commandes problématiques \n",odd)
if len(Default)!=0:
    index_to_remove=odd[odd["CUSTOMER_TIME_WINDOW_FROM_MIN"]!=int(odd["CUSTOMER_TIME_WINDOW_FROM_MIN"].head(1))].index
    print("On nettoie le set")
    df=df.drop(index=index_to_remove)
assert(len(check_customer_intervals(df))==0)
print("Le tableur est revisité , on somme maintenant toute les commandes de chaque client")
new_df=sum_customer_commands(df)
new_df['pos']=[(new_df['CUSTOMER_LATITUDE'].iloc[i],new_df['CUSTOMER_LONGITUDE'].iloc[i]) for i in range(0,len(new_df))]


old_eff=len(new_df)
axs[0,0].boxplot(new_df["TOTAL_WEIGHT_KG"])
axs[0,0].set_title("Avant lissage")
assert(len(check_customer_intervals(new_df))==0)
new_df=smooth_customers_commands(new_df)
"""
###Distances###
print("On procède au calcul des distances entre chaque point \n")
df_distance=pd.DataFrame(columns=['from','to','distance(km)','time'])
for i in tqdm(range(0,len(new_df))):
    for j in range(0,len(new_df)):
        if i!=j:
            departure=new_df['CUSTOMER_CODE'].iloc[i]
            arrival=new_df['CUSTOMER_CODE'].iloc[j]
            z_1=new_df['pos'].iloc[i]
            z_2=new_df['pos'].iloc[j]
            dict={'from':departure,'to':arrival,'distance(km)':get_distance(z_1,z_2)}
            dict['time']=(dict['distance(km)']/v)*60
            df_distance=df_distance.append([dict])
print(df_distance)
axs[1,0].boxplot(df_distance["time"])

"""

axs[0,1].boxplot(new_df["TOTAL_WEIGHT_KG"])
axs[0,1].set_title("Après lissage")
axs[1,0].set_title("Temps")
new_eff=len(new_df)
print("Le ratio de conservation : ",new_eff/old_eff)
new_df=new_df[new_df.columns[0:-1]]
new_df=new_df.drop(['NUMBER_OF_ARTICLES','TOTAL_VOLUME_M3'],axis=1)
new_df['pos']=[utm.from_latlon(new_df['CUSTOMER_LATITUDE'].iloc[i],new_df['CUSTOMER_LONGITUDE'].iloc[i])[:2]  for i in range(0,len(new_df))]

new_df.to_excel("table_2_customers_features.xls")
fig.suptitle("Diagramme boîte du poids des commandes et des distance")
