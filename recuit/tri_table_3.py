# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:47:13 2022

@author: anton
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
df= pd.read_excel("3_detail_table_vehicles.xls")

def check_vehicles_features(df):
    """
    

    Parameters
    ----------
    df : Dataframe customers

    Returns
    -------
    Default : List of vehicles where features are undefined

    """
    Default_1=[]
    Default_2=[]
    Default_3=[]
    customers=pd.unique(df["VEHICLE_CODE"])
    for custom in customers : 
        m=df[df["VEHICLE_CODE"]==custom]
        coeff= pd.unique(m["VEHICLE_VARIABLE_COST_KM"])
        weight=pd.unique(m["VEHICLE_TOTAL_WEIGHT_KG"])
        volume=pd.unique(m["VEHICLE_TOTAL_VOLUME_M3"])
        int_min=pd.unique(m["VEHICLE_AVAILABLE_TIME_FROM_MIN"])
        int_max=pd.unique(m["VEHICLE_AVAILABLE_TIME_TO_MIN"])
        if( len((int_max))!=1 or len(int_min)!=1):
            Default_1.append(custom)
        if( len((volume))!=1):
            Default_2.append(custom)
        if( len((weight))!=1 or len(coeff) !=1) :
            Default_3.append(custom)
    return Default_1,Default_2,Default_3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
Default_1,Default_2,Default_3=check_vehicles_features(df)
odd=df[df["VEHICLE_CODE"].isin(Default_1)]
print("Voici les commandes problématiques \n",odd)
###Pas de problème au niveau des capacités###
df=df[["VEHICLE_CODE","VEHICLE_TOTAL_WEIGHT_KG","VEHICLE_VARIABLE_COST_KM"]]

def prunning_vehicules_features(df):
    new_df=pd.DataFrame(columns=df.columns)
    customers=pd.unique(df["VEHICLE_CODE"])
    for i in tqdm(range(0,len(customers))) : 
        custom=customers[i]
        m=df[df["VEHICLE_CODE"]==custom]
        dict=m.head(1).to_dict('records')
        new_df=new_df.append(dict)
    return new_df
new_df= prunning_vehicules_features(df)
assert(len(pd.unique(new_df["VEHICLE_CODE"]))==len(new_df))
plt.boxplot(new_df["VEHICLE_TOTAL_WEIGHT_KG"])
plt.title("Diagramme boîte des capacités de nos camions")
new_df=new_df.sort_values(by="VEHICLE_TOTAL_WEIGHT_KG",ascending=False)
new_df.to_excel("table_3_cars_features.xls")
    