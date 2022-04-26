""" Import librairies """
import pandas as pd
from os.path import join

""" Import utilities """
from Utility.common import create_G, set_root_dir


set_root_dir()


class Database:
    Customers = []
    Vehicles = []
    Depots = []

    def __init__(self, speed=20):
        customer_path = join('Dataset', 'Tables', 'table_2_customers_features.xls')
        vehicles_path = join('Dataset', 'Tables', 'table_3_cars_features.xls')

        self.df_customers = pd.read_excel(customer_path)
        self.df_vehicles = pd.read_excel(vehicles_path)
        self.df_vehicles=self.df_vehicles.drop(['Unnamed: 0'],axis=1)
        self.df_customers=self.df_customers.drop(['Unnamed: 0'],axis=1)
        self.speed=speed
        self.refresh()

    def refresh(self):
        
        self.graph=create_G(self.df_customers,self.df_vehicles,self.speed)
        number_customers=len(self.graph)
        self.Customers = [self.graph.nodes[i] for i in range(1,number_customers)]
        self.Vehicles = self.graph.nodes[0]['Camion']
        set_to_keep=set(self.graph.nodes[0].keys())-set(['Camion'])
        self.Depots = [{ your_key: self.graph.nodes[0][your_key] for your_key in set_to_keep }]
      
        message = 'Read {} customers, {} depots, and {} vehicles from the database'
        print(message.format(len(self.Customers), len(self.Depots), len(self.Vehicles['VEHICLE_CODE'])))
