""" Import librairies """
import pandas as pd
from os.path import join

""" Import utilities """
from Utility.common import create_graph, set_root_dir

set_root_dir()


class Database:
    Graph = []

    def __init__(self, vehicle_speed=50):
        customer_path = join('Dataset', 'Tables', 'table_2_customers_features.csv')
        vehicles_path = join('Dataset', 'Tables', 'table_3_cars_features.csv')

        df_customers = pd.read_csv(customer_path)
        # TODO: remove once generated solutions in .pkl file have the same number of customers
        #  as in the dataset
        df_customers.drop(index=[len(df_customers) - 1], inplace=True)
        self.df_customers = df_customers

        self.df_vehicles = pd.read_csv(vehicles_path)
        self.vehicle_speed = vehicle_speed

        self.Graph = create_graph(self.df_customers, self.df_vehicles, self.vehicle_speed)
        nbr_of_customer = len(self.df_customers)
        nbr_of_vehicles = len(self.df_vehicles)

        message = 'Loaded {} customers and {} vehicles from the dataset'
        print(message.format(nbr_of_customer, nbr_of_vehicles))


