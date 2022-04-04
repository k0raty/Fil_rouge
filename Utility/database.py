""" Import librairies """
from csv import reader
import os

""" Import utilities """
from Utility.customer import Customer
from Utility.vehicle import Vehicle
from Utility.depot import Depot

os.chdir('..')


class Database:
    Customers = []
    Vehicles = []
    Depots = []

    def __init__(self):
        self.CUSTOMER_PATH = os.path.join('Dataset', 'Tables', 'table_customers_v1.csv')
        self.DEPOT_PATH = os.path.join('Dataset', 'Tables', 'table_depots.csv')
        self.VEHICLE_PATH = os.path.join('Dataset', 'Tables', 'table_vehicles.csv')

        self.refresh()

    def refresh(self):
        print('current path', os.getcwd())
        self.Customers = []
        self.Vehicles = []
        self.Depots = []

        with open(self.CUSTOMER_PATH, newline='') as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            next(csv_reader)
            index_customer = 1

            for row in csv_reader:
                new_customer = Customer(row, index_customer)
                index_customer += 1
                self.Customers.append(new_customer)

        with open(self.DEPOT_PATH, newline='') as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            next(csv_reader)

            for row in csv_reader:
                new_depot = Depot(row)
                self.Depots.append(new_depot)

        with open(self.VEHICLE_PATH, newline='') as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            next(csv_reader)

            for row in csv_reader:
                new_vehicle = Vehicle(row)
                self.Vehicles.append(new_vehicle)

        message = 'Read {} customers, {} depots, and {} vehicles from the database'
        print(message.format(len(self.Customers), len(self.Depots), len(self.Vehicles)))
