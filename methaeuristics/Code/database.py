from csv import reader
import xlrd as xl
from customer import Customer
from vehicle import Vehicle
from depot import Depot


class Database:
    CUSTOMER_PATH = '../../data_PTV_Fil_rouge/table_customers.csv'
    DEPOT_PATH = '../../data_PTV_Fil_rouge/table_depots.csv'
    VEHICLE_PATH = '../../data_PTV_Fil_rouge/table_vehicles.csv'

    Customers = []
    Vehicles = []
    Depots = []

    def __init__(self):
        with open(self.CUSTOMER_PATH, newline='') as csv_file:
            csv_reader = reader(csv_file, delimiter=',')
            next(csv_reader)

            for row in csv_reader:
                new_customer = Customer(row)
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
