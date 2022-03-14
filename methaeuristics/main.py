from genetic_algorithm import GeneticAlgorithm
from database import Database


class Main:
    def __init__(self):
        self.Database = Database

        self.genetic_algorithm = GeneticAlgorithm(
            self.Database.Customers,
            self.Database.Depots,
            self.Database.Vehicles,
        )
