import random as rd
import numpy as np
import matplotlib.pyplot as ppt
from math import pi, cos, sqrt, asin
from database import Database


class GeneticAlgorithm:
    PROBA_CROSSING = 0.8
    TOURNAMENT_SIZE = 3
    PENALTY: int = 2
    POPULATION_SIZE = 50
    MAX_ITERATION = 50

    def __init__(self):
        self.Database = Database()

        nbr_of_sites = len(self.Database.Customers)
        cost_matrix = np.zeros((nbr_of_sites, nbr_of_sites))

        for i in range(nbr_of_sites):
            customer_i = self.Database.Customers[i]

            for j in range(nbr_of_sites):
                customer_j = self.Database.Customers[j]
                lat_i = float(customer_i.CUSTOMER_LATITUDE)
                lon_i = float(customer_i.CUSTOMER_LONGITUDE)
                lat_j = float(customer_j.CUSTOMER_LATITUDE)
                lon_j = float(customer_j.CUSTOMER_LONGITUDE)

                cost_matrix[i, j] = self.distance(lat_i, lon_i, lat_j, lon_j)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_SITES = nbr_of_sites
        self.PROBA_MUTATION = 1 / nbr_of_sites

        self.Customers = self.Database.Customers
        self.Depots = self.Database.Depots
        self.Vehicles = self.Database.Vehicles[0]

    def main(self):
        iteration = 0
        population = self.generate_population()
        fitness_history = []

        while iteration < self.MAX_ITERATION and self.fitness_change(fitness_history):
            iteration += 1

            """ Choose the individuals that survive from the previous generation """
            population_survivors = self.stochastic_selection(population)  # other methods available
            population = population_survivors
            rd.shuffle(population)

            """ Cross the survivors between them and keep their children """
            population_crossed = []

            for index in range(0, self.POPULATION_SIZE, 2):
                father = population[index]
                mother = population[index + 1]

                if rd.random() < self.PROBA_CROSSING:
                    children = self.pmx_cross_over(father, mother)  # other methods available
                    population_crossed.append(children[0])
                    population_crossed.append(children[1])
                else:
                    population_crossed.append(father)
                    population_crossed.append(mother)

            """ Apply a mutation to some individuals """
            population_mutated = []

            for index in range(self.POPULATION_SIZE):
                element = population_crossed[index]

                if rd.random() < self.PROBA_MUTATION:
                    element_mutated = self.mutate(element)
                    population_mutated.append(element_mutated)
                else:
                    population_mutated.append(element)

            population = population_mutated

            """ Display each generation properties """
            fitness_sum = 0

            for index in range(self.POPULATION_SIZE):
                individual = population[index]
                fitness_sum += self.fitness(individual)

            fitness_history.append(fitness_sum)
            print('Iteration {}, fitness sum {}'.format(iteration, fitness_sum))

        self.draw_fitness(iteration, fitness_history)

    """
    Use a stochastic method ("Technique de la roulette") to select individuals from the current generation
    to build the next generation

    @param{list} population - current population
    @return{list} selected population
    """

    def stochastic_selection(self, population: list) -> list:
        population_survivors = []

        while len(population_survivors) != self.POPULATION_SIZE:
            contestants = rd.choices(population, k=self.TOURNAMENT_SIZE)
            contestants = sorted(contestants, key=lambda x: self.fitness(x))
            winner = contestants[0]
            population_survivors.append(winner)

        return population_survivors

    """
    Use a deterministic method to select individuals from the current generation
    to build the next generation
    Sort the population by their fitness score
    Then remove the weakest
    Finally randomly select individuals to generate the new population of the same size

    @param{list} population - current population
    @return{list} selected population
    """

    def determinist_selection(self, population: list) -> list:
        population_survivors = sorted(population, key=lambda x: self.fitness(x))
        population_survivors = population_survivors[:self.POPULATION_SIZE - self.TOURNAMENT_SIZE]
        population_survivors = rd.choices(population_survivors, k=self.POPULATION_SIZE)
        return population_survivors

    """
    Apply a crossover between the father and mother configuration, using a pmx-like crossover

    @param{list} father - one configuration
    @param{list} mother - an other configuration
    @return{list} return two new configurations created from the 2 parents
    """

    def pmx_cross_over(self, father, mother):
        """ first we remove all previous depots """
        new_father = []

        for index_site in range(len(father)):
            site = father[index_site]

            if type(site).__name__ != 'Depot':
                new_father.append(site)

        new_mother = []

        for index_site in range(len(mother)):
            site = mother[index_site]

            if type(site).__name__ != 'Depot':
                new_mother.append(site)

        point = rd.randint(0, self.NBR_OF_SITES)

        brother = new_father[:point]
        sister = new_mother[:point]

        for index in range(self.NBR_OF_SITES):
            if new_mother[index] not in brother:
                brother.append(new_mother[index])

            if new_father[index] not in sister:
                sister.append(new_father[index])

        brother = self.valid_depot_localization(brother)
        sister = self.valid_depot_localization(sister)

        return [brother, sister]

    """
    Apply a mutation to a configuration by inverting 2 sites

    @param{list} individual - list of all the visited sites from the first to the last visited
    @return{list} new configuration with the mutation
    """

    def mutate(self, individual: list) -> list:
        """ first we remove all previous depots """
        individual_len = len(individual)
        filtered = []

        for index_site in range(individual_len):
            site = individual[index_site]

            if type(site).__name__ != 'Depot':
                filtered.append(site)

        n = self.NBR_OF_SITES - 1
        i = rd.randint(0, n - 1)
        j = rd.randint(i + 1, n)

        site_i = filtered[i]
        site_j = filtered[j]

        result = filtered[:i] + [site_j] + filtered[i + 1: j] + [site_i] + filtered[j + 1:]
        result = self.valid_depot_localization(result)

        return result

    """
    Evaluate the cost of visiting sites with this configuration, depending on the number of cars
    and the cost from one site to the next one

    @param{list} individual - list of all the visited sites from the first to the last visited
    @return{int} value of the cost of this configuration
    """

    def fitness(self, individual: list) -> int:
        vehicle_cost = self.PENALTY * self.nbr_of_vehicles(individual)

        travel_cost = 0

        individual_len = len(individual)

        for index in range(individual_len - 1):
            site_from = individual[index]
            site_to = individual[index + 1]

            class_name_from = type(site_from).__name__
            class_name_to = type(site_to).__name__

            if class_name_from != 'Depot' and class_name_to != 'Depot':
                travel_cost += self.COST_MATRIX[int(site_from.CUSTOMER_ID), int(site_to.CUSTOMER_ID)]
            elif class_name_from == 'Depot':
                distance = self.distance(
                    float(site_to.CUSTOMER_LATITUDE),
                    float(site_to.CUSTOMER_LONGITUDE),
                    float(site_from.DEPOT_LATITUDE),
                    float(site_from.DEPOT_LONGITUDE),
                )
                travel_cost += distance
            elif class_name_to != 'Depot':
                distance = self.distance(
                    float(site_from.CUSTOMER_LATITUDE),
                    float(site_from.CUSTOMER_LONGITUDE),
                    float(site_to.DEPOT_LATITUDE),
                    float(site_to.DEPOT_LONGITUDE),
                )
                travel_cost += distance

        return vehicle_cost + travel_cost

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    @param{int} size - the number of individuals in the population
    @return{list} the population
    """

    def generate_population(self) -> list:
        population = []
        seed = range(self.NBR_OF_SITES)

        for index_individual in range(self.POPULATION_SIZE):
            """ first we generate random roads between customers """
            order = rd.sample(seed, k=self.NBR_OF_SITES)
            individual = [self.Customers[i] for i in order]
            """ then we add a depot when needed"""
            individual = self.valid_depot_localization(individual)
            population.append(individual)

        return population

    def valid_depot_localization(self, individual):
        weight = 0
        volume = 0
        weight_vehicle = float(self.Vehicles.VEHICLE_TOTAL_WEIGHT_KG)
        volume_vehicle = float(self.Vehicles.VEHICLE_TOTAL_VOLUME_M3)
        result = []

        for customer in individual:
            weight += float(customer.TOTAL_WEIGHT_KG)
            volume += float(customer.TOTAL_VOLUME_M3)

            if weight > weight_vehicle or volume > volume_vehicle:
                closest_depot = self.find_closest_depot(customer)
                result.append(closest_depot)
                weight = 0
                volume = 0

            result.append(customer)

        return result

    def find_closest_depot(self, customer):
        nbr_of_depots = len(self.Depots)
        distance_min = 0
        closest_depot = self.Depots[0]

        for index_depot in range(nbr_of_depots):
            depot = self.Depots[index_depot]
            distance = self.distance(
                float(customer.CUSTOMER_LATITUDE),
                float(customer.CUSTOMER_LONGITUDE),
                float(depot.DEPOT_LATITUDE),
                float(depot.DEPOT_LONGITUDE),
            )

            if distance_min == 0:
                distance_min = distance
            elif distance < distance_min:
                closest_depot = depot
                distance_min = distance

        return closest_depot

    @staticmethod
    def nbr_of_vehicles(individual: list) -> int:
        return len(individual)

    @staticmethod
    def distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
        deg_2_rad = pi / 180
        a = 0.5 - cos((lat_2 - lat_1) * deg_2_rad) / 2
        b = cos(lat_1 * deg_2_rad) * cos(lat_2 * deg_2_rad) * (1 - cos((lon_2 - lon_1) * deg_2_rad)) / 2
        r_earth = 6371

        return 2 * r_earth * asin(sqrt(a + b))

    @staticmethod
    def fitness_change(history):
        if len(history) < 5:
            return True
        if history[-1] == history[-2] and history[-1] == history[-3]:
            return False
        return True

    @staticmethod
    def draw_fitness(iteration, history):
        ppt.close()
        ppt.plot(range(iteration), history)
        ppt.show()
