import random as rd
import numpy as np
import matplotlib.pyplot as ppt
from math import pi, cos, sqrt, asin


class GeneticAlgorithm:
    PROBA_CROSSING = 0.8
    TOURNAMENT_SIZE = 3
    PENALTY: int = 2
    POPULATION_SIZE = 50
    MAX_ITERATION = 50

    def __init__(self, customers, depots, vehicles):
        nbr_of_sites = len(customers)
        cost_matrix = np.zeros((nbr_of_sites, nbr_of_sites))

        for i in range(nbr_of_sites):
            customer_i = customers[i]

            for j in range(nbr_of_sites):
                customer_j = customers[j]
                lat_i = float(customer_i.CUSTOMER_LATITUDE)
                lon_i = float(customer_i.CUSTOMER_LONGITUDE)
                lat_j = float(customer_j.CUSTOMER_LATITUDE)
                lon_j = float(customer_j.CUSTOMER_LONGITUDE)

                cost_matrix[i, j] = self.distance(lat_i, lon_i, lat_j, lon_j)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_SITES = nbr_of_sites
        self.PROBA_MUTATION = 1 / nbr_of_sites
        self.Customers = customers
        self.Vehicles = vehicles[0]

    def main(self):
        # Initialization
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
        point = rd.randint(0, self.NBR_OF_SITES)

        brother = father[:point]
        sister = mother[:point]

        for index in range(self.NBR_OF_SITES):
            if mother[index] not in brother:
                brother.append(mother[index])

            if father[index] not in sister:
                sister.append(father[index])

        return [brother, sister]

    """
    Apply a mutation to a configuration by inverting 2 sites

    @param{list} individual - list of all the visited sites from the first to the last visited
    @return{list} new configuration with the mutation
    """

    def mutate(self, individual: list) -> list:
        n = self.NBR_OF_SITES - 1
        i = rd.randint(0, n - 1)
        j = rd.randint(i + 1, n)

        site_i = individual[i]
        site_j = individual[j]

        result = individual[:i] + [site_j] + individual[i + 1: j] + [site_i] + individual[j + 1:]

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

        for index in range(self.NBR_OF_SITES - 1):
            site_from = individual[index]
            site_to = individual[index + 1]

            travel_cost += self.COST_MATRIX[site_from, site_to]

        return vehicle_cost + travel_cost

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    @param{int} size - the number of individuals in the population
    @return{list} the population
    """

    def generate_population(self) -> list:
        population = []
        seed = range(self.NBR_OF_SITES)

        weight = 0
        volume = 0

        sub_route = []

        for index_customer in range(self.POPULATION_SIZE):
            customer = self.Customers[index_customer]

            weight += customer.TOTAL_WEIGHT_KG
            volume += customer.TOTAL_VOLUME_M3

            if (weight > self.Vehicles.VEHICLE_TOTAL_WEIGHT_KG
                    or volume > self.Vehicles.VEHICLE_TOTAL_VOLUME_M3):
                individual.append(sub_route)
                weight = 0
                volume = 0

            individual = rd.sample(seed, k=self.NBR_OF_SITES)
            population.append(individual)

        return population

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
