""" Import librairies """
import random as rd
from math import floor
from seaborn import color_palette
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import time
import os

""" Import utilities """
from Utility.database import Database
from Utility.common import compute_cost_matrix, distance


os.chdir(os.path.join('..', '..'))

class GeneticAlgorithm:
    PROBA_CROSSING: float = 0.8
    TOURNAMENT_SIZE: int = 3
    PENALTY: int = 2
    POPULATION_SIZE: int = 50
    MAX_ITERATION: int = 50

    def __init__(self, customers=None, depots=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depots = database.Depots
            cost_matrix = compute_cost_matrix(customers)

        self.solution = None

        nbr_of_sites = len(customers)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_SITES = nbr_of_sites
        self.PROBA_MUTATION = 1 / nbr_of_sites

        self.Customers = customers
        self.Depots = depots
        self.Vehicles = vehicles[0]

    """
    Run the GeneticAlgorithm
    """
    def main(self, initial_solution=None):
        plt.close('all')
        timestamp = floor(time.time())
        path = os.path.join('..', 'Graphs', 'test_{}'.format(timestamp))
        os.mkdir(path)

        iteration = 0
        population = self.generate_population(initial_solution)
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

            sorted_population = sorted(population, key=lambda x: self.fitness(x))
            solution = sorted_population[0]

            self.solution = self.adapt_solution_format(solution)

            filepath = os.path.join(path, 'img_{}.png'.format(iteration))
            self.draw_solution(solution, filepath)

        self.draw_fitness(iteration, fitness_history, filepath)

        return self.solution, self.fitness(self.solution)

    """
    Use a stochastic method ("Technique de la roulette") to select individuals from the current generation
    to build the next generation

    :param population: list - current population
    :return population_survivors: list - selected population
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

        brother = self.add_depot(brother)
        sister = self.add_depot(sister)

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
        result = self.add_depot(result)

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

            elif class_name_from == 'Depot' and class_name_to == 'Depot':
                continue

            elif class_name_from == 'Depot':
                travel_cost += distance(
                    float(site_to.CUSTOMER_LATITUDE),
                    float(site_to.CUSTOMER_LONGITUDE),
                    float(site_from.DEPOT_LATITUDE),
                    float(site_from.DEPOT_LONGITUDE),
                )

            elif class_name_to != 'Depot':
                travel_cost += distance(
                    float(site_from.CUSTOMER_LATITUDE),
                    float(site_from.CUSTOMER_LONGITUDE),
                    float(site_to.DEPOT_LATITUDE),
                    float(site_to.DEPOT_LONGITUDE),
                )

        return vehicle_cost + travel_cost

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    :param size: int - the number of individuals in the population
    :param solution: list - an initial solution to the problem
    :return population: list - the population
    """
    def generate_population(self, solution=None) -> list:
        population = []
        population_size = self.POPULATION_SIZE
        seed = range(self.NBR_OF_SITES)

        if solution is not None:
            initial_solution = []

            for sub_road in solution:
                initial_solution.append(*[customer for customer in sub_road if not 0])

            population.append(initial_solution)
            population_size -= 1

        for index_individual in range(population_size):
            """ first we generate random roads between customers """
            order = rd.sample(seed, k=self.NBR_OF_SITES)
            individual = [self.Customers[i] for i in order]

            """ then we add a depot when needed"""
            individual = self.add_depot(individual)
            population.append(individual)

        return population

    """
    Go through the list of sites and add a return to depot everytime the car is empty
    
    :param individual: list - list of all the visited sites from the first to the last visited
    """
    def add_depot(self, individual):
        weight = 0
        volume = 0

        weight_vehicle = float(self.Vehicles.VEHICLE_TOTAL_WEIGHT_KG)
        volume_vehicle = float(self.Vehicles.VEHICLE_TOTAL_VOLUME_M3)

        result = [self.Depots[0]]

        for customer in individual:
            weight += float(customer.TOTAL_WEIGHT_KG)
            volume += float(customer.TOTAL_VOLUME_M3)

            if weight > weight_vehicle or volume > volume_vehicle:
                result.append(self.Depots[0])
                weight = 0
                volume = 0

            result.append(customer)

        result.append(self.Depots[0])

        return result

    """
    Draw the graph showing all the customers sites and the depots, with the road taken to go through them
    
    :param solution: list - an individual of the population with the lowest fitness score
    """
    def draw_solution(self, solution, filepath):
        plt.figure(figsize=[25, 15])

        vehicle_working = 0

        colors = color_palette(n_colors=self.NBR_OF_VEHICLES)

        latitudes = []
        longitudes = []

        for index_site in range(len(solution)):
            site = solution[index_site]

            if type(site).__name__ == 'Customer':
                latitudes.append(site.CUSTOMER_LATITUDE)
                longitudes.append(site.CUSTOMER_LONGITUDE)

                plt.annotate("{}".format(index_site),
                             (site.CUSTOMER_LATITUDE, site.CUSTOMER_LONGITUDE),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             )
            elif type(site).__name__ == 'Depot':
                latitudes.append(site.DEPOT_LATITUDE)
                longitudes.append(site.DEPOT_LONGITUDE)

                if vehicle_working < self.NBR_OF_VEHICLES - 1:
                    vehicle_working += 1
                else:
                    vehicle_working = 0

                color = colors[vehicle_working]

                plt.plot(latitudes, longitudes,
                         marker='o',
                         markerfacecolor='blue',
                         markeredgecolor='blue',
                         linestyle='solid',
                         linewidth=0.5,
                         color=color,
                         )
                latitudes = []
                longitudes = []

        plt.plot(self.Depots[0].DEPOT_LATITUDE, self.Depots[0].DEPOT_LONGITUDE, 'rs')

        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.title('Solution graph')

        depot_legend = lines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Depot')
        customer_legend = lines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10,
                                       label='Customer')
        road_legend = lines.Line2D([], [], color='green', marker='None', linestyle='-', linewidth=1, label='Road')
        plt.legend(handles=[customer_legend, depot_legend, road_legend])

        fig = plt.gcf()
        fig.savefig(filepath, format='png')

    """
    Draw the graph of the sum of the fitness values of the population at every iteration
    
    :param iteration: int - the number of the iteration in the algorithm shown
    :param history: list - the list of the fitness values for each previous iteration
    """
    @staticmethod
    def draw_fitness(iteration, history, filepath):
        plt.figure()
        plt.plot(range(iteration), history)
        fig = plt.gcf()
        plt.show()
        filepath = os.path.join(filepath, 'img_{}.png'.format(iteration + 1))
        fig.savefig(filepath, format='png')

    @staticmethod
    def nbr_of_vehicles(individual: list) -> int:
        return len(individual)

    """
    Check that the fitness value is still changing, if no then the return will stop the algorithm in the main function
    
    :param history: list - the list of all the previous fitness values
    """
    @staticmethod
    def fitness_change(history):
        if len(history) < 5:
            return True
        if history[-1] == history[-2] and history[-1] == history[-3]:
            return False
        return True

    @staticmethod
    def adapt_solution_format(solution):
        adapted_solution = [[]]
        sub_road_index = 0

        for site in solution:
            if site.INDEX == 0:
                adapted_solution[sub_road_index].append(site.INDEX)
                adapted_solution.append([site.INDEX])
                sub_road_index += 1

            else:
                adapted_solution[sub_road_index].append(site.INDEX)

        return adapted_solution
