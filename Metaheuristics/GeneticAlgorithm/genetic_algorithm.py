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
from Utility.common import compute_cost_matrix, compute_fitness, classe

os.chdir(os.path.join('', '..'))


class GeneticAlgorithm:
    PROBA_CROSSING: float = 0.8
    TOURNAMENT_SIZE: int = 4
    PENALTY: int = 2
    POPULATION_SIZE: int = 50
    MAX_ITERATION: int = 50

    def __init__(self, customers=None, depots=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depots = database.Depots
            cost_matrix = compute_cost_matrix(customers, depots[0])

        self.solution = None

        nbr_of_summits = len(customers)

        self.COST_MATRIX = cost_matrix
        self.NBR_OF_VEHICLES = len(vehicles)
        self.NBR_OF_SITES = nbr_of_summits
        self.PROBA_MUTATION = 1 / nbr_of_summits

        self.Customers = customers
        self.Depots = depots[0]
        self.Vehicles = vehicles[0]

    """
    Run the GeneticAlgorithm
    """
    def main(self, initial_solution=None):
        plt.close('all')
        timestamp = floor(time.time())
        path = os.path.join('Metaheuristics', 'GeneticAlgorithm', 'Graphs', 'test_{}'.format(timestamp))
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
                fitness_sum += compute_fitness(self.adapt_solution_format(individual), self.COST_MATRIX)

            fitness_history.append(fitness_sum)

            print('Iteration {}, fitness sum {}'.format(iteration, fitness_sum))

            sorted_population = sorted(population, key=lambda x: compute_fitness(self.adapt_solution_format(x), self.COST_MATRIX))
            solution = sorted_population[0]

            self.solution = self.adapt_solution_format(solution)

            filepath = os.path.join(path, 'img_{}.png'.format(iteration))
            self.draw_solution(solution, filepath)

        self.draw_fitness(iteration, fitness_history, path)

        return self.solution, compute_fitness(self.solution, self.COST_MATRIX)

    """
    Use a stochastic method ("Technique de la roulette") to select individuals from the current generation
    to build the next generation

    Parameters
    ----------
    population: list - current population
    ----------
    
    Returns
    -------
    population_survivors: list - selected population
    -------
    """

    def stochastic_selection(self, population: list) -> list:
        population_survivors = []

        while len(population_survivors) != self.POPULATION_SIZE:
            contestants = rd.choices(population, k=self.TOURNAMENT_SIZE)
            contestants = sorted(contestants, key=lambda x: compute_fitness(self.adapt_solution_format(x), self.COST_MATRIX))
            winner = contestants[0]
            population_survivors.append(winner)

        return population_survivors

    """
    Use a deterministic method to select individuals from the current generation
    to build the next generation
    Sort the population by their fitness score
    Then remove the weakest
    Finally randomly select individuals to generate the new population of the same size

    :param population: list - current population
    :return population_survivors: list - selected population
    """
    def determinist_selection(self, population: list) -> list:
        population_survivors = sorted(population, key=lambda x: compute_fitness(self.adapt_solution_format(x), self.COST_MATRIX))
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

        for index_summit in range(len(father)):
            summit = father[index_summit]

            if classe(summit) != 'Depot':
                new_father.append(summit)

        new_mother = []

        for index_summit in range(len(mother)):
            summit = mother[index_summit]

            if classe(summit) != 'Depot':
                new_mother.append(summit)

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
    Apply a mutation to a configuration by inverting 2 summits

    :param individual: list - list of all the visummitd summits from the first to the last visummitd
    :return : list - new configuration with the mutation
    """
    def mutate(self, individual: list) -> list:
        """ first we remove all previous depots """
        filtered = [summit for summit in individual if classe(summit) != 'Depot']

        n = self.NBR_OF_SITES - 1
        i = rd.randint(0, n - 1)
        j = rd.randint(i + 1, n)

        summit_i = filtered[i]
        summit_j = filtered[j]

        result = filtered[:i] + [summit_j] + filtered[i + 1: j] + [summit_i] + filtered[j + 1:]
        result = self.add_depot(result)

        return result

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    :param size: int - the number of individuals in the population
    :param solution: list - an initial solution to the problem
    :param proportion: float - the proportion of the given solution in the population
    :return population: list - the population
    """
    def generate_population(self, solution=None, proportion=0) -> list:
        population = []
        population_size = self.POPULATION_SIZE
        seed = range(self.NBR_OF_SITES)

        if solution is not None:
            initial_solution = []

            for delivery in solution:
                initial_solution.append(*[customer for customer in delivery if not 0])

            nbr_of_solution = int(proportion * population_size)

            for index in range(nbr_of_solution):
                population.append(initial_solution)

            population_size -= nbr_of_solution

        for index_individual in range(population_size):
            """ first we generate random roads between customers """
            order = rd.sample(seed, k=self.NBR_OF_SITES)
            individual = [self.Customers[i] for i in order]

            """ then we add a depot when needed"""
            individual = self.add_depot(individual)
            population.append(individual)

        return population

    """
    Go through the list of summits and add a return to depot everytime it is needed
    
    Parameters
    ----------
    individual: list - list of all the visited summits from the first to the last visited
    ----------
    
    Returns
    -------
    individual_with_depot: list - the same solution but with returns to depot where needed
    -------
    """

    def add_depot(self, individual):
        weight = 0
        volume = 0

        weight_vehicle = float(self.Vehicles.VEHICLE_TOTAL_WEIGHT_KG)
        volume_vehicle = float(self.Vehicles.VEHICLE_TOTAL_VOLUME_M3)

        individual_with_depot = [self.Depots]

        for customer in individual:
            weight += float(customer.TOTAL_WEIGHT_KG)
            volume += float(customer.TOTAL_VOLUME_M3)

            if weight > weight_vehicle or volume > volume_vehicle:
                individual_with_depot.append(self.Depots)
                weight = 0
                volume = 0

            individual_with_depot.append(customer)

        individual_with_depot.append(self.Depots)

        return individual_with_depot

    """
    Draw the graph showing all the customers summits and the depots, with the road taken to go through them
    
    Parameters
    ----------
    solution: list - an individual of the population with the lowest fitness score
    ----------
    """

    def draw_solution(self, solution, filepath):
        plt.figure(figsize=[25, 15])

        vehicle_working = 0

        colors = color_palette(n_colors=self.NBR_OF_VEHICLES)

        latitudes = []
        longitudes = []

        for index_summit in range(len(solution)):
            summit = solution[index_summit]

            if summit.INDEX != 0:
                latitudes.append(summit.LATITUDE)
                longitudes.append(summit.LONGITUDE)

                plt.annotate("{}".format(index_summit),
                             (summit.LATITUDE, summit.LONGITUDE),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             )

            else:
                latitudes.append(summit.LATITUDE)
                longitudes.append(summit.LONGITUDE)

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

        plt.plot(self.Depots.LATITUDE, self.Depots.LONGITUDE, 'rs')

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

    """
    :param individual: list - a solution
    """
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

    """
    :param solution: list - solution with the genetic algorithm format
    :return formatted_solution: list - solution with a format readable by other algorithms
    """
    @staticmethod
    def adapt_solution_format(solution):
        formatted_solution = [[]]
        delivery_index = 0

        for summit in solution:
            if summit.INDEX == 0:
                formatted_solution[delivery_index].append(summit.INDEX)
                formatted_solution.append([summit.INDEX])
                delivery_index += 1

            else:
                formatted_solution[delivery_index].append(summit.INDEX)

        return formatted_solution
