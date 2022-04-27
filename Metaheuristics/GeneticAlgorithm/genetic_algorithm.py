""" Import librairies """
import random as rd
from numpy import mean, argmin
from seaborn import color_palette
import matplotlib.lines as lines
from copy import deepcopy
from os.path import join


""" Import utilities """
from Utility.database import Database
from Utility.common import *


set_root_dir()


class GeneticAlgorithm:
    PROBA_CROSSING: float = 0.8
    TOURNAMENT_SIZE: int = 4
    PENALTY: int = 2
    POPULATION_SIZE: int = 50
    MAX_ITERATION: int = 20

    fitness: float = 0
    solution: list = []

    """
    Initialize the genetic algorithm with proper parameters and problem's data
    
    Parameters
    ----------
    customers: list - the list of customers to be served
    vehicles: list - the list of vehicles that can be used to deliver
    depot: Utility.depot - the unique depot of the delivering company
    cost_matrix: numpy.ndarray - the cost of the travel from one summit to another
    ----------
    """

    def __init__(self, cost_matrix=None, graph=None, vehicle_speed=40):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph
            cost_matrix = compute_cost_matrix(graph)

        self.COST_MATRIX = cost_matrix
        self.Graph = graph

        self.NBR_OF_CUSTOMER = len(graph) - 1  # the depot is not a customer
        self.NBR_OF_VEHICLE = len(graph.nodes[0]['Vehicles'])

        self.PROBA_MUTATION = 1 / self.NBR_OF_CUSTOMER

    """
    Run the GeneticAlgorithm
    
    Parameters
    ----------
    initial_solution: list - a solution already given by an algorithm
    ----------
    """

    def main(self, initial_solution=None, speedy=True):
        initial_solution_set_path = join('Dataset', 'Initialized', 'ordre_50_it.pkl')
        initial_solution_set = pd.read_pickle(initial_solution_set_path)

        iteration = 0
        population = list(initial_solution_set.iloc[0])
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
            fitness_list = []

            for individual in population:
                fitness_list.append(compute_fitness(individual, self.COST_MATRIX, self.Graph))

            fitness_mean = mean(fitness_list)
            fitness_history.append(fitness_mean)

            index_min = argmin(fitness_list)
            self.solution = population[index_min]

            print('Iteration {}, fitness {}'.format(iteration, fitness_list[index_min]))

        self.fitness = compute_fitness(self.solution, self.COST_MATRIX, self.Graph)
        self.draw_fitness(iteration, fitness_history)

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
            contestants = sorted(contestants, key=lambda x: compute_fitness(x, self.COST_MATRIX, self.Graph))
            winner = contestants[0]
            population_survivors.append(winner)

        return population_survivors

    """
    Use a deterministic method to select individuals from the current generation
    to build the next generation
    Sort the population by their fitness score
    Then remove the weakest
    Finally randomly select individuals to generate the new population of the same size

    Parameters
    ----------
    population: list - current population
    ----------
    
    Returns
    -------
    population_survivors: list - selected population
    -------
    """

    def determinist_selection(self, population: list) -> list:
        population_survivors = sorted(population, key=lambda x: compute_fitness(x, self.COST_MATRIX, self.Graph))
        population_survivors = population_survivors[:self.POPULATION_SIZE - self.TOURNAMENT_SIZE]
        population_survivors = rd.choices(population_survivors, k=self.POPULATION_SIZE)

        return population_survivors

    """
    Apply a crossover between the father and mother configuration, using a pmx-like crossover

    Parameters
    ----------
    father: list - a solution
    mother: list - an other solution
    ----------
    
    Returns
    -------
    brother: list - a new solution
    sister: list - an other new solution
    -------
    """

    def pmx_cross_over(self, father, mother):
        father_customers = []
        mother_customers = []

        for delivery in father:
            father_customers += [summit for summit in delivery if summit != 0]

        for delivery in mother:
            mother_customers += [summit for summit in delivery if summit != 0]

        point = rd.randint(0, self.NBR_OF_VEHICLE)

        brother_order = father_customers[:point]
        sister_order = mother_customers[:point]

        for index_customer in range(self.NBR_OF_CUSTOMER):
            if mother_customers[index_customer] not in brother_order:
                brother_order.append(mother_customers[index_customer])

            if father_customers[index_customer] not in sister_order:
                sister_order.append(father_customers[index_customer])

        brother = self.generate_solution(brother_order)
        sister = self.generate_solution(sister_order)

        return [brother, sister]

    """
    Apply a mutation to a configuration by inverting 2 summits

    Parameters
    ----------
    individual: list - list of all the visited summits from the first to the last visited summits
    ----------
    
    Returns
    -------
    mutated_individual : list - new configuration with the mutation
    -------
    """

    def mutate(self, individual: list) -> list:
        mutated_individual = deepcopy(individual)

        index_vehicle_i = rd.randint(0, self.NBR_OF_VEHICLE - 1)
        index_vehicle_j = rd.randint(0, self.NBR_OF_VEHICLE - 1)

        delivery_i = individual[index_vehicle_i]
        delivery_j = individual[index_vehicle_j]

        index_summit_i = rd.randint(0, len(delivery_i) - 1)
        index_summit_j = rd.randint(0, len(delivery_j) - 1)

        summit_i = delivery_i[index_summit_i]
        summit_j = delivery_j[index_summit_j]

        mutated_individual[index_vehicle_i][index_summit_i] = summit_j
        mutated_individual[index_vehicle_j][index_summit_j] = summit_i

        return mutated_individual

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    Parameters
    ----------
    size: int - the number of individuals in the population
    solution: list - an initial solution to the problem
    proportion: float - the proportion of the given solution in the population
    ----------
    
    Returns
    -------
    population: list - a population containing several solution to the problem
    -------
    """

    def generate_population(self, initial_solution=None, proportion=0.05) -> list:
        population = []
        population_size = self.POPULATION_SIZE

        if initial_solution is not None:
            nbr_of_solution = int(proportion * population_size)

            population = [initial_solution for i in range(nbr_of_solution)]

            population_size -= nbr_of_solution

        for index_individual in range(population_size):
            individual = self.generate_solution()
            population.append(individual)

        return population

    """
    Generate a solution to the problem
    
    Parameters
    ----------
    order: list - the order of the summits to deliver
    ----------
    
    Returns
    -------
    solution: list - a randomly generated solution to the problem
    -------
    """
    # utiliser une fonction qui génère la solution du problème de voyageur de commerce
    # puis la segmenter par véhicule, en changeant la méthode de segmentation
    def generate_solution(self, order=None):
        if order is None:
            seed = range(1, self.NBR_OF_CUSTOMER + 1)  # as depot's index is 0
            order = rd.sample(seed, k=self.NBR_OF_CUSTOMER)

        solution = []

        nbr_of_customer_by_vehicle = self.NBR_OF_CUSTOMER // self.NBR_OF_VEHICLE
        leftover = self.NBR_OF_CUSTOMER % self.NBR_OF_VEHICLE

        for index_vehicle in range(self.NBR_OF_VEHICLE):
            weight = 0
            delivery = []

            start = index_vehicle * nbr_of_customer_by_vehicle
            end = start + nbr_of_customer_by_vehicle

            for index_order in range(start, end):
                index_customer = order[index_order]
                customer = self.Graph.nodes[index_customer]

                weight += customer['TOTAL_WEIGHT_KG']

                if weight > self.Graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG'][index_vehicle]:
                    weight = customer['TOTAL_WEIGHT_KG']
                    delivery.append(0)

                delivery.append(order[index_order])

            solution.append([0, *delivery, 0])

        if leftover > 0:
            start = self.NBR_OF_VEHICLE * nbr_of_customer_by_vehicle
            end = self.NBR_OF_CUSTOMER
            weight = 0

            for index_order in range(start, end):
                index_customer = order[index_order]
                customer = self.Graph.nodes[index_customer]

                weight += customer['TOTAL_WEIGHT_KG']

                if weight > self.Graph.nodes[0]['Vehicles']['VEHICLE_TOTAL_WEIGHT_KG'][0]:
                    weight = customer['TOTAL_WEIGHT_KG']
                    solution[0].append(0)

                solution[0].append(order[index_order])

        return solution

    """
    Draw the Graph showing all the customers summits and the depots, with the road taken to go through them
    
    Parameters
    ----------
    solution: list - an individual of the population with the lowest fitness score
    ----------
    """

    def draw_solution(self, solution, filepath):
        plt.figure(figsize=[25, 15])

        vehicle_working = 0

        colors = color_palette(n_colors=self.NBR_OF_VEHICLE)

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

                if vehicle_working < self.NBR_OF_VEHICLE - 1:
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

        depot = self.Graph.nodes[0]
        plt.plot(depot['LATITUDE'], depot['LONGITUDE'], 'rs')

        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.title('Solution Graph')

        depot_legend = lines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Depot')
        customer_legend = lines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10,
                                       label='Customer')
        road_legend = lines.Line2D([], [], color='green', marker='None', linestyle='-', linewidth=1, label='Road')
        plt.legend(handles=[customer_legend, depot_legend, road_legend])

        fig = plt.gcf()
        fig.savefig(filepath, format='png')

    """
    Draw the Graph of the sum of the fitness values of the population at every iteration
    
    :param iteration: int - the number of the iteration in the algorithm shown
    :param history: list - the list of the fitness values for each previous iteration
    """

    @staticmethod
    def draw_fitness(iteration, history):
        plt.figure()
        plt.plot(range(iteration), history)
        fig = plt.gcf()
        plt.show()

    """
    Check that the fitness value is still changing, if no then the return will stop the algorithm in the main function
    
    Parameters
    ----------
    history: list - the list of all the previous fitness values
    ----------
    """

    @staticmethod
    def fitness_change(history):
        if len(history) < 5:
            return True

        return history[-1] != history[-2] or history[-1] != history[-3]
