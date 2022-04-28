""" Import librairies """
from numpy import mean, argmin
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from copy import deepcopy
from os.path import join

""" Import utilities """
from Utility.database import Database
from Utility.common import compute_fitness, set_root_dir
from Utility.plotter import plot_solution

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

    def __init__(self, graph=None, vehicle_speed=40):
        if graph is None:
            database = Database(vehicle_speed)
            graph = database.Graph

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
        initial_solution_path = join('Dataset', 'Initialized', 'valid_initial_solution.pkl')
        initial_solution_df = pd.read_pickle(initial_solution_path)

        initial_solution_set = list(initial_solution_df.iloc[0])
        population = rd.choices(initial_solution_set, k=self.POPULATION_SIZE)

        fitness_history = []
        iteration = 0

        while iteration < self.MAX_ITERATION and self.fitness_change(fitness_history):
            iteration += 1
            print('iteration', iteration)
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
                    brother = self.pmx_cross_over(father, mother)  # other methods available
                    sister = self.pmx_cross_over(father, mother)

                    population_crossed.append(brother)
                    population_crossed.append(sister)

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
                fitness_list.append(compute_fitness(individual, self.Graph))

            fitness_mean = mean(fitness_list)
            fitness_history.append(fitness_mean)

            index_min = argmin(fitness_list)
            self.solution = population[index_min]

            message = 'Iteration {}, best fitness {}, mean fitness {}'
            print(message.format(iteration, fitness_list[index_min], fitness_mean))

        self.fitness = compute_fitness(self.solution, self.Graph)
        plot_solution(self.solution, self.Graph)

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
            contestants = sorted(contestants, key=lambda x: compute_fitness(x, self.Graph))
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
        population_survivors = sorted(population, key=lambda x: compute_fitness(x, self.Graph))
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
        point = rd.randint(0, len(father))

        embryo = [father[point], mother[point]]
        children = []

        customers_visited = []
        customers_not_visited = [index for index in range(1, len(self.Graph))]

        for index_delivery in range(len(father)):
            delivery = embryo[index_delivery]
            children.append([0])

            for index_customer in range(1, len(delivery) - 1):
                customer = delivery[index_customer]

                if customer in customers_visited:
                    if len(customers_not_visited) > 0:
                        random_index = rd.randint(0, len(customers_not_visited) - 1)
                        new_customer = customers_not_visited.pop(random_index)
                        customers_visited.append(new_customer)
                        children[-1].append(new_customer)
                else:
                    customers_visited.append(customer)
                    customers_not_visited.pop(customers_not_visited.index(customer))
                    children[-1].append(customer)

            children[-1].append(0)

        return children

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

        index_summit_i = rd.randint(1, len(delivery_i) - 2)
        index_summit_j = rd.randint(1, len(delivery_j) - 2)

        summit_i = delivery_i[index_summit_i]
        summit_j = delivery_j[index_summit_j]

        mutated_individual[index_vehicle_i][index_summit_i] = summit_j
        mutated_individual[index_vehicle_j][index_summit_j] = summit_i

        return mutated_individual

    """
    Draw the Graph of the sum of the fitness values of the population at every iteration
    
    :param iteration: int - the number of the iteration in the algorithm shown
    :param history: list - the list of the fitness values for each previous iteration
    """

    @staticmethod
    def draw_fitness(iteration, history):
        plt.figure()
        plt.plot(range(iteration), history)

        plt.xlabel('iteration')
        plt.ylabel('Mean fitness')
        plt.title('Evolution of the population mean fitness')

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
