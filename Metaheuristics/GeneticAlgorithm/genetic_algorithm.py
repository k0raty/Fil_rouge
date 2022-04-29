""" Import librairies """
from numpy import mean, argmin
import pandas as pd
import random as rd
from copy import deepcopy
from os.path import join
from tqdm import tqdm

from Utility.database import Database
from Utility.common import compute_fitness, set_root_dir
from Utility.plotter import plot_solution

set_root_dir()


class GeneticAlgorithm:
    PROBA_CROSSING: float = 0.8
    TOURNAMENT_SIZE: int = 4
    POPULATION_SIZE: int = 50

    fitness: float = 0
    fitness_evolution: list = []
    fitness_mean_evolution: list = []
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

    def __init__(self, graph=None, max_iteration=30):
        if graph is None:
            database = Database()
            graph = database.Graph

        self.Graph = graph
        self.NBR_OF_VEHICLE = len(graph.nodes[0]['Vehicles'])
        self.MAX_ITERATION = max_iteration
        self.PROBA_MUTATION = 1 / (len(graph) - 1)

    """
    Run the GeneticAlgorithm
    
    Parameters
    ----------
    initial_solution: list - a solution already given by an algorithm
    ----------
    """

    def main(self, initial_solution=None):
        initial_solution_path = join('Dataset', 'Initialized', 'valid_initial_solution.pkl')
        initial_solution_df = pd.read_pickle(initial_solution_path)

        initial_solution_set = list(initial_solution_df.iloc[0])
        population = rd.choices(initial_solution_set, k=self.POPULATION_SIZE)

        self.fitness_evolution = []
        self.fitness_mean_evolution = []

        iteration = 0
        progress_bar = tqdm(desc='Genetic algorithm', total=self.MAX_ITERATION, colour='green')

        while iteration < self.MAX_ITERATION and self.is_fitness_evolving():
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
                    brother = self.order_cross_over(father, mother)  # other methods available
                    sister = self.order_cross_over(father, mother)

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

            fitness_list = []

            for individual in population:
                fitness_list.append(compute_fitness(individual, self.Graph))

            """ Select the best individual in the current generation """
            index_min = argmin(fitness_list)
            self.solution = population[index_min]
            self.fitness_evolution.append(fitness_list[index_min])
            self.fitness_mean_evolution.append(mean(fitness_list))

            progress_bar.update(1)

        self.fitness = compute_fitness(self.solution, self.Graph)

        plot_solution(self.solution, self.Graph, title='Genetic solution')

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

    @staticmethod
    def pmx_cross_over(father, mother):
        return father

    @staticmethod
    def order_cross_over(father, mother):
        point = rd.randint(0, len(father) - 1)
        visited_customers = []
        children = []

        for index_delivery in range(len(father)):
            if index_delivery == point:
                continue

            delivery = father[index_delivery]
            children.append([])

            for index_customer in range(len(delivery)):
                customer = delivery[index_customer]
                children[-1].append(customer)

                if customer != 0:
                    visited_customers.append(customer)

        children.append([0])

        for index_delivery in range(len(mother)):
            delivery = mother[index_delivery]

            for index_customer in range(len(delivery)):
                customer = delivery[index_customer]

                if customer not in visited_customers and customer != 0:
                    children[-1].append(customer)
                    visited_customers.append(customer)

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
    Check that the fitness value is still changing, if no then the return will stop the algorithm in the main function
    
    Parameters
    ----------
    history: list - the list of all the previous fitness values
    ----------
    """

    def is_fitness_evolving(self):
        if len(self.fitness_evolution) < 5:
            return True

        last_fitness = self.fitness_evolution[-1]

        index = len(self.fitness_evolution) - 2

        while index >= 0:
            fitness = self.fitness_evolution[index]

            if fitness != last_fitness:
                return True

            index -= 1

        return False


