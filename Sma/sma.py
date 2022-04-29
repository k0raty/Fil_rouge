""" Import librairies """
from mesa import Agent, Model
from mesa.time import RandomActivation  # agents activated in random order at each step
from mesa.datacollection import DataCollector
import numpy as np

from Utility.database import Database
from Utility.pool import Pool
from Utility.common import set_root_dir, compute_gini

from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.Tabou.tabou import Tabou
from Metaheuristics.SimulatedAnnealing.simulated_annealing import Annealing

set_root_dir()


class AgentMeta(Agent):
    fitness = np.inf
    solution = []
    initial_solution = None

    def __init__(self, unique_id: str, model, meta):
        super().__init__(unique_id, model)

        self.meta = meta

    def step(self):
        self.meta.main(self.initial_solution)
        self.solution = self.meta.solution
        self.fitness = self.meta.fitness


class ModelSma(Model):
    def __init__(self, nbr_of_genetic=0, nbr_of_tabou=0, nbr_of_recuit=0, vehicle_speed=40, speedy=True):
        super().__init__()

        self.Database = Database(vehicle_speed)
        self.Graph = self.Database.Graph
        self.Pool = Pool(self.Graph)

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={'Gini': compute_gini},
            agent_reporters={'Fitness': 'fitness', 'Solution': 'solution'},
        )
        self.nbr_of_agent = nbr_of_recuit + nbr_of_tabou + nbr_of_genetic

        for index_agent in range(nbr_of_genetic):
            unique_id = 'genetic_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, GeneticAlgorithm(graph=self.Graph))
            self.schedule.add(agent)

        for index_agent in range(nbr_of_tabou):
            unique_id = 'tabou_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Tabou(graph=self.Graph))
            self.schedule.add(agent)

        for index_agent in range(nbr_of_recuit):
            unique_id = 'recuit_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Annealing(graph=self.Graph, speedy=speedy))
            self.schedule.add(agent)

    """
    Define what is done at each iteration of the SMA
    
    Parameters
    ----------
    solution :list - a list of initial solution ot the problem
    ----------
    """

    def step(self, solution_list=None):
        if solution_list is None:
            for index_agent in range(self.nbr_of_agent):
                self.schedule.agents[index_agent].initial_solution = solution_list

        elif self.nbr_of_agent == len(solution_list):
            for index_agent in range(self.nbr_of_agent):
                self.schedule.agents[index_agent].initial_solution = solution_list[index_agent]

        else:
            for index_agent in range(self.nbr_of_agent):
                self.schedule.agents[index_agent].initial_solution = solution_list[0]

        self.schedule.step()
        self.datacollector.collect(self)

        solutions = self.datacollector.get_agent_vars_dataframe()

        step_solutions = solutions.iloc[-self.nbr_of_agent:]

        for index_solution in range(step_solutions.shape[0]):
            solution = step_solutions.iloc[index_solution, 1]
            fitness = step_solutions.iloc[index_solution, 0]

            self.Pool.inject_in_pool(solution, fitness)
