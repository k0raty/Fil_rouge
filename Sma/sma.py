""" Import librairies """
from mesa import Agent, Model
from mesa.time import RandomActivation  # agents activated in random order at each step
from mesa.datacollection import DataCollector
import numpy as np

""" Import utilities """
from Utility.database import Database
from Utility.pool import Pool
from Utility.common import compute_cost_matrix, set_root_dir

""" Import metaheuristics """
from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.Tabou.tabou import Tabou
from Metaheuristics.SimulatedAnnealing.recuit_classe import Annealing

set_root_dir()

""" Define agents """


class AgentMeta(Agent):
    """An agent with fixed initial wealth."""

    fitness = np.inf
    solution = []
    initial_solution = None

    def __init__(self, unique_id, model, meta, speedy):
        super().__init__(unique_id, model)
        self.meta = meta
        self.speedy = speedy

    def step(self):
        self.meta.main(self.initial_solution, self.speedy)
        self.solution = self.meta.solution
        self.fitness = self.meta.fitness


"""
Define Gini

Parameters
----------
model: ModelSma - the model gathering the agents containing the metaheuristics
----------

Returns 
-------
gini: float - the gini score
-------
"""


def compute_gini(model) -> float:
    agents_fitness = sorted([agent.fitness for agent in model.schedule.agents])

    total_fitness = sum(agents_fitness)

    A = model.nbr_of_agent * total_fitness
    B = sum(fitness * (model.nbr_of_agent - index) for index, fitness in enumerate(agents_fitness))

    gini = 1 + (1 / model.nbr_of_agent) - 2 * A / B

    return gini


""" Define model """


class ModelSma(Model):
    def __init__(self, nbr_of_genetic=0, nbr_of_tabou=0, nbr_of_recuit=0, vehicle_speed=40, speedy=True):
        self.Database = Database(vehicle_speed)

        vehicles = self.Database.Vehicles
        graph = self.Database.Graph
        cost_matrix = compute_cost_matrix(graph)

        self.Pool = Pool(cost_matrix, vehicles)

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={'Gini': compute_gini},
            agent_reporters={'Fitness': 'fitness', 'Solution': 'solution'},
        )
        self.nbr_of_agent = nbr_of_recuit + nbr_of_tabou + nbr_of_genetic

        for index_agent in range(nbr_of_genetic):
            unique_id = 'genetic_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, GeneticAlgorithm(vehicles, cost_matrix, graph), speedy)
            self.schedule.add(agent)

        for index_agent in range(nbr_of_tabou):
            unique_id = 'tabou_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Tabou(vehicles, cost_matrix, graph), speedy)
            self.schedule.add(agent)

        for index_agent in range(nbr_of_recuit):
            unique_id = 'recuit_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Annealing(graph=graph), speedy)
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

        elif self.nbr_of_agent == len(solution_list):  # scénario Best Solutions
            for index_agent in range(self.nbr_of_agent):
                self.schedule.agents[index_agent].initial_solution = solution_list[index_agent]

        else:  # autres scénarios
            for index_agent in range(self.nbr_of_agent):
                self.schedule.agents[index_agent].initial_solution = solution_list[0]

        self.schedule.step()  # speedy define whether you want a quick sol or not
        self.datacollector.collect(self)

        solutions = self.datacollector.get_agent_vars_dataframe()

        step_solutions = solutions.iloc[-self.nbr_of_agent:]

        for index_solution in range(step_solutions.shape[0]):
            solution = step_solutions.iloc[index_solution, 1]
            fitness = step_solutions.iloc[index_solution, 0]

            self.Pool.inject_in_pool(solution, fitness)
