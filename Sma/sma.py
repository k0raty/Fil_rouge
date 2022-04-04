""" Import librairies """
from mesa import Agent, Model
from mesa.time import RandomActivation  # agents activated in random order at each step
from mesa.datacollection import DataCollector
import numpy as np

""" Import utilities """
from Utility.database import Database
from Utility.pool import Pool
from Utility.common import distance

""" Import metaheuristics """
from Metaheuristics.GeneticAlgorithm.Code.genetic_algorithm import GeneticAlgorithm
from Metaheuristics.Tabou.Code.tabou import Tabou
from Metaheuristics.SimulatedAnnealing.simulated_annealing import Annealing

""" Define problem parameters """
nbr_of_iteration = 10

""" Define agents """


class AgentMeta(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, meta):
        super().__init__(unique_id, model)
        self.fitness = np.inf
        self.solution = []
        self.meta = meta

    def step(self, initial_solution=None):
        self.solution, self.fitness = self.meta.main(initial_solution)


""" Define Gini """


def compute_gini(model_sma):
    agents_fitness = sorted([agent.fitness for agent in model_sma.schedule.agents])
    nbr_of_agent = model_sma.num_agents

    total_fitness = sum(agents_fitness)

    A = nbr_of_agent * total_fitness
    B = sum(fitness * (nbr_of_agent - index) for index, fitness in enumerate(agents_fitness))

    return 1 + (1 / nbr_of_agent) - 2 * A / B


""" Define model """


class ModelSma(Model):
    """A model with some number of agents."""

    def __init__(self, nbr_of_genetic, nbr_of_tabou, nbr_of_recuit):
        self.Database = Database()

        customers = self.Database.Customers
        depots = self.Database.Depots
        vehicles = self.Database.Vehicles[0]

        nbr_of_customer = len(customers)

        cost_matrix = np.zeros((nbr_of_customer, nbr_of_customer))

        for i in range(nbr_of_customer):
            customer_i = customers[i]

            for j in range(nbr_of_customer):
                customer_j = customers[j]
                lat_i = float(customer_i.CUSTOMER_LATITUDE)
                lon_i = float(customer_i.CUSTOMER_LONGITUDE)
                lat_j = float(customer_j.CUSTOMER_LATITUDE)
                lon_j = float(customer_j.CUSTOMER_LONGITUDE)

                cost_matrix[i, j] = distance(lat_i, lon_i, lat_j, lon_j)

        self.Pool = Pool()

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={'Gini': compute_gini},
            agent_reporters={'Wealth': 'fitness'},
        )

        # Create agents
        for index_agent in range(nbr_of_genetic):
            unique_id = 'genetic_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, GeneticAlgorithm(customers, depots, vehicles, cost_matrix))
            self.schedule.add(agent)

        for index_agent in range(nbr_of_tabou):
            unique_id = 'tabou_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Tabou(customers, depots, vehicles, cost_matrix))
            self.schedule.add(agent)

        for index_agent in range(nbr_of_recuit):
            unique_id = 'recuit_{}'.format(index_agent)
            agent = AgentMeta(unique_id, self, Annealing(customers, depots, vehicles, cost_matrix))
            self.schedule.add(agent)

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()


""" Run the SMA """
model = ModelSma(nbr_of_iteration)

for iteration in range(nbr_of_iteration):
    model.step()

model_dataframe = model.datacollector.get_model_vars_dataframe()
agent_dataframe = model.datacollector.get_agent_vars_dataframe()

print(model_dataframe)
print(agent_dataframe)
