""" Import librairies """
import random as rd

""" Import SMA """
from sma import ModelSma


class Scenarios:
    def __init__(self):
        self.NBR_OF_ITERATION = 2

    """
    Run the SMA without any communication and sharing between the agents

    Parameters
    ----------
    nbr_of_iteration: int - the number of steps the SMA should operate
    ----------
    """

    @staticmethod
    def no_interaction(nbr_of_iteration=2, speedy=True):
        model_sma = ModelSma(nbr_of_genetic=1, nbr_of_tabou=1, nbr_of_recuit=0, speedy=True)

        for iteration in range(nbr_of_iteration):
            model_sma.step()

        model_dataframe = model_sma.datacollector.get_model_vars_dataframe()
        agent_dataframe = model_sma.datacollector.get_agent_vars_dataframe()

        print(model_dataframe)
        print(agent_dataframe)

    """
    Run the SMA all algorithms taking as an initial solution the best solution from the pool

    Parameters
    ----------
    nbr_of_iteration: int - the number of steps the SMA should operate
    ----------
    """

    @staticmethod
    def friend_interaction_best_fitness(nbr_of_iteration=5):
        model_sma = ModelSma(nbr_of_genetic=1, nbr_of_tabou=1, nbr_of_recuit=0)

        for iteration in range(nbr_of_iteration):
            if len(model_sma.Pool.pool) > 0:
                solution = [model_sma.Pool.pool[0]]
                model_sma.step(solution)

            else:
                model_sma.step()

        model_dataframe = model_sma.datacollector.get_model_vars_dataframe()
        agent_dataframe = model_sma.datacollector.get_agent_vars_dataframe()

        print(model_dataframe)
        print(agent_dataframe)

    """
    Run the SMA all algorithms taking as an initial solution the best solutions from the pool for each agent

    Parameters
    ----------
    nbr_of_iteration: int - the number of steps the SMA should operate
    ----------
    """

    @staticmethod
    def friend_interaction_best_solutions(nbr_of_iteration=5):
        model_sma = ModelSma(nbr_of_genetic=1, nbr_of_tabou=1, nbr_of_recuit=0)

        for iteration in range(nbr_of_iteration):
            if len(model_sma.Pool.pool) > 0:
                solution_list = [model_sma.Pool.pool[index] for index in range(model_sma.nbr_of_agent)]
                model_sma.step(solution_list)

            else:
                model_sma.step()

        model_dataframe = model_sma.datacollector.get_model_vars_dataframe()
        agent_dataframe = model_sma.datacollector.get_agent_vars_dataframe()

        print(model_dataframe)
        print(agent_dataframe)

    """
    Run the SMA all algorithms taking as an initial solution a random solution from the pool

    Parameters
    ----------
    nbr_of_iteration: int - the number of steps the SMA should operate
    ----------
    """

    @staticmethod
    def friend_interaction_random_solution(nbr_of_iteration=5):
        model_sma = ModelSma(nbr_of_genetic=1, nbr_of_tabou=1, nbr_of_recuit=0)

        for iteration in range(nbr_of_iteration):
            if len(model_sma.Pool.pool) > 0:
                index_solution = rd.randint(0, len(model_sma.Pool.pool))
                solution = [model_sma.Pool.pool[index_solution]]
                model_sma.step(solution)

            else:
                model_sma.step()

        model_dataframe = model_sma.datacollector.get_model_vars_dataframe()
        agent_dataframe = model_sma.datacollector.get_agent_vars_dataframe()

        print(model_dataframe)
        print(agent_dataframe)

    """
    Run the SMA all algorithms taking as an initial solution random solutions from the pool for each agent

    Parameters
    ----------
    nbr_of_iteration: int - the number of steps the SMA should operate
    ----------
    """

    @staticmethod
    def friend_interaction_random_solutions(nbr_of_iteration=5):
        model_sma = ModelSma(nbr_of_genetic=1, nbr_of_tabou=1, nbr_of_recuit=0)

        solution = []
        for iteration in range(nbr_of_iteration):
            if len(model_sma.Pool.pool) > 0:
                for index in range(model_sma.nbr_of_agent):
                    i = rd.randint(0, len(model_sma.Pool.pool) - 1)
                    solution_i = model_sma.Pool.pool[i]
                    solution.append(solution_i)
                model_sma.step(solution)

            else:
                model_sma.step()

        model_dataframe = model_sma.datacollector.get_model_vars_dataframe()
        agent_dataframe = model_sma.datacollector.get_agent_vars_dataframe()

        print(model_dataframe)
        print(agent_dataframe)
