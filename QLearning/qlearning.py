""" Import librairies """
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from Utility.common import compute_fitness
from Utility.validator import is_solution_valid
from voisinage import *
from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from Utility.database import Database
from Utility.plotter import plot_solution


class Qlearning:
    NBR_OF_ACTION = 8

    solution = []
    fitness_evolution = []
    fitness = 0

    epsilon = 0

    def __init__(self, graph=None, max_iteration=50, epsilon=0.8, alpha=0.1, gamma=0.9):
        if graph is None:
            database = Database()
            graph = database.Graph

        # TODO: Replace the defaults values taken from the lectures by more efficient values
        self.MAX_ITERATION = max_iteration

        self.Graph = graph

        self.Q = np.zeros((self.NBR_OF_ACTION, self.NBR_OF_ACTION))

        self.initial_epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    """
    Fonction principale, applique l'apprentissage à notre problème
    
    Returns
    -------
    solution: list - the best solution to the problem
    -------
    """

    def main(self, initial_solution=None):
        if initial_solution is None:
            ga = GeneticAlgorithm(graph=self.Graph)
            ga.main()
            initial_solution = ga.solution

        self.Q = np.zeros((self.NBR_OF_ACTION, self.NBR_OF_ACTION))
        self.epsilon = self.initial_epsilon

        self.fitness_evolution = []
        no_improvement = 0

        best_solution = initial_solution
        current_solution = initial_solution

        best_fitness = compute_fitness(best_solution, self.Graph)

        iteration = 0
        progress_bar = tqdm(desc='Q-Learning', total=self.MAX_ITERATION, colour='green')

        while iteration < self.MAX_ITERATION and self.is_fitness_evolving():
            iteration += 1

            visited_states = []

            next_state = rd.randint(0, self.NBR_OF_ACTION - 1)
            new_solution = perform_action(next_state, current_solution)

            visited_states.append(next_state)

            if is_solution_valid(new_solution, self.Graph):
                current_solution = new_solution

            current_solution_fitness = compute_fitness(current_solution, self.Graph)

            if current_solution_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_solution_fitness
            else:
                is_not_stuck = True

                while is_not_stuck:
                    current_state = next_state

                    if no_improvement == 0:
                        next_state = self.epsilon_greedy(next_state)
                    else:
                        next_state = rd.randint(0, self.NBR_OF_ACTION - 1)

                    current_solution = perform_action(next_state, current_solution)
                    current_solution_fitness = compute_fitness(current_solution, self.Graph)

                    if next_state not in visited_states:
                        visited_states.append(next_state)

                    if current_solution_fitness < best_fitness and is_solution_valid(current_solution, self.Graph):
                        best_solution = current_solution
                        best_fitness = current_solution_fitness

                        no_improvement = 0
                        reward = (10 ** 8) / best_fitness

                        self.Q = self.compute_qtable(current_state, next_state, reward)
                    else:
                        no_improvement += 1

                        if no_improvement > self.MAX_ITERATION and len(visited_states) == self.NBR_OF_ACTION:
                            is_not_stuck = False

                self.epsilon = self.compute_epsilon(iteration)

            self.fitness_evolution.append(best_fitness)
            progress_bar.update(1)

        self.solution = best_solution
        self.fitness = best_fitness

        plot_solution(self.solution, self.Graph, title='Q-Learning solution')

    """
    Select an action to perform that maximize the reward
     
    Parameters
    ----------
    current_state: int - the index of the current action performed
    ----------
    
    Returns
    -------
    next_state: int - the index of the next action to perform
    -------
    """

    def select_best_action(self, current_state):
        reward_max = 0
        next_state = 0

        for index_state in range(len(self.Q)):
            if self.Q[current_state][index_state] > reward_max:
                reward_max = self.Q[current_state][index_state]
                next_state = index_state

        return next_state

    """
    Select the next action to perform, which can either be randomly selected or chosen to maximize reward
    
    Parameters
    ----------
    current_state: int - the current action performed
    ----------
    
    Returns
    -------
    next_state: int - the next action to perform
    -------
    """

    def epsilon_greedy(self, current_state):
        random_proba = rd.random()

        if random_proba < self.epsilon:
            next_state = rd.randint(0, self.NBR_OF_ACTION - 1)
        else:
            next_state = self.select_best_action(current_state)

        return next_state

    """
    Update the Q value
    
    Parameters
    ----------
    state
    next_state
    reward
    ----------
    """

    def compute_qtable(self, state, next_state, reward):
        qtable = deepcopy(self.Q)

        qtable_max = max(self.Q[next_state - 1])
        increase = self.alpha * (reward + self.gamma * qtable_max - self.Q[state - 1][next_state - 1])
        qtable[state - 1][next_state - 1] += np.around(increase, decimals=2)

        return qtable

    def compute_epsilon(self, nbr_of_iteration):
        decay_rate = 1 / (1 + np.sqrt(nbr_of_iteration))
        return self.epsilon * decay_rate

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
