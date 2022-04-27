""" Import librairies """
from copy import deepcopy

""" Import utilities """
from Utility.common import *
from Utility.validator import *
from voisinage import *


class Qlearning:
    NBR_OF_ACTION = 8
    solution = []
    fitness = 0

    def __init__(self, graph, cost_matrix, initial_solution, max_iter=30, epsilon=0.8, alpha=0.1, gamma=0.9):
        # TODO : Replace the defaults values taken from the lectures by more efficient values
        self.MAX_ITER = max_iter

        self.Graph = graph
        self.cost_matrix = cost_matrix
        self.initial_solution = initial_solution

        self.Q = np.zeros((self.NBR_OF_ACTION, self.NBR_OF_ACTION))

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    """
    Fonction principale, applique l'apprentissage à notre problème
    
    Returns
    -------
    solution: list - the best solution to the problem
    -------
    """

    def main(self):
        self.init_qtable()

        is_improving = True
        no_improvement = 0

        current_solution = self.initial_solution
        best_solution = self.initial_solution
        best_solution_fitness = compute_fitness(best_solution, self.cost_matrix, self.Graph)

        history = [best_solution_fitness]
        number_of_round = 1

        while is_improving:
            next_state = rd.randint(0, self.NBR_OF_ACTION - 1)
            modified_solution = perform_action(next_state, current_solution)

            if is_solution_valid(modified_solution, self.Graph):
                current_solution = modified_solution

            current_solution_fitness = compute_fitness(current_solution, self.cost_matrix, self.Graph)

            if current_solution_fitness < best_solution_fitness:
                best_solution = current_solution
                best_solution_fitness = current_solution_fitness
                history.append(best_solution_fitness)
            else:
                states_visited_count = 1
                visited_state = [next_state]
                list_solution = []

                for index_action in range(self.NBR_OF_ACTION):
                    if next_state == index_action:
                        continue

                    solution_dict = self.state_goal_enhancement(next_state, index_action, states_visited_count,
                                                                visited_state, current_solution,
                                                                no_improvement, best_solution,
                                                                best_solution_fitness, is_improving)
                    list_solution.append(solution_dict)

                for index_solution in range(len(list_solution)):
                    if list_solution[index_solution][1] < best_solution_fitness:
                        best_solution = list_solution[index_solution][0]
                        best_solution_fitness = list_solution[index_solution][1]
                        is_improving = list_solution[index_solution][3]
                        self.Q = list_solution[index_solution][2]

                        print('fitness {}'.format(best_solution_fitness))

                number_of_round += 1
                self.epsilon = self.compute_epsilon(number_of_round)

        return best_solution

    """
    Init the Q matrix with random values
    """

    def init_qtable(self):
        for i in range(len(self.Q)):
            for j in range(len(self.Q[i])):
                self.Q[i][j] = rd.randint(0, 50)

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

        for i in range(len(self.Q)):
            if self.Q[current_state - 1][i] > reward_max:
                reward_max = self.Q[current_state - 1][i]
                next_state = i

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
        qtable[state - 1][next_state - 1] += increase

        return qtable

    def compute_epsilon(self, nbr_of_iteration):
        decay_rate = 1 / (1 + sqrt(nbr_of_iteration))
        return self.epsilon * decay_rate

    @staticmethod
    def fitness_change(history):
        if len(history) < 5:
            return True

        return history[-1] != history[-2] or history[-1] != history[-3]

    """
    """

    def state_goal_enhancement(self, next_state, state_goal, states_visited_count, visited_state, current_solution,
                               no_improvement, best_solution, best_solution_fitness, is_improving):
        reward = 0
        qtable = deepcopy(self.Q)

        while next_state != state_goal:
            if no_improvement == 0:
                state = next_state
                next_state = self.epsilon_greedy(state)
            else:
                next_state = rd.randint(0, self.NBR_OF_ACTION - 1)

            modified_solution = perform_action(next_state, current_solution)

            if is_solution_valid(modified_solution, self.Graph):
                current_solution = modified_solution
                current_solution_fitness = compute_fitness(current_solution, self.cost_matrix, self.Graph)
            else:
                current_solution_fitness = best_solution_fitness + 1

            visited_state.append(next_state)

            print('best fitness', best_solution_fitness)
            print('current fitness', current_solution_fitness)

            if best_solution_fitness > current_solution_fitness:
                best_solution = current_solution
                best_solution_fitness = current_solution_fitness
                reward = reward + current_solution_fitness
                no_improvement = 0
                qtable = self.compute_qtable(state, next_state, reward)
            else:
                no_improvement += 1

                if state in visited_state:
                    states_visited_count += 1

                if no_improvement > self.MAX_ITER and states_visited_count == self.NBR_OF_ACTION:
                    is_improving = False

        return [best_solution, best_solution_fitness, qtable, is_improving]

    def adaptive_local_search_qlearning(self):
        self.init_qtable()

        iteration = 0
        is_improving = True
        no_improvement = 0

        best_solution = self.initial_solution
        current_solution = self.initial_solution

        best_solution_fitness = compute_fitness(best_solution, self.cost_matrix, self.Graph)

        while is_improving:
            iteration += 1
            print('iteration qlearning {}, best fitness {}'.format(iteration, best_solution_fitness))

            reward = 0
            visited_states = []

            next_state = rd.randint(0, self.NBR_OF_ACTION - 1)
            new_solution = perform_action(next_state, current_solution)

            visited_states.append(next_state)

            if is_solution_valid(new_solution, self.Graph):
                current_solution = new_solution

            current_solution_fitness = compute_fitness(current_solution, self.cost_matrix, self.Graph)

            if current_solution_fitness < best_solution_fitness:
                best_solution = current_solution
                best_solution_fitness = current_solution_fitness
                reward = best_solution_fitness
            else:
                is_stuck = False

                while is_improving and not is_stuck:
                    if no_improvement == 0:
                        state = next_state
                        next_state = self.epsilon_greedy(state)
                    else:
                        next_state = rd.randint(0, self.NBR_OF_ACTION - 1)

                    current_solution = perform_action(next_state, current_solution)
                    current_solution_fitness = compute_fitness(current_solution, self.cost_matrix, self.Graph)

                    if next_state not in visited_states:
                        visited_states.append(next_state)

                    if current_solution_fitness < best_solution_fitness:
                        best_solution = current_solution
                        best_solution_fitness = current_solution_fitness

                        is_improving = True
                        no_improvement = 0
                        reward += best_solution_fitness
                        self.Q = self.compute_qtable(state, next_state, reward)
                    else:
                        no_improvement += 1

                        if no_improvement > self.MAX_ITER and len(visited_states) == self.NBR_OF_ACTION:
                            print('no improvement : {}'.format(no_improvement))
                            is_improving = False
                            is_stuck = True

                        elif no_improvement > self.MAX_ITER:
                            is_stuck = True

                self.epsilon = self.compute_epsilon(iteration)

            print('current best solution fitness : {}'.format(best_solution_fitness))

        self.solution = best_solution
        self.fitness = best_solution_fitness

        print('Best solution fitness {}'.format(self.fitness))
