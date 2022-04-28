""" Import class """
from Metaheuristics.SimulatedAnnealing.recuit_classe import *

database = Database(vehicle_speed=5000)
graph = database.Graph


def test_pkl_solutions():
    initial_solution_df_path = join('Dataset', 'Initialized', 'ordre_50_it.pkl')
    initial_solution_df = pd.read_pickle(initial_solution_df_path)
    initial_solution_set = list(initial_solution_df.iloc[0])

    for index in range(len(initial_solution_set)):
        solution = initial_solution_set[index]
        flag = is_solution_valid(solution, graph)
        print('solution valid ? ', flag)


def test_annealing_init():
    annealing = Annealing()
    solution = annealing.generate_initial_solution()
    flag = is_solution_valid(solution, annealing.Graph)
    print('solution valid ? ', flag)
