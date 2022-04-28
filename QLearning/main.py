""" Import class """
from Utility.database import *
from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from QLearning.qlearning import Qlearning

""" Define parameters """
vehicle_speed = 50

""" Load dataset """
database = Database(vehicle_speed)
graph = create_graph(database.df_customers, database.df_vehicles, vehicle_speed)

""" Generate an initial solution """
genetic_algorithm = GeneticAlgorithm(graph)
genetic_algorithm.main()
initial_solution = genetic_algorithm.solution

""" Run Q-Learning """
qlearning = Qlearning(graph, initial_solution)
qlearning.adaptive_local_search_qlearning()

print('QLearning solution : ', qlearning.solution)
