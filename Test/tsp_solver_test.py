""" Import class """
from Utility.database import Database
from Utility.plotter import plot_solution
from Utility.tsp_solver import solve_tsp

database = Database()
graph = database.Graph
tsp_solution = solve_tsp(graph)

plot_solution([tsp_solution], graph)
