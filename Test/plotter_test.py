""" Import class """
from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from Utility.plotter import plot_solution

""" Generate a solution """
ga = GeneticAlgorithm()
ga.main()
solution = ga.solution

""" Test plotting solution """
plot_solution(solution, ga.Graph)
