# Problem definition :
import random as rd
import numpy as np

PROBA_CROSSING = 0.2
PROBA_MUTATION = 0.1
MAX_ITERATION = 20
PENALTY = 2
COST_MATRIX = np.array([
    [0, 1, 2, 1],
    [1, 0, 3, 6],
    [5, 2, 0, 1],
    [1, 3, 7, 0],
])


# Functions

def nbr_of_vehicles(individual):
    return 1


def fitness(individual):
    vehicle_cost = PENALTY * nbr_of_vehicles(individual)
    travel_cost = 0
    n = len(individual) - 1

    for index in range(n):
        i = individual[index]
        j = individual[index + 1]
        travel_cost += COST_MATRIX[i, j]

    return vehicle_cost + travel_cost


def mutate(individual):
    n = len(individual) - 1
    i = rd.randint(0, n - 1)
    j = rd.randint(i + 1, n)

    result = individual[:i] + \
             [individual[j]] + \
             individual[i + 1: j] + \
             [individual[i]] + \
             individual[j + 1:]

    return result


# Individual crossing functions

def one_point_cross_over(father, mother):
    n = len(father) - 1
    point = rd.randint(0, n)

    brother = father[:point] + mother[point:]
    sister = mother[:point] + father[point:]

    return [brother, sister]


def two_points_cross_over(father, mother):
    point_1 = rd.randint(0, len(father))
    point_2 = rd.randint(point_1, len(father))

    brother = father[:point_1] + mother[point_1:point_2] + father[point_2:]
    sister = mother[:point_1] + father[point_1:point_2] + mother[point_2:]

    return [brother, sister]


def uniform_cross_over(father, mother):
    mask = []
    for index in range(len(father)):
        mask.append(rd.randint(0, 1))

    brother = []
    sister = []

    for index in range(len(mask)):
        if mask[index]:
            brother.append(father[index])
            sister.append(mother[index])
        else:
            brother.append(mother[index])
            sister.append(father[index])

    return [brother, sister]


# Main algorithm :

def main():
    # Initialization
    iteration = 0
    population = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 3, 1],
        [0, 2, 1, 3],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 2, 0, 3],
        [2, 1, 0, 3],
    ]
    N = len(population)

    while iteration < MAX_ITERATION:
        iteration += 1

        # determinist method
        population = sorted(population, key=lambda x: fitness(x))

        # stochastic method (technique de la roulette)
        # population = stochasticSort(population)

        # survivors selection
        population = population[:N]
        rd.shuffle(population)

        # individuals cross
        population_next = []

        for index in range(0, len(population) // 2, 2):
            father = population[index]
            mother = population[index + 1]

            if rd.random() < PROBA_CROSSING:
                children = one_point_cross_over(father, mother)
                population_next.append(children[0])
                population_next.append(children[1])
            else:
                population_next.append(father)
                population_next.append(mother)

        # individuals mutation
        population = []

        for index in range(len(population_next)):
            element = population_next[index]

            if rd.random() < PROBA_MUTATION:
                mutated_element = mutate(element)
                population.append(mutated_element)
            else:
                population.append(element)

        print('Iteration {} : {}'.format(iteration, population))


main()
