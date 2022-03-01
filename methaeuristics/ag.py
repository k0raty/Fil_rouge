# Problem definition :
import random as rd

PROBA_CROISEMENT = 0.2
PROBA_MUTATION = 0.1
MAX_ITERATION = 10
PENALTY = 2

# Initial population

iteration = 0
population = []
N = len(population)

# Functions

def nbrOfVehicles(individual):
    return 1


def cost(i, j):
    return i + j


def fitness(individual):
    vehicleCost = PENALTY * nbrOfVehicles(individual)
    travelCost = 0

    for index in range(len(individual)):
        i = individual[index]
        j = individual[index + 1]
        travelCost += cost(i, j)

    return vehicleCost + travelCost


def mutate(individual):
    i = rd.randint(0, len(individual))
    j = rd.randint(i, len(individual))

    result = individual[:i] + individual[j] + individual[i + 1: j] + individual[i] + individual[j + 1:]
    return result

# Individual crossing functions

def onePointCrossOver(father, mother):
    point = rd.randint(0, len(father))

    brother = father[:point] + mother[point + 1:]
    sister = mother[:point] + father[point + 1:]

    return [brother, sister]

def twoPointsCrossOver(father, mother):
    point_1 = rd.randint(0, len(father))
    point_2 = rd.randint(point_1, len(father))

    brother = father[:point_1] + mother[point_1:point_2] + father[point_2:]
    sister = mother[:point_1] + father[point_1:point_2] + mother[point_2:]

    return [brother, sister]

def uniformCrossOver(father, mother):
    mask = []
    for index in range(len(father)):
        mask.append(rd.randint(0, 1))

    brother = []
    sister = []

    for index in range(len(mask)):
        if (mask[index]) :
            brother.append(father[index])
            sister.append(mother[index])
        else :
            brother.append(mother[index])
            sister.append(father[index])

    return [brother, sister]

# Main algorithm :

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
    populationNext = []

    for index in range(0, len(population) // 2, 2):
        father = population[index]
        mother = population[index + 1]

        if rd.random() < PROBA_CROISEMENT:
            children = onePointCrossOver(father, mother)
            populationNext.append(children)
        else:
            populationNext.append(father)
            populationNext.append(mother)

    # individuals mutation
    population = []

    for index in range(len(populationNext)):
        element = populationNext[index]

        if rd.random() < PROBA_MUTATION:
            mutatedElement = mutate(element)
            population.append(mutatedElement)
        else:
            population.append(element)
