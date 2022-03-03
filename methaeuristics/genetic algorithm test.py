# Problem definition :
import random as rd
import numpy as np

PROBA_CROSSING = 0.8
PROBA_MUTATION = 0.25
TOURNAMENT_SIZE = 3
MAX_ITERATION = 11
PENALTY = 2
NBR_OF_SITES = 10
POPULATION_SIZE = 10
COST_MATRIX = np.random.randint(1, 20, size=(NBR_OF_SITES, NBR_OF_SITES))

# Functions

"""
Generate the initial population of a certain size, with randomly arranged individuals

@param{int} size - the number of individuals in the population
@return{list} the population
"""


def generate_population() -> list:
    population = []
    seed = range(NBR_OF_SITES)

    for index_individual in range(POPULATION_SIZE):
        individual = rd.sample(seed, k=NBR_OF_SITES)
        population.append(individual)

    return population


def nbr_of_vehicles(individual: list) -> int:
    return 1


"""
Evaluate the cost of visiting sites with this configuration, depending on the number of cars
and the cost from one site to the next one

@param{list} individual - list of all the visited sites from the first to the last visited
@return{int} value of the cost of this configuration
"""


def fitness(individual: list) -> int:
    vehicle_cost = PENALTY * nbr_of_vehicles(individual)

    travel_cost = 0

    for index in range(NBR_OF_SITES - 1):
        site_from = individual[index]
        site_to = individual[index + 1]

        travel_cost += COST_MATRIX[site_from, site_to]

    return vehicle_cost + travel_cost


"""
Apply a mutation to a configuration by inverting 2 sites

@param{list} individual - list of all the visited sites from the first to the last visited
@return{list} new configuration with the mutation
"""


def mutate(individual: list) -> list:
    n = NBR_OF_SITES - 1
    i = rd.randint(0, n - 1)
    j = rd.randint(i + 1, n)

    site_i = individual[i]
    site_j = individual[j]

    result = individual[:i] + [site_j] + individual[i + 1: j] + [site_i] + individual[j + 1:]

    return result


"""
Apply a crossover between the father and mother configuration, in one point

@param{list} father - one configuration
@param{list} mother - an other configuration
@return{list} return two new configurations created from the 2 parents
"""


def one_point_cross_over(father: list, mother: list) -> list:
    point = rd.randint(1, NBR_OF_SITES - 2)

    brother = father[:point] + mother[point:]
    sister = mother[:point] + father[point:]

    return [brother, sister]


"""
Apply a crossover between the father and mother configuration, in 2 points

@param{list} father - one configuration
@param{list} mother - an other configuration
@return{list} return two new configurations created from the 2 parents
"""


def two_points_cross_over(father, mother):
    point_1 = rd.randint(0, NBR_OF_SITES)
    point_2 = rd.randint(point_1, NBR_OF_SITES)

    brother = father[:point_1] + mother[point_1:point_2] + father[point_2:]
    sister = mother[:point_1] + father[point_1:point_2] + mother[point_2:]

    return [brother, sister]


"""
Apply a crossover between the father and mother configuration, using a randomly generated mask

@param{list} father - one configuration
@param{list} mother - an other configuration
@return{list} return two new configurations created from the 2 parents
"""


def uniform_cross_over(father, mother):
    mask = []
    for index in range(NBR_OF_SITES):
        mask.append(rd.randint(0, 1))

    brother = []
    sister = []

    for index in range(NBR_OF_SITES):
        if mask[index]:
            brother.append(father[index])
            sister.append(mother[index])
        else:
            brother.append(mother[index])
            sister.append(father[index])

    return [brother, sister]


"""
Apply a crossover between the father and mother configuration, using a pmx-like crossover

@param{list} father - one configuration
@param{list} mother - an other configuration
@return{list} return two new configurations created from the 2 parents
"""


def pmx_cross_over(father, mother):
    point = rd.randint(0, NBR_OF_SITES)

    brother = father[:point]
    sister = mother[:point]

    for index in range(NBR_OF_SITES):
        if mother[index] not in brother:
            brother.append(mother[index])

        if father[index] not in sister:
            sister.append(father[index])

    return [brother, sister]


"""
Use a deterministic method to select individuals from the current generation
to build the next generation
Sort the population by their fitness score
Then remove the weakest
Finally randomly select individuals to generate the new population of the same size

@param{list} population - current population
@return{list} selected population
"""


def determinist_selection(population: list) -> list:
    population_survivors = sorted(population, key=lambda x: fitness(x))
    population_survivors = population_survivors[:POPULATION_SIZE - TOURNAMENT_SIZE]
    population_survivors = rd.choice(population_survivors, k=POPULATION_SIZE)
    return population_survivors


"""
Use a stochastic method ("Technique de la roulette") to select individuals from the current generation
to build the next generation

@param{list} population - current population
@return{list} selected population
"""


def stochastic_selection(population: list) -> list:
    population_survivors = []

    while len(population_survivors) != POPULATION_SIZE:
        contestants = rd.choices(population, k=TOURNAMENT_SIZE)
        contestants = sorted(contestants, key=lambda x: fitness(x))
        winner = contestants[0]
        population_survivors.append(winner)

    return population_survivors


# Main algorithm :

def main():
    # Initialization
    iteration = 0
    population = generate_population()

    while iteration < MAX_ITERATION:
        iteration += 1

        """ Choose the individuals that survive from the previous generation """
        population_survivors = stochastic_selection(population)  # other methods available
        population = population_survivors
        rd.shuffle(population)

        """ Cross the survivors between them and keep their children """
        population_crossed = []

        for index in range(0, POPULATION_SIZE, 2):
            father = population[index]
            mother = population[index + 1]

            if rd.random() < PROBA_CROSSING:
                children = pmx_cross_over(father, mother)  # other methods available
                population_crossed.append(children[0])
                population_crossed.append(children[1])
            else:
                population_crossed.append(father)
                population_crossed.append(mother)

        """ Apply a mutation to some individuals """
        population_mutated = []

        for index in range(POPULATION_SIZE):
            element = population_crossed[index]

            if rd.random() < PROBA_MUTATION:
                element_mutated = mutate(element)
                population_mutated.append(element_mutated)
            else:
                population_mutated.append(element)

        population = population_mutated

        """ Display each generation properties """
        fitness_sum = 0

        for index in range(POPULATION_SIZE):
            individual = population[index]
            fitness_sum += fitness(individual)

        print('Iteration {}, fitness sum {}'.format(iteration, fitness_sum))


main()
