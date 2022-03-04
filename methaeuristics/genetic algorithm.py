import random as rd
import numpy as np
import xlrd as xl


class GeneticAlgorithm:
    PROBA_CROSSING = 0.8
    PROBA_MUTATION = 0.25
    TOURNAMENT_SIZE = 3
    PENALTY: int = 2
    POPULATION_SIZE = 10

    def __init__(self, max_iteration=15, nbr_of_sites=107):
        self.MAX_ITERATION = max_iteration
        self.NBR_OF_SITES = nbr_of_sites
        doc_dep = xl.open_workbook("4_detail_table_depots.xls")
        feuille_dep = doc_dep.sheet_by_index(0)
        lat_dep = feuille_dep.cell_value(rowx=1,colx=3)
        long_dep = feuille_dep.cell_value(rowx=1,colx=4)
        doc_cust = xl.open_workbook("2_detail_table_customers.xls")
        feuille_cust = doc_cust.sheet_by_index(0)
        self.COST_MATRIX = np.zeros(size=(nbr_of_sites, nbr_of_sites))
        for i in range(1,108):
            lat_cust = feuille_cust.cell_value(rowx=i,colx=3)
            long_cust = feuille_cust.cell_value(rowx=i,colx=4)
            self.COST_MATRIX[0][i]=np.sqrt((lat_cust-lat_dep)**2 + (long_cust-long_dep)**2)*0.2
        for i in range(2,108):
            for j in range(i+1,108):
                lat_i = feuille_cust.cell_value(rowx=i,colx=3)
                long_i = feuille_cust.cell_value(rowx=i,colx=4)
                lat_j = feuille_cust.cell_value(rowx=j,colx=3)
                long_j = feuille_cust.cell_value(rowx=j,colx=4)
                dist = np.sqrt((lat_i-lat_j)**2 + (long_i-long_j)**2)
                self.cost_MATRIX[i][j]=dist*0.2
                self.cost_MATRIX[j][i]=dist*0.2



    def main(self):
        # Initialization
        iteration = 0
        population = self.generate_population()

        while iteration < self.MAX_ITERATION:
            iteration += 1

            """ Choose the individuals that survive from the previous generation """
            population_survivors = self.stochastic_selection(population)  # other methods available
            population = population_survivors
            rd.shuffle(population)

            """ Cross the survivors between them and keep their children """
            population_crossed = []

            for index in range(0, self.POPULATION_SIZE, 2):
                father = population[index]
                mother = population[index + 1]

                if rd.random() < self.PROBA_CROSSING:
                    children = self.pmx_cross_over(father, mother)  # other methods available
                    population_crossed.append(children[0])
                    population_crossed.append(children[1])
                else:
                    population_crossed.append(father)
                    population_crossed.append(mother)

            """ Apply a mutation to some individuals """
            population_mutated = []

            for index in range(self.POPULATION_SIZE):
                element = population_crossed[index]

                if rd.random() < self.PROBA_MUTATION:
                    element_mutated = self.mutate(element)
                    population_mutated.append(element_mutated)
                else:
                    population_mutated.append(element)

            population = population_mutated

            """ Display each generation properties """
            fitness_sum = 0

            for index in range(self.POPULATION_SIZE):
                individual = population[index]
                fitness_sum += self.fitness(individual)

            print('Iteration {}, fitness sum {}'.format(iteration, fitness_sum))

    """
    Use a stochastic method ("Technique de la roulette") to select individuals from the current generation
    to build the next generation

    @param{list} population - current population
    @return{list} selected population
    """

    def stochastic_selection(self, population: list) -> list:
        population_survivors = []

        while len(population_survivors) != self.POPULATION_SIZE:
            contestants = rd.choices(population, k=self.TOURNAMENT_SIZE)
            contestants = sorted(contestants, key=lambda x: self.fitness(x))
            winner = contestants[0]
            population_survivors.append(winner)

        return population_survivors

    """
    Use a deterministic method to select individuals from the current generation
    to build the next generation
    Sort the population by their fitness score
    Then remove the weakest
    Finally randomly select individuals to generate the new population of the same size

    @param{list} population - current population
    @return{list} selected population
    """

    def determinist_selection(self, population: list) -> list:
        population_survivors = sorted(population, key=lambda x: self.fitness(x))
        population_survivors = population_survivors[:self.POPULATION_SIZE - self.TOURNAMENT_SIZE]
        population_survivors = rd.choices(population_survivors, k=self.POPULATION_SIZE)
        return population_survivors

    """
    Apply a crossover between the father and mother configuration, using a pmx-like crossover

    @param{list} father - one configuration
    @param{list} mother - an other configuration
    @return{list} return two new configurations created from the 2 parents
    """

    def pmx_cross_over(self, father, mother):
        point = rd.randint(0, self.NBR_OF_SITES)

        brother = father[:point]
        sister = mother[:point]

        for index in range(self.NBR_OF_SITES):
            if mother[index] not in brother:
                brother.append(mother[index])

            if father[index] not in sister:
                sister.append(father[index])

        return [brother, sister]

    """
    Apply a mutation to a configuration by inverting 2 sites

    @param{list} individual - list of all the visited sites from the first to the last visited
    @return{list} new configuration with the mutation
    """

    def mutate(self, individual: list) -> list:
        n = self.NBR_OF_SITES - 1
        i = rd.randint(0, n - 1)
        j = rd.randint(i + 1, n)

        site_i = individual[i]
        site_j = individual[j]

        result = individual[:i] + [site_j] + individual[i + 1: j] + [site_i] + individual[j + 1:]

        return result

    """
    Evaluate the cost of visiting sites with this configuration, depending on the number of cars
    and the cost from one site to the next one

    @param{list} individual - list of all the visited sites from the first to the last visited
    @return{int} value of the cost of this configuration
    """

    def fitness(self, individual: list) -> int:
        vehicle_cost = self.PENALTY * self.nbr_of_vehicles(individual)

        travel_cost = 0

        for index in range(self.NBR_OF_SITES - 1):
            site_from = individual[index]
            site_to = individual[index + 1]

            travel_cost += self.COST_MATRIX[site_from, site_to]

        return vehicle_cost + travel_cost

    """
    Generate the initial population of a certain size, with randomly arranged individuals

    @param{int} size - the number of individuals in the population
    @return{list} the population
    """

    def generate_population(self) -> list:
        population = []
        seed = range(self.NBR_OF_SITES)

        for index_individual in range(self.POPULATION_SIZE):
            individual = rd.sample(seed, k=self.NBR_OF_SITES)
            population.append(individual)

        return population

    @staticmethod
    def nbr_of_vehicles(individual: list) -> int:
        return len(individual)
