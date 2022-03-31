import matplotlib.pyplot as plt


class Pool:
    pool = []

    def __init__(self, pr):
        self.pr = pr

    """Calcule le paramètre lambda qui donne le nombre d'arcs non-communs entre les deux solutions"""

    def count_shared_arc(self, solution_1, solution_2):
        list_arc_1 = []
        list_arc_2 = []

        for sub_road in solution_1:
            nbr_of_site = len(sub_road)
            list_arc_1 += [(sub_road[i], sub_road[i + 1]) for i in range(nbr_of_site)]

        for sub_road in solution_2:
            nbr_of_site = len(sub_road)
            list_arc_2 += [(sub_road[i], sub_road[i + 1]) for i in range(nbr_of_site)]

        counter = 0
        for arc in list_arc_1:
            if arc not in list_arc_2:
                counter += 1

        for arc in list_arc_2:
            if arc not in list_arc_1:
                counter += 1

        return counter

    """Calcule la distance entre 2 solutions à partir de leur paramètre lambda"""

    def distance_between_solution(self, solution_1, solution_2):
        coeff_lambda = self.count_shared_arc(solution_1, solution_2)

        if coeff_lambda <= self.pr:
            return 1 - (coeff_lambda / self.pr)

        return 0

    """ 
    Évalue la proximité de solutions S d'un pool avec la solution L.
    Plus la somme est grande, plus les solutions sont proches, et donc il faudra en supprimer
    S: Pool de solutions (ex:[S1,S2,...])
    L: Une solution qui pourrait être ajoutée au pool         
    """

    def assert_injection_in_pool(self, new_solution):
        total_distance = 0
        neighborhood = []

        pool_occupation = len(self.pool)

        for index_solution in range(pool_occupation):
            solution = self.pool[index_solution]
            distance = self.distance_between_solution(new_solution, solution)

            # si la solution S[j] est jugée "trop proche" de L
            if distance <= self.pr:
                # indices des solutions proches selon le critère pr
                neighborhood.append(index_solution)

            # somme des phi entre la liste L et les différentes solutions présentes dans S
            total_distance += distance

        return total_distance, neighborhood

    """ Trace le graphe représentant la distance des solutions contenues dans S à la solution L """

    def graph(self, new_solution):
        pool_occupation = len(self.pool)

        X = [1 for i in range(pool_occupation)]
        Y = [self.count_shared_arc(new_solution, self.pool[i]) for i in range(pool_occupation)]

        plt.scatter(X, Y, s=100, alpha=0.5)
        plt.scatter(1, 0, s=150, c='red')
        plt.axhline(self.pr, color='black', linestyle='dashdot')

        plt.title("Graphe de similarité entre la solution L et le pool de solutions S")
        plt.ylabel('$\lambda_{new solution - solutions in the pool}$')
        plt.legend(['pr'], loc='best')

        plt.show()
