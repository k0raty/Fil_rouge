""" Import librairies """
from os.path import join
import pandas as pd
import random as rd

""" Import utilities """
from Utility.common import set_root_dir

set_root_dir()


class Manager:
    solution_df_path = [
        join('Dataset', 'Initialized', 'ordre_50_it.pkl'),
        join('Dataset', 'Initialized', 'valid_initial_solution.pkl')
    ]

    def __init__(self, index_path):
        dataframe = pd.read_pickle(self.solution_df_path[index_path])

        self.solution_df = dataframe

    def pick_solution(self):
        index_solution = rd.randint(0, self.solution_df.size)
        solution = list(self.solution_df.iloc[0])[index_solution]

        return solution

    def drop_invalid_solution(self):
        index_to_keep = []
        solution_list = list(self.solution_df.iloc[0])

        for index_solution in range(self.solution_df.size):
            solution = solution_list[index_solution]

            if [0, 0] in solution:
                print('Error empty delivery, index : {}'.format(index_solution))
            elif self.is_duplicate_in_solution(solution):
                print('Error duplicated customer, index : {}'.format(index_solution))
            else:
                index_to_keep.append(index_solution)

        if len(index_to_keep) < len(solution_list):
            new_solution_df = self.solution_df[index_to_keep].reset_index(drop=True, inplace=False)
            self.solution_df = new_solution_df

    def export_solution_as_pkl(self):
        path = join('Dataset', 'Initialized', 'valid_initial_solution.pkl')
        self.solution_df.to_pickle(path)
        print('solutions exported in a .pkl file')

    @staticmethod
    def is_duplicate_in_solution(solution):
        customers = []

        for delivery in solution:
            for customer in delivery:
                if customer in customers:
                    return True

        return False
