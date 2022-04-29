""" Import librairies """
import unittest

from Utility.database import Database
from Utility.validator import *


class TestValidator(unittest.TestCase):
    def setUpClass(self) -> None:
        self.Database = Database()
        self.Graph = self.Database.Graph

    def test_is_solution_shape_valid(self):
        valid_solution = [[0, 1, 0], [0, 2, 3, 0]]
        invalid_solution = [[1, 0]]

        message = 'should valid the shape of a valid solution'
        self.assertTrue(is_solution_shape_valid(valid_solution, self.Graph), message)

        message = 'should not valid the shape of an invalid solution'
        self.assertFalse(is_solution_shape_valid(invalid_solution, self.Graph), message)
