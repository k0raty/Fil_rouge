""" Import librairies """
from Utility.database import Database
from Utility.common import compute_cost_matrix


class Meta:
    """
    Initialize the genetic algorithm with proper parameters and problem's data

    Parameters
    ----------
    customers: list - the list of customers to be served
    vehicles: list - the list of vehicles that can be used to deliver
    depot: Utility.depot - the unique depot of the delivering company
    cost_matrix: numpy.ndarray - the cost of the travel from one summit to another
    ----------
    """

    def __init__(self, customers=None, depot=None, vehicles=None, cost_matrix=None):
        if customers is None:
            database = Database()

            customers = database.Customers
            vehicles = database.Vehicles
            depot = database.Depots[0]
            cost_matrix = compute_cost_matrix(customers, depot)

        self.COST_MATRIX = cost_matrix

        self.Customers = customers
        self.Depot = depot
        self.Vehicles = vehicles

        self.NBR_OF_CUSTOMER = len(customers)
        self.NBR_OF_VEHICLES = len(vehicles)

        self.solution = [[] for vehicle in vehicles]

        self.PROBA_MUTATION = 1 / self.NBR_OF_CUSTOMER
