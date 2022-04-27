""" Import class """
from QLearning.voisinage import *

""" Test """
solution = [[0,1,2,3,0],[0,4,5,6,0]]
print(solution)
print(intra_route_swap(solution))
print(InterRouteSwap(solution))
print(IntraRouteShift(solution))
print(inter_route_shift(solution))
print()

solution1 = [[0,1,2,3,4,5,6,7,0],[0,8,9,10,11,12,13,0]]
print(solution1)
print(two_intra_route_swap(solution1))
print(two_intra_route_shift(solution1))
print()

solution2 = [[0,1,2,3,4,0],[0,5,6,7,0],[0,8,9,10,11,0]]
print(solution2)
print(RemoveSmallestRoad(solution2))
print()

solution3 = [[0,1,2,3,4,0],[0,5,6,7,0],[0,8,9,10,11,0]]
print(solution3)
print(remove_random_road(solution3))
