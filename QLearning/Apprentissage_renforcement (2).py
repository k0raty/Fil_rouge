import random
from math import sqrt
from Metaheuristics.GeneticAlgorithm.genetic_algorithm import GeneticAlgorithm
from voisinage import *
from Utility2.common import *
import numpy as np
from Utility2.database import *
from Utility2.validator import *
import pandas as pd

class Apprentissage:

    def __init__(self, max_iter, x0, q_size, epsilon = 0.8, alpha = 0.1, gamma = 0.9):
        """Les valeurs par défaut sont des valeurs arbitraires, issues du diapo de cours"""
        self.MAX_ITER =  max_iter
        self.solution_init = x0
        self.q_size = q_size
        self.Q = np.zeros((q_size,q_size))   
        self.epsilon = epsilon                 
        self.alpha = alpha
        self.gamma = gamma                   #Valeurs du diapo mais il faudra certainement les modifier
        
    def init_Q(self):
        """Initialisation de la matrice Q. On choisit d'initialiser avec des valeurs entières aléatoires."""
        for i in range(len(self.Q)):
            for j in range(len(self.Q[i])):
                self.Q[i][j] = random.randint(0,50)    #Initialisation aléatoire mais avec des valeurs relativement grandes (d'après la littérature)
        
    def max_action(self, current_state): 
        """On choisit l'action dont la récompense est la plus élevée (par apprentissage).
        Retourne l'indice de l'action à réaliser."""
        maxi = 0
        next_state = 0
        for i in range(len(self.Q)):
            if self.Q[current_state-1][i] > maxi :
                maxi = self.Q[current_state-1][i]
                next_state = i
        return next_state
    
    def epsilon_greedy(self, current_state):
        """Fonction qui détermine une action suivante suivant une probabilité.
        Renvoi l'indice de l'action suivante à réaliser."""
        random_proba = random.random()
        if random_proba < self.epsilon:
            next_state = random.randint(1,self.q_size)       
        else:                                                  
            next_state = self.max_action(current_state)
        return next_state
    
    def choose_action(self, state, type_function):      #Pour info, j'ai juste modifié l'initialisation de next_state et la boucle else, 
    # j'ai considéré qu'on allait pas avoir de problème donc j'ai fait sauté les "sécurités"
        """Fonction utilisée dans l'algorithme d'apprentissage, qui détermine l'action suivante"""
        if type_function == 1 :
            next_state = self.epsilon_greedy(state)
        else:
            next_state = random.randint(1,self.q_size)
        return next_state

    def calculate_Q_value(self,state,next_state,reward):
        """Fonction d'actualisation de Q."""
        #Calcul du max
        maxQ = max(self.Q[next_state-1])       #Faut mettre des "indices-1" parce que le tableau va de 0 à 7 et le state de 1 à 8 

        #Actualisation de Q
        self.Q[state-1][next_state-1] += self.alpha * (reward + self.gamma * maxQ - self.Q[state-1][next_state-1])

    def fitness_change(self,history):               #Fonction d'Alexandre que j'ai remise pour remplacer le state_goal, parce que je vois pas comment
        if len(history) < 5:                    # le définir autrement, en gros mon idée est de dire "si tu réduis plus assez le fitness, c'est
            return True                         # terminé, on arrête l'algo" parce qu'on a pas de possibilité d'aller sur un state "final" ou "goal"
        return history[-1] != history[-2] or history[-1] != history[-3]     # ou alors faut le créer mais je vois pas comment
    
    def state_goal_enhancement(self,next_state,state_goal,states_visited_count,visited_state,Q,current_x,fitness_current_x,reward,no_improvement,best_x,fitness_best_x,improved):
        print("etape de state_goal_enhancement")
        while next_state != state_goal :

            if no_improvement == 0:
                state = next_state
                next_state = self.choose_action(state,1) 
            else: 
                next_state = self.choose_action(0,2) 
            

            modified_x = action_state(next_state, current_x) #Fonction dans voisinage.py
            
            if check_constraint(modified_x, G): #boucle de vérification des différentes contraintes de la solution suite aux modifications
                current_x = modified_x 
                fitness_current_x = compute_fitness(current_x, cost_matrix, vehicles)
            else:
                fitness_current_x = fitness_best_x + 1

            visited_state.append(next_state)

            if fitness_best_x > fitness_current_x:
                best_x = current_x
                fitness_best_x = fitness_current_x
                reward = reward + fitness_current_x
                no_improvement = 0
                improved = True
                Q = self.calculate_Q_value(state, next_state, reward)
            else: 
                no_improvement +=1
                
                if state in visited_state:  
                    states_visited_count += 1
                
                if no_improvement > self.MAX_ITER and states_visited_count == self.q_size:
                    improved = False

        return [[best_x,fitness_best_x,reward],improved]

    def main(self, cost_matrix, vehicles, G):
        """Fonction principale, applique l'apprentissage à notre problème
        Retourne la meilleure solution X"""
        self.init_Q()
        improved = True
        no_improvement = 0
        best_x = self.solution_init
        current_x = self.solution_init
        history = []
        fitness_best_x = compute_fitness(best_x, cost_matrix, vehicles)
        history.append(fitness_best_x)        #Ajout à history
        number_of_round = 1
        print("début de la fonction main")
        
        while improved:
            reward = 0
            states_visited_count = 0
            next_state = self.choose_action(0,2) 
            visited_state = [next_state]
            modified_x = action_state(next_state, current_x)
            print("premier choix d'état suivant")
            
            
            if check_constraint(modified_x, G): #boucle de vérification des différentes contraintes de la solution suite aux modifications
                current_x = modified_x 
                fitness_current_x = compute_fitness(current_x, cost_matrix, vehicles)
                print("contraintes respectées")
            else:
                fitness_current_x = fitness_best_x + 1
                print("contraintes pas respectées")
        
            if fitness_best_x > fitness_current_x:
                best_x = current_x
                fitness_best_x = fitness_current_x
                history.append(fitness_best_x)        #On ajoute les fitness calculés
                reward = fitness_current_x
            else:
                states_visited_count += 1
                list_x = []

                for i in range(1,9):
                    print("DANS LA BOUCLE AVEC CHACUN DES STATES GOAL")
                    state_goal = i
                    Q_copy = self.Q.copy()
                    list_x.append(self.state_goal_enhancement(next_state,state_goal,states_visited_count,visited_state,Q_copy,current_x,fitness_current_x,reward,no_improvement,best_x,fitness_best_x,improved))    
                
                best_list_x = 0
                best_fitness_list_x = fitness_best_x
                best_reward_list_x = 0
                
                for j in range(len(list_x)):
                    if list_x[j][0][1] < best_fitness_list_x :
                        best_list_x = list_x[j][0][0]
                        best_fitness_list_x = list_x[j][0][1]
                        best_reward_list_x = list_x[j][0][2]
                        improved = list_x[j][1]
                        Q = list_x[j][2]

                best_x = best_list_x
                fitness_best_x = best_fitness_list_x
                reward = best_reward_list_x
                self.Q = Q
                
                number_of_round +=1
                self.epsilon = 1/(1+sqrt(number_of_round))
                print("numero du round " + number_of_round)
                
        return(best_x)

### TEST ###

v=50 #Vitesse des véhicules
df_customers= pd.read_excel(r"Dataset\table_2_customers_features.xls")
df_vehicles=pd.read_excel(r"Dataset\table_3_cars_features.xls")

database = Database(v)
customers = database.Customers
vehicles = database.Vehicles
depot = database.Depots[0]
G = create_G(df_customers, df_vehicles, v)
cost_matrix = compute_cost_matrix(G)

from Metaheuristics.GeneticAlgorithm import *

ga = GeneticAlgorithm()
solution_init = ga.main()

a = Apprentissage(20, solution_init, 8)
solution = a.main(cost_matrix,vehicles,G)
print(solution)