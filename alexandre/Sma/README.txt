J'ai reussi à rajouter le recuit au sma , cependant j'ai modifié quelques méthode d'initialisation ce qui fait que je ne peux me permettre de relier les autres 
et c'est donc a toi alex de changer ça : 
-database.py et modifié , vehicles , customers ect ne sont plus exactements les même mais contiennent les même informations ,ils dépendent maintenant du graph

-la matrice des poids est construite a partir du graph également 
-compute_fitness a été réécrite en fonction du nouveau self.vehicles
-dans les agents du sma , le paramètre speedy indique si oui on non , on effectue seulement une ité pour le recuit  , c'est par défaut oui. 
-Dans l'algo génétique , j'ai rajouté les 50 sol initiales lors de l'initialisation de l'agent. 

-> a toi de réadapter les deux autres algo en fonction de ces nouveaux paramètres. 