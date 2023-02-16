# EVENT GRAPH PROJECT

Ce dossier contient le code pour la génération de graphes et l'entrainement de réseaux de neurones sur graphes sur des données provenant de caméras DVS.
Ce dossier contient les ficiers suivants : 
+ gat.py
+ gcn.py
+ dataset.py
+ KNNGraph.py
+ test.py
+ train_gat.py
+ train_gcn.py
+ utils_dataset.py
+ utils.py
+ visualize_graph.py

Le fichier **test.py** permet de générer la train_dataset et test_dataset des données provenant des caméras DVS. La train_dataset et la test_dataset contiennent les événements et les labels pour les gestes d'entrainement et de validation. Ceux-ci sont implémenté par les classes du dossier *src/mldvs/datawrappers*. Dans ce projet le travail s'est centré sur la base de données DVSGestureV2. 

Pour exécuter le code, les fichiers doivent figurer dans le chemin %(HOME_PATH)s/event-gnn/examples/scripts. 
Le fichier **KNNGraph.py** contient le KNNGraph permettant d'implémenter le KNNGraph pour générer des edges index entre les noeuds générés par le KMeans.
Le fichier **dataset.py** contient le code pour générer les graphes à partir des données test_dataset et train_dataset, l'implémentation du code se base sur les modules dans **utils.py** et **utils_dataset.py** qui est une version ajustée de l'implémentation de la classe Dataset de la librairie torch_geometric. 
Les fichiers **gat.py** et **gcn.py** implémentent les réseaux de neurones sur graphe de type *GCN *et *GAT*.
Les fichiers **train_gat.py** et **test_gat.py** implémentent l'entrainement des réseaux de neurones sur graphe d'évènements. 