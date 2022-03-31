# Projet-classifiction-esp-ces-v-g-tales
installer rasterio sur linux
$ sudo add-apt-repository ppa:ubuntugis/ppa
$ sudo apt-get update
$ sudo apt-get install python-numpy gdal-bin libgdal-dev
$ pip install rasterio

sur le terminal machine de l'insa
pip install rasterio

Importation des données et on voit qu'on a bien plus de 50% des données où on a proba appartenance à une classe >0.5. A voir si sa classe unclassed (numéro 0) c'est pas juste là où il y a du noir sur la photo (à vérifier) et correspondrait ainsi pas vraiment à une vraie classe rejet au sens de celle qu'on veut créer. 

J'ai essayé d'utiliser un gmm tout codé que j'ai trouvé sur internet mais mon kernel plante (j'ai du me tromper je re regarderai). Vous avez le site dont je l'ai extrait. https://python-course.eu/machine-learning/expectation-maximization-and-gaussian-mixture-models-gmm.php

projet_data_extraction : tout sur l'extraction, la construction d'un dictionnaire... 
tentative gmm : la même chose + tentative de code gmm qui marche pas (en tout cas sur mon ordi)

clustering2.py : réalise le clustering sur le petit jeu de données => à présenter aux profs

clustering_small_data.py : même code que clustering2.py mais réutilisable dans d'autres algos (code mis en forme avec des fonctions)

clustering_kmeans_v2.py : même code que clustering2.py mais sans la manipulation relou du dico

TO DO LIST : 
- comparer les 10 itérations (demande des profs)
- extraire les caches (données réelles) et voir comment s'en servir pour vérifier les résultats 
- réfléchir à la prise en compte du contexte
- comparer les résultats obtenus avec les différentes règles de classification pour la construction du set d'entrainement
- évaluation des algos, on peut calculer différentes choses : le nombre de pixels rejetés, le ratio nombre_rejets/nombre_pixels_apredire, NMI (normalized Mutual Information)
- trouver comment fixer un rayon qui prenne en compte la variance
- tester plusieurs seuils T sur kmeans 
- extraire les données du set d'entraînement et les plot
------------------------------------------------------------------------------------------------------------------------------------------ 
Lien vers l'overleaf : https://fr.overleaf.com/4173155583tgsvqjsnwzsj
