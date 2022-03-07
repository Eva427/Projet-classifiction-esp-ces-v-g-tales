import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


## petit jeu de données :
dataset = rasterio.open("./Data/combined_svm_rbf_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./Data/combined_svm_rbf.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()    
    
    
# dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
# dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

##### extraction des classes à partir des probas avec utilisation de kmeans-----------------------------
nb_class = np.shape(dataset)[0] 
nb_raws = np.shape(dataset)[1] 
nb_columns = np.shape(dataset)[2]

mat_pre_classif = 2*np.ones((nb_raws,nb_columns)) 
##  mat_pre_classif : matrice qui contient des int indiquant le niveau de classification de la matrice d'origine :
    # 0 = pixel d'ombre
    # 1 = pixel bien classé (avec une proba > 0.5) => va constituer l'échantillon de tests
    # 2 = pixel mal classé (avec une proba < 0.5) qu'il faudra prédire
# on commence par une matrice remplie de 2 => les pixels mal classés sont ainsi rentrés par défaut
#### En fait cette matrice m'a pas servi pour le moment mais elle servira par la suite pour repérer les indices des arbres à prédire je pense

#Repérer les indices des pixels bien classés et des zones d'ombres : 
_, abs_pixels_classes, ord_pixels_classes = np.where(dataset[:,:,:]>0.5) #pixels bien classés
somme_10mat = np.sum(dataset,axis=0) #somme des 10 matrices de proba => là où la somme fait 0 on a des vecteurs d'ombre
abs_pixels_ombres, ord_pixels_ombres = np.where(somme_10mat==0) #pixels d'ombre

mat_pre_classif[abs_pixels_classes, ord_pixels_classes]=1
mat_pre_classif[abs_pixels_ombres, ord_pixels_ombres]=0

#qd on fait un kmeans sur une matrice il faut que la matrice soit de la forme suivante : 
    # - n_observations en lignes (ici c'est le nombre de pixels)
    # - n_clusters en colonnes (ici 10)
    # cond on transpose notre matrice 
kmeans = KMeans(n_clusters=nb_class, random_state=0).fit(dataset[:,abs_pixels_classes, ord_pixels_classes].T) 
labels = kmeans.labels_  #il va falloir vérifier à quel arbre correspond chaque cluster


##### extraction des classes à partir du fichier fourni dataset2-----------------------------------------
classes_connues = dataset2[:,abs_pixels_classes, ord_pixels_classes].T
dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "inconnu"}
classes_connues = np.vectorize(dic_arbres.get)(classes_connues) #remplacer les chiffres par les noms 

##### trouver les correspondances entre les clusters de kmeans et les classes connues---------------------
# Matrice de confusion pour Kmeans : mat_result_kmeans
# en colonne : les arbres connus 
# en ligne : le nombre de fois où ils sont associés au ieme label donné par Kmeans 
mat_result_kmeans = np.zeros((nb_class,nb_class))
j = 0
for arbre in dic_arbres.values():
    pos = np.where(classes_connues==arbre)[0]
    for i in range(nb_class) :
        mat_result_kmeans[i,j] = len(np.where(labels[pos]==i)[0])
    j+=1

newkey = np.argmax(mat_result_kmeans, axis=0) #n° classes qui sont les plus associées à chaque arbre
new_dic_arbres = dict(zip(newkey, dic_arbres.values())) #arbre associé à chaque cluster de kmeans

# pourcentage de réussite de kmeans : 
reussite_kmeans =  mat_result_kmeans[newkey,range(nb_class)]/np.sum(mat_result_kmeans,axis=0)   
print(reussite_kmeans)

# prédiction des pixels mal classés 
abs_pixels_apred,ord_pixels_apred = np.where(mat_pre_classif==2)
prediction = kmeans.predict(dataset[:,abs_pixels_apred, ord_pixels_apred].T)

#comment vérifier le résultat ? Il faudrait reconvertir les clusters fixés par kmeans en leur classe correspondante 
# dans les numéros des arbres ou l'inverse. 
# verif = dataset2[0,abs_pixels_apred,ord_pixels_apred]
# verif2 = np.zeros(np.shape(verif))
# oldkey = np.array(list(dic_arbres.keys()))
# for i in range(nb_class):
#     verif2[verif==oldkey[i]]=newkey[i]

# print(np.where(verif2!=prediction))

# pour la prédiction il faudrait définir un seuil de distance autour des centres des clusters, et si le point prédit 
# ne tombe pas dans le cluster alors c'est un outlier => chercher comment définir cette distance. 





