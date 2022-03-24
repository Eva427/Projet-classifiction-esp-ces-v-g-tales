#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:10:24 2022

@author: julie
- Rq : même version que clstering_kmeans_v2 mais avec le rejet
- Rq : la fonction predict renvoie abs_pixels_apred et abs_pixels_apred, ce qui est la 
même chose que abs_nonclass et ord_nonclass renvoyés par rule_select_training_data. Peut-être supprimer 
abs_pixels_apred et abs_pixels_apred on s'en sert pas.
"""

#importation de librairies
import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
import scipy
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

#importation d'autres codes
import rule_select_training_data as RSTD 


#*******************************************************************************************************
##### EXTRACTION DES DONNEES :
#*******************************************************************************************************

## petit jeu de données :
dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()

## gros jeu de données  lrl : 
# dataset = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
# dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  svm : 
# dataset = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img") #matrice des probas
# dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "Mais"}

#*******************************************************************************************************
###### DEFINITION DES FONCTIONS :
#*******************************************************************************************************

##### extraction des classes à partir des probas avec utilisation de kmeans-----------------------------
def apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes):
    #qd on fait un kmeans sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # donc on transpose notre matrice 
    kmeans = KMeans(n_clusters=nb_class, random_state=0).fit(dataset[:,abs_pixels_classes, ord_pixels_classes].T) 
    labels = kmeans.labels_  #il va falloir vérifier à quel arbre correspond chaque cluster
    clusters = kmeans.cluster_centers_
    return kmeans,labels, clusters 

##### extraction des classes à partir du fichier fourni dataset2-----------------------------------------
def extract_classes_connues (dataset2,abs_pixels_classes, ord_pixels_classes) :
    classes_connues = dataset2[:,abs_pixels_classes, ord_pixels_classes].T
    return classes_connues

##### trouver les correspondances entre les clusters de kmeans et les classes connues---------------------
# Matrice de confusion pour Kmeans : mat_result_kmeans
# en colonne : les arbres connus 
# en ligne : le nombre de fois où ils sont associés au ieme label donné par Kmeans 

def map_cluster(nb_class,classes_connues,labels) :
    mat_result_kmeans = np.zeros((nb_class,nb_class))
    num_classes = np.array(np.unique(classes_connues),dtype=int) #n° originla des espèces d'arbre
    j = 0
    for num in num_classes:
        pos = np.where(classes_connues==num)[0]
        for i in range(nb_class) :
            mat_result_kmeans[i,j] = len(np.where(labels[pos]==i)[0])
        j+=1
    newclass = np.argmax(mat_result_kmeans, axis=0) #n° classes qui sont les plus associées à chaque arbre
    return newclass,mat_result_kmeans

# pourcentage de réussite de kmeans : 
def eval_kmean(nb_class,mat_result_kmeans,newkey): 
    reussite_kmeans =  mat_result_kmeans[newkey,range(nb_class)]/np.sum(mat_result_kmeans,axis=0)   
    return reussite_kmeans

# prédiction des pixels mal classés :
def predict(mat_pre_classif,kmeans):
    abs_pixels_apred,ord_pixels_apred = np.where(mat_pre_classif==2)
    prediction = kmeans.predict(dataset[:,abs_pixels_apred, ord_pixels_apred].T)
    return abs_pixels_apred,ord_pixels_apred,prediction

##### création d'une matrice contenant le résultat du clustering------------------------------------------ 
def matrice_cluster(labels,nb_raws,nb_columns,abs_pixels_classes,ord_pixels_classes) :
    # matrice_cluster associe à chaque pixel bien classé le cluster obtenu    
    mat_cluster = (-1)*np.ones((nb_raws,nb_columns)) #-1 = pas de cluster associé à ce pixel
    mat_cluster[abs_pixels_classes,ord_pixels_classes] = labels       
    return mat_cluster

##### définition des fonctions pour le rejet-------------------------------------------------------------- 

# définition d'une distance :
def wasserstein (x,y):
    # calule la distance de wasserstein définie entre 2 mesures de probabilités
    return scipy.stats.wasserstein_distance(x,y)

# Calcul du rayon de chaque cluster : 
def calcul_radius(clusters,dataset,mat_cluster,mesure,nb_class):
    # calcule le rayon de chaque cluster en utilisant la mesure entrée en argument
    # rayon du cluster k = distance entre le centre du clusteur k et le point le + éloigné du cluster k
    # kmeans suppose que les clusters sont sphériques. 
    rayons = np.zeros(nb_class)
        
    for k in range(nb_class): 
        abs_points_cluster,ord_points_cluster = np.where(mat_cluster == k) #coordonées des points associés au cluster k
        data_cluster_k = dataset[:,abs_points_cluster,ord_points_cluster] #points de dataset associés au cluster k    
        # distance_to_center_k est une matrice contenant les distances de chaque point au centre du cluster k :
        distance_to_center_k = np.zeros(np.shape(data_cluster_k)[1]) 
        for i in range(np.shape(data_cluster_k)[1]): #pour chaque donnée i du cluster k...
            #...on calcule la distance entre la donnée i et le centre du cluster k
            distance_to_center_k[i] = mesure(clusters[k,:],data_cluster_k[:,i]) #matrice "clusters" of shape (n_clusters, n_features)
       
        rayons[k] = np.max(distance_to_center_k)#rayon = distance entre centre de k et point le + éloigné de k           
    return rayons

# Implémentation du rejet : On rejette lorsque distance(centre du cluster k, donnée du cluster k) > seuil*(rayon du cluster k)
def rejet(dataset,rayons,clusters,abs_nonclass,ord_nonclass,mesure,nb_class,mat_cluster): 
    # fonction qui crée renvoie les abscisses et ordonnées des pixels rejetés. 
    # On lui rentre en argument les coordonées des pixels non classés qui ne sont pas de l'ombre
    
    data_non_classee = dataset[:,abs_nonclass,ord_nonclass] #points de dataset qui sont mal classé et non ombre
    distances_to_center = np.zeros((np.shape(data_non_classee))) #lignes = n° du cluster,colonnes = la donnée pour laquelle on calcule la distance au centre du clusteur
    best_cluster = np.zeros((2,np.shape(data_non_classee)[1])) #pour chaque data en colonne, on a : ligne 0 = n° du cluster le plus proche
                                                                                                   #ligne 1 = distance entre la donnée et le cluster le + proche
    abs_data_rejet = [] #abscisses et ordonnées des data rejetées. 
    ord_data_rejet = []
    
    T = 0.7 #seuil de rejet à calibrer (se rapporte à 2*sigma ou 3*sigma)
        
    for i in range(np.shape(abs_nonclass)[0]) : #pour chaque donnée non classée...
        for k in range(nb_class): #...on calcule sa distance au centre de chaque cluster
            distances_to_center[k,i] = mesure(clusters[k,:], data_non_classee[:,i]) #attention à bien prendre les lignes dans clusters pour calculer la distance
        best_cluster[0,i] = np.argmin(distances_to_center[:,i])#on retient le n° du cluster le plus proche. C'est le cluster auquel appartient la donnée.
        best_cluster[1,i] = np.min(distances_to_center[:,i])#on retient la valeur de cette distance la plus proche
        
        if best_cluster[1,i] >= T*rayons[int(best_cluster[0,i])]: #si la distance donnée-centre cluster > T*rayon cluster
            abs_data_rejet.append(abs_nonclass[i])#on retient l'abscisse et l'ordonnée de la donnée que l'on va rejeter
            ord_data_rejet.append(ord_nonclass[i])
            
    mat_cluster[abs_data_rejet,ord_data_rejet] = -2 #pixels rejetés
    return mat_cluster


#*******************************************************************************************************
###### TRACER LES CARTES :
#*******************************************************************************************************

## fonction pour avoir les mêmes numéros associés aux clusters et aux arbres du jeu ded données datset2
## + l'ombre est mise à -1 et le rejet à -2
def map_matrice(mat_cluster,newclass,classes_connues,val_ombre,val_rejet,nb_class):
    mat_copy = np.copy(mat_cluster)
    num_classes = np.array(np.unique(classes_connues),dtype=int) #n° originla des espèces d'arbre
    #map les numéros des classes aux numéros des clusters
    for i in range(nb_class):
        pos = np.where(mat_cluster==newclass[i])
        mat_copy[pos] = num_classes[i]
    #ajout classe rejet et ombre
    mat_copy[np.where(mat_cluster==val_ombre)] = -1
    mat_copy[np.where(mat_cluster==val_rejet)] = -2
    return mat_copy

## fonction pour plot la matrice obtenue (dans les mêmes couleurs pour la matrice des clusters détermniés 
## par l'algo et pour la matrice des classes de dataset2)
def plot_map_matrice(dataset,dic_arbres,nb_class,title):
    #VERIFIER AVEC QGIS QUE LA LEGENDE EST BIEN ASSOCIEE AUX BONS ARBRES !!  
    
    #pour la légende au graphe :
    values = list(dic_arbres.keys())
    names = list(dic_arbres.values())
    
    ### définition de la map de couleur :
    #------ les valeurs dans dataset ne se suivent pas, elles valent : [1,2,3,4,7,9,13,14,15,16] 
    #------ donc on ajoute 4 fois la même couleur entre la classe 9 et 13 par exemple pour pallier à ce pb
    color_list= ['red','black','black','gold','saddlebrown','green','blueviolet','blueviolet','blueviolet',
                                   'skyblue', 'skyblue','chartreuse','chartreuse','chartreuse','chartreuse',
                                   'lightpink','darkgrey','royalblue','ivory']
    if len(np.unique(dataset)) == nb_class+1 : #si c'est une image sans le rejet
        cmap2 = colors.ListedColormap(color_list[2:])
        values = np.insert(values,0,-1) #ajoute la classe d'ombre (valeur -1)
        names = np.insert(names,0,'0mbre') #ajoute la classe d'ombre
    else : #si c'est une image avec le rejet
        cmap2 = colors.ListedColormap(color_list)
        values = np.insert(values,0,-1) #ajoute la classe d'ombre (valeur -1)
        values = np.insert(values,0,-2) #ajoute la classe de rejet (valeur -2)
        names = np.insert(names,0,'0mbre') #ajoute la classe d'ombre
        names = np.insert(names,0,'Rejet') #ajoute la classe de rejet
            
    
    ### tracé du graphe : 
    plt.figure(figsize=(15,15))
    im = plt.imshow(dataset,cmap = cmap2)
    plt.title (title)
    #plt.colorbar() #barre de couleurs
    #------ get the colors of the values, according to the colormap used by imshow
    couleur = [im.cmap(im.norm(value)) for value in values]
    #------ put those patched as legend-handles into the legend
    patches = [ mpatches.Patch(color=couleur[i], label= names[i]) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    #plt.savefig('Classes_classification.png', dpi=600, bbox_inches='tight')
    plt.show()
    
#*******************************************************************************************************
##### TEST DU PROGRAMME
#*******************************************************************************************************
# mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = info_pre_classif(dataset,nb_raws,nb_columns)
# labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
# classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
# newkey, mat_result_kmeans = map_cluster(nb_class,classes_connues,dic_arbres,labels)
# reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newkey)

def test(dataset,dataset2):
    nb_class = np.shape(dataset)[0] 
    nb_raws = np.shape(dataset)[1] 
    nb_columns = np.shape(dataset)[2]
    val_ombre = -1
    val_rejet = -2
    
    #### Renormalisation des données avant d'appliquer le clustering
    """
    # On exclut les pixels d'ombre qui ne sont pas considérés comme des données.
    datasetR = dataset #copie du dataset pour renormalisation
    abs_pixels_no_ombre = np.concatenate((abs_pixels_classes,abs_nonclass))
    ord_pixels_no_ombre = np.concatenate((ord_pixels_classes,ord_nonclass))
    to_renorm = dataset[:,abs_pixels_no_ombre,ord_pixels_no_ombre]#matrice de taille nb_class x (nb de pixels non ombre) = données à renormaliser
    to_renorm = preprocessing.normalize(to_renorm, axis=0) #normalisation selon les colonnes de to_renorm
    datasetR[:,abs_pixels_no_ombre,ord_pixels_no_ombre] = to_renorm #datasetR est maintenant le dataset renormalisé.
    """
    
    #### Appel du clustering 
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_raws,nb_columns,RSTD.rule05)
    kmeans,labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newclass)
    
    #### Appel du rejet 
    mat_cluster = matrice_cluster(labels,nb_raws,nb_columns,abs_pixels_classes,ord_pixels_classes)
    mesure = wasserstein
    rayons = calcul_radius(clusters,dataset,mat_cluster,mesure,nb_class)
    mat_cluster =  rejet(dataset,rayons,clusters,abs_nonclass,ord_nonclass,mesure,nb_class,mat_cluster)
    mat_cluster = map_matrice(mat_cluster,newclass,classes_connues,val_ombre,val_rejet,nb_class)
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans,mat_cluster


labels, clusters, mat_result_kmeans, reussite_kmeans,mat_cluster = test(dataset,dataset2)   

# # Tracés des graphiques
nb_class = np.shape(dataset)[0] 
plot_map_matrice(dataset2[0,:,:],dic_arbres,nb_class,"Classes déterminiées par l'algorithme de classification log reg l1")
plot_map_matrice(mat_cluster,dic_arbres,nb_class, "Classes déterminiées par l'algorithme Kmeans log reg l1")

