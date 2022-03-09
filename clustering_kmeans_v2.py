#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:34:39 2022

@author: julie
"""
### Meme code que dans clustering_small_data mais en enlevant le dictionnaire => plus facle à manipuler.

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans

## petit jeu de données :
dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()

def info_pre_classif(dataset,nb_raws,nb_columns) :
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
    return mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres

##### extraction des classes à partir des probas avec utilisation de kmeans-----------------------------
def apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes):
    #qd on fait un kmeans sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # cond on transpose notre matrice 
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
    abs_pixels_apred,ord_pixels_apre = np.where(mat_pre_classif==2)
    prediction = kmeans.predict(dataset[:,abs_pixels_apred, ord_pixels_apre].T)
    return abs_pixels_apred,ord_pixels_apre,prediction

############# test du programme ###################################################
# mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = info_pre_classif(dataset,nb_raws,nb_columns)
# labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
# classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
# newkey, mat_result_kmeans = map_cluster(nb_class,classes_connues,dic_arbres,labels)
# reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newkey)

def test(dataset,dataset2):
    nb_class = np.shape(dataset)[0] 
    nb_raws = np.shape(dataset)[1] 
    nb_columns = np.shape(dataset)[2]
    
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = info_pre_classif(dataset,nb_raws,nb_columns)
    kmeans,labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newclass)
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans

labels, clusters, mat_result_kmeans, reussite_kmeans = test(dataset,dataset2)   