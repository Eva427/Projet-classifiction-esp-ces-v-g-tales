#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:10:24 2022

@author: julie
-Rq : Meme code que dans clustering_small_data mais en enlevant le dictionnaire => plus facile à manipuler
"""

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
import rule_select_training_data as RSTD 

##### EXTRACTION DES DONNEES :

## petit jeu de données :
# dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
# dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  lrl : 
# dataset = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
# dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  svm : 
dataset = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()


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
    
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_raws,nb_columns,RSTD.rule05)
    kmeans,labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newclass)
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans

labels, clusters, mat_result_kmeans, reussite_kmeans = test(dataset,dataset2)   