#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:46:12 2022

@author: julie
https://stackoverflow.com/questions/28862334/k-means-with-selected-initial-centers
"""

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
import clustering_small_data as csd

## petit jeu de données :---------------------------------------------------------------
small_dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
small_dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
small_dataset = small_dataset.read()
small_dataset2 = small_dataset2.read()

#### kmeans avec centroïdes précalulés :-----------------------------------------------
def semi_supervised_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes,clusters1):
    #qd on fait un kmeans sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # cond on transpose notre matrice 
    kmeans = KMeans(n_clusters=nb_class, random_state=0, init=clusters1).fit(dataset[:,abs_pixels_classes, ord_pixels_classes].T) 
    labels = kmeans.labels_
    clusters = kmeans.cluster_centers_
    return labels, clusters 

## gros jeu de données :----------------------------------------------------------------
dataset = rasterio.open("./Data/combined_svm_rbf_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./Data/combined_svm_rbf.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()
dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "inconnu",
              17: "inconnu2"}

nb_class = np.shape(dataset)[0] 
nb_raws = np.shape(dataset)[1] 
nb_columns = np.shape(dataset)[2]

##### calcul des clusters obtenus sur notre petit jeu de données : 
labels1, clusters1, mat_result_kmeans1, reussite_kmeans1 = csd.test(small_dataset,small_dataset2)
cluster_inconnu = np.zeros(np.shape(clusters1)[0])
cluster_inconnu2 = np.zeros(nb_class).reshape((1,nb_class))
clusters1 = np.c_[clusters1,cluster_inconnu]
clusters1 = np.r_[clusters1,cluster_inconnu2]

mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = csd.info_pre_classif(dataset,nb_raws,nb_columns)
labels, clusters = semi_supervised_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes, clusters1)
classes_connues = csd.extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres) 
newkey, new_dic_arbres, mat_result_kmeans = csd.map_cluster(nb_class,classes_connues,dic_arbres,labels)
reussite_kmeans = csd.eval_kmean(nb_class,mat_result_kmeans,newkey)






