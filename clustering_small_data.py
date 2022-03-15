#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 21:03:25 2022

@author: julie
"""

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import rule_select_training_data as RSTD 


## petit jeu de données lrl :
# dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
# dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  lrl : 
dataset = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()
'''
Notre d'ictionnaire d'arbres : implémenté dans la fonction test
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

nb_class = np.shape(dataset)[0] 
nb_raws = np.shape(dataset)[1] 
nb_columns = np.shape(dataset)[2]
'''

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
def extract_classes_connues (dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres) :
    classes_connues = dataset2[:,abs_pixels_classes, ord_pixels_classes].T
    classes_connues = np.vectorize(dic_arbres.get)(classes_connues) #remplacer les chiffres par les noms 
    return classes_connues

##### trouver les correspondances entre les clusters de kmeans et les classes connues---------------------
# Matrice de confusion pour Kmeans : mat_result_kmeans
# en colonne : les arbres connus 
# en ligne : le nombre de fois où ils sont associés au ieme label donné par Kmeans 
def map_cluster(nb_class,classes_connues,dic_arbres,labels) :
    mat_result_kmeans = np.zeros((nb_class,nb_class))
    j = 0
    for arbre in dic_arbres.values():
        pos = np.where(classes_connues==arbre)[0]
        for i in range(nb_class) :
            mat_result_kmeans[i,j] = len(np.where(labels[pos]==i)[0])
        j+=1
    
    newkey = np.argmax(mat_result_kmeans, axis=0) #n° classes qui sont les plus associées à chaque arbre
    new_dic_arbres = dict(zip(newkey, dic_arbres.values())) #arbre associé à chaque cluster de kmeans
    return newkey, new_dic_arbres, mat_result_kmeans

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
# classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres) 
# newkey, new_dic_arbres, mat_result_kmeans = map_cluster(nb_class,classes_connues,dic_arbres,labels)
# reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newkey)

def test(dataset,dataset2):
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

    nb_class = np.shape(dataset)[0] 
    nb_raws = np.shape(dataset)[1] 
    nb_columns = np.shape(dataset)[2]
    
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_raws,nb_columns,RSTD.rule05)
    kmeans,labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = extract_classes_connues (dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres)
    newkey, new_dic_arbres, mat_result_kmeans = map_cluster(nb_class,classes_connues,dic_arbres,labels)
    reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newkey)
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans

labels, clusters, mat_result_kmeans, reussite_kmeans = test(dataset,dataset2)   





