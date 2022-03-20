#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:48:49 2022

@author: julie
"""
###### BUT DE CE FICHIER : Implémenter les différentes règles qu'on pourrait mettre en place pour
# sélectionner les données que l'on consièdre comme étant bien classées. Suggestions : 
# - règle d'une proba > 0.5
# - règle de la plus grande proba > 0.5 + d'un écart suffisant entre les 1ère et 2ème plus grandes probas 
#   => pour éviter la confusion entre deux classes + pouvoir rejeter efficacement les bordures des arbres. 
# - règle que le pixel sorte classé par les 3 algos (en cours d'implémentation chez Lélé *je crois*)
#### Rq : - Code général pour renvoyer la matrice de pré classification quelque soit la règle chosie
####       => il faut donc que les fonctions rule respectent un certains template
####      - On pourra aussi mettre plusieurs règles en cascade si on veut de fortes restrictions

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans

##### EXTRACTION DES DONNEES ###########################################################

## petit jeu de données :
# dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
# dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  lrl : 
dataset = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()

## gros jeu de données  svm : 
# dataset = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img") #matrice des probas
# dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

##### IMPLEMNTATION DES FONCTIONS RULE ##################################################

def rule05 (dataset,nb_raws,nb_columns,diff=0) :
    #diff inutile mais ça permet d'avoir les mêmes arguments dans mes deux fonctions
    _, abs_pixels_classes, ord_pixels_classes = np.where(dataset[:,:,:]>0.5)
    return abs_pixels_classes, ord_pixels_classes

def rule05_ecartProbas (dataset,nb_raws,nb_columns,diff=0) :
    abs_pixels_classes05, ord_pixels_classes05 = rule05(dataset,nb_raws,nb_columns)
    mat05 = dataset[:,abs_pixels_classes05, ord_pixels_classes05] #contient les vecteurs de probas des pixels classés selon la règle >0.5
    first,second = (-mat05).argsort(axis=0)[:2]
    #first : indices des positions des plus grosses valeurs de proba pour chaque pixel
    #second : indices des positions des 2èmes plus grosses valeurs de proba pour chaque pixel
    ind = np.arange(0,len(abs_pixels_classes05)) #pour sélectionner la bonne valeur dans first et second et la bonne colonne de mat05
    list_diff = np.abs(mat05[first[ind],ind]-mat05[second[ind],ind]) #différence entre la plus grosse et la plus petite valeur
    ind_diff = np.where(list_diff>diff) #sélectionne les indices pour lesquels la différence est > à un seuil
    abs_pixels_classesdiff = abs_pixels_classes05[ind_diff]
    ord_pixels_classesdiff= ord_pixels_classes05[ind_diff]
    return abs_pixels_classesdiff,ord_pixels_classesdiff

##### DETERMINATION DES ECHANTILLONS D'ENTRAINEMENT #####################################
def info_pre_classif(dataset,nb_raws,nb_columns,rule,diff=0) :
    #rule sera remplacé par le nom de la fonction dont on choisit la règle
    mat_pre_classif = 2*np.ones((nb_raws,nb_columns)) 
    ##  mat_pre_classif : matrice qui contient des int indiquant le niveau de classification de la matrice d'origine :
        # 0 = pixel d'ombre
        # 1 = pixel bien classé (avec une proba > 0.5) => va constituer l'échantillon de tests
        # 2 = pixel mal classé (avec une proba < 0.5) qu'il faudra prédire
    # on commence par une matrice remplie de 2 => les pixels mal classés sont ainsi rentrés par défaut
    #### En fait cette matrice m'a pas servi pour le moment mais elle servira par la suite pour repérer les indices des arbres à prédire je pense
    
    #Repérer les indices des pixels bien classés et des zones d'ombres : 
    abs_pixels_classes, ord_pixels_classes = rule (dataset,nb_raws,nb_columns,diff) #pixels bien classés
    somme_10mat = np.sum(dataset,axis=0) #somme des 10 matrices de proba => là où la somme fait 0 on a des vecteurs d'ombre
    abs_pixels_ombres, ord_pixels_ombres = np.where(somme_10mat==0) #pixels d'ombre
    
    mat_pre_classif[abs_pixels_classes, ord_pixels_classes]=1
    mat_pre_classif[abs_pixels_ombres, ord_pixels_ombres]=0
    abs_nonclass, ord_nonclass=np.where(mat_pre_classif==2) #coordonnées 
    return mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres,abs_nonclass, ord_nonclass


#### TESTS DU PROGRAMME : ################################################################
# nb_raws = np.shape(dataset)[1] 
# nb_columns = np.shape(dataset)[2]

### test règle 0.5: 
#mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = info_pre_classif(dataset,nb_raws,nb_columns,rule05)

### test règle 0.5 +  diff : 
#diff = 0.1
#mat_pre_classif2, abs_pixels_classes2, ord_pixels_classes2, abs_pixels_ombres2, ord_pixels_ombres2, abs_nonclass2, ord_nonclass2 = info_pre_classif(dataset,nb_raws,nb_columns,rule05_ecartProbas,diff)


