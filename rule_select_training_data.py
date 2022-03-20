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
datasetrbf = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas pour la méthode logreg
dataset2rbf = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes pour la méthode logreg
datasetsvm=rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img") #matrice des probas pour la méthode SVM 
dataset2svm=rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes pour la méthode SVM
#remarque : l'ombre dans les deux résultats est bien placée au même endroit, tout va bien !!! 

dataset2svm=dataset2svm.read()
datasetrbf = datasetrbf.read()
dataset2rbf = dataset2rbf.read()
datasetsvm=datasetsvm.read()

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

def mix2data(data1, data2, class1, class2, abs_pixels_ombres, ord_pixels_ombres): #on lui donne la matrice pre classif de chaque méthode et les classes sortantes de chaque méthode, et nous redonne deux vecteurs : 
    #1 avec les indices communs aux deux méabs_pixels_ombres, ord_pixels_ombresthodes considéréss comme bien classés. et un avec les indices parmi les précédents indices dont la classe entre les deux méthodes est la même 
    mat_classif_commune_vec=2*np.ravel(np.ones(np.shape(data1))) #même taille que data1 avec que des 2
    mat_classif_communeclasse=2*(np.ones(np.shape(data1)))
    mat_classif_communeclasse[abs_pixels_ombres, ord_pixels_ombres]=0
    value, count=np.unique(np.concatenate((np.where(np.ravel(data1)==1)[0], np.where(np.ravel(data2)==1)[0])), return_counts=True) 
    #on regarde combien de fois apparaissent chaque coefficient de la matrice considéré comme bien classifié. Si un même indice apparait deux fois, cela signifie qu'il est considéré comme classfié presque surement par les deux méthodes. 
    indicesur=value[np.where(count==2)[0]] #on récupère les indices des pixels classés au dessus du seuil (0.5) 2 fois c'est à dire validé par 2 méthodes^
    print("il y a bien ", np.shape(value[np.where(count==1)])[0], " classés différement dans les 2 méthodes")
    mat_classif_commune_vec[list(indicesur)]=1 #on met à 1 lespixels bien classés par les deux méthodes 
    mat_classif_commune_vec=mat_classif_commune_vec.reshape(np.shape(data1))
    mat_classif_commune_vec[abs_pixels_ombres, ord_pixels_ombres]=0
    
    absc, ordo=np.where(mat_classif_commune_vec==1)
    indicesurclassediff=np.where(class1!=class2)
    indicesurmemeclasse=np.where(np.ravel(class1[0,:,:])==np.ravel(class2[0,:,:]))[0]
    print(indicesurmemeclasse)
    value2, count2=np.unique(np.concatenate((indicesur, indicesurmemeclasse)), return_counts=True) 
    indicesurbienclasse=value2[np.where(count2==2)[0]]
    mat_classif_communeclasse=np.ravel(mat_classif_communeclasse)
    #on y met tous les indices des pixels considérés comme bien classés et dont la classe obtenue est identique pour les deux algos 
    mat_classif_communeclasse[list(indicesurbienclasse)]=1
    mat_classif_communeclasse=mat_classif_communeclasse.reshape(np.shape(data1))
    mat_classif_commune_vec=mat_classif_commune_vec.reshape(np.shape(data1))
    
    return mat_classif_commune_vec, mat_classif_communeclasse

def info_pre_classifbis(data1, data2, class1, class2,nb_raws,nb_columns,rule,choix=1, diff=0) : 
    #choix=0 on veut tous ceux qui ont été classés >0.5 par tous les algo
    #choix=1 on veut tous ceux qui ont été classés >0.5 et avec les mêmes classes 
    #choix 2 : pixels qui ont eu la même classfication dans les deux méthodes 
    #rule sera remplacé par le nom de la fonction dont on choisit la règle
    
    ##  mat_pre_classif : matrice qui contient des int indiquant le niveau de classification de la matrice d'origine :
        # 0 = pixel d'ombre
        # 1 = pixel bien classé (avec une proba > 0.5) => va constituer l'échantillon de tests
        # 2 = pixel mal classé (avec une proba < 0.5) qu'il faudra prédire
    # on commence par une matrice remplie de 2 => les pixels mal classés sont ainsi rentrés par défaut
    #### En fait cette matrice m'a pas servi pour le moment mais elle servira par la suite pour repérer les indices des arbres à prédire je pense
    mat_pre_classif1, abs_pixels_classes1, ord_pixels_classes1, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass1, ord_nonclass1= info_pre_classif(data1,nb_raws,nb_columns,rule)
    mat_pre_classif2, abs_pixels_classes2, ord_pixels_classes2, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass2, ord_nonclass2 = info_pre_classif(data2,nb_raws,nb_columns,rule)

    matrice=mix2data(mat_pre_classif1, mat_pre_classif2, dataset2rbf, dataset2svm, abs_pixels_ombres, ord_pixels_ombres)
    mat_pre_classif = matrice[choix]
    print(np.shape(matrice), np.shape(matrice[choix]))
    #Repérer les indices des pixels bien classés et des zones d'ombres : 
    abs_pixels_classes, ord_pixels_classes = np.where(mat_pre_classif==1) #pixels bien classés
    abs_pixels_ombres, ord_pixels_ombres = np.where(mat_pre_classif==0) #pixels d'ombre
    abs_nonclass, ord_nonclass = np.where(mat_pre_classif==2) #pixels d'ombre
    
    return mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres,abs_nonclass, ord_nonclass


#### TESTS DU PROGRAMME : ################################################################
nb_raws = np.shape(datasetrbf)[1] 
nb_columns = np.shape(datasetrbf)[2]

### test règle 0.5: 
mat_pre_classifrbf, abs_pixels_classesrbf, ord_pixels_classesrbf, abs_pixels_ombres, ord_pixels_ombres, abs_nonclassrbf, ord_nonclassrbf = info_pre_classif(datasetrbf,nb_raws,nb_columns,rule05)
mat_pre_classifsvm, abs_pixels_classessvm, ord_pixels_classessvm, abs_pixels_ombres, ord_pixels_ombres, abs_nonclasssvm, ord_nonclasssvm = info_pre_classif(datasetsvm,nb_raws,nb_columns,rule05)

### test règle 0.5 +  diff : 
diff = 0.1
mat_pre_classif2rbf, abs_pixels_classes2rbf, ord_pixels_classes2rbf, abs_pixels_ombres2rbf, ord_pixels_ombres2rbf, abs_nonclass2rbf, ord_nonclass2rbf = info_pre_classif(datasetrbf,nb_raws,nb_columns,rule05_ecartProbas,diff)


mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres,abs_nonclass, ord_nonclass=info_pre_classifbis( datasetrbf, datasetsvm, dataset2rbf, dataset2svm,nb_raws,nb_columns,rule05_ecartProbas) 