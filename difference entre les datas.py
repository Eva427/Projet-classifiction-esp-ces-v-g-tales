#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance 
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

## gros jeu de données :----------------------------------------------------------------
dataset = rasterio.open("./insa/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./insa/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes")
datasetsvm=rasterio.open("./insa/none_ite_0_proba_SVM_rbfcombined_mean_proba.img")
dataset = dataset.read()
dataset2 = dataset2.read()
datasetsvm=datasetsvm.read()
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

from scipy.special import rel_entr


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
    abs_pixels_ombres, ord_pixels_ombres = np.where(somme_10mat==0.) #pixels d'ombre
    
    mat_pre_classif[abs_pixels_classes, ord_pixels_classes]=1
    mat_pre_classif[abs_pixels_ombres, ord_pixels_ombres]=0
    abs_nonclass, ord_nonclass=np.where(mat_pre_classif==2)
    return mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass

mat=np.zeros((nb_raws,nb_columns)) #matrice composée des max pour chaque pixel du vecteur de proba de taille 10
matarg=np.zeros((nb_raws,nb_columns)) #matrice des classes correspondantes aux max 
for i in range (nb_columns):
    mat[:,i]=np.max(dataset[:,:,i],0)
    matarg[:,i]=np.argmax(dataset[:,:,i],0)




#array([[   0,    0,    0, ..., 2515, 2515, 2515],
#[   0,    1,    2, ..., 2470, 2471, 2472]])

mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass=info_pre_classif(dataset,nb_raws,nb_columns)
mat_pre_classif2, abs_pixels_classes2, ord_pixels_classes2, abs_pixels_ombres2, ord_pixels_ombres2, abs_nonclass2, ord_nonclass2=info_pre_classif(datasetsvm,nb_raws,nb_columns)
#définition d'un échantillon test et d'apprentissage 
X_train, X_test, Y_train, Y_test=train_test_split(dataset[:,abs_pixels_classes, ord_pixels_classes].T,dataset2[:,abs_pixels_classes, ord_pixels_classes].T,test_size=0.25, random_state=11)

classif2vec=np.ravel(mat_pre_classif2)  #transforme la matrice en vecteur
classifvec=np.ravel(mat_pre_classif)

nb_pixels_sans_ombre=np.shape(dataset)[1]*np.shape(dataset)[2]-np.shape(np.where(mat_pre_classif==0))[1]
indice_pas_en_commun=set(np.where(classifvec==1.0)[0]).symmetric_difference(set((np.where(classif2vec==1.0)[0])))
print("nb élements considérés bien classifiés dans la classif par rbf et pas dans svm : ", len(set(np.where(classifvec==1.0)[0]).difference(set(np.where(classif2vec==1.0)[0]))))
print("nb élements considérés bien classifiés dans la classif par svm et pas dans rbf : ", len(set(np.where(classif2vec==1.0)[0]).difference(set(np.where(classifvec==1.0)[0]))))
print("nombre d'élements totaux mal classifiés dans une des deux méthodes : ", len(indice_pas_en_commun), " soit ", len(set(np.where(classifvec==1.0)[0]).symmetric_difference(set((np.where(classif2vec==1.0)[0]))))/nb_pixels_sans_ombre, "% du nombres de pixels en dehors de ceux correspondants à l'ombre")

#permet d'obtenir classification plus certaine pour l'entrainement (suelement pixels avec proba >0.5)
mat_classif_commune_vec=np.copy(classifvec)
mat_classif_commune_vec[list(indice_pas_en_commun)]=2
mat_classif_commune_vec.reshape(np.shape(mat_pre_classif))


from scipy.stats import wasserstein_distance

def KL_divergence(data1, data2):
    DivKL=np.zeros((nb_raws, nb_columns))
    Distwass=np.zeros((nb_raws, nb_columns))
    for i in range (nb_raws): 
        for j in range (nb_columns):
            DivKL[i,j]=sum(rel_entr(data1[:,i,j], data2[:,i,j]))
            #Distwass[i,j]=wasserstein_distance(data1[:,i,j], data2[:,i,j])
            
    DivKL[abs_pixels_ombres, ord_pixels_ombres]=10000 
    Distwass[abs_pixels_ombres, ord_pixels_ombres]=10000 
    #mettre l'indicateur très grand là où il y a de l'ombre
    return DivKL, Distwass
div, wass=KL_divergence(dataset, datasetsvm)

print("summary sur la KL divergence entre svm et rbf pour les données qui sont pas l'ombre")
KL=pd.Series(np.ravel(np.ravel(div[div!=10000])))
print(KL.describe())#permet de voir les valeurs remarquables de la KL_divergence 

#wass=pd.Series(np.ravel(np.ravel(wass[wass!=10000])))
#print(wass.describe())#permet de voir les valeurs remarquables de la KL_divergence 

