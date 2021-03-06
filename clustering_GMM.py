# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:53:45 2022

@author: Lila
"""
import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from exploration_data_complet import data_proba, datat_class

## petit jeu de données :
#dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
#dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
#dataset = dataset.read()
#dataset2 = dataset2.read()

dataset = data_proba
dataset2 = datat_class
'''
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


##### extraction des classes à partir des probas avec utilisation de GMM-----------------------------
def apply_GMM(dataset,nb_class,abs_pixels_classes, ord_pixels_classes):
    #qd on fait un GMM sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # donc on transpose notre matrice 
    gmm = GaussianMixture(n_components=nb_class, random_state=0).fit_predict(dataset[:,abs_pixels_classes, ord_pixels_classes].T)
    return gmm

##### extraction des classes à partir du fichier fourni dataset2-----------------------------------------
def extract_classes_connues (dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres) :
    classes_connues = dataset2[:,abs_pixels_classes, ord_pixels_classes].T
    classes_connues = np.vectorize(dic_arbres.get)(classes_connues) #remplacer les chiffres par les noms 
    return classes_connues

##### trouver les correspondances entre les clusters de GMM et les classes connues---------------------
# Matrice de confusion pour GMM : mat_result_gmm
# en colonne : les arbres connus 
# en ligne : le nombre de fois où ils sont associés au ieme label donné par GMM 
def map_cluster(nb_class,classes_connues,dic_arbres,labels) :
    mat_result_gmm = np.zeros((nb_class,nb_class))
    j = 0
    for arbre in dic_arbres.values():
        pos = np.where(classes_connues==arbre)[0]
        for i in range(nb_class) :
            mat_result_gmm[i,j] = len(np.where(labels[pos]==i)[0])
        j+=1
    
    newkey = np.argmax(mat_result_gmm, axis=0) #n° classes qui sont les plus associées à chaque arbre
    new_dic_arbres = dict(zip(newkey, dic_arbres.values())) #arbre associé à chaque cluster de kmeans
    return newkey, new_dic_arbres, mat_result_gmm

# pourcentage de réussite de kmeans : 
def eval_GMM(nb_class,mat_result_gmm,newkey): 
    reussite_gmm =  mat_result_gmm[newkey,range(nb_class)]/np.sum(mat_result_gmm,axis=0)   
    return reussite_gmm

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
    
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = info_pre_classif(dataset,nb_raws,nb_columns)
    labels_gmm = apply_GMM(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes, dic_arbres) 
    newkey_gmm, new_dic_arbres_gmm, mat_result_gmm = map_cluster(nb_class,classes_connues,dic_arbres,labels_gmm)
    reussite_gmm = eval_GMM(nb_class,mat_result_gmm,newkey_gmm)
    
    return labels_gmm,  mat_result_gmm, reussite_gmm

labels_gmm, mat_result_gmm, reussite_gmm = test(dataset,dataset2)