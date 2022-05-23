#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:10:24 2022

@author: lea
#ALGORITHME SVM OVO AND OVR 
"""

# importation de librairies
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
import scipy
#from sklearn.cluster import gmm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping



# importation d'autres codes
import rule_select_training_data as RSTD
import maskutilisation as MASKS
import file_data_rasterio as FDR




# *******************************************************************************************************
# DEFINITION DES FONCTIONS :
# *******************************************************************************************************

from sklearn.calibration import CalibratedClassifierCV

def apply_svm_OVO(dataset, dataset2, nb_class, abs_pixels_classes, ord_pixels_classes, abs_nonclass, ord_nonclass, test=False):
    if test==False:
        XX = dataset[:, abs_pixels_classes, ord_pixels_classes].T
        YY = dataset2[:, abs_pixels_classes, ord_pixels_classes].T
        
        Yt = dataset2[:, abs_nonclass, ord_nonclass].T
        Xt = dataset[:, abs_nonclass, ord_nonclass].T
    
    else:
        XX = abs_pixels_classes
        YY = ord_pixels_classes
        
        Xt = abs_nonclass
        Yt = ord_nonclass
    
    clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(XX, np.ravel(YY))

# tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
    #score_onesvm =model.score(Xt, Yt)
    #confu_svm = pd.DataFrame(confusion_matrix(Yt, predict_one_svm))
    predict_one_svm = clf.predict(Xt)
    decision_one_svm = clf.decision_function(Xt)
# tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
    score_onesvm = clf.score(Xt, Yt)
    confu_svm = pd.DataFrame(confusion_matrix(Yt, predict_one_svm))
    #renvoyer model.predict_proba(Xt) à la place de decision
    return confu_svm, score_onesvm,decision_one_svm, predict_one_svm


def apply_svm_OVR(dataset, dataset2, nb_class, abs_pixels_classes, ord_pixels_classes, abs_nonclass, ord_nonclass,test=False):
    if test==False:
        XX = dataset[:, abs_pixels_classes, ord_pixels_classes].T
        YY = dataset2[:, abs_pixels_classes, ord_pixels_classes].T
        
        Yt = dataset2[:, abs_nonclass, ord_nonclass].T
        Xt = dataset[:, abs_nonclass, ord_nonclass].T
    
    else:
        XX = abs_pixels_classes
        YY = ord_pixels_classes
        
        Xt = abs_nonclass
        Yt = ord_nonclass
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(XX, np.ravel(YY))

# on est obligé d'utiliser np.ravel pour transformer en vecteur de taille (n_samples,) et plus (n_samples, 1)
    predict_one_svm = clf.predict(Xt)
    decision_one_svm = clf.decision_function(Xt)
# tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
    score_onesvm = clf.score(Xt, Yt)
    confu_svm = pd.DataFrame(confusion_matrix(Yt, predict_one_svm))

    return confu_svm, score_onesvm, decision_one_svm, predict_one_svm

# extraction des classes à partir du fichier fourni dataset2logreg-----------------------------------------


def extract_classes_connues(dataset2logreg, abs_pixels_classes, ord_pixels_classes):
    classes_connues = dataset2logreg[:,
                                     abs_pixels_classes, ord_pixels_classes].T
    return classes_connues

# trouver les correspondances entre les clusters de gmm et les classes connues---------------------clustering
# Matrice de confusion pour gmm : mat_result_gmm
# en colonne : les arbres connus
# en ligne : le nombre de fois où ils sont associés au ieme label donné par gmm


def map_cluster(nb_class, classes_connues, labels):
    mat_result_gmm = np.zeros((nb_class, nb_class))
    # n° originla des espèces d'arbre
    num_classes = np.array(np.unique(classes_connues), dtype=int)
    j = 0
    for num in num_classes:
        pos = np.where(classes_connues == num)[0]
        for i in range(nb_class):
            mat_result_gmm[i, j] = len(np.where(labels[pos] == i)[0])
        j += 1

    # n° classes qui sont les plus associées à chaque arbre
    newclass = np.argmax(mat_result_gmm, axis=0)
    return newclass, mat_result_gmm

# pourcentage de réussite de gmm :


def eval_gmm(nb_class, mat_result_gmm, newkey):
    reussite_gmm = mat_result_gmm[newkey, range(
        nb_class)]/np.sum(mat_result_gmm, axis=0)
    return reussite_gmm

# prédiction des pixels mal classés :


def predict(dataset, mat_pre_classif, GMM):
    abs_pixels_apred, ord_pixels_apred = np.where(mat_pre_classif == 2)
    prediction = GMM.predict(dataset[:, abs_pixels_apred, ord_pixels_apred].T)
    probaprediction = GMM.predict_proba(
        dataset[:, abs_pixels_apred, ord_pixels_apred].T)
    return prediction, probaprediction

# création d'une matrice contenant le résultat du clustering------------------------------------------


def matrice_cluster(labels, nb_raws, nb_columns, abs_pixels_classes, ord_pixels_classes):
    # matrice_cluster associe à chaque pixel bien classé le cluster obtenu
    # -1 = pas de cluster associé à ce pixel
    mat_cluster = (-1)*np.ones((nb_raws, nb_columns))

    for i in range(len(abs_pixels_classes)):
        mat_cluster[abs_pixels_classes[i], ord_pixels_classes[i]] = labels[i]
    return mat_cluster


# *******************************************************************************************************
# DEFINITION DES GRAPHIQUES :
# *******************************************************************************************************

# Graphique des résultats de l'algorithme de classification------------------------------------------

# Graphique des résultats de l'algorithme de Clustering & Rejet--------------------------------------


def plot_resultat_cluster(mat_cluster, nb_class, name, nametitre):
    # Cette procédure trace les résultats du clustering et du rejet

    # définition de la map de couleur :
    # ------ les valeurs dans mat_cluster se suivent, elles valent : [-1,-2,0,1,2,3,4,5,6,7,9]
    # ------ on ne double donc pas les couleurs.
    cmap2 = colors.ListedColormap(['red', 'black', 'gold', 'saddlebrown', 'green', 'blueviolet',
                                   'skyblue', 'chartreuse',
                                   'lightpink', 'darkgrey', 'royalblue', 'ivory'])

    # tracé du graphe :
    plt.figure(figsize=(15, 15))
    im2 = plt.imshow(mat_cluster, cmap=cmap2)
    plt.title(nametitre)

    # ajout de la légende au graphe :
    # ------ définition des labels
    values_cluster = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2]  # 10 clusters,

    names_cluster = ["Platane", "Saule", "Peuplier", "Chene", "Aulnes", "Robiniers",
                     "Melanges_Herbaces", "Melanges_Arbustifs", "Renouees_du_Japon", "Mais"]

    names_cluster.append("Ombre")  # nom de la classe ombre
    names_cluster.append("Rejet")  # nom de la classe rejet

    # ------ get the colors of the values, according to the colormap used by imshow
    couleur = [im2.cmap(im2.norm(value)) for value in values_cluster]
    # ------ put those patched as legend-handles into the legend
    patches = [mpatches.Patch(color=couleur[i], label=names_cluster[i])
               for i in range(len(values_cluster))]
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)
    title="Classes_clustering_"+str(name)+".png"
    plt.grid(True)
    plt.show()
    plt.savefig(title, dpi=600, bbox_inches='tight')
    
    

def plot_resultat_clusterplat(mat_cluster, nb_class, name, nametitre):
    # Cette procédure trace les résultats du clustering et du rejet

    # définition de la map de couleur :
    # ------ les valeurs dans mat_cluster se suivent, elles valent : [-1,-2,0,1,2,3,4,5,6,7,9]
    # ------ on ne double donc pas les couleurs.
    cmap2 = colors.ListedColormap(['red', 'black', 'saddlebrown', 'green', 'blueviolet',
                                   'skyblue', 'chartreuse',
                                   'lightpink', 'darkgrey', 'royalblue', 'ivory'])

    # tracé du graphe :
    plt.figure(figsize=(15, 15))
    im2 = plt.imshow(mat_cluster, cmap=cmap2)
    plt.title(nametitre)

    # ajout de la légende au graphe :
    # ------ définition des labels
    values_cluster = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2]  # 10 clusters,

    names_cluster = ["Saule", "Peuplier", "Chene", "Aulnes", "Robiniers",
                     "Melanges_Herbaces", "Melanges_Arbustifs", "Renouees_du_Japon", "Mais"]

    names_cluster.append("Ombre")  # nom de la classe ombre
    names_cluster.append("Rejet")  # nom de la classe rejet

    # ------ get the colors of the values, according to the colormap used by imshow
    couleur = [im2.cmap(im2.norm(value)) for value in values_cluster]
    # ------ put those patched as legend-handles into the legend
    patches = [mpatches.Patch(color=couleur[i], label=names_cluster[i])
               for i in range(len(values_cluster))]
    plt.legend(handles=patches, bbox_to_anchor=(
        1.05, 1), loc=2, borderaxespad=0.)
    title="Classes_clustering_"+str(name)+".png"
    plt.grid(True)
    plt.show()
    plt.savefig(title, dpi=600, bbox_inches='tight')




#FDR.save_img(mat_cluster, "svm.img", meta, nb_rows, nb_columns)
#taille image=3 ==> 376988 pixels rejetés 
#taille image =2==> 371153 pixels rejetés 
#taille image =1==>387345 pixels rejetés 