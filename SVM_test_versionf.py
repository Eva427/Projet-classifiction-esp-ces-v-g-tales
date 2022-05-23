#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:10:24 2022

@author: julie
- Rq : même version que clstering_gmm_v2 mais avec le rejet
- Rq : la fonction predict renvoie abs_pixels_apred et abs_pixels_apred, ce qui est la 
même chose que abs_nonclass et ord_nonclass renvoyés par rule_select_training_data. Peut-être supprimer 
abs_pixels_apred et abs_pixels_apred on s'en sert pas.
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
import maskutilisation as MASKSS

import file_data_rasterio as FDR

import CONTEXTE_FINAL as CTF
import SVM_versionf as SVMfield
from sklearn.model_selection import cross_val_score




###############################################################DATASETS CHARGEMENT##########################################################################
############################################################################################################################################################
# matrice des probas pour la méthode logregl1
logreg = "./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img"
# matrice des classes pour la méthode logregl1
logreg2 = "./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img"

datasetlogreg, metalogreg, dataset2logreg, meta2logreg = FDR.extract_data(
    logreg, logreg2)


# matrice des probas pour la méthode SVM
svm = "./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img"
# matrice des classes pour la méthode SVM
svm2 = "./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img"
datasetsvm, metasvm, dataset2svm, meta2svm = FDR.extract_data(svm, svm2)

# matrice des probas pour la méthode RF
rf = "./bonnes_data/none_ite_0_proba_RFcombined_mean_proba.img"
rf2="./bonnes_data/none_ite_0_proba_RFrejection_class.img"
# dataset2rf=rasterio.open("./bonnes_data/none_ite_0_proba_RFrejection_class.img") #matrice des probas pour la méthode RF
datasetrf, metarf, dataset2rf, meta2rf = FDR.extract_data(
    rf, rf2)

svmbis = "./bonnes_data/none_ite_0_proba_SVM_linearcombined_mean_proba.img"
svm2bis = "./bonnes_data/none_ite_0_proba_SVM_linearrejection_class.img"
# matrice des probas pour la méthode logregl2
datasetsvmbis, metasvmbis, dataset2svmbis, meta2svmbis = FDR.extract_data(
    svmbis, svm2bis)

logreg2 = "./bonnes_data/none_ite_0_proba_Log_reg_l2combined_mean_proba.img"
# matrice des classes pour la méthode logregl2
log2reg2 = "./bonnes_data/none_ite_0_proba_Log_reg_l2rejection_class.img"
datasetlogreg2, metalogreg2, dataset2logreg2, meta2logreg2 = FDR.extract_data(
    logreg2, log2reg2)

nb_class = np.shape(datasetlogreg)[0]
nb_raws = np.shape(datasetlogreg)[1]
nb_columns = np.shape(datasetlogreg)[2]
array=np.arange(nb_raws*nb_columns)




#comparaison des datasets
val, co=np.unique(np.concatenate((np.where(np.ravel(dataset2logreg)-np.ravel(dataset2logreg2)==0)[0], np.where(np.ravel(dataset2logreg)-np.ravel(dataset2svm)==0)[0], np.where(np.ravel(dataset2logreg)-np.ravel(dataset2svmbis)==0)[0],np.where(np.ravel(dataset2logreg)-np.ravel(dataset2rf)==0)[0])), return_counts=True)
5538672-4698859
val, co=np.unique(np.concatenate((np.where(np.ravel(dataset2logreg)-np.ravel(dataset2logreg2)==0)[0], np.where(np.ravel(dataset2logreg)-np.ravel(dataset2svm)==0)[0], np.where(np.ravel(dataset2logreg)-np.ravel(dataset2svmbis)==0)[0])), return_counts=True)
(5805799-4698859)/6222068
val[co==3]
array=0*array
array[np.where(co==3)]=-3
array.reshape(np.shape(dataset2logreg[0,:,:]))

dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "Mais"
              }

# Tailles

nb_class = np.shape(datasetlogreg)[0]
nb_raws = np.shape(datasetlogreg)[1]
nb_columns = np.shape(datasetlogreg)[2]




################################################## TESTS SUR OVO DIFFERENCE RULE AVEC CONTEXTE #######################################################
#mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classifbis(
    #datasetlogreg, datasetsvm, dataset2logreg, dataset2svm, nb_raws, nb_columns, RSTD.rule05 , choix=1, diff=0.1)
diff = 0.1
mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(datasetrf,nb_raws,nb_columns,RSTD.rule05_ecartProbas,0.05)
confu_svm, score_onesvm, decision_one_svm, predict_one_svm = SVMfield.apply_svm_OVO(
   datasetrf, dataset2rf, nb_class, abs_pixels_classes, ord_pixels_classes, abs_nonclass, ord_nonclass)
diffdecision1et2 = np.sort(
    ((-decision_one_svm.T+np.max(decision_one_svm, 1))/np.max(decision_one_svm, 1)).T, axis=1)[:, 1]
#predict_one_svm[np.where(diffdecision1et2 < np.median(diffdecision1et2))] = -2 #ligne à décommenter si on veut la médiane
predict_one_svm[np.where(diffdecision1et2 < np.percentile(diffdecision1et2, 75))] = -2 #ligne à décommenter si on veut le quartile
matcluster = np.zeros((nb_raws, nb_columns))
matcluster[abs_pixels_ombres, ord_pixels_ombres] = -1
matcluster[np.where(mat2sur == 2)[0], np.where(
    mat2sur == 2)[1]] = predict_one_svm
matcluster[abs_pixels_classes, ord_pixels_classes] = dataset2rf[0,
                                                                    abs_pixels_classes, ord_pixels_classes]


print("OVO", MASKSS.compare_masque(matcluster))

clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 1, abs_nonclass, ord_nonclass, matcluster, 0.6, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1

print("1,0.6  : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 2, abs_nonclass, ord_nonclass, matcluster, 0.6, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
    
print("2,0.6  : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 3, abs_nonclass, ord_nonclass, matcluster, 0.6, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
print("3,0.6  : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 4, abs_nonclass, ord_nonclass, matcluster, 0.6, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
print("4,0.6  : ", MASKSS.compare_masque(clasbis))

clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 1, abs_nonclass, ord_nonclass, matcluster, 0.4, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1

print("1,0.4  : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 2, abs_nonclass, ord_nonclass, matcluster, 0.4, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
    
print("2,0.4 : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 3, abs_nonclass, ord_nonclass, matcluster, 0.4, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
print("3,0.4  : ", MASKSS.compare_masque(clasbis))


clas, rejected = CTF.contexte_post_class(
    mat2sur, dataset2svm, 4, abs_nonclass, ord_nonclass, matcluster, 0.4, nb_raws, nb_columns)
l=0
datacolor=[1,2, 3, 4, 7, 9, 13, 14, 15, 16]
clasbis=np.copy(clas)
for i in np.arange(0,10):
    absc, ordo = np.where(clas == i)
    clasbis[absc, ordo] = datacolor[l]
    l += 1
print("4,0.4  : ", MASKSS.compare_masque(clasbis))



########################################################################"JEUX DE DONNEES AVEC UNE MISSING CLASS###################################################

####################################################################### 1) sans les platanes 
#Jeux de données dans lesquelles les platanes ou renouées ont été supprimés : 
#1- SANS PLATANES :
dataset_path_lrl1_noplat = "./bonnes_data/HS_no_plat_ite_0_proba_Log_reg_l1.img" #matrice des probas pour la méthode logreg1
dataset_path2_lrl1_noplat = "./bonnes_data/HS_no_plat_ite_0_class_Log_reg_l1.img" #matrice des classes pour la méthode logreg
dataset_path_svmrbf_noplat = "./bonnes_data/HS_no_plat_ite_0_proba_SVM_rbf.img" #matrice des probas pour la méthode logreg1
dataset_path2_svmrbf_noplat = "./bonnes_data/HS_no_plat_ite_0_class_SVM_rbf.img" #matrice des classes pour la méthode logreg
#2- SANS RENOUEES :
dataset_path_lrl1_noren = "./bonnes_data/HS_no_rey_ite_0_proba_Log_reg_l1.img" #matrice des probas pour la méthode logreg1
dataset_path2_lrl1_noren = "./bonnes_data/HS_no_rey_ite_0_class_Log_reg_l1.img" #matrice des classes pour la méthode logreg
dataset_path_svmrbf_noren = "./bonnes_data/HS_no_rey_ite_0_proba_SVM_rbf.img" #matrice des probas pour la méthode logreg1
dataset_path2_svmrbf_noren = "./bonnes_data/HS_no_rey_ite_0_class_SVM_rbf.img" #matrice des classes pour la méthode logreg



dataset_path_lrl1_norenlele, metanorenlr1, dataset_path2_lrl1_norenlele, meta2= FDR.extract_data(dataset_path_lrl1_noplat, dataset_path2_lrl1_noplat )
dataset_path_svm_norenlele, metanorensvm, dataset_path2_svm_norenlele, meta2bis= FDR.extract_data(dataset_path_svmrbf_noplat , dataset_path2_svmrbf_noplat )
dataset_path_lrl1_noren, metanorenlr1, dataset_path2_lrl1_noren, meta2= FDR.extract_data(dataset_path_lrl1_noren, dataset_path2_lrl1_noren )
dataset_path_svm_noren, metanorensvm, dataset_path2_svm_noren, meta2bis= FDR.extract_data(dataset_path_svmrbf_noren , dataset_path2_svmrbf_noren)




nb_class,nb_rows,nb_columns = np.shape(dataset_path_lrl1_norenlele)

#mat2surnoplat, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_pixels_ombresnoplat, ord_pixels_ombresnoplat, abs_nonclassnoplat, ord_nonclassnoplat = RSTD.info_pre_classifbis(
    #dataset_path_lrl1_norenlele, dataset_path_svm_norenlele,dataset_path2_lrl1_norenlele,dataset_path2_svm_norenlele, nb_raws, nb_columns, RSTD.rule05_ecartProbas)
mat2surnoplat, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_pixels_ombresnoplat, ord_pixels_ombresnoplat, abs_nonclassnoplat, ord_nonclassnoplat = RSTD.info_pre_classif(
    dataset_path_svm_norenlele,nb_raws,nb_columns,RSTD.rule05_ecartProbas, 0.1)
#mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(datasetlogreg,nb_raws,nb_columns,RSTD.rule05_ecartProbas,diff)
confu_svm, score_onesvm, decision_one_svm, predict_one_svm = SVMfield.apply_svm_OVR(
    dataset_path_svm_norenlele, dataset_path2_svm_norenlele, nb_class, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_nonclassnoplat, ord_nonclassnoplat)
diffdecision1et2 = np.sort(
    ((-decision_one_svm.T+np.max(decision_one_svm, 1))/np.max(decision_one_svm, 1)).T, axis=1)[:, 1]
#predict_one_svm[np.where(diffdecision1et2 < np.median(diffdecision1et2))] = -2
predict_one_svm[np.where(
      diffdecision1et2 < np.percentile(diffdecision1et2, 75))] = -2
#predict_one_svm[np.where(
   #diffdecision1et2 < np.median(diffdecision1et2))] = -2
matcluster = np.zeros((nb_raws, nb_columns))
matcluster[abs_pixels_ombresnoplat, ord_pixels_ombresnoplat] = -1
matcluster[np.where(mat2surnoplat == 2)[0], np.where(
    mat2surnoplat == 2)[1]] = predict_one_svm
matcluster[abs_pixels_classesnoplat, ord_pixels_classesnoplat] = dataset_path2_svm_norenlele[0,
                                                                    abs_pixels_classesnoplat, ord_pixels_classesnoplat]



print("OVO",MASKSS.compare_masque_platanus_renouees(matcluster,1) )

datacolor = np.arange(0, 10)
matclusteraffich = np.copy(matcluster)
l = 0
for i in [2, 3, 4, 7, 9, 13, 14, 15, 16]:
    absc, ordo = np.where(matclusteraffich == i)
    matclusteraffich[absc, ordo] = datacolor[l]
    l += 1
name = "OVO_0.75_quantile_0.1diff_logreg_svm_withoutplatane" + \
    str(np.shape(np.where(matcluster == -2))[1])+"rejected"
nametitre = "OVO_0.75_quantile_0.1_diff_svm_logreg_rejet, rejected= " + \
    str(np.shape(np.where(matcluster == -2))[1])
SVMfield.plot_resultat_cluster(matclusteraffich, 10, name, nametitre)

SVMfield.plot_resultat_clusterplat(matclusteraffich, 9, "hoy", "hoy")

#list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svmrbf,dataset_path_rf]
#list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svmrbf,dataset_path2_rf]
#mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classifbis(
    #datasetlogreg, datasetsvm, dataset2logreg, dataset2svm, nb_raws, nb_columns, RSTD.rule05_ecartProbas)

mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(
    datasetlogreg,nb_raws,nb_columns,RSTD.rule05_ecartProbas, 0.1)


#--------------------------------------------------
####################################################################### 2) sans les renouées 
#------------------------------------------------
nb_class,nb_rows,nb_columns = np.shape(dataset_path_lrl1_noren)

#mat2surnoplat, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_pixels_ombresnoplat, ord_pixels_ombresnoplat, abs_nonclassnoplat, ord_nonclassnoplat = RSTD.info_pre_classifbis(
    #dataset_path_lrl1_noren, dataset_path_svm_noren,dataset_path2_lrl1_noren,dataset_path2_svm_noren, nb_raws, nb_columns, RSTD.rule05_ecartProbas)
mat2surnoplat, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_pixels_ombresnoplat, ord_pixels_ombresnoplat, abs_nonclassnoplat, ord_nonclassnoplat = RSTD.info_pre_classif(
    dataset_path_lrl1_noren,nb_raws,nb_columns,RSTD.rule05_ecartProbas, 0.1)
#mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(datasetlogreg,nb_raws,nb_columns,RSTD.rule05_ecartProbas,diff)
confu_svm, score_onesvm, decision_one_svm, predict_one_svm = SVMfield.apply_svm_OVR(
    dataset_path_lrl1_noren, dataset_path2_lrl1_noren, nb_class, abs_pixels_classesnoplat, ord_pixels_classesnoplat, abs_nonclassnoplat, ord_nonclassnoplat)
diffdecision1et2 = np.sort(
    ((-decision_one_svm.T+np.max(decision_one_svm, 1))/np.max(decision_one_svm, 1)).T, axis=1)[:, 1]
#predict_one_svm[np.where(diffdecision1et2 < np.median(diffdecision1et2))] = -2
#predict_one_svm[np.where(
    #diffdecision1et2 < np.percentile(diffdecision1et2, 75))] = -2
predict_one_svm[np.where(
     diffdecision1et2 < np.median(diffdecision1et2))] = -2
matcluster = np.zeros((nb_raws, nb_columns))
matcluster[abs_pixels_ombresnoplat, ord_pixels_ombresnoplat] = -1
matcluster[np.where(mat2surnoplat == 2)[0], np.where(
    mat2surnoplat == 2)[1]] = predict_one_svm
matcluster[abs_pixels_classesnoplat, ord_pixels_classesnoplat] = dataset_path2_lrl1_noren[0,
                                                                    abs_pixels_classesnoplat, ord_pixels_classesnoplat]


print("OVO",MASKSS.compare_masque_platanus_renouees(matcluster,2) )

datacolor = np.arange(0, 10)
matclusteraffich = np.copy(matcluster)
l = 0
for i in [2, 3, 4, 7, 9, 13, 14, 15, 16]:
    absc, ordo = np.where(matclusteraffich == i)
    matclusteraffich[absc, ordo] = datacolor[l]
    l += 1
name = "OVR_0.75_quantile_0.1diff_logreg_svm_withoutplatane" + \
    str(np.shape(np.where(matcluster == -2))[1])+"rejected"
nametitre = "OVR_0.75_quantile_0.1_diff_svm_logreg_rejet, rejected= " + \
    str(np.shape(np.where(matcluster == -2))[1])
SVMfield.plot_resultat_cluster(matclusteraffich, 10, name, nametitre)


#list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svmrbf,dataset_path_rf]
#list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svmrbf,dataset_path2_rf]
#mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classifbis(
    #datasetlogreg, datasetsvm, dataset2logreg, dataset2svm, nb_raws, nb_columns, RSTD.rule05_ecartProbas)

#
mat2sur, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(
    datasetsvm,nb_raws,nb_columns,RSTD.rule05_ecartProbas, 0.1)






