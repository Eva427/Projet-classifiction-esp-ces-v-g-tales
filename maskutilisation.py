#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 17:40:15 2022

@author: camusat
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


from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping



# importation d'autres codes
import rule_select_training_data as RSTD




# *******************************************************************************************************
# EXTRACTION DES DONNEES :
# *******************************************************************************************************
# gros jeu de données  svm : *
# gros jeu de données :----------------------------------------------------------------

# matrice des probas pour la méthode logregl1
datasetlogreg = rasterio.open(
    "./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img")
# matrice des classes pour la méthode logregl1
dataset2logreg = rasterio.open(
    "./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img")
# matrice des probas pour la méthode SVM
datasetsvm = rasterio.open(
    "./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img")
# matrice des classes pour la méthode SVM
dataset2svm = rasterio.open(
    "./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img")
# matrice des probas pour la méthode RF
datasetrf = rasterio.open(
    "./bonnes_data/none_ite_0_proba_RFcombined_mean_proba.img")
# dataset2rf=rasterio.open("./bonnes_data/none_ite_0_proba_RFrejection_class.img") #matrice des probas pour la méthode RF
datasetsvmbis = rasterio.open(
    "./bonnes_data/none_ite_0_proba_SVM_linearcombined_mean_proba.img")
dataset2svmbis = rasterio.open(
    "./bonnes_data/none_ite_0_proba_SVM_linearcombined_mean_proba.img")
# matrice des probas pour la méthode logregl2
datasetlogreg2 = rasterio.open(
    "./bonnes_data/none_ite_0_proba_Log_reg_l2combined_mean_proba.img")
# matrice des classes pour la méthode logregl2
dataset2logreg2 = rasterio.open(
    "./bonnes_data/none_ite_0_proba_Log_reg_l2rejection_class.img")

 
# remarque : l'ombre dans les deux résultats est bien placée au même endroit, tout va bien !!!
dataset2svm = dataset2svm.read()
dataset2logreg = dataset2logreg.read()
datasetlogreg = datasetlogreg.read()
datasetsvm = datasetsvm.read()
dataset2svmbis = dataset2svmbis.read()
datasetsvmbis = datasetsvmbis.read()
dataset2logreg2 = dataset2logreg2.read()
datasetlogreg2 = datasetlogreg2.read()
datasetrf = datasetrf.read()



#télécharger les masques, attention pour ne pas avoir d'erreur (fatale), 
#merci de mettre dans le dossier bonnes data tous les fichiers
# base de donnée juillet.dbf, .cpg, .shp... car sinon ça plante

#je pense que le.tif ça doit etre l'image en noir et blanc et non pas en couleur 


from rasterio.plot import show
import pyproj



def creationtest(tif="./bonnes_data/nbimage.tif", shp='./bonnes_data/base_de_donnee_juillet_v2.shp'):
    
    shapefile = gpd.read_file(shp)
    #shapefile.crs #propre au système de coordonnées
    geoms = shapefile.geometry.values #tous les polygones, ici il y en a 391
    
    # let's grab a single shapely geometry to check
    #geometry = geoms[0] 
    #print(type(geometry))
    #print(geometry) #tous les sommets (coorodnnées géographiques du polygone 0)


    # extract the raster values within the polygon 
    y=np.zeros((2516, 2473)) #matrice qu'on va créer avec des 0 là où il y a pas de polygones et la valeurs des classes là où il y en a
    X = np.zeros((2516, 2473)) # pixels for trainingk, on pourra surement la supprimer 
    with rasterio.open(tif) as src:
        band_count = src.count #ici 4 peut etre les niveaux de gris 
        for index, geom in enumerate(geoms): #boucle qui itère sur les 391 polygones 
            feature = [mapping(geom)] #récupérer tous les coordonnées dans l'image des différents sommets du polygone 
            coord=np.max(np.array(feature[0]['coordinates'][0]), 0)
            
            # the mask function returns an array of the raster pixels within this feature
            out_image, out_transform = mask(src, feature, crop=False) #ne jamais remettre ça à True, sinon ca rogne l'image c'est catastrophique #des heures pour trouver ça... 
           
            out_image=out_image[0,:,:] #récupérer la première image, car en renvoie 4 dont 3 avec exactement la même chose pour chacune 
            
            y[out_image!=0]=shapefile["id_class"][index] #là où c'est non nul, ça veut dire que ça fait partie du polygone on met la classe idclass (classe validée sur le terrain)
            
            X = X+out_image #surement à supprimer 
        plt.figure()
        plt.imshow(y)
    return y

y=creationtest()
#création de la matrice cluster test 
cluster_test=np.copy(y)
for i in [10,11,12,17]:
    cluster_test[np.where(y==i)]=-2
for i in [5, 6,8]:
    cluster_test[np.where(y==i)]=-3
#cluster_test est donc une matrice de test, avec des 0 là où on sait pas, des -2 là où on rejette car c'est pas les bonnes classes, et ailleurs les classes 

def taux_meme_class_test(matclust, cluster_test):
    nb_test=np.shape(np.where(cluster_test!=0))[1]
    #récupérer les parties labelisées, c'es-à-dire où on a des polygones : 
    predit_pixels_polys=matclust[np.where(cluster_test!=0)]
    classes_reelles_polys=cluster_test[np.where(cluster_test!=0)]
    #enlever l'ombre (certains morceaux des polygones sont sur des pixels d'ombre) : 
    predit_pixels_polys=predit_pixels_polys[np.where(predit_pixels_polys!=-1)] 
    classes_reelles_polys=classes_reelles_polys[np.where(predit_pixels_polys!=-1)]
    
    percent_meme_predict=np.shape(np.where(predit_pixels_polys==classes_reelles_polys))[1]/len(predit_pixels_polys)
    return percent_meme_predict, predit_pixels_polys, classes_reelles_polys

np.savetxt("testbis.txt", cluster_test)
M=np.loadtxt("testbis.txt")
print(taux_meme_class_test(M, M))

#autre version permettant de récupérer une matrice avec des 0 et des 1, là où il y a des polygones là où il y en a pas 
# import fiona 
# matpolygone=np.zeros((2516, 2473))
# with fiona.open('./bonnes_data/base_de_donnee_juillet_v2.shp', "r") as shapefile:
#     print(np.shape(shapefile))
    
#     shapes=[feature["geometry"] for feature in shapefile]
# with rasterio.open("./bonnes_data/nbimage.tif") as src:
#     out_image, out_transform=rasterio.mask.mask(src, shapes, crop=False) 
#     #crop=True ça a l'air pour rogner l'image au max pour voir seulement tous les polygones
#     absc, ordo=np.where(out_image[0,:,:]!=0)
#     matpolygone[absc, ordo]=y
#     plt.figure()
#     plt.imshow(matpolygone) #matrice avec des 1 seulement là où on a notre échantillon test 
#     out_meta=src.meta
# out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width":out_image.shape[2], "transform": out_transform})


# plt.figure()
# plt.imshow(out_image[0,:,:])

# absc, ordo=np.where(out_image[0,:,:]!=0)

# full_dataset=rasterio.open("./bonnes_data/nbimage.tif")
# clipped_img = full_dataset.read()[:,:,:]
# print(clipped_img.shape)
# print(full_dataset.transform)
# fig, ax=plt.subplots(figsize=(10,7))
# show(clipped_img, ax=ax, transform=full_dataset.transform)



