#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:26:40 2022

@author: julie
"""
import rasterio 
import numpy as np 

### extract_data : permet d'ouvrir les fichiers avec Rasterio. 
#Arguments : 
    # - path_dataset et path_dataset2 : path des deux fichiers à ouvrir 
    # - dataset = matrice de taille nb*nb_raws*nb_columns : contient les probas
    # - dataset2 = matrice de taille 1*nb_raws*nb_columns : contient les classes
#Sorties : 
    # - dataset et dataset2
    # - meta est renvoyé car il est utilisé par lasuite pour enregistrer l'image
def extract_data(path_dataset, path_dataset2):
    with rasterio.open(path_dataset) as src:
        dataset = src.read()
        meta = src.meta
        meta.update(count = 1)
    with rasterio.open(path_dataset2) as src2:
        dataset2 = src2.read()
        meta2 = src2.meta
        meta2.update(count = 1)
    return dataset,meta,dataset2,meta2

### save_img : permet de sauvegarder la matrice des résultats obtenus : 
#Arguments : 
    # - mat_result : matrice qui contient les numéros des classes obtenues avec le rejet et les numéros réadaptés (-2,-1,1,2...,16)
    # - path_img : path où on doit stocker l'image (permet aussi de définir le nom de l'image)
    # - meta : juste pour la sauvegarde
    # - nb_raws et nb_columns pour resize
def save_img(mat_result,path_img,meta,nb_raws,nb_columns):
    mask = mat_result.reshape(1,nb_raws, nb_columns)
    mask = mask.astype(np.float32)
    with rasterio.open(path_img, "w", **meta) as dest:
        dest.write(mask)


### TEST : comment se servir des fonctions : 
## Ouvrir les fichiers : 
# path_dataset = "./Data/proba_log_reg_l1_extract.tif"
# path_dataset2 = "./Data/class_log_reg_l1_extract.tif"
# dataset,meta,dataset2,meta2 = extract_data(path_dataset, path_dataset2)

## Enregistrer l'image voulue : 
# nb_raws = np.shape(dataset2)[1] 
# nb_columns = np.shape(dataset2)[2] 
# path_img="image_to_save"
# #mat_cluster c'est la matrice qui contient les numéros des classes obtenues avec
# #le rejet et les numéros réadaptés (-2,-1,1,2...,16)
# save_img(mat_cluster,path_img,meta,nb_raws,nb_columns)