#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:58:17 2022

@author: julie
"""

import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
import matplotlib.pyplot as plt

#-- Matrice des proba ----------------------------------------
## sélectionner un des deux sets de données ci-dessous
data_proba = rasterio.open('combined_svm_rbf_mean_proba.img')
#data_proba = rasterio.open('rlr_l1_combined_mean_proba.img')
data_proba = data_proba.read()
p=np.sum(data_proba,axis=0)
n=len(np.where(np.where(p!=0))[0]) #repérer les cases qui ne sont pas de l'ombre

nb_sorted_pixels2 = len(np.where(data_proba>=0.5)[0]) #nb de pixels qui ont été classés avec proba >0.5
percent_sorted_pixels = nb_sorted_pixels2/n*100

#-- Matrice des classes --------------------------------------
datat_class = rasterio.open('combined_svm_rbf.img')
#datat_class = rasterio.open('rlr_l1_combined.img') #autre set
datat_class = datat_class.read()

keys = list(set(datat_class[np.where(datat_class!=0)])) #repérer les n° des classes (le set sert à avoir une seule occurence)
classes = dict() #va contenir les effectifs par classe
for i in keys :
    classes[i] = len(np.where(datat_class==i)[0])
plt.figure()
plt.bar(classes.keys(), classes.values(), color='g')
plt.title("effectifs par classe")
plt.xticks(keys,keys)
plt.show()
