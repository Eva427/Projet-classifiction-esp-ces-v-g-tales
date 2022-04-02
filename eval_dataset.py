#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 08:42:06 2022

@author: julie
"""
#importation de librairies
import rasterio 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

#importation d'autres codes
import rule_select_training_data as RSTD 
import file_data_rasterio as FDR 

#importation des données 
# 1. log reg l1
# dataset_lrl1_path = "./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img" #matrice des probas pour la méthode logreg
# dataset_lrl1_path2 = "./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img" #matrice des classes pour la méthode logreg
# datasetlrl1,metalrl1,datasetlrl12,metalrl12 = FDR.extract_data(dataset_lrl1_path, dataset_lrl1_path2)

# 2. log reg l2
# dataset_lrl2_path ="./bonnes_data/none_ite_0_proba_Log_reg_l2combined_mean_proba.img"
# dataset_lrl2_path2 = "./bonnes_data/none_ite_0_proba_Log_reg_l2rejection_class.img"
# datasetlrl2,metalrl2,datasetlrl22,metalrl22 = FDR.extract_data(dataset_lrl2_path, dataset_lrl2_path2)

# 3. svm
dataset_svm_path = "./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img"#matrice des probas pour la méthode SVM 
dataset_svm_path2 = "./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img" #matrice des classes pour la méthode SVM
datasetsvm,metasvm,datasetsvm2,metasvm2 = FDR.extract_data(dataset_svm_path, dataset_svm_path2)

# # 4. RF 
# datasetrf_path = "./bonnes_data/none_ite_0_proba_RFcombined_mean_proba.img" #matrice des probas pour la méthode logreg
# dataset2rf_path = "./bonnes_data/none_ite_0_proba_RFrejection_class.img" #matrice des classes pour la méthode logreg
#datasetrf,metarf,datasetrf2,metarf2 = FDR.extract_data(datasetrf_path, dataset2rf_path)


def plot_train(mat_train,title,path_img):
    ### Plot de la carte de rejet : -------------------------------------------
    cmap2 = colors.ListedColormap(['white','black']) # définition map de couleur
    ### tracé du graphe :
    plt.figure(figsize=(15,15))
    im = plt.imshow(mat_train,cmap = cmap2)
    plt.title (title)
    
    #légende du graphe
    values = [0, 1] 
    names = ["autre","set d'entrainement"]
    couleur = [im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=couleur[i], label= names[i]) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    plt.savefig(path_img, dpi=600, bbox_inches='tight')
    plt.show()
    

def eval_training_set(path_img,meta,nb_rows,nb_columns,dataset,rule,title, diff=0):
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres,abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_rows,nb_columns,rule,diff)
    mat_train = np.zeros(np.shape(mat_pre_classif))
    mat_train[abs_pixels_classes, ord_pixels_classes] = 1
    #FDR.save_img(mat_train,path_img,meta,nb_rows,nb_columns)
    
    nb_pixels_train = len(abs_pixels_classes)
    ratio_train = nb_pixels_train/(nb_pixels_train+len(abs_nonclass))
    #plot_train(mat_train,title,path_img)
    return mat_train,ratio_train

def Test_eval_1dataset(dataset,dataset2,nom,meta):
    list_ratio_train =[]
    ### sur image rbf : 
    _,nb_rows,nb_columns = np.shape(dataset)
    #règel 0.5
    mat_train05,ratio_train = eval_training_set('./train_set_rules/train_set_rule05_'+nom,meta,nb_rows,nb_columns,dataset,RSTD.rule05,'rule05_'+nom)
    list_ratio_train.append(ratio_train)
    #règle 0.5diff :
    mat_train05diff,ratio_train = eval_training_set('./train_set_rules/train_set_rule05diff_'+nom,meta,nb_rows,nb_columns,dataset,RSTD.rule05_ecartProbas,'rule05diff_'+nom,0.1)
    list_ratio_train.append(ratio_train)
    
    difference = np.abs(mat_train05-mat_train05diff)
    nb_pixels_diff = len(np.where(difference!=0)[0])
    #FDR.save_img(difference,'./train_set_rules/train_set_diff_'+nom,meta,nb_rows,nb_columns)
    #plot_train(difference,'difference_'+nom,'./train_set_rules/difference_'+nom)
    return list_ratio_train,nb_pixels_diff

def eval_training_set_mix2data(data1, data2, class1, class2,rule,nom,meta,choix=1, diff=0):
    _,nb_rows,nb_columns = np.shape(data1)
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres,abs_nonclass, ord_nonclass = RSTD.info_pre_classifbis(data1, data2, class1, class2,nb_rows,nb_columns,rule,choix, diff)
    mat_train = np.zeros(np.shape(mat_pre_classif))
    mat_train[abs_pixels_classes, ord_pixels_classes] = 1
    FDR.save_img(mat_train,'./train_set_rules/'+nom,meta,nb_rows,nb_columns)
    
    nb_pixels_train = len(abs_pixels_classes)
    ratio_train = nb_pixels_train/(nb_pixels_train+len(abs_nonclass))
    plot_train(mat_train,'train_set_'+nom,'./train_set_rules/'+nom)
    
    # on compare avec la matrice mat_train_comp calculée avec la rule05 ou avec la rule05diff selon la règle qu'on a choisit ici :
    # difference = np.abs(mat_train-mat_train_comp)
    # FDR.save_img(difference,'./train_set_rules/train_set_diff_'+nom,meta,nb_rows,nb_columns)
    # plot_train(difference,'difference_'+nom,'./train_set_rules/difference_'+nom)
    return ratio_train

#### TEST
# 1. log reg l1
#list_nb_pixels_train_lrl1, list_ratio_train_lrl1, pixels_diff_lrl1= Test_eval_1dataset(datasetlrl1,datasetlrl12,"log_reg_l1",metalrl1)

# 2. log reg l2
#list_nb_pixels_train_lrl2, list_ratio_train_lrl2, pixels_diff_lrl2 = Test_eval_1dataset(datasetlrl2,datasetlrl22,"log_reg_l2",metalrl2)

# 3. svm
list_ratio_train_svm, nb_pixels_diff_svm = Test_eval_1dataset(datasetsvm,datasetsvm2,"svm",metasvm)
 
# mix2data :
# log reg l1 et svm :
#mnb_pixels_train_mix2,ratio_train_mix2 = eval_training_set_mix2data(datasetlrl1, datasetsvm, datasetlrl12, datasetsvm,RSTD.rule05,"mix2data_rule05_svm_lrl1",metalrl1,choix=1)
