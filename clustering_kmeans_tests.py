#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:14:37 2022

@author: julie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#importation de librairies
import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
import matplotlib.pyplot as plt


#importation d'autres codes
import rule_select_training_data as RSTD 
import file_data_rasterio as FDR 
import clustering_kmeans_v3 as Km
import maskutilisation as MASKS

#*******************************************************************************************************
##### EXTRACTION DES DONNEES :
#*******************************************************************************************************
#1. log reg l1
dataset_path_lrl1 = "./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img" #matrice des probas pour la méthode logreg1
dataset_path2_lrl1 = "./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img" #matrice des classes pour la méthode logreg

# 2. log reg l2
dataset_path_lrl2 ="./bonnes_data/none_ite_0_proba_Log_reg_l2combined_mean_proba.img"#matrice des probas pour la méthode logreg2
dataset_path2_lrl2 = "./bonnes_data/none_ite_0_proba_Log_reg_l2rejection_class.img" #matrice des classes pour la méthode logreg2

# 3. svm linéaire
dataset_path_svml = "./bonnes_data/none_ite_0_proba_SVM_linearcombined_mean_proba.img"#matrice des probas pour la méthode SVM 
dataset_path2_svml = "./bonnes_data/none_ite_0_proba_SVM_linearrejection_class.img" #matrice des classes pour la méthode SVM

# 4. svm rbf
dataset_path_svfrbf = "./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img"#matrice des probas pour la méthode SVM 
dataset_path2_svfrbf = "./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img" #matrice des classes pour la méthode SVM

# 5. RF 
dataset_path_rf = "./bonnes_data/none_ite_0_proba_RFcombined_mean_proba.img" #matrice des probas pour la méthode logreg
dataset_path2_rf = "./bonnes_data/none_ite_0_proba_RFrejection_class.img" #matrice des classes pour la méthode logreg

noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svfrbf,dataset_path_rf]
list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svfrbf,dataset_path2_rf]


dataset,meta,dataset2,meta2 = FDR.extract_data(dataset_path_lrl1,dataset_path2_lrl1)
datasetp,metap,dataset2p,meta2p = FDR.extract_data(dataset_path_lrl2,dataset_path2_lrl2)
# dataset_path=list_dataset[0]
# dataset_path2=list_dataset2[0]
# dataset,meta,dataset2,meta2 = FDR.extract_data(dataset_path, dataset_path2)
# nb_class,nb_rows,nb_columns = np.shape(dataset)

#*********************************************************************************************************
#### VARIABLES GLOBALES :
#****************************************************************************************************

rules = [RSTD.rule05,RSTD.rule05_ecartProbas]
rules_names = ['rule05','rule05diff']

dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "Mais"}
#*******************************************************************************************************


#*******************************************************************************************************
##### TESTS DU PROGRAMME
#*******************************************************************************************************
### 1. RESULTATS DE KMEANS SUR LE SET D'ENTRAINEMENT ET CONSTRUCTION DU SET D'ENTRAINEMENT OPTIMAL
def test_training_set(rule,diff,dataset,dataset2):
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #### Appel du clustering 
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_rows,nb_columns,rule,diff)
    kmeans,labels, clusters = Km.apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = Km.extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = Km.map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans,purity = Km.eval_kmean(nb_class,mat_result_kmeans,newclass)
    return purity

# OPTIMISATION DU SEUIL de différence pour la construction de l'échantillon d'entrainement :
def opti_seuil_diff() : 
    diffs = np.linspace(0.,0.1,11)
    rule = RSTD.rule05_ecartProbas
    
    plt.figure()
    #for d in range(len(list_dataset)) :
    for d in range(1) :
        print(d)
        purity = np.zeros(len(diffs))
        dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[d], list_dataset2[d])
        nb_class,nb_rows,nb_columns = np.shape(dataset)
        for i in range(len(diffs)) : 
            purity[i] = test_training_set(rule,diffs[i],dataset,dataset2)
        plt.plot(diffs,purity,label=noms_algos[d])
    plt.title("Evolution of purity according to the value of the difference for the rule05diff")
    plt.legend()
    plt.show()
    plt.close()

#test :
#opti_seuil_diff(list_dataset,list_dataset2)

#*******************************************************************************************************
### 2. RECHERCHE DU SEUIL DE REJET T OPTIMAL

def opti_seuil_rejet(num_dataset,num_rule):
    
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[num_dataset], list_dataset2[num_dataset])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    ratios_rejets =  []
    ## paramètres actuels (à changer selon les tests) ----------------
    nom_algo = noms_algos[num_dataset]
    rule = rules[num_rule]
    rule_name = rules_names[num_rule]
    diff = 0.1
    
    for i in range(len(seuils)):
        T = T_list[i]
        seuil = seuils[i]
        
        path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name
        labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_image,meta)
        ratios_rejets.append(ratio_rejet) #on garde en mémoire la quantité de rejet
        
        #Tracés des graphiques
        Km.plot_map_matrice(mat_cluster,dic_arbres,nb_class, "Classes déterminées par Kmeans sur dataset " + nom_algo + " avec un seuil T="+str(T)+" ("+rule_name+")" ,path_image)
        
        #affichage et enregistrement du rejet :
        path_rejet = path_image + 'rejet'
        mat_rejet,nb_pixels_rejetes = Km.extract_rejet(mat_cluster,path_rejet,meta,nb_rows,nb_columns,"rejet kmeans sur le dataset "+nom_algo+ " avec un seuil T="+str(T)+" ("+rule_name+")")
    
    #graphique de la classification pour les algos de base
    Km.plot_map_matrice(dataset2[0,:,:],dic_arbres,nb_class,"Classes déterminées par l'algorithme de classification "+ nom_algo + " ("+rule_name+")",path_image)

#test :
#opti_seuil_rejet(0,0)

#*******************************************************************************************************
### 2. EVALUATION DU REJET AVEC LES MASQUES 
def evaluer_rejet(num_dataset,num_rule,num_seuil):
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[num_dataset], list_dataset2[num_dataset])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #liste des paramètres : 
    noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    #choix des pramètres : 
    nom_algo = noms_algos[num_dataset]
    seuil = seuils[num_seuil]
    T = T_list[num_seuil]
    rule = rules[num_rule]
    rule_name = rules_names[num_rule]
    diff = 0.1
    path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name

    #appel de la fonction test de kmeans :
    labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_image,meta)
    
    #utilisation des masques
    M=np.loadtxt("testbis.txt")
    percent_meme_predict,predit_pixels_polys,classes_reelles_polys=MASKS.taux_meme_class_test(mat_cluster,M)
    l,count=np.unique(np.concatenate((list(np.where(predit_pixels_polys==-2)[0]), list(np.where(classes_reelles_polys==-2)[0]))), return_counts=True)
    print("il y a au total TPR ", np.shape(l[count==2])[0]/np.shape(np.where(classes_reelles_polys==-2))[1])
    print("il y a au total FPR ", (np.shape(np.where(predit_pixels_polys==-2))[1]-np.shape(l[count==2])[0])/np.shape(np.where(classes_reelles_polys!=-2))[1])
    return percent_meme_predict,predit_pixels_polys,classes_reelles_polys,l,count
#a,b,c,l,count=evaluer_rejet(0,0,3)

num_dataset=0
num_rule = 0
num_seuil = 3
dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[num_dataset], list_dataset2[num_dataset])
nb_class,nb_rows,nb_columns = np.shape(dataset)

#liste des paramètres : 
noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
seuils = ['T06','T07','T08','T09','T1']
T_list = [0.6,0.7,0.8,0.9,1]

#choix des pramètres : 
nom_algo = noms_algos[num_dataset]
seuil = seuils[num_seuil]
T = T_list[num_seuil]
rule = rules[num_rule]
rule_name = rules_names[num_rule]
diff = 0.1
path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name
    
'''
#liste des paramètres : 
noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
seuils = ['T06','T07','T08','T09','T1']
T_list = [0.6,0.7,0.8,0.9,1]

#choix des pramètres : 
nom_algo = noms_algos[num_dataset]
seuil = seuils[num_seuil]
T = T_list[num_seuil]
rule = rules[num_rule]
rule_name = rules_names[num_rule]
diff = 0.1
path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name

#appel de la fonction test de kmeans :
labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_image,meta)
    
#utilisation des masques
M=np.loadtxt("testbis.txt")
a,b,c=MASKS.taux_meme_class_test(mat_cluster,M)
#a : pourcentage de réussite
#b : classes prédites
#c : classes réelles 
l,count=np.unique(np.concatenate((list(np.where(b==-2)[0]), list(np.where(c==-2)[0]))), return_counts=True)
np.shape(l[count==2])
print("il y a au total TPR ", np.shape(l[count==2])[0]/np.shape(np.where(c==-2))[1])
print("il y a au total FPR ", (np.shape(np.where(b==-2))[1]-np.shape(l[count==2])[0])/np.shape(np.where(c!=-2))[1])

c2 = c[c!=-3]
b2 = b[c!=-3]
c2 = c[c!=-3]
b2 = b[c!=-3]

l2,count2 = np.unique(np.c_[np.where(c2!=-2),np.where(b2!=-2)],return_counts=True)
bien_classe = len(np.where(c2[l2]==b2[l2])[0])
pas_a_rejeter = len(np.where(c2!=-2)[0])
non_rejected_accuracy = bien_classe/pas_a_rejeter
'''

#appel de la fonction test de kmeans :
labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_image,meta)

#utilisation des masques
M=np.loadtxt("testbis.txt")
percent_meme_predict,predit_pixels_polys,classes_reelles_polys=MASKS.taux_meme_class_test(mat_cluster,M)

#1. calcul du non-rejected accuracy : 
#repérer les indices bien classés et ce qu'il ne faut pas rejeter selon les polygones : 
l,count = np.unique(np.concatenate((list(np.where(classes_reelles_polys!=-2)[0]),list(np.where(predit_pixels_polys!=-2)[0]))),return_counts=True)
ind = l[count==2] #les indices en communs (autre que le rejet) sont ceux qui apparaissent deux fois dans la liste 
bien_classe = len(np.where(classes_reelles_polys[ind]==predit_pixels_polys[ind])[0]) #ces indices communs sont bien classés si on a la même classe dans les deux vecteurs
pas_a_rejeter = len(np.where(classes_reelles_polys!=-2)[0]) #pixels pas à rejeter selon les polygones
non_rejected_accuracy = bien_classe/pas_a_rejeter

#2. calcul du true rejected rate et false rejected rate : 
rejetReel = list(np.where(classes_reelles_polys==-2)[0])
rejetPredit = list(np.where(predit_pixels_polys==-2)[0])
lRejet,countRejet = np.unique(np.concatenate((rejetReel,rejetPredit)),return_counts=True)
indRejet = lRejet[countRejet==2] #les indices de rejet en communs sont ceux qui apparaissent deux fois dans la liste => NE VAUT QUE 31 ??? 
TRR = len(indRejet)/len(rejetReel) #true positive reject = pixels bien rejetés / nombre de pixels que l'on doit rejeter
FRR = (len(rejetPredit)-len(lRejet[countRejet==2]))/len(list(np.where(classes_reelles_polys!=-2)[0]))
# print("il y a au total TPR ", np.shape(l[count==2])[0]/np.shape(np.where(classes_reelles_polys==-2))[1])
# print("il y a au total FPR ", (np.shape(np.where(predit_pixels_polys==-2))[1]-np.shape(l[count==2])[0])/np.shape(np.where(classes_reelles_polys!=-2))[1])

