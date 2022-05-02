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
import matplotlib.colors as colors
import matplotlib.patches as mpatches


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
dataset_path_svmrbf = "./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img"#matrice des probas pour la méthode SVM 
dataset_path2_svmrbf = "./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img" #matrice des classes pour la méthode SVM

# 5. RF 
dataset_path_rf = "./bonnes_data/none_ite_0_proba_RFcombined_mean_proba.img" #matrice des probas pour la méthode logreg
dataset_path2_rf = "./bonnes_data/none_ite_0_proba_RFrejection_class.img" #matrice des classes pour la méthode logreg

noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svmrbf,dataset_path_rf]
list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svmrbf,dataset_path2_rf]


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
    for d in range(len(list_dataset)) :
        print(d)
        purity = np.zeros(len(diffs))
        dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[d], list_dataset2[d])
        nb_class,nb_rows,nb_columns = np.shape(dataset)
        for i in range(len(diffs)) : 
            purity[i] = test_training_set(rule,diffs[i],dataset,dataset2)
            if purity[i]==1 : 
                print(diffs[i])
        plt.plot(diffs,purity,label=noms_algos[d])
    plt.title("Evolution of purity according to the value of the difference for the rule05diff")
    plt.legend()
    plt.show()
    plt.close()

#test :
#opti_seuil_diff()
#*******************************************************************************************************
### 2. RECHERCHE DU SEUIL DE REJET T OPTIMAL

def opti_seuil_rejet(num_dataset,num_rule,diff):
    
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

#test : avec les sets d'entrainement optimaux
#opti_seuil_rejet(0,1,0.08) #lrl1
#opti_seuil_rejet(1,1,0.08) #lrl2
#opti_seuil_rejet(2,1,0.05) #svm linear
#opti_seuil_rejet(3,1,0.07) #svm rbf
#opti_seuil_rejet(4,1,0.05) #rf
#*******************************************************************************************************
### 2. EVALUATION DU REJET AVEC LES MASQUES 
def test_poly(T,rule,diff,dataset,dataset2,path_img,meta):
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    val_ombre = -1
    val_rejet = -2
    
    #### Renormalisation des données avant d'appliquer le clustering
    """
    # On exclut les pixels d'ombre qui ne sont pas considérés comme des données.
    datasetR = dataset #copie du dataset pour renormalisation
    abs_pixels_no_ombre = np.concatenate((abs_pixels_classes,abs_nonclass))
    ord_pixels_no_ombre = np.concatenate((ord_pixels_classes,ord_nonclass))
    to_renorm = dataset[:,abs_pixels_no_ombre,ord_pixels_no_ombre]#matrice de taille nb_class x (nb de pixels non ombre) = données à renormaliser
    to_renorm = preprocessing.normalize(to_renorm, axis=0) #normalisation selon les colonnes de to_renorm
    datasetR[:,abs_pixels_no_ombre,ord_pixels_no_ombre] = to_renorm #datasetR est maintenant le dataset renormalisé.
    """
    
    #### Appel du clustering 
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif_poly(dataset,nb_rows,nb_columns,rule,diff)
    kmeans,labels, clusters = Km.apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = Km.extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = Km.map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans,purity = Km.eval_kmean(nb_class,mat_result_kmeans,newclass)
    
    #### Appel du rejet 
    mat_cluster = Km.matrice_cluster(labels,nb_rows,nb_columns,abs_pixels_classes,ord_pixels_classes)
    mesure = Km.wasserstein
    rayons = Km.calcul_radius(clusters,dataset,mat_cluster,mesure,nb_class)
    mat_cluster,n_rejet =  Km.rejet(T,dataset,rayons,clusters,abs_nonclass,ord_nonclass,mesure,nb_class,mat_cluster)
    mat_cluster = Km.map_matrice(mat_cluster,newclass,classes_connues,val_ombre,val_rejet,nb_class)
    ratio_rejet = n_rejet/(len(abs_pixels_classes)+len(abs_nonclass)) #ratio de pixels rejetés 
    #exportation de l'image résultat en rasterio :
    FDR.save_img(mat_cluster,path_img+'.img',meta,nb_rows,nb_columns)    
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet



def evaluer_rejet_poly2(num_dataset,num_rule):
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[num_dataset], list_dataset2[num_dataset])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #liste des paramètres : 
    noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    #choix des pramètres : 
    nom_algo = noms_algos[num_dataset]
    rule_name = rules_names[num_rule]
    rule = rules[num_rule]
    diff = 0.08
    
    
    seuil = seuils[3]
    T = T_list[3]
        
    path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name

    #appel de la fonction test de kmeans :
    labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = test_poly(T,rule,diff,dataset,dataset2,path_image,meta)
    
    #utilisation des masques
    M=np.loadtxt("testbis.txt")
    a, b, c = MASKS.taux_meme_class_test(mat_cluster, M)
    
    l, count = np.unique(np.concatenate((list(np.where(b == -2)[0]), list(np.where(c == -2)[0]))), return_counts=True)
    #np.shape(l[count == 2])
    TPR = np.shape(l[count == 2])[0]/np.shape(np.where(c == -2))[1]
    FPR = (np.shape(np.where(b == -2))[1]-np.shape(l[count == 2])[0])/(np.shape(np.where(c != -2))[1])
    print("il y a au total TPR ", np.shape(l[count == 2])[0]/np.shape(np.where(c == -2))[1])
    print("il y a au total FPR ", (np.shape(np.where(b == -2))[1]-np.shape(l[count == 2])[0])/np.shape(np.where(c != -2))[1])
    
    l,count = np.unique(np.concatenate((list(np.where(c!=-2)[0]),list(np.where(b!=-2)[0]))),return_counts=True)
    ind = l[count==2] #les indices en communs (autre que le rejet) sont ceux qui apparaissent deux fois dans la liste
    bien_classe = len(np.where(c[ind]==b[ind])[0]) #ces indices communs sont bien classés si on a la même classe dans les deux vecteurs
    pas_a_rejeter = len(np.where(c!=-2)[0]) #pixels pas à rejeter selon les polygones
    non_rejected_accuracy = bien_classe/pas_a_rejeter
    return non_rejected_accuracy,ratio_rejet,TPR,FPR
#non_rejected_accuracy,ratio_rejet,TPR,FPR = evaluer_rejet_poly2(0,1)


#version où on ne force pas les masques à être dans le set de test : 
    
def compare_masque(mat_cluster) :
    # ATTENTION : c'est uniquement sur la surface des masques qu'on évalue les résultats dans cette fonction. 
    # on met tout ce qui n'est pas couvert par les masques à 0 dans mat_cluster (comme de l'ombre)
    M = np.loadtxt("testbis.txt")
    abs_Mombre, ord_Mombre = np.where(M ==0) #ombre dans M = tout ce qui n'est pas couvert par les masques
    mat_cluster[abs_Mombre, ord_Mombre] = 0 #on met l'ombre dans mat_cluster
    
    abs_rejet, ord_rejet = np.where(mat_cluster==-2) #coordonnées des points rejetés par le clustering
    abs_Mrejet, ord_Mrejet = np.where(M==-2) #coordonnées dans M de la classe à rejeter
    
    #intersection entre les coordonnées des pixels rejetés dans mat_cluster et à rejeter selon M_masque :
    mat_rejet = np.array([abs_rejet,ord_rejet]).T
    mat_Mrejet = np.array([abs_Mrejet,ord_Mrejet]).T
    set_rejet = set([tuple(x) for x in mat_rejet])
    set_Mrejet = set([tuple(x) for x in mat_Mrejet])
    inter_rejet = np.array([x for x in set_rejet & set_Mrejet])
    abs_inter_rejet = inter_rejet[:,0]; ord_inter_rejet = inter_rejet[:,1] #pixels rejetés sur les masques et par notre méthode
    
    # 1. true rejected rate = pixels qu'on a bien rejeté / pixels qu'il faut rejeter
    TRR = len(abs_inter_rejet)/len(abs_Mrejet) 
    
    # 2. false rejected rate = pixels qu'on a mal rejeté / pixels qu'il ne faut pas rejeter
    pas_a_rejeter = abs(np.size(M)-len(abs_Mrejet)-len(abs_Mombre)) #nombre de pixels pas à rejeter (=tout sauf l'ombre et sauf le rejet)
    FRR = (len(abs_rejet)-len(abs_inter_rejet))/pas_a_rejeter # (pixels rejetés - pixels bien rejetés) / pixels qu'il ne faut pas rejeter
    
    # 3. non rejected accuracy = pixels pas à rejeter et bien classés / pixels qu'il ne faut rejeter
    #on met ce qui est rejet et ombre dans une même catégorie (-3) pour ne pas les compter
    Mcopy = np.copy(M)
    Mcopy[abs_Mombre, ord_Mombre] = -3
    Mcopy[abs_Mrejet, ord_Mrejet] = -3
    bien_classes_pas_rejete = len(np.where(Mcopy==mat_cluster)[0])
    NRA = bien_classes_pas_rejete/pas_a_rejeter
    
    return TRR,FRR,NRA   
    
def evaluer_rejet_poly(num_dataset,num_rule,diff):
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset[num_dataset], list_dataset2[num_dataset])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #liste des paramètres : 
    noms_algos = ['lrl1','lrl2','svmlinear','svmrbf','rf']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    #choix des pramètres : 
    nom_algo = noms_algos[num_dataset]
    rule_name = rules_names[num_rule]
    rule = rules[num_rule]
    
    seuil = seuils[3]
    T = T_list[3]
        
    path_image = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name

    #appel de la fonction test de kmeans :
    labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_image,meta)
    
    #utilisation des masques
    TRR,FRR,NRA = compare_masque(mat_cluster)
    return TRR,FRR,NRA

#test : avec les sets d'entrainement optimaux
TRR,FRR,NRA = evaluer_rejet_poly(0,1,0.08) #lrl1
#TRR,FRR,NRA = evaluer_rejet_poly(1,1,0.08) #lrl2
#TRR,FRR,NRA  = evaluer_rejet_poly(2,1,0.05) #svm linear
#TRR,FRR,NRA  = evaluer_rejet_poly(3,1,0.07) #svm rbf
#TRR,FRR,NRA  = evaluer_rejet_poly(4,1,0.05) #rf

#*******************************************************************************************************
### 3. EVALUATION DU REJET EN ENLEVANT les PLATANES (n°1) OU LES RENOUEES (n°15)
# Méthode utilisée : 
# 1. on réalise le clustering et le rejet comme d'habitude sur le jeu de données
# 2. on charge l'ancien fichier dans lequel on avait toutes les classes et on repère les coordonnées 
#    des arbres (dont le numéro est passé en argument par num_class) dont on est sûrs. 
# 3. on regarde si on a rejeté les pixels de notre nouveau jeu de données en ces coordonnées

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

list_dataset_new = [dataset_path_lrl1_noplat,dataset_path_svmrbf_noplat,dataset_path_lrl1_noren,dataset_path_svmrbf_noren]
list_dataset2_new = [dataset_path2_lrl1_noplat,dataset_path2_svmrbf_noplat,dataset_path2_lrl1_noren,dataset_path2_svmrbf_noren]


dic_arbres_noplat = {2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "Mais"}

dic_arbres_noren =  {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              16: "Mais"}

def get_coords_a_rejeter(num_dataset_old,num_rule,num_class,diff=0) :
    # charger l'ancien jeu de données :
    dataset_old,meta_old,dataset2_old,meta2_old = FDR.extract_data(list_dataset[num_dataset_old], list_dataset2[num_dataset_old])
    nb_class,nb_rows,nb_columns = np.shape(dataset_old)
    rule = rules[num_rule]
    
    #repérer sur l'ancien jeu de données les pixels bien classés et les pixels de la classe à supprimer : 
    mat_pre_classif_old, abs_pixels_classes_old, ord_pixels_classes_old, abs_pixels_ombres_old, ord_pixels_ombres_old,abs_nonclass_old, ord_nonclass_old = RSTD.info_pre_classif(dataset_old,nb_rows,nb_columns,rule,diff)
    abs_class_old, ord_class_old = np.where(dataset2_old[0,:,:]==num_class) #repère les coordonnées de la classe à supprimer
    
    #intersection entre les coordonnées des pixels dont la classe est sûre et ceux qui appartiennent à la classe qu'on veut regarder :
    mat_class_old = np.array([abs_class_old,ord_class_old]).T
    mat_sur_old = np.array([abs_pixels_classes_old,ord_pixels_classes_old]).T
    set_class_old = set([tuple(x) for x in mat_class_old])
    set_sur_old = set([tuple(x) for x in mat_sur_old])
    inter = np.array([x for x in set_class_old & set_sur_old])
    abs_inter = inter[:,0]; ord_inter = inter[:,1]
    return abs_inter,ord_inter


def eval_rejet_1class(num_dataset_new,num_dataset_old,num_rule,num_class,diff):
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset_new[num_dataset_new], list_dataset2_new[num_dataset_new])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #liste des paramètres : 
    noms_algos_new = ['lrl1_noplat','svmrbf_noplat','lrl1_noren','svmrbf_noren']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    #choix des pramètres : 
    nom_algo = noms_algos_new[num_dataset_new]
    rule = rules[num_rule]
    rule_name = rules_names[num_rule]
    
    TRR=np.zeros(len(seuils))
    for num_seuil in range(len(seuils)) :
        seuil = seuils[num_seuil]
        T = T_list[num_seuil]
        path_img = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name
        title = "rejet kmeans sur le dataset "+nom_algo+ " avec un seuil T="+str(T)+" ("+rule_name+")"
        print(path_img)
        
        if num_class == 1 : # pas de platane 
            dic = dic_arbres_noplat
        else :
            dic = dic_arbres_noren
    
        #appel de la fonction test de kmeans :
        labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = Km.test(T,rule,diff,dataset,dataset2,path_img,meta)
        Km.plot_map_matrice(mat_cluster,dic,nb_class,title,path_img)
        Km.plot_map_matrice(dataset2[0,:,:],dic,nb_class,"Classes déterminées par l'algorithme de classification "+ nom_algo + " ("+rule_name+")",path_img)
        
        #calcul des coordonnées à rejeter :
        abs_inter,ord_inter = get_coords_a_rejeter(num_dataset_old,num_rule,num_class,diff)
        
        #2. calcul du true rejected rate :
        bonrejet = len(np.where(mat_cluster[abs_inter,ord_inter]==-2)[0]) #nombre de pixels bien rejetés
        TRR[num_seuil] = bonrejet/len(abs_inter) #true positive reject = pixels bien rejetés / nombre de pixels que l'on doit rejeter
    return TRR


# list_dataset_new = [dataset_path_lrl1_noplat,dataset_path_svmrbf_noplat,dataset_path_lrl1_noren,dataset_path_svmrbf_noren]
# list_dataset2_new = [dataset_path2_lrl1_noplat,dataset_path2_svmrbf_noplat,dataset_path2_lrl1_noren,dataset_path2_svmrbf_noren]

# list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svmrbf,dataset_path_rf]
# list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svmrbf,dataset_path2_rf]

'''
num_dataset_new = 3
num_dataset_old = 3
num_rule = 1
num_class = 15
#num_seuil = 3
diff = 0.07
TRR = eval_rejet_1class(num_dataset_new,num_dataset_old,num_rule,num_class,diff)
'''

### TEST avec mix2data antre svmrbf et lrl1

def testmix2data(T,rule,diff,dataset,dataset2,path_img,meta):
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    val_ombre = -1
    val_rejet = -2
    
    #### Renormalisation des données avant d'appliquer le clustering
    """
    # On exclut les pixels d'ombre qui ne sont pas considérés comme des données.
    datasetR = dataset #copie du dataset pour renormalisation
    abs_pixels_no_ombre = np.concatenate((abs_pixels_classes,abs_nonclass))
    ord_pixels_no_ombre = np.concatenate((ord_pixels_classes,ord_nonclass))
    to_renorm = dataset[:,abs_pixels_no_ombre,ord_pixels_no_ombre]#matrice de taille nb_class x (nb de pixels non ombre) = données à renormaliser
    to_renorm = preprocessing.normalize(to_renorm, axis=0) #normalisation selon les colonnes de to_renorm
    datasetR[:,abs_pixels_no_ombre,ord_pixels_no_ombre] = to_renorm #datasetR est maintenant le dataset renormalisé.
    """
    
    #### Appel du clustering :
    data1,meta1,class1,meta2 = FDR.extract_data(list_dataset[0], list_dataset2[0]) #lrl1
    data2,meta12,class2,meta22 = FDR.extract_data(list_dataset[3], list_dataset2[3]) #svmrbf
    mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classifinfo_pre_classifbis(data1, data2, class1, class2,nb_rows,nb_columns,rule,choix=1, diff=0) 
    kmeans,labels, clusters = Km.apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
    classes_connues = Km.extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
    newclass, mat_result_kmeans = Km.map_cluster(nb_class,classes_connues,labels)
    reussite_kmeans,purity = Km.eval_kmean(nb_class,mat_result_kmeans,newclass)
    
    #### Appel du rejet 
    mat_cluster = Km.matrice_cluster(labels,nb_rows,nb_columns,abs_pixels_classes,ord_pixels_classes)
    mesure = Km.wasserstein
    rayons = Km.calcul_radius(clusters,dataset,mat_cluster,mesure,nb_class)
    mat_cluster,n_rejet =  Km.rejet(T,dataset,rayons,clusters,abs_nonclass,ord_nonclass,mesure,nb_class,mat_cluster)
    mat_cluster = Km.map_matrice(mat_cluster,newclass,classes_connues,val_ombre,val_rejet,nb_class)
    ratio_rejet = n_rejet/(len(abs_pixels_classes)+len(abs_nonclass)) #ratio de pixels rejetés 
    #exportation de l'image résultat en rasterio :
    FDR.save_img(mat_cluster,path_img+'.img',meta,nb_rows,nb_columns)    
    
    return labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet, abs_pixels_classes, ord_pixels_classes, abs_nonclass, ord_nonclass


def eval_rejet_1class_mix2data(num_dataset_new,num_dataset_old,num_rule,num_class,diff):
    #sélection du jeu de données : 
    dataset,meta,dataset2,meta2 = FDR.extract_data(list_dataset_new[num_dataset_new], list_dataset2_new[num_dataset_new])
    nb_class,nb_rows,nb_columns = np.shape(dataset)
    
    #liste des paramètres : 
    noms_algos_new = ['lrl1_noplat','svmrbf_noplat','lrl1_noren','svmrbf_noren']
    seuils = ['T06','T07','T08','T09','T1']
    T_list = [0.6,0.7,0.8,0.9,1]
    
    #choix des pramètres : 
    nom_algo = noms_algos_new[num_dataset_new]
    rule = rules[num_rule]
    rule_name = rules_names[num_rule]
    
    TRR=np.zeros(len(seuils))
    for num_seuil in range(len(seuils)) :
        seuil = seuils[num_seuil]
        T = T_list[num_seuil]
        path_img = './resultats_kmeans/'+nom_algo+'/'+nom_algo+seuil+'_'+rule_name
        title = "rejet kmeans sur le dataset "+nom_algo+ " avec un seuil T="+str(T)+" ("+rule_name+")"
        print(path_img)
        
        if num_class == 1 : # pas de platane 
            dic = dic_arbres_noplat
        else :
            dic = dic_arbres_noren
    
        #appel de la fonction test de kmeans :
        labels, clusters, mat_result_kmeans, reussite_kmeans, purity, mat_cluster, ratio_rejet = testmix2data(T,rule,diff,dataset,dataset2,path_img,meta)
        Km.plot_map_matrice(mat_cluster,dic,nb_class,title,path_img)
        Km.plot_map_matrice(dataset2[0,:,:],dic,nb_class,"Classes déterminées par l'algorithme de classification "+ nom_algo + " ("+rule_name+")",path_img)
        
        #calcul des coordonnées à rejeter :
        abs_inter,ord_inter = get_coords_a_rejeter(num_dataset_old,num_rule,num_class,diff)
        
        #2. calcul du true rejected rate :
        bonrejet = len(np.where(mat_cluster[abs_inter,ord_inter]==-2)[0]) #nombre de pixels bien rejetés
        TRR[num_seuil] = bonrejet/len(abs_inter) #true positive reject = pixels bien rejetés / nombre de pixels que l'on doit rejeter
    return TRR

# list_dataset_new = [dataset_path_lrl1_noplat,dataset_path_svmrbf_noplat,dataset_path_lrl1_noren,dataset_path_svmrbf_noren]
# list_dataset2_new = [dataset_path2_lrl1_noplat,dataset_path2_svmrbf_noplat,dataset_path2_lrl1_noren,dataset_path2_svmrbf_noren]

# list_dataset = [dataset_path_lrl1,dataset_path_lrl2,dataset_path_svml,dataset_path_svmrbf,dataset_path_rf]
# list_dataset2 = [dataset_path2_lrl1,dataset_path2_lrl2,dataset_path2_svml,dataset_path2_svmrbf,dataset_path2_rf]
'''
num_dataset_new = 0
num_dataset_old = 0
num_rule = 1
num_class = 1
#num_seuil = 3
diff = 0.08
TRR = eval_rejet_1class_mix2data(num_dataset_new,num_dataset_old,num_rule,num_class,diff)
'''