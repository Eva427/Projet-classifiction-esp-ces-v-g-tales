

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:10:24 2022

@author: julie
-Rq : Meme code que dans clustering_small_data mais en enlevant le dictionnaire => plus facile à manipuler
"""

import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
import numpy.linalg as npl
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from random import sample
from sklearn import preprocessing
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

#paralléliser le code
#attention à installer la version 0.53 
# se placer dans le bon environnement : conda activate [nom environnement] (optionnel si pas plusuers environnements)
# installer : conda install numba==0.53
from numba import jit, prange
import time

import rule_select_training_data as RSTD 

##### EXTRACTION DES DONNEES :

## petit jeu de données :
# dataset = rasterio.open("./Data/proba_log_reg_l1_extract.tif") #matrice des probas
# dataset2 = rasterio.open("./Data/class_log_reg_l1_extract.tif") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()

## gros jeu de données  lrl : 
dataset = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1combined_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_Log_reg_l1rejection_class.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()

## gros jeu de données  svm : 
# dataset = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfcombined_mean_proba.img") #matrice des probas
# dataset2 = rasterio.open("./bonnes_data/none_ite_0_proba_SVM_rbfrejection_class.img") #matrice des classes
# dataset = dataset.read()
# dataset2 = dataset2.read()


##### extraction des classes à partir des probas avec utilisation de kmeans-----------------------------
def apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes):
    #qd on fait un kmeans sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # cond on transpose notre matrice 
    kmeans = KMeans(n_clusters=nb_class, random_state=0).fit(dataset[:,abs_pixels_classes, ord_pixels_classes].T) 
    labels = kmeans.labels_  #il va falloir vérifier à quel arbre correspond chaque cluster
    clusters = kmeans.cluster_centers_
    return kmeans,labels, clusters 

def apply_GMM(dataset,nb_class,abs_pixels_classes, ord_pixels_classes):
    #qd on fait un GMM sur une matrice il faut que la matrice soit de la forme suivante : 
        # - n_observations en lignes (ici c'est le nombre de pixels)
        # - n_clusters en colonnes (ici 10)
        # donc on transpose notre matrice 
    gmm = GaussianMixture(n_components=nb_class, random_state=0).fit(dataset[:,abs_pixels_classes, ord_pixels_classes].T)
    labels_gmm = gmm.predict(dataset[:,abs_pixels_classes, ord_pixels_classes].T)
    
    means_gmm = gmm.means_ #mean of each mixture component : array-like of shape (n_components, n_features) 
    cov_gmm = gmm.covariances_ #covariance of each mixture component. If covariance_type = full the shape is (n_components, n_features, n_features)
    return labels_gmm,means_gmm,cov_gmm

##### extraction des classes à partir du fichier fourni dataset2-----------------------------------------
def extract_classes_connues (dataset2,abs_pixels_classes, ord_pixels_classes) :
    classes_connues = dataset2[:,abs_pixels_classes, ord_pixels_classes].T
    return classes_connues

##### trouver les correspondances entre les clusters de kmeans et les classes connues---------------------
# Matrice de confusion pour Kmeans : mat_result_kmeans
# en colonne : les arbres connus 
# en ligne : le nombre de fois où ils sont associés au ieme label donné par Kmeans 

def map_cluster(nb_class,classes_connues,labels) :
    mat_result_kmeans = np.zeros((nb_class,nb_class))
    num_classes = np.array(np.unique(classes_connues),dtype=int) #n° originla des espèces d'arbre
    j = 0
    for num in num_classes:
        pos = np.where(classes_connues==num)[0]
        for i in range(nb_class) :
            mat_result_kmeans[i,j] = len(np.where(labels[pos]==i)[0])
        j+=1

    newclass = np.argmax(mat_result_kmeans, axis=0) #n° classes qui sont les plus associées à chaque arbre
    return newclass,mat_result_kmeans

# pourcentage de réussite de kmeans : 
def eval_kmean(nb_class,mat_result_kmeans,newkey): 
    reussite_kmeans =  mat_result_kmeans[newkey,range(nb_class)]/np.sum(mat_result_kmeans,axis=0)   
    return reussite_kmeans

# prédiction des pixels mal classés :
def predict(mat_pre_classif,kmeans):
    abs_pixels_apred,ord_pixels_apre = np.where(mat_pre_classif==2)
    prediction = kmeans.predict(dataset[:,abs_pixels_apred, ord_pixels_apre].T)
    return abs_pixels_apred,ord_pixels_apre,prediction

##### création d'une matrice contenant le résultat du clustering------------------------------------------ 
def matrice_cluster(labels,nb_raws,nb_columns,abs_pixels_classes,ord_pixels_classes) :
    # matrice_cluster associe à chaque pixel bien classé le cluster obtenu    
    mat_cluster = (-1)*np.ones((nb_raws,nb_columns)) #-1 = pas de cluster associé à ce pixel
    
    for i in range(len(abs_pixels_classes)):
        mat_cluster[abs_pixels_classes[i],ord_pixels_classes[i]] = labels[i]          
    return mat_cluster


def plot_resultat_cluster(mat_cluster,nb_class):
    #Cette procédure trace les résultats du clustering et du rejet  
    
    ### définition de la map de couleur :
    #------ les valeurs dans mat_cluster se suivent, elles valent : [-2,-1,0,1,2,3,4,5,6,7,9] 
    #------ on ne double donc pas les couleurs.
    cmap2 = colors.ListedColormap(['black','gold','saddlebrown','green','blueviolet',
                                   'skyblue','chartreuse',
                                   'lightpink','darkgrey','royalblue','ivory'])
    
    ### tracé du graphe : 
    plt.figure(figsize=(15,15))
    im2 = plt.imshow(mat_cluster,cmap = cmap2)
    plt.title ("Classes déterminées par le clustering avec rejet")
    
    #### ajout de la légende au graphe :
    #------ définition des labels  
    values_cluster = list(np.arange(0,10)) #10 clusters
    values_cluster.append(-1) #ajout de la classe ombre
    names_cluster = []   
    
    for i in range(nb_class): # on attribue un nom aux 10 clusters
        names_cluster.append("Cluster {l}".format(l=i)) 
        
    names_cluster.append("Ombre") #nom de la classe ombre
    
    #------ get the colors of the values, according to the colormap used by imshow
    couleur = [im2.cmap(im2.norm(value)) for value in values_cluster]
    #------ put those patched as legend-handles into the legend
    patches = [mpatches.Patch(color=couleur[i], label= names_cluster[i]) for i in range(len(values_cluster)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    plt.savefig('Classes_clustering.png', dpi=600, bbox_inches='tight')
    plt.show()
    

def plot_resultat_cluster_rejet(mat_cluster,nb_class):
    #Cette procédure trace les résultats du clustering et du rejet  
    
    ### définition de la map de couleur :
    #------ les valeurs dans mat_cluster se suivent, elles valent : [-2,-1,0,1,2,3,4,5,6,7,9] 
    #------ on ne double donc pas les couleurs.
    cmap2 = colors.ListedColormap(['red','black','gold','saddlebrown','green','blueviolet',
                                   'skyblue','chartreuse',
                                   'lightpink','darkgrey','royalblue','ivory'])
    
    ### tracé du graphe : 
    plt.figure(figsize=(15,15))
    im2 = plt.imshow(mat_cluster,cmap = cmap2)
    plt.title ("Classes déterminées par le clustering avec rejet")
    
    #### ajout de la légende au graphe :
    #------ définition des labels  
    values_cluster = list(np.arange(0,nb_class)) #10 clusters
    values_cluster = np.insert(values_cluster,0,-1) #ajout de la classe ombre
    values_cluster = np.insert(values_cluster,0,-2) #ajout de la classe rejet outlier
    names_cluster = []   
    
    for i in range(nb_class): # on attribue un nom aux 10 clusters
        names_cluster.append("Cluster {l}".format(l=i)) 
        
    names_cluster = np.insert(names_cluster,0,"Ombre") #nom de la classe ombre
    names_cluster = np.insert(names_cluster,0,"Rejet") #nom de la classe rejet outlier

    
    #------ get the colors of the values, according to the colormap used by imshow
    couleur = [im2.cmap(im2.norm(value)) for value in values_cluster]
    #------ put those patched as legend-handles into the legend
    patches = [mpatches.Patch(color=couleur[i], label= names_cluster[i]) for i in range(len(values_cluster)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    plt.savefig('Classes_clustering_peintures_T1_0_50.png', dpi=900, bbox_inches='tight')
    plt.show()
    

############# test du programme ###################################################
# mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres = info_pre_classif(dataset,nb_raws,nb_columns)
# labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_classes, ord_pixels_classes)
# classes_connues = extract_classes_connues(dataset2,abs_pixels_classes, ord_pixels_classes) 
# newkey, mat_result_kmeans = map_cluster(nb_class,classes_connues,dic_arbres,labels)
# reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newkey)

#def test(dataset,dataset2):
nb_class = np.shape(dataset)[0] 
nb_raws = np.shape(dataset)[1] 
nb_columns = np.shape(dataset)[2]

#### Renormalisation des données avant d'appliquer le clustering
 
mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass = RSTD.info_pre_classif(dataset,nb_raws,nb_columns,RSTD.rule05)

# Abscisses et ordonnées des pixels qui ne sont pas de l'ombre (= pixels bien classés et mal classés)
abs_pixels_no_ombre = np.concatenate((abs_pixels_classes,abs_nonclass))
ord_pixels_no_ombre = np.concatenate((ord_pixels_classes,ord_nonclass))

# On exclut de la renormalisation les pixels d'ombre qui ne sont pas considérés comme des données.
to_renorm = dataset[:,abs_pixels_no_ombre,ord_pixels_no_ombre]#matrice de taille nb_class x (nb de pixels non ombre) = données à renormaliser
to_renorm = preprocessing.normalize(to_renorm, axis=0) #normalisation selon les colonnes de to_renorm
dataset[:,abs_pixels_no_ombre,ord_pixels_no_ombre] = to_renorm #dataset est maintenant renormalisé


# Pour cet algorithme, on applique le clustering sur tous les pixels qui ne sont pas de l'ombre
# et non plus uniquement sur les pixels bien classés

kmeans,labels, clusters = apply_kmeans(dataset,nb_class,abs_pixels_no_ombre, ord_pixels_no_ombre)
#labels,means_gmm,cov_gmm = apply_GMM(dataset, nb_class, abs_pixels_no_ombre, ord_pixels_no_ombre)

classes_connues = extract_classes_connues(dataset2,abs_pixels_no_ombre, ord_pixels_no_ombre) 
newclass, mat_result_kmeans = map_cluster(nb_class,classes_connues,labels)
reussite_kmeans = eval_kmean(nb_class,mat_result_kmeans,newclass)

mat_cluster = matrice_cluster(labels,nb_raws,nb_columns,abs_pixels_no_ombre,ord_pixels_no_ombre)
    
    #return labels, clusters, mat_result_kmeans, reussite_kmeans

#labels, clusters, mat_result_kmeans, reussite_kmeans = test(dataset,dataset2)   

plot_resultat_cluster(mat_cluster,nb_class)

#*******************************************************************************************************
###### REJET PEINTURES :
#*******************************************************************************************************
    
#rq : si je ne rescale pas les data au début, les matrices de variance-covariance sont trop proches d'une matrice singulière
# et en mettant "allow_singular=False" dans le calcul de multivariate_normal.pdf alors ça nous met des matrices singular.
#a singular covariance matrix indicates that at least one component of a random vector is extraneous 
#or there is a linear interdependances among the variables.
#Some frequent particular situations when the correlation/covariance matrix of variables is singular:
#Two or more variables sum up to a constant 
#https://stackoverflow.com/questions/35273908/scipy-stats-multivariate-normal-raising-linalgerror-singular-matrix-even-thou



### Calcul du rejet 
#-------------------------------------------------------------------------------------------------------

@jit(parallel=True) 
def rejet(mat_cluster,abs_pixels_no_ombre,nb_class,Niter):
        #mat_rejet est une matrice contenant le résultat du clustering + rejet. Elle peut prendre les valeurs : 
            # de 0 à 9 : valeur d'un cluster
            # -1 : ombre 
            # -2 : rejet comme outlier
        
    mat_rejet = mat_cluster
    
    T1 = 0.90 #seuil pour le rejet d'un pixel outlier
    total_pts_clusters = len(abs_pixels_no_ombre) #nombre total de points impliqués dans le clustering
    
    
    for i in range(Niter):
        for M in prange(nb_class):
            abs_cluster_M,ord_cluster_M = np.where(mat_rejet==M) #abscisses et ordonnées des points du cluster k  
            
            #---on re-calcule les paramètres des clusters k
            print(M)
                
            #calcul de la moyenne et matrice de variance-covaiance de chaque cluster (on en aura besoin pour la suite)
            moyennes_clusters =[] #liste contenant les vecteurs moyennes des données de chaque cluster 
            cov_clusters = []  #liste contenant les matrices de variance-covariance des données de chaque cluster
            P_M = np.zeros(nb_class) #vecteur contenant P(M) pour chaque cluster M
            for k in prange(nb_class): 
                abs_cluster_k,ord_cluster_k = np.where(mat_rejet == k) #abscisses et ordonnées des points du cluster k
                moy_k = np.mean(dataset[:,abs_cluster_k,ord_cluster_k],axis = 1)  #moyenne sur les colonnes
                #rq : moy_k est environ égal aux coordonées du centre du cluster k
                cov_k = np.cov(dataset[:,abs_cluster_k,ord_cluster_k],rowvar=True) #calcule la matrice de var-cov du cluster k                           
                #rowvar = True : each row represents a variable, with observations in the columns.  
                moyennes_clusters.append(moy_k)
                cov_clusters.append(cov_k)
                P_M[k] = len(abs_cluster_k)/total_pts_clusters #len(abs_cluster_k)=nb de points dans le cluster k
                
            #R1.append(moyennes_clusters)
            #R2.append(cov_clusters)
                
            #---pour chaque pixel appartenant au cluster k:
            for l in prange(len(abs_cluster_M)):
                i = abs_cluster_M[l]
                j = ord_cluster_M[l]      #i,j coordonées d'un pixel du cluster k dans le dataset 
            #P_ui_M vecteur contenant la valeur P(ui|M) où M représente le cluster k et i,j le pixel ui
            #pour tous les clusters M 
                P_ui_M = np.zeros(nb_class)
                P_ui = 0
                P_M_ui = np.zeros(nb_class)
                for m in prange(nb_class): #calcul de P(M|ui) pour un ui donnée et pour tous les cluster M
                    P_ui_M[m] = multivariate_normal.pdf(x=dataset[:,i,j], 
                              mean = moyennes_clusters[m] ,cov = cov_clusters[m], allow_singular=True)
                    # Calcul de P(ui) = somme P(ui|M)*P(M) pour tous les clusteurs M (probabilités totales)
                    P_ui = np.sum(P_ui_M*P_M) # * : produit terme à terme
                    
                    if P_ui != 0: 
                    #si P(ui) = 0 alors toutes les valeurs de P(ui|M) = 0 donc toutes les valeurs de P(M|ui) = 0
                    #pour un ui donnée et pour les M clusters.
                    #Dans ce cas, on laisse P_M_ui à 0 sinon on uptade la valeur : 
                        # Calcul de P(M|ui) pour tous les clusters M
                        P_M_ui = P_ui_M*P_M/P_ui # * : produit terme à terme     
                    
                    #R.append(P_M_ui)
                    if P_M_ui[M] < T1 : 
                        mat_rejet[i,j] = -2 #on rejette le pixel comme outlier
                    else :
                        mat_rejet[i,j] = M #le pixel est bon et on lui associe sa classe k 
        return mat_rejet
        

start = time.time()
mat_rejet = rejet(mat_cluster,abs_pixels_no_ombre,nb_class,Niter = 3)
end = time.time()
print("Temps de calcul = %s" % (end - start))


plot_resultat_cluster_rejet(mat_rejet,nb_class)
   
    
 
#*******************************************************************************************************
###### DOCUMENTATION :
#*******************************************************************************************************

# --- calcul du vecteur moyenne de et la matrice de variance-coviance d'après la vidéo :
    #vidéo YT à 4:50 : https://www.youtube.com/watch?v=qMTuMa86NzU
    #on va calculer les maximum likelihood estimates
    #dans le texte, ils disent d'utiliser Maximum similarity estimation mais ça n'existe pas sur internet...

# --- mutlvariate normal (vecteur gaussien):
#multivariate_normal.pdf(x=dataset[:,i,j], mean = moyennes_clusters[m] ,cov = cov_clusters[m], allow_singular=True) 
#si je renormalise pas avant les données, les matrices sont toujours singulières. Mais même en normalisant,
#il peut arriver parfois (pas à tous les run c'est aléatoire) que la matrice soit singulière.   
# C'est pourquoi on met allow_singular=True    
#https://www.delftstack.com/fr/api/scipy/scipy-scipy.stats.multivariate_normal-method/


# --- paralléliser le code avec numba :
#https://numba.pydata.org/numba-doc/dev/user/parallel.html
















