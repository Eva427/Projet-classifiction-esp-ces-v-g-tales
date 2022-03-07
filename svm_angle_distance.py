#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance 


## gros jeu de données :----------------------------------------------------------------
dataset = rasterio.open("./Data_new/combined_svm_rbf_mean_proba.img") #matrice des probas
dataset2 = rasterio.open("./Data_new/combined_svm_rbf.img") #matrice des classes
dataset = dataset.read()
dataset2 = dataset2.read()
dic_arbres = {1: "Platane",
              2: "Saule",
              3: "Peuplier",
              4: "Chene",
              7: "Aulnes",
              9: "Robiniers",
              13: "Melanges_Herbaces",
              14: "Melanges_Arbustifs",
              15: "Renouees_du_Japon",
              16: "inconnu",
              17: "inconnu2"}

nb_class = np.shape(dataset)[0] 
nb_raws = np.shape(dataset)[1] 
nb_columns = np.shape(dataset)[2]

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
    abs_nonclass, ord_nonclass=np.where(mat_pre_classif==2)
    return mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass


mat_pre_classif, abs_pixels_classes, ord_pixels_classes, abs_pixels_ombres, ord_pixels_ombres, abs_nonclass, ord_nonclass=info_pre_classif(dataset,nb_raws,nb_columns)
#définition d'un échantillon test et d'apprentissage 
X_train, X_test, Y_train, Y_test=train_test_split(dataset[:,abs_pixels_classes, ord_pixels_classes].T,dataset2[:,abs_pixels_classes, ord_pixels_classes].T,test_size=0.25, random_state=11)
#essai d'un OnevsOne classifier utilisant SVM (avec hypothèse de séparation linéaire) 
#essai avec données tests du jeu de données d'apprentissage 
clf=OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, np.ravel(Y_train))
#on est obligé d'utiliser np.ravel pour transformer en vecteur de taille (n_samples,) et plus (n_samples, 1)
predict_one_svm=clf.predict(X_test)
decision_one_svm=clf.decision_function(X_test) 
#tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
score_onesvm=clf.score(X_test, Y_test)


################################SUR DONNEES TESTS : NE MARCHE PAS CAR LES DONNEES DE DATASET2 NE SONT PAS BONNES #########################
#essai d'un OnevsOne classifier utilisant SVM (avec hypothèse de séparation linéaire) 
#essai sur données test 
XX=dataset[:,abs_pixels_classes, ord_pixels_classes].T
YY=dataset2[:,abs_pixels_classes, ord_pixels_classes].T
Yt=dataset2[:,abs_nonclass, ord_nonclass].T
Xt=dataset[:,abs_nonclass, ord_nonclass].T
clf=OneVsOneClassifier(LinearSVC(random_state=0)).fit(XX, np.ravel(YY))
#on est obligé d'utiliser np.ravel pour transformer en vecteur de taille (n_samples,) et plus (n_samples, 1)
predict_one_svm=clf.predict(Xt)
decision_one_svm=clf.decision_function(Xt) 
#tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
score_onesvm=clf.score(Xt, Yt)
confu_svm=pd.DataFrame(confusion_matrix(Yt, predict_one_svm))

##########################CODES AVEC D'AUTRES NOYAUX : MARCHENT BIEN SUR PETIT JEU MAIS TROP LONG ICI
#on prend un kernel rbf car c'est celui en général utilisé pour les signaux.
'''method=SVC(gamma=0.2, C=0.5, kernel="rbf")
method.fit(X_train, np.ravel(Y_train))
score=method.score(X_test,Y_test)
ypred=method.predict(X_test)
confu_svm=pd.DataFrame(confusion_matrix(Y_test, ypred), index=dic_arbres.values(), columns=dic_arbres.values())

from sklearn.multiclass import OneVsRestClassifier
who=OneVsRestClassifier(SVC()).fit(X_train, np.ravel(Y_train))
predict_rest_svm=who.predict(X_test)
decision_rest_svm=who.decision_function(X_test) 
#tableau avec des niveaux de confiance pour lever l'ambiguité si il y a le même nombre de votes pour plusieurs classes
score_onesvm=who.score(X_test, Y_test)'''

##### en essayant d'utiliser la distance de Mahanabolis 
#1st step : trouver le vecteur moyen pour chaque classe
ordre=np.argsort(np.ravel(Y_train))
effectifs_par_classe=np.zeros(nb_class+1)
n=0
for i in np.unique(np.sort(Y_train)): 
    effectifs_par_classe[n]=np.count_nonzero(Y_train==i) #effectif dans chaque classe, les classes étant rangées par ordre croissant
    n+=1

X_rangé_par_classe=X_train[ordre, :] #on range les lignes de classe par modalité de classe croissante
i=0
matrix_mean=np.zeros((nb_class+1, np.shape(X_train)[1])) #matrice avec les vecteurs moyens pour chaque classe
n=0
for j in (effectifs_par_classe.astype(int)):
    matrix_mean[n, :]=np.mean(X_rangé_par_classe[i:i+j,:], axis=0)
    n+=1
    i=j


def IVcov():  #renvoie l'inverse de la matrice de covariance
    x=[]
    n=0
    i=0
    for j in (effectifs_par_classe.astype(int)):
        x.append((np.cov(np.array(X_rangé_par_classe[i:i+j,:]).T)))
        n+=1
        i=j

    return x

vecteur_IVcov=IVcov()
#test avec la distance de mahalanobis : donne des mauvais résultats    
ypred=np.zeros(np.shape(X_test)[0])
for i in range (np.shape(X_test)[0]): 
    distance_vec=[]
    j=0  
    for IV in vecteur_IVcov:
        distance_vec.append((distance.mahalanobis(matrix_mean[j, :], X_test[i,:], IV)))
        j+=1
    ypred[i]=(np.unique(np.sort(Y_train))[np.argmin(distance_vec)])
    
        
#avec la norme euclidienne 
ypred_normeeucli=np.zeros(np.shape(X_test)[0])
for i in range (np.shape(X_test)[0]): 
    distance_vec=[]
    j=0  
    for j in range(np.shape(matrix_mean)[1]):
        distance_vec.append(np.linalg.norm(matrix_mean[j, :]- X_test[i,:]))
        j+=1
    ypred_normeeucli[i]=(np.unique(np.sort(Y_train))[np.argmin(distance_vec)])

#avec le cosinus entre les deux vecteurs 
ypredangle=np.zeros(np.shape(X_test)[0])
for i in range (np.shape(X_test)[0]): 
    angle_vec=[]
    j=0  
    for j in range(np.shape(matrix_mean)[1]):
        print(i,j)
        angle_vec.append((np.dot(matrix_mean[j, :], X_test[i,:]))/(np.linalg.norm(matrix_mean[j, :])*np.linalg.norm(X_test[i,:])))
        j+=1
    ypredangle[i]=(np.unique(np.sort(Y_train))[np.argmax(np.abs(angle_vec))])
    