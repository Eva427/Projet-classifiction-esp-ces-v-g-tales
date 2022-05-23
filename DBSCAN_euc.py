import rasterio 
import rasterio.features 
import rasterio.warp 
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import scipy
import matplotlib.colors as colors
import matplotlib.patches as mpatches
sns.set()

### extract_data : permet d'ouvrir les fichiers avec Rasterio. 
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
def save_img(mat_result,path_img,meta,nb_raws,nb_columns):
    mask = mat_result.reshape(1,nb_raws, nb_columns)
    mask = mask.astype(np.float32)
    with rasterio.open(path_img, "w", **meta) as dest:
        dest.write(mask)

### chargement des données
path_dataset = "./Data/proba_log_reg_l1_extract.tif" #matrice des probas 
path_dataset2 = "./Data/class_log_reg_l1_extract.tif" #matrice des classes 
data_proba,meta,dataset2,meta2 = extract_data(path_dataset, path_dataset2)


"""def select_n_per_100(abscisses, ordonnees, n) : 
    index_selected = np.random.randint(0, len(abscisses)-1, int(len(abscisses)/n)).tolist()
    index_selected = np.sort(index_selected, axis=0)
    short_abs_pixels_classes = abscisses[index_selected]
    short_ord_pixels_classes = ordonnees[index_selected]

    return short_abs_pixels_classes, short_ord_pixels_classes"""

#Calcul des distances de chaque point aux autres points
def compute_distances(dataset, abscisses, ordonnees) :
    X = dataset[:,abscisses, ordonnees].T
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    return distances, indices 

#Choix optimal d'eps obtenu en trouvant le eps pour lequel la différence de pente entre 2 points est supérieure à un 1%
def best_eps(dataset, abscisses, ordonnees, percentage) :
    distances, indices = compute_distances(dataset, abscisses, ordonnees) 
    slopes = [distances[1]-distances[0]] #/ (0+1) - 0 = 1
    bool = True 
    for i in range(1, len(distances)-2) :
        slope = (distances[i+1]-distances[i]) #/ (i+1) - i = 1
        slopes.append(slope)
        diff_slope = slopes[i-1]-slopes[i]
        if (diff_slope*100)>=percentage and bool :  #observation d'une différence de pente superieure à percentage
            ind_save = i #on sauvegarde la 1ere valeur de i pour laquelle la différence de pente est superieure à percentage
            bool = False 
    return ind_save

#Cette fonction renvoie les indices des pixels rejettés par DBSCAN
def reject_by_DBSCAN(dataset, abscisses, ordonnees, epsilon, method, min_points) : 
    clustering = DBSCAN(eps=epsilon, metric=method, min_samples=min_points).fit(dataset[:,abscisses, ordonnees].T) 
    labels = clustering.labels_ 
    index_to_reject = np.where(labels==-1)[0].tolist()
    abs_rejected = abscisses[index_to_reject]
    ord_rejected= ordonnees[index_to_reject]

    return abs_rejected, ord_rejected, labels

somme_10mat = np.sum(data_proba,axis=0) #somme des 10 matrices de proba : là où la somme fait 0 => vecteurs d'ombre
abs_pixels_classes, ord_pixels_classes = np.where(somme_10mat!=0) #pixels classes 
#short_abs_pixels_classes, short_ord_pixels_classes = select_n_per_100(abs_pixels_classes, ord_pixels_classes, 3)
distances, indices = compute_distances(data_proba, abs_pixels_classes, ord_pixels_classes)
plt.plot(distances)
ind_epsilon = best_eps(data_proba, abs_pixels_classes, ord_pixels_classes, 0.01)
epsilon = distances[ind_epsilon]
#plt.axvline(x=ind_epsilon, color="red")
ax = plt.axes()
ax.set_facecolor("white")
plt.axhline(y=0.04, color="red", linestyle="--")
plt.xlabel("Point number", fontweight="bold")
plt.ylabel("eps", fontweight="bold", )
plt.grid(color="lightgray")
plt.plot(distances, color="black")
plt.savefig('./epsilon.png', dpi=600)

#La bdd étant trop lourde, nous la divisons en 3 sous parties
#On utilise ici la métrique euclidienne mais à changer si besoin
epsilon = 0.04

a,m,n = np.shape(data_proba)
somme_10mat = np.sum(data_proba,axis=0) 
abs_pixels_classes, ord_pixels_classes = np.where(somme_10mat!=0) #pixels classes 
l = np.shape(abs_pixels_classes)[0]


###Application de DBSCAN sur les differentes parties de l'image
abs_pixels_q1 = np.array(abs_pixels_classes[:l//3])
ord_pixels_q1 = np.array(ord_pixels_classes[:l//3])
datasetq1 = data_proba[:,abs_pixels_q1, ord_pixels_q1] 
clustering1 = DBSCAN(eps=epsilon, metric='euclidean', min_samples=115).fit(datasetq1.T)  
labels1 = clustering1.labels_

abs_pixels_q2 = np.array(abs_pixels_classes[l//3:2*(l//3)])
ord_pixels_q2 = np.array(ord_pixels_classes[l//3:2*(l//3)])
datasetq2 = data_proba[:,abs_pixels_q2, ord_pixels_q2] 
clustering2 = DBSCAN(eps=epsilon, metric='euclidean', min_samples=200).fit(datasetq2.T)  
labels2 = clustering2.labels_

abs_pixels_q3 = np.array(abs_pixels_classes[2*(l//3):])
ord_pixels_q3 = np.array(ord_pixels_classes[2*(l//3):])
datasetq3 = data_proba[:,abs_pixels_q3, ord_pixels_q3] 
clustering3 = DBSCAN(eps=epsilon, metric='euclidean', min_samples=150).fit(datasetq3.T)  
labels3 = clustering3.labels_

#Labels donnés aux pixels par l'aglo DBSCAN 
labels = np.r_[labels1,labels2,labels3]

#Recuperation des abs et ord des pixels rejetes par DBSCAN 
index_to_reject = np.where(labels==-1)[0].tolist()
abs_rejected = abs_pixels_classes[index_to_reject]
ord_rejected= ord_pixels_classes[index_to_reject]

#Affichage des pixels rejetes sur la carte
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

#cmap2 = colors.ListedColormap(['#CDCDCD','#e6e8ff', '#FFFFB5', '#FEE1E8', '#FED7C3', '#FFD8BE','#D4F0F0', '#ECEAE4', '#FFDBCC', 'red'])
cmap2 = colors.ListedColormap(['red','red','black','gold','saddlebrown','green','blueviolet','blueviolet','blueviolet',
                                   'skyblue', 'skyblue','chartreuse','chartreuse','chartreuse','chartreuse',
                                   'lightpink','darkgrey','royalblue','ivory'])

                        
# Preparation des donnees 
nb_raws,nb_columns = np.shape(data_proba)[1:]
mat_cluster1 = np.zeros((nb_raws,nb_columns))
for i in range(nb_raws) :
    for j in range(nb_columns) :
        mat_cluster1[i,j] = dataset2[:,i,j]

mat_cluster1[abs_rejected,ord_rejected] = -2
mat_cluster1[np.where(mat_cluster1==0)] = -1

## fonction pour plot la matrice obtenue (dans les mêmes couleurs pour la matrice des clusters détermniés 
## par l'algo et pour la matrice des classes de dataset2)
def plot_map_matrice(dataset,dic_arbres,nb_class,title):
    #VERIFIER AVEC QGIS QUE LA LEGENDE EST BIEN ASSOCIEE AUX BONS ARBRES !!  
    
    #pour la légende au graphe :
    values = list(dic_arbres.keys())
    names = list(dic_arbres.values())
    
    ### définition de la map de couleur :
    #------ les valeurs dans dataset ne se suivent pas, elles valent : [1,2,3,4,7,9,13,14,15,16] 
    #------ donc on ajoute 4 fois la même couleur entre la classe 9 et 13 par exemple pour pallier à ce pb
    color_list= ['red','black','black','gold','saddlebrown','green','blueviolet','blueviolet','blueviolet',
                                   'skyblue', 'skyblue','chartreuse','chartreuse','chartreuse','chartreuse',
                                   'lightpink','darkgrey','royalblue','ivory']
    if len(np.unique(dataset)) == nb_class+1 : #si c'est une image sans le rejet
        cmap2 = colors.ListedColormap(color_list[2:])
        values = np.insert(values,0,-1) #ajoute la classe d'ombre (valeur -1)
        names = np.insert(names,0,'Ombre') #ajoute la classe d'ombre
    else : #si c'est une image avec le rejet
        cmap2 = colors.ListedColormap(color_list)
        values = np.insert(values,0,-1) #ajoute la classe d'ombre (valeur -1)
        values = np.insert(values,0,-2) #ajoute la classe de rejet (valeur -2)
        names = np.insert(names,0,'Ombre') #ajoute la classe d'ombre
        names = np.insert(names,0,'Rejet') #ajoute la classe de rejet

    ### tracé du graphe : 
    plt.figure(figsize=(15,15))
    im = plt.imshow(dataset,cmap = cmap2)
    plt.title (title)
    #plt.colorbar() #barre de couleurs
    #------ get the colors of the values, according to the colormap used by imshow
    couleur = [im.cmap(im.norm(value)) for value in values]
    #------ put those patched as legend-handles into the legend
    patches = [ mpatches.Patch(color=couleur[i], label= names[i]) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.grid(True)
    if len(np.unique(dataset)) == nb_class+1 : 
        plt.savefig('Classes_classification.png', dpi=600, bbox_inches='tight')
    else :
        plt.savefig('./images_rejet/Classes_clustering_Euc.png', dpi=600, bbox_inches='tight')
    plt.show()

nb_class = np.shape(data_proba)[0] 
plot_map_matrice(dataset2[0,:,:],dic_arbres,nb_class,"Classes déterminiées par l'algorithme de classification svm")
plot_map_matrice(mat_cluster1,dic_arbres,nb_class, "Classes déterminées par l'algorithme de DBSCAN - Distance Eulidienne")

#Sauvegarder la matrice des resultats 
np.savetxt('./resultats_matrices/matrix_euc', mat_cluster1, delimiter=",") 
