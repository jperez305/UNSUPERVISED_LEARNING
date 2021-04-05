import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, silhouette_score, homogeneity_score, v_measure_score, adjusted_mutual_info_score,homogeneity_completeness_v_measure
import random 
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
import imblearn
from scprep import reduce
from sklearn.decomposition import PCA, FastICA 
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier



LE = LabelEncoder()



mushroom_data = pd.read_csv("C:/Users/Joey/Downloads/archive(2)/mushrooms.csv")
#mushroom_data = pd.get_dummies(mushroom_data)
mushroom_coded = mushroom_data.iloc[:,:].apply(lambda col: LE.fit_transform(col))

X = mushroom_coded.iloc[:,1:]
Y = mushroom_coded['class']




x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state  = 2047)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

import time



def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def my_kMeans(X, Y):
    distance_k = list(np.arange(2,30,1))  
    
    time = []
    score = []
    homogen_score = []
    vm_score = []
    ami = []
    wss = []
    for i in distance_k:
        k_means_ = KMeans(n_clusters = i, random_state = 3333).fit(X)
#        
#        centroids = k_means_.cluster_centers_
#        predict_clust = k_means_.predict(X)
#        curr_sse = 0
#        for id in range(len(X)):
#            curr_center = centroids[predict_clust[id]]
#            curr_sse = curr_sse +  (X.iloc[id, 0] - curr_center[0]) ** 2 + (X.iloc[id, 1] - curr_center[1]) ** 2
#   
#        score.append(curr_sse)
        wss.append(k_means_.inertia_)
        score.append(silhouette_score(X, k_means_.labels_))
        homogen_score.append(homogeneity_score(Y, k_means_.labels_))
        vm_score.append(v_measure_score(Y, k_means_.labels_))
        ami.append(adjusted_mutual_info_score(Y, k_means_.labels_))
        
    plt.plot(distance_k, score)
    plt.title("Mushroom Data - Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()


    plt.plot(distance_k, homogen_score)
    plt.title("Homogeneity Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Homogeneity Score")
    plt.show()
    
    plt.plot(distance_k, vm_score)
    plt.title("VM Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("VM Score")
    plt.show()
    
    plt.plot(distance_k, ami)
    plt.title("ami Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("ami Score")
    plt.show()
    

def my_EM(X,Y):
    distance_k = list(np.arange(2,10,1))   
    
    time = []
    score = []
    homogen_score = []
    ami = []
    vm_score = []
    for dist in distance_k:
        
        ex_max = GaussianMixture(n_components = dist, random_state = 2047, covariance_type = "diag")
        ex_max.fit(X)
        
        pred_labels = ex_max.predict(X)
        
        score.append(silhouette_score(X, pred_labels))
        homogen_score.append(homogeneity_score(Y, pred_labels))
        vm_score.append(v_measure_score(Y, pred_labels))
        ami.append(adjusted_mutual_info_score(Y, pred_labels))
    
   
    
    
    plt.plot(distance_k, score)
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()


    plt.plot(distance_k, homogen_score)
    plt.title("Homogeneity Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Homogeneity Score")
    plt.show()
    
    plt.plot(distance_k, vm_score)
    plt.title("VM Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("VM Score")
    plt.show()
    
    plt.plot(distance_k, ami)
    plt.title("ami Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("ami Score")
    plt.show()
    
    
    
# PCA     
    

pca = PCA(random_state = 2047)
pca.fit(X)

plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100 ))
plt.title("Survery Response - PCA: EXPLAINED VARIANCE RATIO")
plt.xlabel("Total Number of Components" )
plt.ylabel("Variance Percentage")
plt.show()

plt.plot(range(len(pca.singular_values_) ), pca.singular_values_  )
plt.title("Survery Response - PCA: Eigenvalues")
plt.xlabel("Total Number of Components" )
plt.ylabel("Eigenvalues")
plt.show()



pca_best = PCA(n_components = 10, whiten =False)
pca_best.fit(X)

plt.scatter(pca_best.transform(X)[:,0], pca_best.transform(X)[:,1], c = Y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Mushroom  Data")
plt.show()




def my_ica(X,Y):
    dim_choice = list(np.arange(2, len(X.columns), 1)) + [len(X.columns)]
    
    my_ica = FastICA(random_state  = 2047)
    kurtosis = []
    
    for i in dim_choice:
        print(i)
        my_ica.set_params(n_components = i)
        df = pd.DataFrame(my_ica.fit_transform(X))
        df = df.kurt(axis = 0)
        kurtosis.append(df.abs().mean())
        
    plt.title("Mushroom Classifcation - ICA: ")     
    plt.plot(dim_choice, kurtosis)
    plt.xlabel("Components")
    plt.ylabel("Average Kurtosis")
    plt.show()



from itertools import product

# Randomzied Projections
dim_choice = list(np.arange(2, len(X.columns), 3)) + [len(X.columns)]
my_dict = defaultdict(dict)

for i, j in product(range(3),dim_choice):
    print(i,j)
    my_rca = SparseRandomProjection(n_components = j, random_state = i)
    df = my_rca.fit_transform(X)
    d1 = pairwise_distances(df)
    d2 = pairwise_distances(X)
    my_dict[j][i] =  1 - np.corrcoef(d1.ravel(), d2.ravel())[0,1]

reconstruction_error = pd.DataFrame(my_dict).T.mean(axis = 1).tolist()
    
plt.plot(dim_choice, reconstruction_error)
plt.title("Reconstruction Error: RCA")
plt.xlabel("Dims")
plt.ylabel("Recon. Error")
plt.show()




# RFC
dim_choice = list(np.arange(2, 25, 5)) 
my_dict = defaultdict(dict)


my_rfc = RandomForestClassifier(n_estimators = 25, random_state = 2047)
my_rfc.fit(X, Y)
feature_importance = my_rfc.feature_importances_
feature_importance2 = np.where(feature_importance > .06)
plt.bar(X.columns[feature_importance2[0].tolist()] ,feature_importance[feature_importance2[0].tolist()])
plt.xticks(rotation = 90)
plt.title("Survey Data - RFC: Feature Importance")
plt.show()





######################################################################################################

# RFC
most_important_feat = np.array(X[X.columns[feature_importance2[0].tolist()]].values)
distance_k = list(np.arange(2,5,1))  

time = []
score = []
homogen_score = []
vm_score = []
ami = []

for i in distance_k:
    k_means_ = KMeans(n_clusters = i, random_state = 2047).fit(X)


    score.append(silhouette_score(most_important_feat, k_means_.labels_))
    homogen_score.append(homogeneity_score(Y, k_means_.labels_))
    vm_score.append(v_measure_score(Y, k_means_.labels_))
    ami.append(adjusted_mutual_info_score(Y, k_means_.labels_))

plt.plot(distance_k, score)
plt.title("K MEANS Survey Response - Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()


plt.plot(distance_k, homogen_score)
plt.title("Homogeneity Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Homogeneity Score")
plt.show()

plt.plot(distance_k, vm_score)
plt.title("VM Score")
plt.xlabel("Number of Clusters")
plt.ylabel("VM Score")
plt.show()

plt.plot(distance_k, ami)
plt.title("AMI Score")
plt.xlabel("Number of Clusters")
plt.ylabel("AMI Score")
plt.show()


# ICA 
dim_choice = list(np.arange(2, len(X.columns), 50)) + [len(X.columns)]
    
my_ica = FastICA(random_state  = 2047)
kurtosis = []
my_ica.set_params(n_components = 21)
ica_df = pd.DataFrame(my_ica.fit_transform(X))

my_kMeans(ica_df, Y)




# PCA
pca_best = PCA(n_components = 10)
pca_df = pca_best.fit_transform(X)
my_kMeans(pca_df, Y)



# RP

my_rca = SparseRandomProjection(n_components = 8, random_state = 2047)
rca_df = my_rca.fit_transform(X)
my_kMeans(rca_df, Y)
###############################################################################

###############################################################################

# RFC
most_important_feat = np.array(X[X.columns[feature_importance2[0].tolist()]].values)
distance_k = list(np.arange(2,5,1))  

time = []
score = []
homogen_score = []
vm_score = []
ami = []

for i in distance_k:
    ex_max = GaussianMixture(n_components = i, random_state = 2047, covariance_type = "diag")
    ex_max.fit(X)
        
    pred_labels = ex_max.predict(X)

    score.append(silhouette_score(X, pred_labels))
    homogen_score.append(homogeneity_score(Y, pred_labels))
    vm_score.append(v_measure_score(Y, pred_labels))
    ami.append(adjusted_mutual_info_score(Y, pred_labels))

plt.plot(distance_k, score)
plt.title("EM Survey Response - Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()


plt.plot(distance_k, homogen_score)
plt.title("Homogeneity Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Homogeneity Score")
plt.show()

plt.plot(distance_k, vm_score)
plt.title("VM Score")
plt.xlabel("Number of Clusters")
plt.ylabel("VM Score")
plt.show()

plt.plot(distance_k, ami)
plt.title("ami Score")
plt.xlabel("Number of Clusters")
plt.ylabel("ami Score")
plt.show()


# ICA 
dim_choice = list(np.arange(2, len(X.columns), 100)) + [len(X.columns)]
    
my_ica = FastICA(random_state  = 2047)
kurtosis = []
my_ica.set_params(n_components = 21)
ica_df = pd.DataFrame(my_ica.fit_transform(X))

my_EM(ica_df, Y)




# PCA
pca_best = PCA(n_components = 10)
pca_df = pca_best.fit_transform(X)
my_EM(pca_df, Y)



# RP

my_rca = SparseRandomProjection(n_components = 8, random_state = 2047)
rca_df = my_rca.fit_transform(X)
my_EM(rca_df, Y)
######################################################################################





# NN


f1_array_train = []
f1_array_test = []
hidden_layer_size = list(range(1, 10,1))
for size in hidden_layer_size:
    clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (size,), random_state = 2047)
    clf.fit(x_train, y_train)
    
    # We use F1 because binary targets
    f1_array_train.append(f1_score(y_train, clf.predict(x_train), pos_label = 1))
    f1_array_test.append(f1_score(y_test, clf.predict(x_test),  pos_label = 1))



plt.plot(hidden_layer_size, f1_array_train, color = 'g', label = "Training data")
plt.plot(hidden_layer_size, f1_array_test, color = 'r', label = "Test data")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("F1 Score")
plt.title("Mushroom Classification: Neural Networks")
plt.legend()



clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (8,), random_state = 2047)
clf.fit(x_train, y_train)


plot_learning_curve(clf, "Mushroom Response: NN",x_train, y_train)


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


train_sizes=np.linspace(.1, 1.0, 5)
test_list = []




pca_best = PCA(n_components = 10)
pca_df = pca_best.fit_transform(X)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
x_train, x_test, y_train, y_test = train_test_split(pca_df, Y, test_size=.3, random_state  = 2047)

plot_learning_curve(clf, "Mushroom Classification: NN w/ PCA",x_train, y_train)




my_ica = FastICA(random_state  = 2047)
my_ica.set_params(n_components = 21)
ica_df = pd.DataFrame(my_ica.fit_transform(X))
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
x_train, x_test, y_train, y_test = train_test_split(ica_df, Y, test_size=.3, random_state  = 2047)

plot_learning_curve(clf, "Mushroom Classification: NN w/ ICA",x_train, y_train)



my_rca = SparseRandomProjection(n_components = 8, random_state = 2047)
rca_df = my_rca.fit_transform(X)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
x_train, x_test, y_train, y_test = train_test_split(rca_df, Y, test_size=.3, random_state  = 2047)

plot_learning_curve(clf, "Mushroom Classification: NN w/ RP",x_train, y_train)



most_important_feat = np.array(X[X.columns[feature_importance2[0].tolist()]].values)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
x_train, x_test, y_train, y_test = train_test_split(most_important_feat, Y, test_size=.3, random_state  = 2047)

plot_learning_curve(clf, "Mushroom Classification: NN w/ RFC",x_train, y_train)






x_train, x_test, y_train, y_test = train_test_split(pca_final_df, Y, test_size=.3, random_state  = 2047)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
plot_learning_curve(clf, "Survery Response: NN w/ PCA and clusters",x_train, y_train)

x_train, x_test, y_train, y_test = train_test_split(ica_final_df, Y, test_size=.3, random_state  = 2047)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
plot_learning_curve(clf, "Survery Response: NN w/ ICA and clusters",x_train, y_train)

x_train, x_test, y_train, y_test = train_test_split(rp_final_df, Y, test_size=.3, random_state  = 2047)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
plot_learning_curve(clf, "Survery Response: NN w/ RP and clusters",x_train, y_train)


x_train, x_test, y_train, y_test = train_test_split(rf_final_df, Y, test_size=.3, random_state  = 2047)
clf = MLPClassifier(solver = 'adam', activation='logistic', hidden_layer_sizes = (10,), random_state = 2047)
plot_learning_curve(clf, "Survery Response: NN w/ RF and clusters",x_train, y_train)
