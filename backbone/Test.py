import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display,clear_output
from scipy.stats import sem
from sklearn.metrics import f1_score






def KNN_accuracy(rep,labels):
    accuracy= []

    for random_state in np.random.randint(1,10000,10):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2, random_state=random_state)

        # Define the model

        neigh = KNeighborsClassifier(n_neighbors = 5)

        #Train

        neigh.fit(X_train, y_train)

        acc = sum([neigh.predict(X_test) == y_test][0])/len(y_test)

        accuracy.append(acc)
    m_accuracy = np.mean(accuracy)
    var = sem(accuracy)

    
    return round(m_accuracy*100,2),round(var*100,2)



def KNN_f1(rep,labels, classes =3 ):
    f1_scores = []
    
    for random_state in np.random.randint(1,10000,50):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2,random_state=random_state)

        # Define the model

        neigh = KNeighborsClassifier(n_neighbors = 5)

        #Train

        neigh.fit(X_train, y_train)
        f1_scored = [None,None,None]
        for i in range(classes):
            f1_scored[i] = f1_score(y_test, neigh.predict(X_test), average=None)[i]
        print(f1_scored)
        f1_scores.append(f1_scored) 
    edge = [a[0] for a in f1_scores]
    ellip = [a[1] for a in f1_scores]
    spir = [a[2] for a in f1_scores]

    
    return (np.mean(edge),np.mean(ellip),np.mean(spir)),(sem(edge),sem(ellip),sem(spir))



def silhuoette(rep,labels):

    #umap = viz.umap(rep,dim = 10,scatter = False)
    umap = viz.pca(data = rep,n_components = min([512,len(rep[1])]), variance = 0.95)
        
    sil = metrics.silhouette_score(rep,labels, metric = "euclidean", n_jobs = -1)
    
    return sil