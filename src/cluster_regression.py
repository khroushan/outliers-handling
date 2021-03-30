# load modules
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def cluster_regression(X, y):
    """ Performs clustering on dataset. Learn a regression model over clusters accumulatively.
    
    input:
    ------
    X: np.array, design matrix
    y: np.array, target vector (labels)
    
    output:
    -------
    model: sklearn trained model, best model
    """
    # TODO: 
    # - right way to decide the number of clusters
    # - devide to validation, test, cross validation 
    # ...
    
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.33, 
                                                        random_state=42)

    # Clustering
    n_clusters = 10
    km = KMeans(n_clusters = n_clusters)
    km.fit(X_train)
    # calculate population of clusters
    cluster_populations = {i:km.labels_[km.labels_ == i].size 
                           for i in range(0,n_clusters)}
    
    # sort cluster based on their population
    cluster_sorted_key = [c[0] for c in sorted(cluster_populations.items(), 
                                               key=lambda el: el[1], 
                                               reverse=True)]
    # Regression
    lr = LinearRegression()
    X_cum = X_train[km.labels_ == cluster_sorted_key[0]]
    y_cum = y_train[km.labels_ == cluster_sorted_key[0]]
    for cl in cluster_sorted_key[1:]:
        lr.fit(X_cum, y_cum)
        print("Score: {:.3}".format(lr.score(X_test, y_test)))
        X_cum = np.r_[X_cum, X_train[km.labels_ == cluster_sorted_key[cl]]]
        y_cum = np.r_[y_cum, y_train[km.labels_ == cluster_sorted_key[cl]]]
