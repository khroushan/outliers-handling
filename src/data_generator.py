import numpy as np
import numpy.random as rnd

from sklearn.datasets import make_blobs


class ToyDataGenerator():
    
    def __init__(self, 
                 n_inliers, 
                 n_features,
                 random_state=0):
        
        self.n_inliers = n_inliers
        self.n_features = n_features
        self.blobs_params = dict(random_state=0, 
                                 n_samples=self.n_inliers, 
                                 n_features=self.n_features)

    def feature_generator(self, 
                          centers=None, 
                          cluster_std=None):
        if not centers:
            centers=np.random.rand(self.n_features, self.n_features)
        if not cluster_std:
            cluster_std = 0.5

        X, clust = make_blobs(centers=centers, 
                           cluster_std=cluster_std,
                           **self.blobs_params)
        return X

    def target_generator_independent(self, 
                                     X, 
                                     degrees=1, 
                                     coeffs=None):
        """ Generate target for given numerical features."""
        y = np.zeros(X.shape[0])

        for dg in range(degrees):
            indxs = np.random.choice(X.shape[1], dg)
            coefs = np.random.randint(-4, 4, size=len(indxs))
            print('Coefficients are', coefs)
            print('indeces are', indxs)
            for c, indx in zip(coefs, indxs):
                y += c*X[:,indx]
        return y

    def outlier_generator(self, 
                          X, 
                          y, 
                          ratio=0.1):
        """ Generate outliers in the range of dataset."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        y_min = y.min()
        y_max = y.max()

        N_outlier = int(X.shape[0]*ratio)
        X_outlier = np.zeros((N_outlier, X.shape[1]))
        for i in range(X.shape[1]):
            X_outlier[:,i] = np.random.uniform(X_min[i], X_max[i], N_outlier)

        y_outlier = np.random.uniform(y_min, y_max, N_outlier)

        return X_outlier, y_outlier
