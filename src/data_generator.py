import numpy as np
import numpy.random as rnd


# parameters 
size_data = 500
ratio_outliers = 0.1
size_outliers = int(ratio_outliers*size_data)
size_inliers = size_data - size_outliers

# Generate one cluster of data for inliers
X_in, _ = make_blobs(n_samples=size_inliers, n_features=2, centers=1)
a1 = 1.2
a2 = 2.1
# target: linear function of features
y_in = a1*X_in[:,0] + a2*X_in[:,1] + rnd.rand(size_inliers)

# outliers
X_out, _ = make_blobs(n_samples=size_outliers, n_features=2, centers=1)
a1 = 1.2
a2 = 2.1
# target: just a random noise in the same range of inlier data
y_out = (y_in.max() - y_in.min()) * rnd.rand(size_outliers) 

X = np.r_[X_in, X_out]
y = np.r_[y_in, y_out]

fig, ax = pl.subplots(1,2, figsize=(12,4))
ax[0].scatter(X[:,0], y)
ax[1].scatter(X[:,1], y)

class DataGenerator:
    """ Class to generate different form of sysnthesis dataset. It is 
    an extension to the `sklearn.data` module for generating
    consistent `inlier`, `outlier` and `novelty` datasets. """
    
    def __init__(self):
        self.inlier_size = inlier_size
        self.outlier_size = outlier_size
        self.novelty_size = novelty_size
        
    def inlier_generator(self):
        """ To generate inlier dataset """
        
        
    def outlier_generator(self):
        """ To generate inlier dataset """
        
    def novelty_generator(self):
        """ To generate inlier dataset """
  
    def target_generator(self, order=1):
        """ To generate target for inlier and outlier according to given order.
        input:
        ------
        order: int, (default = 1), order of dependency to the features
        """
        
        
        self.y = y