import random

import numpy as np
import skdim
from sklearn.decomposition import PCA


def id_score(representations,sub_sample = 0.3, Nsamples = 5,verbose = False):
        twonn = skdim.id.TwoNN()
        RepSize = len(representations)
        IDs = []
        representations = np.array(representations)
        for i in range(Nsamples):
            indices = random.sample(range(len(representations)),
                                min(5000,int(len(representations)*sub_sample))
                               )            
            ID = twonn.fit_transform(representations[indices,:])
            IDs.append(ID)
            if verbose:
                print("ID :",ID)
        return np.mean(IDs, axis = 0),np.std(IDs,axis = 0, ddof=1)


def pca_dimension(representations, variance = 0.95):
        """Number of PCA components needed to explain `variance` of the total variance."""
        return PCA(n_components = variance, svd_solver = "full").fit(np.array(representations)).n_components_
