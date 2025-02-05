import numpy as np

from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
import random
import Distributions as dist

random.seed(42)

def scale_and_sample(pca_features, sub_sample_size = 8000, n_output_features = 20, seed = 42):

    #obtain an aray of the elements of highest variance
    first_elements = pca_features[:,0]
    #compute the mean and std
    mean = np.mean(first_elements)
    std_dev = np.std(first_elements)
    np.seed = seed
        
    #scale the representations
    scaled_rep = np.array([(representation-mean)/std_dev for representation in pca_features])

    #Subsample
    
    sampled = scaled_rep[np.random.choice(pca_features.shape[0], sub_sample_size, replace=False)]
    #Reduce the dimension, get only the first n features 
    smaller = []
    for sample in sampled:
        smaller.append(sample[:n_output_features])

    return smaller


def two_point(data, bins, method='standard',
              data_R=None, random_state=42, metric = "manhattan"):
    """Two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    method : string
        "standard" or "landy-szalay".
    data_R : array_like (optional)
        if specified, use this as the random comparison sample
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    n_samples, n_features = data.shape

    # shuffle all but one axis to get background distribution
    if data_R is None:
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')
    import Distributions as dist

    factor = len(data_R) * 1. / len(data)

    # Fast two-point correlation functions added in scikit-learn v. 0.14
    KDT_D = KDTree(data,metric = metric)
    KDT_R = KDTree(data_R, metric = metric)

    counts_DD = KDT_D.two_point_correlation(data, bins)
    counts_RR = KDT_R.two_point_correlation(data_R, bins)

    DD = np.diff(counts_DD)
    RR = np.diff(counts_RR)

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor ** 2 * DD / RR - 1
    elif method == 'landy-szalay':
        counts_DR = KDT_R.two_point_correlation(data, bins)

        DR = np.diff(counts_DR)

        corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    #
    corr[RR_zero] = np.nan
    #dist.scatter_points(data_R,alpha = 0.1,title =  "Gaussian space sparse")  


    return corr, data_R



def bootstrap_two_point(data, bins, Nbootstrap=10,
                        method='standard', return_bootstraps=False,
                        random_state=None,data_R = None,sub_sample_fraction = 0.1):
    """Bootstrapped two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    Nbootstrap : integer
        number of bootstrap resamples to perform (default = 10)
    method : string
        "standard" or "landy-szalay".
    return_bootstraps: bool
        if True, return full bootstrapped samples
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr, corr_err : ndarrays
        the estimate of the correlation function and the bootstrap
        error within each bin. shape = Nbins
    """
    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    if Nbootstrap < 2:
        raise ValueError("Nbootstrap must be greater than 1")

    n_samples, n_features = data.shape

    # get the baseline estimate
    #corr, data_R = two_point(data, bins, method=method, random_state=rng,data_R = data_R)
    

    bootstraps = np.zeros((Nbootstrap, len(bins[1:])))

    if data_R is not None:
    
        for i in range(Nbootstrap):
            indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
            bootstraps[i],_ = two_point(data[indices, :], bins, method=method,
                                      random_state=rng,data_R = data_R[indices, :])
            print(i)

    else:
    
        for i in range(Nbootstrap):
            indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
            bootstraps[i],_ = two_point(data[indices, :], bins, method=method,
                                      random_state=rng)
            
            
  
    # use masked std dev in case of NaNs
    corr_err = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))

    if return_bootstraps:
        return bootstraps
    else:
        return corr, corr_err