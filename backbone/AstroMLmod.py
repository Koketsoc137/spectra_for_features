import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
import random
import backbone.Distributions as dist
import skdim
import backbone.VISUAL as viz

random.seed(42)

def norm(observed, errors = None):
    
    norm = np.nansum([observed**2])

    #removel all invalid entries

    valid = (~np.isnan(observed))&(~np.isnan(errors))
    
    observed = observed[valid]
    errors = errors[valid]

    number_of_bins  = np.count_nonzero(~np.isnan(observed))

    norm = np.nansum([observed**2])

    #
    norm_error = np.sum([abs(2*o*e) for o,e in zip(observed,errors)])

    return (norm/number_of_bins, norm_error/number_of_bins)

def reduced_chi_square(observed, errors, expected = None):
    #Mask all the invalid bins
    
    print(sum(errors ==0))
    valid = (~np.isnan(observed))&(~np.isnan(errors)) & (errors != 0)
    observed = observed[valid]
    errors = errors[valid]
    
    chi_square = np.sum([o**2/error**2 for o,error in zip(observed,errors)])
    print(len(observed))
    return chi_square/len(observed)



def first_order_structure(bootstraps):

    Nbootstrap = len(bootstraps)
    squared_bootstraps = np.zeros((Nbootstrap, len(bootstraps[0])))
                          
    for i in range(Nbootstrap):
                              
        squared_bootstraps[i] = np.array([o**2 for o in bootstraps[i]])
    
    #Compute the mean and error, ignore the nans
    
    corr = np.ma.masked_invalid(squared_bootstraps).mean(0)
    error = np.ma.masked_invalid(squared_bootstraps).std(0)

    return corr,reduced_nonchi_square(corr, error, expected = None)

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
              data_R=None, random_state=42, metric = "euclidean"):
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
                        random_state=None,data_R = None,sub_sample_fraction = 0.1,
                       flatten_reps = True,representations =None):
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


    n_samples, n_features = data.shape


    if Nbootstrap ==0:
        # get the baseline estimate
        corr, data_R = two_point(data, bins, method=method, random_state=rng,data_R = data_R)
        return corr, data_R
    

    bootstraps = np.zeros((Nbootstrap, len(bins[1:])))

    if data_R is not None:
        #If the rerpesentations are to be flattened using UMAP
        if flatten_reps:
            for i in range(Nbootstrap):
                
                data = viz.umap(representations,scatter = True,name = "UMAP", dim = 2, min_dist = 0.6, n_neighbors = 50,alpha = 0.2)
                data = data/max(np.max(data),abs(np.min(data)))

                indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
                bootstraps[i],_ = two_point(data, bins, method=method,
                                          random_state=i,data_R = data_R[indices, :])
            

        else:
    
            for i in range(Nbootstrap):
                indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
                bootstraps[i],_ = two_point(data, bins, method=method,
                                          random_state=i,data_R = data_R[indices, :])


    else:
    
        for i in range(Nbootstrap):
            indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
            bootstraps[i],_ = two_point(data[indices, :], bins, method=method,
                                      random_state=rng)
            
            
  
    # use masked std dev in case of NaNs
    corr = np.ma.masked_invalid(bootstraps).mean(0)
    corr_err = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))


    if return_bootstraps:
        return bootstraps
    else:
        return corr, corr_err


def correlate_and_plot(data = list,max_dist = 1.5,min_dist=0,
                    bin_number = 100,plot = True, label = "correlation on features",fig_name ="tpcor",return_corr = False, representations = []):

    data = data/max(np.max(data),abs(np.min(data)))

    Eff_mean = np.mean(data, axis = 0)
    Eff_cov = np.cov(data,rowvar = False)

    length, dimension = data.shape

    # Sample covariance matrices
    Nbootstrap = 5
  
    background = dist.generate_gaussian_points(Eff_mean, Eff_cov,Nbootstrap*len(data))
    

    max_dist = np.percentile(np.linalg.norm(background-Eff_mean, axis=1), 68)*2 #probe to the 99th percentile from the mean
    bins = np.linspace(min_dist, max_dist, bin_number)

    
    corr, dcorr= bootstrap_two_point(data, bins, 
                                    data_R = background,Nbootstrap=Nbootstrap,
                                    sub_sample_fraction =1/Nbootstrap,
                                    method = 'standard',  
                                    return_bootstraps =False,
                                    flatten_reps = True,
                                    representations = representations
                                    
                                    )

    
    StructureScore = reduced_chi_square(corr, dcorr, expected = None)
    NormScore = norm(corr,dcorr)
    print("Chi-Square score: ",StructureScore)
    print("L2 Norm score: ",NormScore)

    
    if plot:
        fig = plt.figure(dpi = 300)
        plt.style.use("default")
        plt.figure(figsize=(15,10))
        plt.rcParams.update({'font.size': 20})
        plt.plot(bins[1:],corr)
        plt.fill_between(bins[1:],corr-dcorr, corr+dcorr, color = "blue",alpha = .3)
        plt.title(label)
        plt.savefig(fig_name+".png")
        plt.show()
        return StructureScore, NormScore

    elif return_corr:
        return corr,dcorr,StructureScore,NormScore
    else:
        return StructureScore,NormScore


def id_score(representations,SubSampleFraction = 0.3, Nsamples = 5,verbose = False):
        twonn = skdim.id.TwoNN()
        RepSize = len(representations)
        IDs = []
        representations = np.array(representations)
        for i in range(Nsamples):
            indices = random.sample(range(RepSize),int(RepSize*SubSampleFraction))
            ID = twonn.fit_transform(representations[indices,:])
            IDs.append(ID)
            if verbose:
                print("ID :",ID)
        return np.mean(IDs, axis = 0),np.std(IDs,axis = 0, ddof=1)




