import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
import random
try:
    import backbone.Distributions as dist
    import backbone.VISUAL as viz
except:
    import spectra_for_features.backbone.Distributions as dist
    import spectra_for_features.backbone.VISUAL as viz
    
import math
import skdim
import time
import importlib
importlib.reload(dist)




"""
Summary statistics for the 2 point correlation function
"""

def norm(observed,
         errors = None,
         background_factor= 10,
         bins =[]):
    

    #removel all invalid entries
    dr = bins[2]-bins[1]

    valid = (~np.isnan(observed))&(~np.isnan(errors))
    
    observed = observed[valid]
    errors = errors[valid]


    number_of_bins  = np.count_nonzero(~np.isnan(observed))
    norm = np.nansum([(o) for b,o,e in zip(bins,observed,errors)])

    #norm = np.nansum([(o)**2 for o in observed])
    print("Background factor", background_factor)
   # print(observed-(background_factor/0.7))

    #
    norm_error = np.sum([abs(2*o*e) for o,e in zip(observed,errors)])/number_of_bins

    print("Number of valid bins: ", number_of_bins)

    return (norm, norm_error)




def two_point(data, 
              bins,
              method='standard',
              data_R=None,
              precomputed_RR = None,
              background_factor = 1,
              sub_sample_fraction =0.7,
              random_state=None):
    
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

    if precomputed_RR is None:
        if data_R is None:
            data_R = data.copy()
            for i in range(n_features - 1):
                rng.shuffle(data_R[:, i])
        else:
            data_R = np.asarray(data_R)
            if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
                raise ValueError('data_R must have same n_features as data')
    
        factor = len(data_R) * 1. / len(data)

    else:
        factor = background_factor/sub_sample_fraction
        print("Bakcground factor from twopoint", background_factor)




    

    # Fast two-point correlation functions added in scikit-learn v. 0.14
    KDT_D = KDTree(data)
    counts_DD = KDT_D.two_point_correlation(data, bins)
    DD = np.diff(counts_DD)

    if precomputed_RR is None:

        if data_R is None:
            raise ValueError("No background; no precomputed RR")
        else:

            KDT_R = KDTree(data_R)
            counts_RR = KDT_R.two_point_correlation(data_R, bins)
            RR = np.diff(counts_RR)
    else:
        RR = precomputed_RR

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor ** 2 * DD / RR - 1
        
    elif method == 'landy-szalay':

        """
        The precompute speeding only works with the standard method
        """
        if precomputed_RR is not None:
            
            raise ValueError(" The precompute speeding only works with the standard method")
        else:
            
            counts_DR = KDT_R.two_point_correlation(data, bins)
    
            DR = np.diff(counts_DR)
    
            corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    corr[RR_zero] = np.nan

    corr_err =  np.asarray([(1+cor)/math.sqrt(d) for cor,d in zip(corr,DD)])
    return corr, corr_err


def bootstrap_two_point(data, 
                        bins, 
                        Nbootstrap=10,
                        method='standard', 
                        return_bootstraps=False,
                        random_state=None,
                        data_R = None,
                        background_factor = 5,
                        sub_sample_fraction =0.7,
                        flatten_reps = True,
                        representations =None,
                        precomputed_RR = None):

    
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

    

    bootstraps = np.zeros((Nbootstrap, len(bins[1:])))


        
    for i in range(Nbootstrap):
        
        stamp_1 = time.time()
        indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))

        bootstraps[i], corr_err = two_point(data[indices, :],
                                  data_R = data_R,
                                  bins = bins, 
                                  method=method,
                                  precomputed_RR=precomputed_RR,
                                  background_factor = background_factor,
                                  sub_sample_fraction = sub_sample_fraction,
                                  random_state=rng)
                        

    if return_bootstraps:
        return bootstraps,corr_err
    else:
        # use masked std dev in case of NaNs
        corr = np.ma.masked_invalid(bootstraps).mean(0)

        
        return corr, corr_err


def correlate_and_plot(data = list,
                       max_dist = 1.5,
                       min_dist=0,
                       bin_number = 100,
                       plot = False, 
                       Nbootstrap = 1,
                       representations = [],
                       precomputed_RR = None,
                       background = None,
                       background_factor = 1,
                       method = "standard",
                       label = "correlation on features",
                       fig_name ="tpcor",
                       return_corr = False,
                       verbose = True):


    """
    Scale the data into a unit block. Center and pull the furthest point in to the edge of such a box

    """
    #Center, scale down the sample

    Eff_mean = np.mean(data, axis = 0)

    #Center to 0,0,0,...
    data = data - Eff_mean
    
    #Scale by finding the furtherst point (or 95th percentile to aviod artifacts or statistical flukes)
    
    #Scaling confuses cluster separation
    
    distances = np.linalg.norm(data, axis=1)
    
    
    max_dist = np.percentile(np.linalg.norm(data, axis=1), 95)*2

    data = data/max_dist

    max_dist = np.percentile(np.linalg.norm(data, axis=1), 70)*2

    print(max_dist)

    #Chopping up the space,importtant
    
    bins = np.linspace(min_dist,
                       max_dist, 
                       bin_number)

    """
    base = 10
    bins = np.logspace(np.log(max_dist/bin_number)/np.log(base),
                       np.log10(max_dist),
                       bin_number,
                       base = 10)
    print(bins)

    """


    




    if precomputed_RR is None:

        if verbose:
            print("Computing background and RR distributions: will be slower")

    
            Eff_cov = np.cov(data,rowvar = False)
        
            length, dimension = data.shape
        
            #Percentile of the scaled data

            if method == "Standard":
            


                precomputed_RR =  dist.precompute_RR(bins = bins,
                                                   dimension = dimension,
                                                   n_points =background_factor*len(data), 
                                                   metric = "euclidean",
                                                   use_stored = False,
                                                   background = None,
                                                   statistics = "Gaussian",
                                                   Eff_cov = Eff_cov,
                                                   )
            else:
                
                background = dist.generate_gaussian_points(mean = Eff_mean, 
                                                             cov = Eff_cov,
                                                             n_points = background_factor*len(data), 
                                                             dimensions = dimension,
                                                             seed = random.randint(0,10000))



    

    """
    bootstraps,poisson_error = bootstrap_two_point(data, bins, 
                                    data_R = background,
                                    background_factor = background_factor,
                                    precomputed_RR = precomputed_RR,
                                    Nbootstrap=Nbootstrap,
                                    sub_sample_fraction =0.8,
                                    method = method,  
                                    return_bootstraps =True,
                                    flatten_reps = False,
                                    representations = representations,
                                    )


    

    
    corr = np.ma.masked_invalid(bootstraps).mean(0)
    dcorr = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))
    """
    corr, dcorr = two_point(data,
                                  data_R = background,
                                  bins = bins, 
                                  method=method,
                                  precomputed_RR=precomputed_RR,
                                  background_factor = background_factor,
                                  sub_sample_fraction =1,
                                  random_state=42)
                        

    NormScore = norm(corr,
                     errors =dcorr,
                     background_factor= background_factor,
                     bins =bins)
        
    
    print("Repley's K: ",NormScore)

    
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
        return NormScore


    else:
        return NormScore


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




