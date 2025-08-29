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




"""
Summary statistics for the 2 point correlation function
"""

def norm(observed, errors = None, bins =[]):
    
    norm = np.nansum([observed**2])

    #removel all invalid entries
    dr = bins[2]-bins[1]

    valid = (~np.isnan(observed))&(~np.isnan(errors))
    
    observed = observed[valid]
    errors = errors[valid]

    number_of_bins  = np.count_nonzero(~np.isnan(observed))


    
    norm = np.sum([dr*abs(1+obs) for obs,r in zip(observed,bins)])

    #norm = np.nansum([dr*(observed)**2])

    #
    norm_error = np.sum([abs(e) for o,e in zip(observed,errors)])

    return (norm, norm_error)

        

def reduced_chi_square(observed, errors, expected = None):
    #Mask all the invalid bins
    
    valid = (~np.isnan(observed))&(~np.isnan(errors)) & (errors != 0)
    observed = observed[valid]
    errors = errors[valid]
    
    chi_square = np.sum([o**2/error**2 for o,error in zip(observed,errors)])
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


def precompute_gaussian_RR(bins = np.linspace(0, 1.5, 100), dimension = 3,n_points =50000, metric = "euclidean"):
        """
        return np.array([1790404, 5328902, 8800108, 12203110, 15524622, 18788670, 21975224,
                25095202, 28169396, 31177772, 34085198, 36959380, 39743652, 42476928,
                45154096, 47748230, 50285228, 52764272, 55149412, 57496774, 59778764,
                61989802, 64179574, 66278740, 68310758, 70279396, 72175208, 74005130,
                75842426, 77578986, 79230000, 80896532, 82467886, 83969200, 85445968,
                86858474, 88219836, 89499032, 90763236, 91949526, 93096398, 94181672,
                95212664, 96190420, 97146650, 98017290, 98838806, 99598856, 100367658,
                101048426, 101646904, 102259252, 102772510, 103265548, 103703002, 104069730,
                104404500, 104690238, 104939278, 105131416, 105262160, 105408324, 105437346,
                105444382, 105405948, 105335388, 105199794, 105066484, 104861080, 104602358,
                104280222, 103976004, 103606536, 103181656, 102749396, 102239502, 101712082,
                101120988, 100494432, 99865736, 99182590, 98459460, 97677952, 96889988,
                96060550, 95175574, 94289944, 93331508, 92358698, 91374356, 90266292,
                89218802, 88116726, 86960964, 85833694, 84611082, 83363458, 82158218,
                80885766])
        """

    

        Eff_mean = np.zeros(dimension)
          
        background = dist.generate_random_points_nd(n_points,s_l = 1,dimension = dimension, seed = 42)

        #Obtrain the distance distribution

        KDT_R = KDTree(background,metric = metric)

        counts_RR = KDT_R.two_point_correlation(background, bins)

        RR = np.diff(counts_RR)

        return RR

def two_point_orig(data, bins, method='standard',
              data_R=None, random_state=None):
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
    KDT_D = KDTree(data)
    KDT_R = KDTree(data_R)

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

    corr[RR_zero] = np.nan

    return corr, np.asarray([(1+cor)/math.sqrt(d) for cor,d in zip(corr,DD)])
        
    




def two_point(data,
              bins,
              method='landy_zalay',
              errors = "poisson",
              counts_RR = None,
              data_R = None,
              random_state=42, 
              metric = "euclidean",
             precomputed_RR = None):
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
    """
    if data_R is None:
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')
    
    if precomputed_RR is not None:

        data_to_random_ratio = r

    """
    print("FIXT THE RATIO PROBLEM!!")

    factor = 2*Nbootstrap*len(data) * 1. / len(data)


    # Fast two-point correlation functions added in scikit-learn v. 0.14

    #distance distribution for the data


    KDT_D = KDTree(data,metric = metric)
    counts_DD = KDT_D.two_point_correlation(data, bins)
    DD = np.diff(counts_DD)

    RR = precomputed_RR


    if RR is None and counts_RR is None and data_R is not None:
        KDT_R = KDTree(data_R, metric = metric)
        counts_RR = KDT_R.two_point_correlation(data_R, bins)
        RR = np.diff(counts_RR)
    elif RR is None:
        RR = np.diff(counts_RR)
        print("No viable background distance distribution distribution")


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
    corr[RR_zero] = 0   #np.nan
    #dist.scatter_points(data_R,alpha = 0.1,title =  "Gaussian space sparse")
    if errors == "poisson":
        DD_zero = (DD == 0) 
        DD[DD_zero] = 1
        
        dcorr = np.asarray([(1+cor)/math.sqrt(d) for cor,d in zip(corr,DD)])

        return corr, dcorr
        

    else:
        return corr, data_R


    



def bootstrap_two_point(data, 
                        bins, 
                        Nbootstrap=10,
                        method='standard', 
                        return_bootstraps=False,
                        random_state=None,
                        data_R = None,
                        sub_sample_fraction = 0.1,
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

    
    if precomputed_RR is None:
        stamp = time.time()
        print("Computing counts from background")
        KDT_R = KDTree(data_R, metric = "euclidean")
        counts_RR = KDT_R.two_point_correlation(data_R, bins)
        print("Counts compute time: ", time.time()-stamp)
    

    bootstraps = np.zeros((Nbootstrap, len(bins[1:])))

    if data_R is not None:
        #If the rerpesentations are to be flattened using UMAP
        if flatten_reps:
            for i in range(Nbootstrap):
                                
                data = viz.umap(representations,scatter = False,name = "UMAP", dim = 2, min_dist = 0.6, n_neighbors = 50,alpha = 0.2)
                data = data/max(np.max(data),abs(np.min(data)))

                indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
                bootstraps[i],_ = two_point(data, bins, method=method,
                                          random_state=i,data_R = data_R[indices, :])
            

        else:
    
            for i in range(Nbootstrap):
                indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))
                bootstraps[i],_ = two_point(data[indices, :], bins, method=method,
                                          random_state=i,counts_RR = counts_RR)

    
    else:
    
        for i in range(Nbootstrap):

            stamp_1 = time.time()

            indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))

            bootstraps[i],_ = two_point(data[indices, :],
                                        bins, 
                                        method=method,
                                        precomputed_RR=precomputed_RR,
                                      random_state=rng)
            print(time.time()-stamp_1)
            
            
  



    if return_bootstraps:
        return bootstraps
    else:
        # use masked std dev in case of NaNs
        corr = np.ma.masked_invalid(bootstraps).mean(0)
        corr_err = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))
        
        return corr, corr_err


def correlate_and_plot(data = list,
                       max_dist = 1.5,
                       min_dist=0,
                       bin_number = 100,
                       plot = False, 
                       Nbootstrap = 5,
                       representations = [],
                       precomputed_RR = None,
                       background = None,
                       label = "correlation on features",
                       fig_name ="tpcor",
                       return_corr = False,
                       verbose = 4):


    """
    Scale the data into a unit block. Center and pull the furthest point in to the edge of such a box

    """
    #Center, scale down the sample

    Eff_mean = np.mean(data, axis = 0)

    #Center to 0,0,0,...
    data = data - Eff_mean
    
    #Scale by finding the furtherst point (or 95th percentile to aviod artifacts or statistical flukes)
    
    distances = np.linalg.norm(data, axis=1)
    scaling_factor = np.percentile(distances, 95)*2
    data = data/scaling_factor
    dist.scatter_points(data, alpha = 0.5)
    




    if precomputed_RR is None:

        if verbose >3:
            print("Computing background and RR distributions, will be slower.")

    
            Eff_cov = np.cov(data,rowvar = False)
            
        
            length, dimension = data.shape
        
            #Percentile of the scaled data
            max_dist = np.percentile(np.linalg.norm(data, axis=1), 95)*2
            
            background = dist.generate_gaussian_points(Eff_mean, 
                                                       Eff_cov,
                                                       2*len(data), 
                                                       dimensions = dimension,
                                                       seed = random.randint(0,10000))
            """
            
            
            background = dist.generate_random_points_nd(2*len(data),
                                                        s_l =max_dist ,
                                                        dimension = 3,
                                                        seed = 42)
            """
            #dist.scatter_overlay(data,background)
            
            
            
            
    
    
    #smax_dist = 1.5
    bins = np.linspace(min_dist, max_dist, bin_number)

    
    
    corr, dcorr = two_point_orig(data, 
                            bins, 
                            method='landy-szalay', 
                          #  errors = "poisson",
                           # data_R=background, 
                            random_state=42,
                            #metric = "euclidean"
                            )

    
    StructureScore  = 1



    """

    
    NormScore = weighted_integral(corr,bins, bootstrap_input = False)
    
    #dist.scatter_points(data, alpha = 0.5)
    #plt.show()
    
    bootstraps= bootstrap_two_point(data, bins, 
                                    data_R = background,
                                    precomputed_RR = precomputed_RR,
                                    Nbootstrap=Nbootstrap,
                                    sub_sample_fraction =0.5,
                                    method = 'standard',  
                                    return_bootstraps =True,
                                    flatten_reps = False,
                                    representations = representations,
                                    )
    """
        

    
    
    #StructureScore = reduced_chi_square(corr, dcorr, expected = None)
    StructureScore  = 1
    
    #NormScore = weighted_integral(bootstraps,bins, bootstrap_input = True)

    corr = np.ma.masked_invalid(bootstraps).mean(0)
    dcorr = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))
    NormScore = norm(corr,dcorr,bins)
    #StructureScore = reduced_chi_square(corr, dcorr, expected = None)

    
    print("Chi-Square score: ",StructureScore)
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
        return 1,1,StructureScore, NormScore

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



