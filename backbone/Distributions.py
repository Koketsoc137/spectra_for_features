import numpy as np
import matplotlib.pyplot as plt
#matplotlib.rc('font', **font)

import numpy as np

def reduced_chi_square(observed, errors, expected = None):
    
    chi_square = np.nansum([o**2/error**2 for o,error in zip(observed,errors)])

    return chi_square/np.count_nonzero(~np.isnan(chi_square))
    
def norm(observed, expected = None):
    
    norm = np.nansum([observed**2])

    return norm/np.count_nonzero(~np.isnan(observed))


def reduced_nonchi_square(observed, errors, expected = None):

    nonchi_square = np.sum(observed)
    nonchi_error = np.sum(errors)

    return nonchi_square/len(observed), nonchi_error/len(errors)#/np.count_nonzero(~np.isnan(observed)),nonchi_error/np.count_nonzero(~np.isnan(nonchi_error))



def first_order_structure(bootstraps):

    Nbootstrap = len(bootstraps)
    squared_bootstraps = np.zeros((Nbootstrap, len(bootstraps[0])))
                          
    for i in range(Nbootstrap):
                              
        squared_bootstraps[i] = np.array([o**2 for o in bootstraps[i]])
    
    #Compute the mean and error, ignore the nans
    
    corr = np.ma.masked_invalid(squared_bootstraps).mean(0)
    error = np.ma.masked_invalid(squared_bootstraps).std(0)

    return corr,reduced_nonchi_square(corr, error, expected = None)



# Step 1: Generate random points in 2D space
# Function to generate random points on a disk
def generate_points_on_disk(radius, n_points, center = [0,0]):
    """
    Generate random points uniformly on a disk of given radius.

    Args:
        radius (float): Radius of the disk.
        n_points (int): Number of random points to generate.

    Returns:
        points (np.ndarray): Array of shape (n_points, 2) containing random points on the disk.
    """
    # Generate random angles uniformly between 0 and 2pi
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    
    # Generate random radii with proper distribution for uniform sampling
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles) +center[0]
    y = radii * np.sin(angles)  +center[0]
    
    # Combine x and y into a single array
    points = np.column_stack((x, y))
    
    return points


def generate_random_points_2d(n_points,seed = 42):
    """
    Generate random points in 2D space.

    Args:
        n_points (int): Number of points to generate.
        grid_size (int): Size of the 2D grid.

    Returns:
        points (np.ndarray): 2D array of points (n_points, 2).
    """
    points = np.random.uniform(-1, 1, size=(n_points, 2))
    
    return points



def generate_gaussian_points(mean, cov, num_points, dimensions=2,seed = 42):
     ##points sampled from a gaussian distribution
    np.random.seed(seed)

    
    points = np.random.multivariate_normal(mean, cov, num_points)

    
    return points


def sample_means_and_covariances(dimensions, mean_range, cov_range, num_samples,seed = 42):

    # Sample means from a uniform distribution
    np.random.seed(seed)
    means = np.random.uniform(mean_range[0], mean_range[1], size=(num_samples, dimensions))
    space_center = 50
    mean_array = []

    space_center = (mean_range[0] - mean_range[1])/2
    
    # Sample covariance matrices
    covariances = []
    #covariances_sparse = []
    for _ in range(num_samples):
        # Generate a random matrix and force it to be positive semi-definite
        A = np.random.uniform(cov_range[0], cov_range[1], size=(dimensions, dimensions))
        covariance_matrix = np.dot(A, A.T)  # A * A.T ensures the matrix is positive semi-definite
        covariances.append(covariance_matrix)

       # sparse_matrix = np.copy(covariance_matrix)

       # np.fill_diagonal(sparse_matrix, np.diagonal(sparse_matrix)*2)
  
       # covariances_sparse.append(sparse_matrix)

       
    covariances = np.array(covariances)

    
  #  covariances_sparse = np.array(covariances_sparse)
    
    return means, covariances#, covariances_sparse



def scatter_points(points,alpha = 0.2, title = ""):
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 
    plt.scatter(points[:, 0], points[:, 1], alpha=alpha)
    plt.title(title)
    #plt.xlim(-00,500)
    #plt.ylim(-500,500)
    plt.axis('equal')
    plt.show()


def scatter_overlay(points1, points2):
    plt.scatter(points1[:, 0], points1[:, 1], color='blue', label='Set 1',s = 0.05)
    plt.scatter(points2[:, 0], points2[:, 1], color='red', label='Set 2', s = 0.05)
    plt.legend()
    plt.show()

# Step 4: Plot the 2-point autocorrelation function
def plot_autocorrelation_2d(separation, correlation,title = "Correlation"):
    """
    Plot the 2-point autocorrelation function.

    Args:
        bin_centers (np.ndarray): Bin centers for the autocorrelation.
        correlation (np.ndarray): Autocorrelation values in each bin.
    """
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 
    plt.plot(separation, correlation, marker='o')
    plt.title(title)
    plt.xlabel('Separation')
    plt.ylabel('Frequency of separation')
    plt.grid(False)
    plt.show()

