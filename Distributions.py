import numpy as np
import matplotlib.pyplot as plt


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


def generate_random_points_2d(n_points, grid_size):
    """
    Generate random points in 2D space.

    Args:
        n_points (int): Number of points to generate.
        grid_size (int): Size of the 2D grid.

    Returns:
        points (np.ndarray): 2D array of points (n_points, 2).
    """
    points = np.random.uniform(-grid_size/2, grid_size/2, size=(n_points, 2))
    return points



def generate_gaussian_points(mean, cov, num_points, dimensions=2):
     ##points sampled from a gaussian distribution
    
    points = np.random.multivariate_normal(mean, cov, num_points)

    
    return points


def sample_means_and_covariances(dimensions, mean_range, cov_range, num_samples):

    # Sample means from a uniform distribution
    means = np.random.uniform(mean_range[0], mean_range[1], size=(num_samples, dimensions))
    
    # Sample covariance matrices
    covariances = []
    for _ in range(num_samples):
        # Generate a random matrix and force it to be positive semi-definite
        A = np.random.uniform(cov_range[0], cov_range[1], size=(dimensions, dimensions))
        covariance_matrix = np.dot(A, A.T)  # A * A.T ensures the matrix is positive semi-definite
        covariances.append(covariance_matrix)
    
    covariances = np.array(covariances)
    
    return means, covariances



def scatter_points(points):
    plt.scatter(points[:, 0], points[:, 1], alpha=0.2)
    #plt.xlim(-00,500)
    #plt.ylim(-500,500)
    plt.axis('equal')
    plt.show()


def scatter_overlay(points1, points2):
    plt.scatter(points1[:, 0], points1[:, 1], color='blue', label='Set 1')
    plt.scatter(points2[:, 0], points2[:, 1], color='red', label='Set 2')
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
    plt.plot(separation, correlation, marker='o')
    plt.title(title)
    plt.xlabel('Separation')
    plt.ylabel('Frequency of separation')
    plt.grid(False)
    plt.show()

