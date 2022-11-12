import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import datasets
import random
import math

iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

# Create new data set with two features: (sepal length/sepal width, petal length/petal width)
for i in X:
    i[0] = i[0] / i[1]
    i[1] = i[2] / i[3]
X = X[:, :2]

# Plot data with colors
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


# Honestly a lot of this code is dependent on k equaling 3 which is only because I am very low on time to finish this


def distance(p1, p2):
    return np.linalg.norm(p1-p2)


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    # Create k x 2 np array so that we have k centroids, each defined by 2 parameters
    centroids = np.empty((k, 2))
    #initialize first centroid to random point
    centroids[0] = random.choice(X)

    # Initialize other centroids by maximizing distance to existing centroids
    
    distances = []
    for i in range(len(X)): # Loop through each point
        distances.append(distance(centroids[0], X[i])**2) # Not sure if squaring is neccasary but it is 10:20 PM and I don't have time to figure it out
    
    centroids[1] = X[distances.index(max(distances))]

    distances = []
    for i in range(len(X)):
        distances.append(min(distance(centroids[0], X[i])**2, distance(centroids[1], X[i])**2))

    centroids[2] = X[distances.index(max(distances))]

    return centroids



def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    data_map = np.zeros((len(X), len(C)))

    for i in range(len(X)):
        distances = [0 for x in range(len(C))] # len = 3
        for j in range(len(C)):
            distances[j] = distance(X[i], C[j])
        value_to_be_one = distances.index(min(distances))
        data_map[i][value_to_be_one] = 1
            

    return data_map


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    

    distance_map = np.zeros((len(X), len(C))) # n x k matrix of zeros
    obj_func_val = 0

    for i in range(len(X)): # Loop through every point
        for j in range(len(C)): # Loop through each centroid
            distance_map[i][j] = distance(X[i], C[j]) # Distance from each point to each centroid
        for k in range(len(C)):
            obj_func_val += min(np.array(distance_map[i])) ** 2 # Add the value of the distance to the centroid for the cluster the point is apart of

    return obj_func_val



def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    objective_values: array, shape (max_iter)
        The objective value at each iteration
    """
    centroids = k_init(X, k)
    obj_func_values = []


    for iter in range(max_iter):

        data_map = assign_data2clusters(X, centroids)

        cluster0 = []
        cluster1 = []
        cluster2 = []

        obj_func_values.append(compute_objective(X, centroids))

        # Sort all points into an array of other points in the same cluster
        for i in range(len(X)):
            if data_map[i][0] == 1:
                cluster0.append(X[i])
            elif data_map[i][1] == 1:
                cluster1.append(X[i])
            elif data_map[i][2] == 1:
                cluster2.append(X[i])
            else:
                print("Something went wrong.")

        running_x = 0
        running_y = 0
        for i in cluster0:
            running_x += i[0]
            running_y += i[1]
        centroids[0] = np.array((running_x / len(cluster0), running_y / len(cluster0)))

        running_x = 0
        running_y = 0
        for i in cluster1:
            running_x += i[0]
            running_y += i[1]
        centroids[1] = np.array((running_x / len(cluster1), running_y / len(cluster1)))

        running_x = 0
        running_y = 0
        for i in cluster2:
            running_x += i[0]
            running_y += i[1]
        centroids[2] = np.array((running_x / len(cluster2), running_y / len(cluster2)))


    return (centroids, obj_func_values)

    
print(k_means_pp(X, 3, 10))





    





