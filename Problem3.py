# Name: Chase Rensberger
# Homework 4 Problem 3
# Honestly a lot of this code is dependent on k equaling 3 which is obviously not the only condition but is what I implemented because I am very low on time to finish this.
# I also picked k=3 because it is the true value of the data, which obviously you wouldn't know it a real case but I implemented here just so what I had would be as relevant as possible.
# Everything else (other than the value of k) is working fine though as far as I can tell.
# There is a seperate pdf in this directory called Problem3Plots.pdf which has the plots showing the actual coloring initially alongside the graphs of the objective function for each iteration
# and then a final plot with the data colored by assigment which can be compared to the first plot

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

# Only want firs to columns
X = X[:, :2]

# Used for plot 1 in the pdf
# Plot data with colors
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.title('True coloring')
# plt.xlabel('sepal length/sepal width')
# plt.ylabel('petal length/petal width')
# plt.show()

# Euclidian distance between two points
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
    centroids = np.zeros((k, 2))
    #initialize first centroid to random point
    centroids[0] = random.choice(X)
    # centroids[0] = X[1] # I used this sometimes in testing for repeatable behavior

    # Initialize other centroids by maximizing distance to existing centroids
    # Want to initialize so that each centroid is allocated by finiding the point with the max distance to the nearest centroid

    for i in range(1, k): # we have k-1 centroids left to allocate
        distances = []
        for j in range(len(X)): # Loop through every data point
            curr_min = 9223372036854775807
            for l in range(len(centroids)):
                if distance(X[j], centroids[l]) < curr_min:
                    curr_min = distance(X[j], centroids[l])
            distances.append(curr_min)
        
        centroids[i] = X[distances.index(max(distances))]

    return centroids


# Testing k_init
# centroids = k_init(X, 3)
# plt.scatter(X[:, 0], X[:, 1])
# plt.scatter(centroids[:, 0], centroids[:, 1])
# plt.show()



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

    # This function outputs a map where the number of rows is the number of points and we put a 1 in the value for the cluster it belongs to
    # We do this by looping through each point and finding its minimum distance to each cluster
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
        
        obj_func_val += min(distance_map[i]) ** 2 # Add the value of the distance to the centroid for the cluster the point is apart of

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
    # These values will be updated during the function execution
    centroids = k_init(X, k)
    obj_func_values = []


    for iter in range(max_iter):

        data_map = assign_data2clusters(X, centroids)

        # ---------------------------------------
        new_y = [-1 for x in range(len(X))]
        for i in range(len(X)):
            if data_map[i][0] == 1:
                new_y[i] = 0
            elif data_map[i][1] == 1:
                new_y[i] = 1
            elif data_map[i][2] == 1:
                new_y[i] = 2
            else:
                print("Something went wrong.")

        plt.scatter(X[:, 0], X[:, 1])
        plt.scatter(centroids[:, 0], centroids[:, 1])
        plt.scatter(X[:, 0], X[:, 1], c=new_y)
        
        plt.show()


        # ---------------------------------------

        clusters = [[] for _ in range(k)]
        

        # Sort each point into a cluster array using data_map
        for i in range(len(X)):
            clusters[np.where(data_map[i] == 1)[0][0]].append(X[i])


        obj_func_values.append(compute_objective(X, centroids))

        # Average each point in each cluster and update the centroids accordingly
        for i in range(len(clusters)):
            running_x = 0
            running_y = 0
            for j in clusters[i]:
                running_x += j[0]
                running_y += j[1]
            if (len(clusters[i]) != 0):
                centroids[i] = np.array((running_x / len(clusters[i]), running_y / len(clusters[i])))


    return (centroids, obj_func_values)


centroids, obj_func_values = k_means_pp(X, 3, 5)




# Used for plots 2 - 6 in the PDF
# plt.plot(obj_func_values)
# plt.title('Objective function over 1000 iterations for k=3')
# plt.xlabel('Iterations')
# plt.ylabel('Objective Function')
# plt.show()


# data_map = assign_data2clusters(X, centroids)
# new_y = [-1 for x in range(len(X))]
# for i in range(len(X)):
#     if data_map[i][0] == 1:
#         new_y[i] = 0
#     elif data_map[i][1] == 1:
#         new_y[i] = 1
#     elif data_map[i][2] == 1:
#         new_y[i] = 2
#     else:
#         print("Something went wrong.")

# Used for plot 7 in the pdf
# Plot data with colors
# plt.scatter(X[:, 0], X[:, 1], c=new_y)
# plt.title('Evaluated coloring')
# plt.xlabel('sepal length/sepal width')
# plt.ylabel('petal length/petal width')
# plt.show()


# Ideally I wouldn't of generalized to k=3 so that I could plot the accuracy as per described in the assignment pdf but I believe that the objective accuracy would be highest around k=3 and decrease as k takes higher or lower values. 
# I believe that k would be the global maximum of the graph of x=number of cluster and y=clustering objective









    





