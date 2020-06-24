#importing libraries
import numpy as np
import pandas as pd
import math

#loading the dataset
df=pd.read_csv("C:/Users/havish/Desktop/iris.csv")

#creating different lists to store data that needed to be clustered
id = df['id'].values
c1 = df['sepallengthcm'].values
c2 = df['sepalwidthcm'].values
c3 = df['petallengthcm'].values
c4 = df['petalwidthcm'].values
species = df['species'].values

original_data=np.array(list(zip(id, species, c1, c2, c3, c4)))
data=np.array(list(zip(c1,c2,c3,c4)))

#function to initialize the centroids 
def init_centroids(k, data):
	c = []
    	s = np.random.randint(low=1, high=len(data), size=k)
    	while (len(s) != len(set(s))):
        	s = np.random.randint(low=1, high=len(data), size=k)
    	for i in s:
        	c.append(data[i])
	return c


#calculating euclidean distance for two input values
def euc_dist(a, b):
	sum = 0
    	for i, j in zip(a, b):
        	a = (i - j) * (i - j)
        	sum = sum + a
    	return math.sqrt(sum)


#calculating the distance from clusters
def cal_dist(centroids, data):
	c_dist = []
    	for i in centroids:
        	temp = []
        	for j in data:
            		temp.append(euc_dist(i, j))
        	c_dist.append(temp)
    	return c_dist

#clustering based on distance table
def perf_clustering(k, dist_table):
	clusters = []
    	for i in range(k):
        	clusters.append([])
    	for i in range(len(dist_table[0])):
        	d = []
        	for j in range(len(dist_table)):
           		 d.append(dist_table[j][i])
        	clusters[d.index(min(d))].append(i)
    	return clusters

#updating the centroids after each iteration
def update_centroids(centroids, cluster_table, data):
	for i in range(len(centroids)):
        	if (len(cluster_table[i]) > 0):
            	temp = []
            for j in cluster_table[i]:
                temp.append(list(data[j]))
            sum = [0] * len(centroids[i])
            for l in temp:
                sum = [(a + b) for a, b in zip(sum, l)]
            centroids[i] = [p / len(temp) for p in sum]

    	return centroids

#implementation of algorithm
def kMeans(k, data, max_iterations):
	dist_mem = []
    cluster_mem = []

    # Initialize centroids
    centroids = init_centroids(k, data)
    # Calculate distance table
    distance_table = cal_dist(centroids, data)
    # Perform clustering based on above generated distance table
    cluster_table = perf_clustering(k, distance_table)
    # Update centroid location based on above generated cluster table
    newCentroids = update_centroids(centroids, cluster_table, data)

    # Add distance and cluster table to memory list
    dist_mem.append(distance_table)
    cluster_mem.append(cluster_table)

    # Repeat from step 2 till stopping criteria is met
    for i in range(max_iterations):
        distance_table = cal_dist(newCentroids, data)
        cluster_table = perf_clustering(k, distance_table)
        newCentroids = update_centroids(newCentroids, cluster_table, data)

        # Check for stopping criteria
        # Maintain memory for past distance table and cluster table
        dist_mem.append(distance_table)
        cluster_mem.append(cluster_table)
        # If distance/cluster has not changed over last 10 iterations, stop, else continue
        if len(dist_mem) > 10:
            dist_mem.pop(0)
            cluster_mem.pop(0)
            if check_n_stop(dist_mem, cluster_mem):
                print("Stopped at iteration #", i)
                break

    # Display the final results
    for i in range(len(newCentroids)):
        print("Centroid #", i, ": ", newCentroids[i])
        print("Members of the cluster: ")
        for j in range(len(cluster_table[i])):
            print(original_data[cluster_table[i][j]])

#checking the clusters for 3 clusters & 10 iterations
print(kmeans(3,data,10))



