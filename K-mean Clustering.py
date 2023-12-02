import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Reading data from a CSV file
# Replace 'path_to_your_file.csv' with your file path and 'x', 'y' with your column names
data = pd.read_csv('path_to_your_file.csv')
X = data[['x', 'y']].values

# Elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):  # Testing for 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph
plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()

# Choose the number of clusters (k) based on the elbow plot
k = int(input("Enter the optimal number of clusters: "))

# Applying KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting the clusters
plt.figure(figsize=(10,5))
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(k):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s = 100, c = colors[i], label = f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], s = 300, c = 'black', marker='x', label = '
