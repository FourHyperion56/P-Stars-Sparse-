import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# set directory
MAIN_DIR = '/Users/isaacperez/Downloads/CS70 Homework/P-Stars_Project'
DATA_DIR = os.path.join(MAIN_DIR, 'team_Sparse')
DATA_DIR2 = os.path.join(MAIN_DIR, 'var_output/v0.1.0')
LC_DIR = os.path.join(MAIN_DIR, 'g_band_lcs') # for folded .dats
LC_OUT = os.path.join(MAIN_DIR, 'sample_lcs')
DATA_DIR3 = '/Users/isaacperez/Downloads/CS70 Homework/P-Stars_Project/team_Sparse/HCV.csv'

file_path = DATA_DIR3  # Replace with the path to your CSV file
data = pd.read_csv(file_path)


def clusterPlotter():
    # Reading data from a CSV file
    # Replace 'path_to_your_file.csv' with your file path and 'x', 'y' with your column names
    data = pd.read_csv(file_path)
    X = data[['chi2', 'mad']].values

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
    plt.xlabel('chi2')
    plt.ylabel('mad')  # Within cluster sum of squares
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

    plt.scatter(centroids[:, 0], centroids[:, 1], s = 300, c = 'black', marker='x', label = 'Centroids')
    plt.title('Clusters of data points')
    plt.xlabel('mad')
    plt.ylabel('chi2')
    plt.legend()
    plt.show()


def newClusterPlotter(): 
    ######## new method
    import pandas as pd
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np


    # Function to apply the elbow method
    def find_optimal_clusters(data):
        wcss = []
        for i in range(1, 11):  # Testing 1 to 10 clusters
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

        # Determine the elbow point
        k = wcss.index(min(wcss, key=lambda x: abs(x - np.mean(wcss)))) + 1
        return k


    # Load the entire CSV file
    groupId = 1040910
    data = pd.read_csv(file_path)

    # Assuming there's a column 'dataset_id' that identifies each dataset
    dataset_ids = data["groupid"].unique()

    for dataset_id in dataset_ids:
        # Filter data for the current dataset
        current_data = data[data["groupid"] == dataset_id]

        # Extract the relevant columns
        X = current_data[['chi2', 'mad']].values

        # Find the optimal number of clusters
        optimal_clusters = find_optimal_clusters(X)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Plotting
        plt.figure(figsize=(10, 5))
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'brown', 'pink', 'gray']
        for i in range(optimal_clusters):
            plt.scatter(X[labels == i, 0], X[labels == i, 1], s=100, c=colors[i], label=f'Cluster {i + 1}')

        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='chi2', label='Centroids')
        plt.title(f'Clusters for Dataset ID {dataset_id}')
        plt.xlabel('chi2')
        plt.ylabel('mad')
        plt.legend()
        plt.show()

newClusterPlotter()
