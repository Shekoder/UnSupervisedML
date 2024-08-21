import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset
file_path = "C:/Users/HP8CG/OneDrive/Documents/PROJECTS/grip/iris.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Select features (excluding the target variable if present)
X = data.iloc[:, :-1]  # Assuming the last column is the target

# Elbow Method
inertia = []
k_range = range(1, 11)  # Testing cluster sizes from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette Score
silhouette_avg = []
for k in k_range[1:]:  # Avoid k=1 as silhouette score is not defined for a single cluster
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(X, cluster_labels))

# Plot Silhouette Scores
plt.figure(figsize=(10, 5))
plt.plot(k_range[1:], silhouette_avg, marker='o')
plt.title('Silhouette Score For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Fit the KMeans model with the optimal number of clusters (for example, k=3)
optimal_k = 3  # Replace with the optimal k from the plots
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title(f'Clusters Visualization with k={optimal_k}')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.show()