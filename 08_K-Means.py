import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data  

k = int(input('Enter k value : '))

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)  

labels = kmeans.labels_
centroids = kmeans.cluster_centers_  

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')  
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100) 
plt.title("K-Means Clustering (Iris Dataset)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
