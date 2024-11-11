import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data 

model = AgglomerativeClustering(n_clusters=2) 
result = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=result, cmap='viridis')
plt.title("Agglomerative Clustering (Iris Dataset)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
print("\n\n")

linked = linkage(X, 'ward')

plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram (Iris Dataset)")
plt.xlabel("Index")
plt.ylabel("Euclidean Distance")
plt.show()
