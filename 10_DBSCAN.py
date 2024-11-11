import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
X = np.concatenate([X, np.random.uniform(low=-2, high=2, size=(50,2))])

plt.scatter(X[:, 0], X[:, 1], c='black', marker='.')
plt.title("Synthetic Data")
plt.show()

dbscan = DBSCAN(eps=0.1, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
plt.title("DBSCAN Clustering Result")
plt.show()
