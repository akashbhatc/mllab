import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
x=load_iris().data
k=3
centroid = X[:K]
def kmean(x,k ,maxiters=100):
    for i in (maxiters):
        expanded_x = x[:, np.newaxis]
        euc = np.linalg.norm ( expanded_x - centroid , axis=2)
        labels= np.argmin(euc , axis=1)
        new= np.array([x[label ==k ] .mean (axis=0) for k in range(k)])
        if np.all(new==centroid):
            break
        centroid = new
    return label, centroid
print("Labels:", labels)
print("Centroids:", centroids)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.show()
