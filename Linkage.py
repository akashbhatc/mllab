import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage 
from sklearn.datasets import load_iris

data = load_iris().data
data = data[:6]
def proximity_matrix(data):
    n = data.shape[0]
    pm = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            pm[i,j]=pm[j,i]=np.linalg.norm(data[i]-data[j])
    return pm
def plot_deno(data,method):
    link_mat = linkage(data,method=method)
    dendrogram(link_mat)
    plt.title(method.capitalize()+" Linkage")
    plt.show()
print('Proximity Matrix is : ')
print(proximity_matrix(data))
plot_deno(data,'single')
plot_deno(data,'complete')
