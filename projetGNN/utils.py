import numpy as np
from test import *
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
from KNNgraph import * 
from torch_geometric.data import Data
import torch 
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import time
from sklearn.cluster import KMeans

os.environ['TORCH'] = torch.__version__
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data correspond aux données d'un geste
# tester avec ou sans normalisation 

def normalize_time(data, coef):
    data[:,3] = data[:,3]/coef
    data[:,3] -= data[:,3].min(axis=0)
    #peut etre ajouter la normalisation suivant l'axe des x et l'axe des y
    return data


def generate_knn_graph_kmeans(data, labels, k): 

    kmeans = KMeans(n_clusters=int(k)).fit(data)
    label = kmeans.predict(data)
    u_labels = np.unique(label)
    centroids = kmeans.cluster_centers_
    data_pos = np.stack((np.array(centroids[:,0]),np.array(centroids[:,1]),np.array(centroids[:,3])))
    means_p = np.array(centroids[:,2])
    data_= Data(x=torch.tensor(centroids), y=labels,pos=torch.tensor(data_pos.T))
    data_to_graph= graphKNN.__call__(data_)
    data_to_graph.to(device)
    return data_to_graph


def generate_knn_graph_means(data, labels, coef_mean):
    data_pos = np.stack((np.array(data[:,int(0)]),np.array(data[:,int(1)]), np.array(data[:,int(3)]))) 
    data_ = np.reshape(data[:,2], (1,data[:,2].shape[0]))

    data_features = np.vstack((data_pos, data_))
    n_points = data.shape[0]

    _means_x = []
    _means_y = []
    _means_t = []
    _means_p = []

    for i in range(0, n_points, coef_mean):
        mean_inter_x = sum(data[i:i+coef_mean,0])/coef_mean
        mean_inter_y = sum(data[i:i+coef_mean,1])/coef_mean
        mean_inter_t = sum(data[i:i+coef_mean,3])/coef_mean
        mean_inter_p = sum(data[i:i+coef_mean,2])/coef_mean
        _means_x.append(mean_inter_x)
        _means_y.append(mean_inter_y)
        _means_t.append(mean_inter_t)
        _means_p.append(mean_inter_p)

    means_p = []
    for i in range(len(_means_p)): 
        if(_means_p[i] > 0.5): 
            means_p.append(1)
        else : 
            means_p.append(0)

    data_pos = np.stack((np.array(_means_x),np.array(_means_y),np.array(_means_t)))
    means_p = np.array(means_p)
    data_features = np.vstack((data_pos, means_p))
    #data_= Data(x=torch.tensor(means_p).to(device), y=labels,pos=torch.tensor(data_pos.T).to(device))
    data_to_graph = Data(x=torch.tensor(data_features.T).to(device), y=labels ,pos=torch.tensor(data_pos.T).to(device))

    data_to_graph= graphKNN.__call__(data_to_graph)
    return data_to_graph
