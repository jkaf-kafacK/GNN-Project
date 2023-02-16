import os.path as osp
import torch
from zipfile import ZipFile
from test import *
import time 
import copy

import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/onajib/event-gnn/examples/scripts/rendu_projet/utils')

from utils_dataset import * 
from utils import *
import matplotlib.pyplot as plt
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Tuple, Union
from test import * 
# https://github.com/NYU-MLDA/OpenABC/blob/master/models/qor/SynthNetV1/netlistDataset.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# https://networkx.org/documentation/stable/auto_examples/3d_drawing/plot_basic.html

color_dictionnary = {
    0: 'red', 
    1: 'blue', 
    2: 'green', 
    3: 'orange',
    4: 'yellow', 
    5: 'cyan', 
    6: 'navy', 
    7: 'blueviolet', 
    8: 'olive', 
    9: 'darkcyan', 
    10: 'crimson'
}

from torch_geometric.utils import to_networkx


def _format_axes(ax):
    """Visualization options for the 3D axes."""
    # Turn gridlines off
    ax.grid(False)
    # Suppress tick labels
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    # Set axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
def visualize_graphe(data, G, id):
    pos = data.pos
    pos = pos.cpu().detach().numpy()
    dict_pos = {}
    for i in range(pos.shape[0]):
        dict_pos[i] = pos[i,:]
    # Extract node and edge positions from the layout
    node_xyz = np.array([dict_pos[v] for v in sorted(G)])
    edge_xyz = np.array([(dict_pos[u], dict_pos[v]) for u, v in G.edges()])
    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    color = data.y
    ax.scatter(*node_xyz.T, s=100, c=color_dictionnary[color],ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    _format_axes(ax)
    fig.tight_layout()
    ax.set_title('Graph KNN label ' + str(color) + ' - example ' + str(id))
    return fig, ax




class CustomDataset(Dataset):
    def __init__(self, root,dataset, transform=False, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)      
        self.dataset = dataset
        if(transform== True): 
            self.transform = self.create_knn_graph_kmeans
        else:
            self.transform = self.create_knn_graph_means
        
    @property
    def len(self):
        return len(self.dataset)

    def num_classes(self): 
        print("we are here")
        y = torch.cat([torch.tensor(data[1]).reshape(1) for data in self.dataset], dim=0)
        if hasattr(self, '_data_list') and self._data_list is not None:
            self._data_list = self.len * [None]
        return self._infer_num_classes(y)  -1 
    
    #def download(self):
    def get(self, idx): 
        return self.dataset[idx]

    def get_graph(self, idx):
        return self.create_graph(idx)

    def get_data_by_label(self, label):
        list_data = [] 
        for i in range(self.len): 
            if(self.get(i)[1] == label): 
                list_data.append(self.__getitem__(i)) 
        return list_data

    def create_knn_graph_kmeans(self, data, coef_time=10000, k=50): 
        data, label = data
        one_gesture, label_gesture = np.array(data).astype("float32"), label
        data = normalize_time(one_gesture, coef_time)
        #labels = generate_labels(data, label)
        data_to= generate_knn_graph_kmeans(data, label, k)
        return data_to

    def create_knn_graph_means(self, data, coef_time=1000): 
        data, label = data
        one_gesture, label_gesture = np.array(data).astype("float32"), label
        data = normalize_time(one_gesture, coef_time)
        data_to= generate_knn_graph_means(data, label, coef_mean=1000)
        data_to.edge_attr
        return data_to

    # def create_radius_graph

    def create_graph(self, idx):
        data = self.__getitem__(idx) 
        G = to_networkx(data, to_undirected=True)
        return G

    def create_graph_from_data(self, data):
        G = to_networkx(data, to_undirected=True)
        return G

newDataset_train = CustomDataset(root="/onajib/event-gnn/src", dataset=train_dataset, transform=True, pre_transform=None, pre_filter=None)
interDataset_train = CustomDataset(root="/onajib/event-gnn/src", dataset=data_train_dataset, transform=True, pre_transform=None, pre_filter=None)
interDataset_test = CustomDataset(root="/onajib/event-gnn/src", dataset=data_test_dataset, transform=True, pre_transform=None, pre_filter=None) 

data_0 = newDataset_train[0]
# data_1 = newDataset_train[1]
# data_2 = newDataset_train[2]
# data_3 = newDataset_train[3]

graph_0 = newDataset_train.create_graph_from_data(data_0)
# graph_1 = newDataset_train.create_graph_from_data(data_1)
# graph_2 = newDataset_train.create_graph_from_data(data_2)
# graph_3 = newDataset_train.create_graph_from_data(data_3)

fig1, ax1 = visualize_graphe(data_0.to('cpu'), graph_0, data_0.y)
# fig2, ax2 = visualize_graph(data_1.to('cpu'), graph_1, data_1.y)
# fig3, ax3 = visualize_graph(data_2.to('cpu'), graph_2, data_2.y)
# fig4, ax4 = visualize_graph(data_3.to('cpu'), graph_3, data_3.y)

plt.show()