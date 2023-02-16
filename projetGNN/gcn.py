import os
import torch
from torch_geometric.nn import aggr
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

num_features = 4
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 128)
        self.conv4 = GCNConv(128, 128)

        #self.global_pool = aggr.SumAggregation()
        self.lin = Linear(128,num_classes)  

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
       # print("shape of x after first conv", x.shape)
        x = self.conv2(x, edge_index)
        x = x.relu()
        #print("shape of x after second conv", x.shape)
        x = self.conv3(x, edge_index)
        x = x.relu()
        #print("shape of x after second conv", x.shape)
        x = self.conv4(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)
        x = torch.sigmoid(x)

       # print("shape of x after linear", x.shape)
        
        return x

model_gcn = GCN(hidden_channels=128).to(device)
print(model_gcn)