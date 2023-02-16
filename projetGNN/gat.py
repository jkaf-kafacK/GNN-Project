import torch
from torch_geometric.nn import Linear
from torch_geometric.nn import aggr
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T

num_features = 4
num_classes = 5

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 64
        self.in_head = 2
        self.out_head = 2
        
        
        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, heads=self.in_head)
        self.conv3 = GATConv(self.hid*self.in_head, self.hid, heads=self.in_head)
        self.conv4 = GATConv(self.hid*self.in_head, self.hid, heads=self.in_head)
        self.lin = Linear(128,num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data,  batch):
        x, edge_index = data.x, data.edge_index
                
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
       # print("shape of x after global mean pool", x.shape)

        # 3. Apply a final classifier
        self.lin = self.lin
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)
model_gat = GAT().to(device)
print(model_gat)