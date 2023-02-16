from dataset import * 
import os
import torch
from torch_geometric.nn import aggr
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from gcn import model_gcn

train_loader = DataLoader(interDataset_train, batch_size=128, shuffle=True)
val_loader = DataLoader(interDataset_test, batch_size=128, shuffle=True)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from pathlib import Path
from joblib import dump
from joblib import load

p = Path("tensorboard/")

p.mkdir(parents=True, exist_ok=True)

# one = next(iter(train_loader))

# Embed each node by performing multiple message passing
# Aggregate node embeddings into a unified graph readout layer
# Train a final classifier on the graph embedding

# choosing model
model_gcn = model_gcn

optimizer = torch.optim.SGD(model_gcn.parameters(), lr = 0.005, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

EPOCH = 15
list_dir = os.listdir("./")

PATH = "/home/onajib/event-gnn/examples/scripts/checkpoints/model_gat.pt"
LOSS = 0.4
start_epoch = 0

torch.save({
            'epoch': EPOCH,
            'model_state_dict': model_gcn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

checkpoint = torch.load(PATH)
model_gcn.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("model state dict", model_gcn.state_dict)
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_gcn.eval()

def train():
    
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.

         out = model_gcn(data.x, data.edge_index, data.batch)  # Perform a single forward pass.  
         loss = criterion(out, data.y.to(device))  # Compute the loss.
         loss.backward()  # Derive gradients.
         
         optimizer.step()  # Update parameters based on gradients.$
         print("---------loss", loss)
         total_loss += loss

    return total_loss/len(train_loader.dataset)

def test(loader):
     model_gcn.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.        
         out = model_gcn(data.x, data.edge_index, data.batch)  
         
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         data.y = data.y.clone().detach()
         print("prediction", pred)
         print("data.y ", data.y)
         pred = pred.clone().detach()
         pred = pred.to(device)
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

train_accs = []
val_accs = []
loss_vals = []

for epoch in range(start_epoch, epoch):
    print("training epoch: ", start_epoch)
    train_loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    loss_vals.append(train_loss)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Test Acc: {val_acc:.4f}')
    # , Test Acc: {test_acc:.4f}')
    writer.add_scalar('train loss', train_loss, epoch)
    writer.add_scalar('train accuracy', train_acc, epoch)
    writer.add_scalar('test accuracy', val_acc, epoch)

writer.close()

dump(train_accs, Path("tensorboard/train_accuracy_3classes_gcn_aggr_mean.txt"))
dump(val_accs, Path("tensorboard/test_accuracy3classes_gcn_mean.txt"))