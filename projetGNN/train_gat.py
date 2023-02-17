import torch 
from gat import model_gat
from dataset import * 
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
from pathlib import Path
from joblib import dump
from joblib import load

p = Path("tensorboard/")

p.mkdir(parents=True, exist_ok=True)
# loading data into batches 
train_loader = DataLoader(interDataset_train, batch_size=128, shuffle=True)
val_loader = DataLoader(interDataset_test, batch_size=128, shuffle=True)


# choosing model
model_gat = model_gat

optimizer = torch.optim.SGD(model_gat.parameters(), lr = 0.005, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

EPOCH = 15
list_dir = os.listdir("./")

PATH = ".../event-gnn/examples/scripts/checkpoints/model_gat.pt"
LOSS = 0.4
start_epoch = 0

torch.save({
            'epoch': EPOCH,
            'model_state_dict': model_gat.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

checkpoint = torch.load(PATH)
model_gat.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("model state dict", model_gat.state_dict)
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_gat.eval()

def train():
    
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
         data.x = data.x.float() 
         out = model_gat(data, data.batch)  # Perform a single forward pass.  
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.$
         print("---------loss", loss)
         total_loss += loss

    return total_loss/len(train_loader.dataset)

def test(loader):
     model_gat.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.        
         out = model_gat(data, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         print("pred", pred)
         data.y = data.y.clone().detach()  
         print("data y", data.y )      
         pred = pred.clone().detach()
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

train_accs = []
val_accs = []
loss_vals = []

for epoch in range(0, 30):
    print("starting training")
    train_loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    loss_vals.append(train_loss)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
    print(f'Epoch: {epoch:03d}, Test Acc: {val_acc:.4f}')
    writer.add_scalar('train loss_gat', train_loss, epoch)
    writer.add_scalar('train accuracy_gat', train_acc, epoch)
    writer.add_scalar('test accuracy_gat', val_acc, epoch)

writer.close()

dump(train_accs, Path("tensorboard/train_accuracy.txt"))
dump(val_accs, Path("tensorboard/test_accuracy.txt"))
