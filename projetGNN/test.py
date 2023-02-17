import torch
import numpy as np
import argparse
import configparser
from torch import nn
import torch_geometric
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchvision.transforms import Compose
import os
import matplotlib.pyplot as plt
import math 
from mldvs.datawrappers import Datasets
from mldvs.trainers.trainers import BaselineTrainer
from mldvs.metrics.cls_metrics import SparseTopKAccuracy 
from mldvs.models.demo_models import EventCountCifar10DVSModel,EncoderFileToImg
from mldvs.callbacks import TensorBoardCallback, ModelCheckpoint
from mldvs.utils.generic_utils import expanded_join, load_params
from mldvs.utils.event_utils import EventToTensor, EventWindowing, EventSubsampling


os.environ['TORCH'] = torch.__version__
print(torch.__version__)
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Code here will not run when just importing the module.
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                    node_color=color, cmap="Set2")
    plt.show()


def calc_distance(events): 
    mat_dist = torch.zeros(events.shape[0], events.shape[0])
    for i in range(events.shape[0]): 
        x = events[i][0]
        y = events[i][1]
        t = events[i][3]
        vect = torch.tensor([x, y, t])
        vect_dist = torch.zeros(events.shape[0],1)
        inter = torch.sqrt((vect[0] - events[:,0]) **2+ (vect[1] - events[:,1])**2 + (vect[2] - events[:,3])**2)
        mat_dist[i,:]=inter

    return mat_dist
def get_args():
    parser = argparse.ArgumentParser(description="Basic script to train a deep conv net on Cifar10-DVS dataset.")
    #print(parser)
    parser.add_argument("--model-config",
                        type=str,
                        dest='mdl_cfg',
                        default=".../event-gnn/model_conf/event_baseline_config.json",
                        help="Path to the json file which contains all training parameters.")

    parser.add_argument("--generic-config",
                        type=str,
                        dest='generic_config',
                        default=".../event-gnn/config.ini",
                        help="Path to the file which contains generic information "
                            "(paths to datasets, standard splitting, ...)")

    return parser.parse_args()



args = get_args()

    # Load parameters (generic configuration and model specificity) and set paths
gen_cfg = configparser.ConfigParser()
gen_cfg.read(expanded_join(args.generic_config))

mdl_cfg = load_params(expanded_join(args.mdl_cfg), summary=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

    # Build data augmentation function:
data_aug = Compose([EventToTensor(),
                        EventSubsampling(sampling_factor=mdl_cfg["sampling_factor"]),
                        EventWindowing(window_size=mdl_cfg["event_window_size"])])
    # Get train/test dataset and some information about it:
dataset = Datasets["classification"]["DVS128-GestureV2"](generic_config=args.generic_config, transforms=data_aug, split=0)

train_dataset = dataset.get_train_set()
test_dataset = dataset.get_test_set()
val_dataset = dataset.get_validation_set()

dataset_params = dataset.get_params()
top_k_list = dataset_params["k"]

model1 = EncoderFileToImg(n_class=dataset.get_class_number()).to(device)

x = model1

top_k_acc = SparseTopKAccuracy(params=top_k_list, device=device)
from torch_geometric.data import Data

tb_callback = TensorBoardCallback(run_id=None,  # Random ID automatically generated
                                    log_folder=None,  # No hardcoded path, let callback reads it from 'config.ini'
                                    cfg_filepath=args.generic_config)

# mdl_callback = ModelCheckpoint(metric_name=top_k_acc.name.format(p=top_k_list[0]),
#                                    ref_model=model,
#                                    run_id=tb_callback.run_id,  # Use same run_id as Tensorboard
#                                    log_folder=None,  # No hardcoded path, let callback reads it from 'config.ini'
#                                    cfg_filepath=args.generic_config)

    # Build data generators for training and testing sets:
train_loader = DataLoader(train_dataset,
                            batch_size=mdl_cfg['train_batch_size'],
                            shuffle=False)
val_loader = DataLoader(val_dataset,
                            batch_size=mdl_cfg['test_batch_size'],
                            shuffle=False)

test_loader = DataLoader(test_dataset,        
                    batch_size=mdl_cfg['test_batch_size'])


data, labels  = next(iter(train_loader))

one_gesture, label_gesture = np.array(data[0,:]).astype("float32"), labels[0].item()

data_train_dataset = []
data_test_dataset = []

for i in range(len(train_dataset)):
    if(train_dataset[i][1] == 0 or train_dataset[i][1] == 2 or train_dataset[i][1] == 1  or train_dataset[i][1] == 3 or train_dataset[i][1] == 4) : 
        data_train_dataset.append(train_dataset[i])

for i in range(len(test_dataset)):
    if(test_dataset[i][1] == 0 or test_dataset[i][1] == 2 or test_dataset[i][1] == 0 or test_dataset[i][1] == 3 or test_dataset[i][1] == 4 ): 
        data_test_dataset.append(test_dataset[i])
        
