import time
import sys
import os
import argparse
import io
from datetime import datetime
import numpy as np
import awkward as awk
#from gnn_encoder import GNNEncoder, collate_fn_gnn
#from gnn_trafo_helper import train_model, evaluate_model, normalize_x, normalize_y, normalize_time, denormalize_x, denormalize_y, get_img_from_matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, DynamicEdgeConv, global_mean_pool

DATA_PATH = "../datasets/iceCube/"  # path to the data

batch_size = 30


# Load the dataset
train_dataset = awk.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset = awk.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset = awk.from_parquet(os.path.join(DATA_PATH, "test.pq"))


# Normalize data and labels
# working with awk arrays is a bit tricky because the ['data'] field can't be assigned in-place,
# so we need to extract the time, x, and y coordinates, normalize them separately,
# and then concatenate them back together.

class AwkScaler:
    def __init__(self):
        self.fitted = False

    def fit(self, x):
        if not self.fitted:
            self.mean = awk.mean(x)
            self.std = awk.std(x)
            self.fitted = True
        else:
            raise RuntimeError("Stupid user already fitted")
    
    def transform(self, x):
        if self.fitted:
            return (x - self.mean)/self.std
        else:
            raise RuntimeError("Stupid user did not fit")
            
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        if self.fitted:
            return (x * self.std) +  self.mean
        else:
            raise RuntimeError("Stupid user did not fit")



times = train_dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                            # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
x = train_dataset["data"][:, 1:2, :]
y = train_dataset["data"][:, 2:3, :]

time_scaler = AwkScaler()
x_scaler = AwkScaler()
y_scaler = AwkScaler()

time_scaler.fit(times)
x_scaler.fit(x)
y_scaler.fit(y)

# Concatenate the normalized data back together


def scale_input(dataset):
    """Open dataset, take apart and scale, put back together.

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    times = dataset["data"][:, 0:1, :] 
    x = dataset["data"][:, 1:2, :]
    y = dataset["data"][:, 2:3, :]
    norm_times = time_scaler.transform(times)
    norm_x = x_scaler.transform(x)
    norm_y = y_scaler.transform(y)
    return awk.concatenate([norm_times, norm_x, norm_y], axis=1)




# -----labels ----------
# Normalize labels (this can be done in-place), e.g. by
label_x_scaler = AwkScaler()
label_y_scaler = AwkScaler()

train_dataset["data"] = scale_input(train_dataset)
train_dataset["xpos"] = label_x_scaler.fit_transform(train_dataset["xpos"])
train_dataset["ypos"] = label_y_scaler.fit_transform(train_dataset["ypos"])

val_dataset["data"] = scale_input(val_dataset)
val_dataset["xpos"] = label_x_scaler.transform(val_dataset["xpos"])
val_dataset["ypos"] = label_y_scaler.transform(val_dataset["ypos"])


test_dataset["data"] = scale_input(test_dataset)
test_dataset["xpos"] = label_x_scaler.transform(test_dataset["xpos"])
test_dataset["ypos"] = label_y_scaler.transform(test_dataset["ypos"])



# Hint: You can define a helper function to normalize the data and you can use the same normalization for the validation and test datasets.

# Create the DataLoader for training, validation, and test datasets
# Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the awk arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    !!!You do not need to change this function.!!!

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)



# Defintion of the GNN model
# Use the DynamicEdgeConv layer from the pytorch geometric package like this:
# MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
# The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
# because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
# The output dimension of the MLP is the new feauture dimension of this graph layer.
from torch_geometric.nn import DynamicEdgeConv
class GNNEncoder(nn.Module):
    def __init__(self, ):
        super(GNNEncoder, self).__init__()
        
        layer = DynamicEdgeConv(
                    MLP(...),
                    aggr='mean', k=k,  # k is the number of nearest neighbors to consider
                )

    def forward(self, data):
        # data is a batch graph item. it contains a list of tensors (x) and how the batch is structured along this list (batch)
        x = data.x
        batch = data.batch

        # loop over the DynamicEdgeConv layers:
        for layer in self.layer_list:
            x = layer(x, batch)

        # the output of the last layer has dimensions (n_batch, n_nodes, graph_feature_dimension)
        # where n_batch is the number of graphs in the batch and n_nodes is the number of nodes in the graph
        # i.e. one output per node (i.e. the hits in the event).
        # To combine all node feauters into single prediction, we recommend to use global pooling
        x = global_mean_pool(x, batch) # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # either your the last graph feature dimension is already the output dimension you want to predict
        # or you need to add a final MLP layer to map the output dimension to the number of labels you want to predict
        x = self.final_mlp(x)

        return x