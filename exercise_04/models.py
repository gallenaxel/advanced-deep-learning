import torch.nn as nn

from torch_geometric.nn import knn_graph, DynamicEdgeConv, global_mean_pool, MLP


# Defintion of the GNN model
# Use the DynamicEdgeConv layer from the pytorch geometric package like this:
# MLP is a Multi-Layer Perceptron that is used to compute the edge features, you still need to define it.
# The input dimension to the MLP should be twice the number of features in the input data (i.e., 2 * n_features),
# because the edge features are computed from the concatenation of the two nodes that are connected by the edge.
# The output dimension of the MLP is the new feauture dimension of this graph layer.

class GNNEncoder(nn.Module):
    def __init__(self, n_edge_features, n_latent_edge_features, k=5, hidden_layers=None, output_dim=128, final_layers=None):
        super(GNNEncoder, self).__init__()

        if hidden_layers is None:
            hidden_layers = [32]
        if final_layers is None:
            final_layers = [32]
        layer = DynamicEdgeConv(
                    MLP([n_edge_features, *hidden_layers, n_latent_edge_features]),
                    aggr='mean', k=k,  # k is the number of nearest neighbors to consider
                )

        self.layer_list = [layer]

        self.final_mlp = nn.Sequential(
            nn.LazyLinear(),
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
        # To combine all node features into single prediction, we recommend to use global pooling
        x = global_mean_pool(x, batch) # -> (n_batch, output_dim)
        # x is now a tensor of shape (n_batch, output_dim)

        # either your the last graph feature dimension is already the output dimension you want to predict
        # or you need to add a final MLP layer to map the output dimension to the number of labels you want to predict
        x = self.final_mlp(x)

        return x