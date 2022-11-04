from math import ceil
from typing import List, Union

import torch as th
import torch.nn.functional as F
from einops import reduce
from einops.layers.torch import Rearrange, Reduce
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric.nn import DenseGCNConv as GCNConv
from torch_geometric.nn import Sequential, dense_diff_pool


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, apply_bn=True, n_hidden_layers=0):
        super(GNN, self).__init__()
        
        def block(in_fea, out_fea):
            if apply_bn: 
                return [
                    (GCNConv(in_fea, out_fea, normalize),'x, adj, mask -> x'),
                    nn.ReLU(),
                    Rearrange('b n c -> b c n'),
                    nn.BatchNorm1d(out_fea),
                    Rearrange('b c n -> b n c'),
                ]
            else:
                return [
                    (GCNConv(in_fea, out_fea, normalize),'x, adj, mask -> x'),
                    nn.ReLU(),
                ]

        assert n_hidden_layers >= 0
        gnn_layers = block(in_channels, hidden_channels)
        for _ in range(n_hidden_layers):
            gnn_layers += block(hidden_channels, hidden_channels)
        gnn_layers += block(hidden_channels, out_channels)

        self.gnn = Sequential('x, adj, mask',gnn_layers)

    def forward(self, x, adj, mask=None):
        # x shape: (batch_size, num_nodes, in_channels)
        
        x = self.gnn(x, adj, mask)

        return x

class DiffPoolEncoder(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        in_nodes: int,
        hidden_channels: int, 
        out_channels: int,
        apply_bn: bool = True,
        n_hidden_layers: int = 0,
    ):
        """A GNN encoder with DiffPool pooling layer.

        Default setting: 1 DiffPool layer, 1 MLP layer

        Args:
            in_channels (int): Number of input features.
            in_nodes (int): Number of input nodes in one graph.
            hidden_channels (int): Number of hidden features.
            out_channels (int): Number of output features.
            apply_bn (bool, optional): Whether to apply batch normalization. Defaults to True.
            n_hidden_layers (int, optional): Number of hidden layers. Defaults to 0.
        """
        super(DiffPoolEncoder, self).__init__()

        num_nodes = ceil(0.25 * in_nodes)
        self.gnn1_pool = GNN(
            in_channels, hidden_channels, num_nodes,
            apply_bn=apply_bn, n_hidden_layers=n_hidden_layers)
        self.gnn1_embed = GNN(in_channels, hidden_channels, hidden_channels,
            apply_bn=apply_bn, n_hidden_layers=n_hidden_layers)

        # num_nodes = ceil(0.25 * num_nodes)
        # self.gnn2_pool = GNN(64, 64, num_nodes)
        # self.gnn2_embed = GNN(64, 64, 64)

        self.gnn2_embed = GNN(hidden_channels, hidden_channels, hidden_channels,
            apply_bn=apply_bn, n_hidden_layers=n_hidden_layers)

        self.mlp = nn.Sequential(
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, adj, mask=None)-> th.Tensor:
        s = self.gnn1_pool(x, adj, mask)    # determines the nodes to keep
        x = self.gnn1_embed(x, adj, mask)   # determines the node embeddings (num features)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        
        # Consider using only one pooling layer

        x = self.gnn2_embed(x, adj)
        x = self.mlp(x)
        return x

class GnnExtractor(BaseFeaturesExtractor):
    """Feature extractor using a GNN encoder.
    """
    def __init__(
        self, 
        observation_space, 
        features_dim=64, 
        embedding_vars: Union[List, None] = None,
        embedding_dim: int = 64,
        encoder=DiffPoolEncoder,
        **kwargs):
        """
        Args:
            embedding_vars (list): List of tuples (n_cls, n_var) for embedding. 
                n_cls is short for number of classes, while n_var is short for 
                the number of variables for the same embedding layer.
                None for no embedding.
        """
        super(GnnExtractor, self).__init__(observation_space, features_dim)

        n_nodes = observation_space.shape[0]
        n_raw_features = observation_space.shape[1]
        self.embedding_dim = embedding_dim

        if embedding_vars is None:
            n_features = n_raw_features
            self.embedding_vars = [n_features]
            self.embedding_layers = nn.ModuleList([nn.Identity()])
        else:
            n_features = embedding_dim
            self.embedding_vars = embedding_vars
            self.embedding_layers = nn.ModuleList([
                nn.Embedding(n_cls+1, embedding_dim, padding_idx=-1)  # pad the output at index -1
                for n_cls, n_var in embedding_vars
            ])

        self.encoder = encoder(
            in_channels=n_features, 
            in_nodes=n_nodes,
            hidden_channels=64, 
            out_channels=features_dim,
            **kwargs
        )

    def forward(self, observations: th.Tensor)->th.Tensor:
        """
        Args:
            observations (th.Tensor): (batch_size, n_nodes, n_fea_and_adj) last dimension
                is a concatenation of the features matrix and the adjacency matrix.
        """
        # Preprocess observations
        batch_size, n_nodes, n_fea_and_adj = observations.size()
        x_raw = observations[:, :, :-n_nodes]
        adj = observations[:, :, -n_nodes:]
        mask = None

        # Embedding
        x = th.zeros(batch_size, n_nodes, self.embedding_dim, device=observations.device)
        var_dim_cum = 0
        for embedding_var, embedding_layer in zip(self.embedding_vars, self.embedding_layers):
            n_cls, n_var = embedding_var
            x_delta = embedding_layer(x_raw[:, :, var_dim_cum:var_dim_cum+n_var].long())
            x += reduce(x_delta, 'b n f h -> b n h', 'sum')
            var_dim_cum += n_var

        return self.encoder(x, adj, mask)

# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True, use_cuda=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = nn.Linear(input_dim, input_dim)
        # self.layer.weight.data = utls.fanin_init(self.layer.weight.data.size())
        self.weight = nn.Parameter(th.FloatTensor(input_dim, output_dim))
        # self.weight.data = init.xavier_uniform(self.weight.data, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(output_dim))
            # self.bias.data = init.constant(self.bias.data, 0.0)
        else:
            self.bias = None

    def forward(self, x, adj):
        x = self.layer(x)
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = th.matmul(adj, x)
        if self.add_self:
            y += x
        y = th.matmul(y,self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

class GcnEncoder(BaseFeaturesExtractor):
    """GCN encoder for graphs. 
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None, use_cuda=True, num_aggs=1):
        super().__init__(observation_space, features_dim)
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=num_aggs
        self.use_cuda = use_cuda

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, use_cuda=self.use_cuda)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, use_cuda=self.use_cuda) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, use_cuda=self.use_cuda)
        return conv_first, conv_block, conv_last


    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [th.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = th.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(self.device)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = th.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = th.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = th.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = th.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)
        #x = self.act(x)
        out, _ = th.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = th.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = th.cat(out_all, dim=1)
        else:
            output = out
        #print(output.size())
        return output

class SoftPoolingGcnEncoder(GcnEncoder):
    """
    GCN from graph RL paper:
        Kamarthi, H. et al.,
        "Influence maximization in unknown social networks: Learning Policies for Effective Graph Sampling."
        AAMAS, pp. 575-583. 2020.
    """
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, num_layers,
            assign_hidden_dim, assign_num_layers=-1, assign_ratio=1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0,
            assign_input_dim=-1, args=None, num_aggs=1, use_cuda=True):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args, num_aggs=num_aggs, use_cuda=use_cuda)
        add_self = not concat
        self.num_pooling = num_pooling
        self.assign_ent = True


        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            

            # next pooling layer
            assign_input_dim = embedding_dim + hidden_dim * (num_layers - 1)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)


        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes=None,compute_loss=True, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        from copy import deepcopy

        # mask
        self.link_loss = th.zeros(1).to(self.device)
        self.entropy_loss = th.zeros(1).to(self.device)
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
                
        self.node_embeddings = embedding_tensor.clone()
        out, _ = th.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = th.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = F.softmax(self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask), -1)
            
            # [batch_size x num_nodes x next_lvl_num_nodes]
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask
            if compute_loss:
                self.link_loss += self.loss(adj)
                self.entropy_loss -= (1/adj.size()[-2]) * th.sum(self.assign_tensor * th.log(self.assign_tensor))

            # update pooled features and adj matrix
            x = th.matmul(th.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = th.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i])


            out, _ = th.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                #out = th.mean(embedding_tensor, dim=1)
                out = th.sum(embedding_tensor, dim=1)
                out_all.append(out)


        if self.concat:
            output = th.cat(out_all, dim=1)
        else:
            output = out
        return output
