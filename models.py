import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.mincut_pool import dense_mincut_pool
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from copy import deepcopy
import numpy as np
import warnings

EPS = 1e-15

class GNN(torch.nn.Module):
    r"""Represents the class of end-to-end GNN-based models that directly generate soft cluster assignments, accounting for both graph connectivity and node features. Consists of GNN layers + MLP layers + loss objective computation. 

    Supports 2 loss objectives: `just_balance` (JBGNN - https://arxiv.org/abs/2207.08779) and `mincut` (MinCutPool - https://arxiv.org/abs/1907.00481).
    """

    def __init__(self, in_channels, hidden_channels, hidden_channels_mlp, out_channels, num_layers, num_layers_mlp, dropout, pooling, no_cluster_mask=None):
        super(GNN, self).__init__()

        out_chan = out_channels - 1 if no_cluster_mask is not None else out_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=False))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, normalize=False))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.mlps = torch.nn.ModuleList()
        self.mlps.append(Linear(hidden_channels, hidden_channels_mlp))
        for _ in range(num_layers_mlp - 2):
            self.mlps.append(Linear(hidden_channels_mlp, hidden_channels_mlp))
        self.mlps.append(Linear(hidden_channels_mlp, out_chan))

        self.dropout = dropout
        # self.activation = activation
        self.no_cluster_mask = no_cluster_mask
        self.pooling = pooling

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x, adj_t):
        for _, conv in enumerate(self.convs):
            x = F.relu(conv(x, adj_t))
            # x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        s = x
        for _, mlp in enumerate(self.mlps[:-1]):
            s = F.relu(mlp(s))
        s = self.mlps[-1](s) # no relu on last.
        
        adj_dense = adj_t.to_dense()

        if self.pooling == "mincut":
            _, _, mincut_loss, ortho_loss = dense_mincut_pool(x, adj_dense, s, self.no_cluster_mask)
            return s.softmax(dim=-1), mincut_loss + ortho_loss
        
        elif self.pooling == "just_balance":
            _, _, b_loss = just_balance_pool(x, adj_dense, s, self.no_cluster_mask)
            return s.softmax(dim=-1), b_loss

def SC(adj, sc_args):
    r"""Compute spectral clustering on the features.

    Args:
        adj (scipy.sparse matrix): graph adjacency matrix, converted into one of the SciPy sparse matrix formats.
        sc_args (dict): see https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sc = SpectralClustering(**sc_args)
        s = sc.fit_predict(adj)
    return sc, s

def just_balance_pool(x, adj, s, mask=None, normalize=True):
    r"""The Just Balance pooling operator from the `"Simplifying Clustering with 
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ paper
    
    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and the following auxiliary objective: 

    .. math::
        \mathcal{L} = - {\mathrm{Tr}(\sqrt{\mathbf{S}^{\top} \mathbf{S}})}

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}` 
            with batch-size :math:`B`, (maximum) number of nodes :math:`N` 
            for each graph, and feature dimension :math:`F`.
        adj (Tensor): Symmetrically normalized adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` 
            with number of clusters :math:`C`. The softmax does not have to be 
            applied beforehand, since it is executed within this method.
        mask (BoolTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    
    # Loss
    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-torch.einsum('ijj->i', ss_sqrt)) # rank3_trace
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss

def src_upsample(adj, features, labels, idx_train, portion, im_class_num):
    r"""Perform upsampling in the raw input domain, i.e. duplicate nodes in the minority classes.
    From: https://github.com/TianxiangZhao/GraphSmote/blob/main/utils.py.

    Args:
        portion (float): indicates the resulting quantity of nodes; 1 = double all nodes in the minority classes, 2 = triple, etc. Setting this to 0 results in even distribution, i.e. all minority classes receive the same amount of new nodes.
        im_class_num (float): The number of classes to rebalance. Also assumes the specified classes are numbered last, e.g. if you have 5 (0-4) classes, setting this to 2 implies classes 3 and 4 should be upsampled. 
    """

    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0) 

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train