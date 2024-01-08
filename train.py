import argparse
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix
from sklearn.metrics import normalized_mutual_info_score, completeness_score
from numpy.random import seed
from collections import Counter

import models
import utils

import importlib
importlib.reload(models)
importlib.reload(utils)

def train():
    model.train()
    optimizer.zero_grad()
    out, loss = model(data.x, data.adj_t)
    loss.backward()
    optimizer.step()

    return loss.item(), out

@torch.no_grad()
def test(out=None, exclude_non_clustered=False):
    model.eval()
    out = model(data.x, data.adj_t)[0] if out is None else out

    if exclude_non_clustered:
        y_true = data.y[data.no_cluster_mask].squeeze().cpu()
        y_pred = out[data.no_cluster_mask].max(1)[1].cpu()
        adj = data.adj_t[data.no_cluster_mask, data.no_cluster_mask]
    else:
        y_true = data.y.squeeze().cpu()
        y_pred = out.max(1)[1].cpu()
        adj = data.adj_t
    
    return calc_metrics(y_true, y_pred, adj)

def calc_metrics(y_true, y_pred, adj):
    return (
        utils.pairwise_acc(y_true, y_pred),
        utils.pairwise_f1(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
        completeness_score(y_true, y_pred),
        utils.conductance(adj, y_pred),
        utils.modularity(adj, y_pred)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_dataset', action='store_false')
    parser.add_argument('--upsample_minorities', action='store_true')
    parser.add_argument('--upsample_portion', type=int, default=0)
    parser.add_argument('--exclude_non_clustered', action='store_false')
    parser.add_argument('--use_gnn', action='store_true')
    parser.add_argument('--random_state', type=int, default=1911)

    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--hidden_channels_mlp', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_mlp', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pooling', type=str, default='just_balance')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--log_steps', type=int, default=10)

    args = parser.parse_args("")
    print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.custom_dataset:
        data = torch.load("data/data_with_refs.pt").to(device)

        if args.upsample_minorities:
            adj, data.x, data.y, new_idx = models.src_upsample(
                data.adj_t,
                data.x,
                data.y.squeeze(),
                torch.tensor(range(data.num_nodes), dtype=torch.long),
                portion=args.upsample_portion,
                im_class_num=5,
            )
            data.adj_t = SparseTensor.from_torch_sparse_coo_tensor(adj)
            data.num_nodes = len(new_idx)

            print("Cluster Distribution: ", Counter(data.y.squeeze().tolist()))

        if args.exclude_non_clustered:
            data.y = data.y - 1
            data.no_cluster_mask = ~(data.y == -1).squeeze()

    else:
        dataset = Planetoid("data/Cora", "Cora", transform=T.Compose([T.ToSparseTensor(), T.NormalizeFeatures()]))
        data = dataset[0]

    if not args.use_gnn: # run the spectral clustering baseline.
        if args.exclude_non_clustered:
            n_clusters = data.y.unique().shape[0] - 1
            adj = data.adj_t[data.no_cluster_mask, data.no_cluster_mask]
            coo = adj.coo()
            x = coo_matrix((coo[2], (coo[0], coo[1])), shape=tuple([data.no_cluster_mask.nonzero().shape[0]]*2))
            y_true = data.y[data.no_cluster_mask].squeeze().numpy()
        else:
            n_clusters = data.y.unique().shape[0]
            adj = data.adj_t
            coo = adj.coo()
            x = coo_matrix((coo[2], (coo[0], coo[1])), shape=tuple([data.num_nodes]*2))
            y_true = data.y.squeeze().numpy()

        sc_args = {
            "n_clusters": n_clusters,
            "eigen_solver": 'amg',
            "random_state": args.random_state,
            # "n_init": 10, # for kmeans.
            "affinity": 'rbf',
            "assign_labels": 'cluster_qr',
            "degree": 3, # for rbf.
            # "verbose": True
        }
        seed(args.random_state) # for AMG solver determinism.

        _, y_pred = models.SC(x, sc_args)

        metrics = calc_metrics(y_true, y_pred, adj)
        print(
            f'F1: {metrics[1]:.4f}, NMI: {metrics[2]:.4f}, Completeness: {metrics[3]:.4f}, \
            Conductance: {metrics[4]:.4f}, Modularity: {metrics[5]:.4f}'
        )
    
    else:
        model = models.GNN(
            in_channels=data.x.shape[1], 
            hidden_channels=args.hidden_channels, 
            hidden_channels_mlp=args.hidden_channels_mlp, 
            out_channels=data.y.unique().shape[0], 
            num_layers=args.num_layers, 
            num_layers_mlp=args.num_layers_mlp,
            dropout=args.dropout,
            no_cluster_mask=data.no_cluster_mask if args.exclude_non_clustered else None,
            pooling=args.pooling
        )

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
        )

        for epoch in tqdm(range(args.epochs)):
            train_loss, out = train()
            metrics = test(out, args.exclude_non_clustered)
            if epoch % args.log_steps == 0:
                tqdm.write(
                    f'Epoch {epoch:02d}, Loss: {train_loss:.4f}, \
                    F1: {metrics[1]:.4f}, NMI: {metrics[2]:.4f}, Completeness: {metrics[3]:.4f}, \
                    Conductance: {metrics[4]:.4f}, Modularity: {metrics[5]:.4f}'
                )