import argparse
import pandas as pd
from math import isnan
from tqdm import tqdm
from itertools import combinations
from collections import Counter
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch_geometric.utils
from sklearn.feature_extraction.text import CountVectorizer
from networkx import degree_histogram, connected_components
from networkx.algorithms.approximation import average_clustering, diameter
import matplotlib.pyplot as plt

def coauthorship_edges(df):
    edge_list = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        edge_list.extend([edge for edge in combinations(sorted(r.auid), 2)])
    
    counts = Counter(edge_list)

    return list(counts.keys()), list(counts.values())

def remapper(edge_list):
    node_map = dict(zip(range(len(USED_SUP_AUIDS)), USED_SUP_AUIDS))
    inv_node_map = {v : k for k, v in node_map.items()}
    edge_list_remap = [(inv_node_map[auid_1], inv_node_map[auid_2]) for (auid_1, auid_2) in tqdm(edge_list)]
    
    return node_map, inv_node_map, edge_list_remap

def label_vector(df, node_map):
    authors_to_labels = df[["auid", "label"]].explode('auid') \
        .groupby("auid")["label"].apply(list) \
        .apply(lambda x: max(Counter([label for sublist in x for label in sublist]).items(), key=lambda x: (x[1], x[0]))[0]).to_dict()
    
    return [authors_to_labels[k] for k in list(node_map.values())[:len(USED_SUP_AUIDS)]]

def node_feats(df, vectorizer_args, feature):
    features = df[[feature, "auid"]].explode('auid').groupby('auid')[feature].apply(list)
    features = features.loc[node_map.values()]
    features_list = ["; ".join(feats) for feats in features]

    vectorizer = CountVectorizer(**vectorizer_args)
    x = vectorizer.fit_transform(features_list)

    return x, vectorizer

def store_data(x, edge_list, y, edge_attr=None):
    data = Data(
        x=torch.tensor(x.toarray(), dtype=torch.float32),
        edge_index=torch.tensor(edge_list, dtype=torch.long).T, 
        y=torch.tensor(y, dtype=torch.long).unsqueeze(0).T,
    )

    assert not torch_geometric.utils.isolated.contains_isolated_nodes(data.edge_index)

    if edge_attr:
        data.edge_weight = torch.tensor(edge_attr, dtype=torch.float32).sigmoid()
        sparsify = T.ToSparseTensor(attr="edge_weight")
    else:
        sparsify = T.ToSparseTensor()

    transform = T.Compose([
        T.ToUndirected(reduce="max"), 
        sparsify,
        T.NormalizeFeatures(),
    ])
    data = transform(data)

    return data

def data_metrics(data):
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)
    G0 = G.subgraph(sorted(connected_components(G), key=len, reverse=True)[0])

    return (
        G0.number_of_nodes(), 
        G0.number_of_edges(), 
        average_clustering(G, trials=1000, seed=1911), 
        degree_histogram(G), 
        diameter(G0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retracted_papers_path", type=str, default='data/hindawi_retracted.parquet.gzip')
    parser.add_argument("--retracted_papers_refs_path", type=str, default='data/hindawi_retracted_refs.parquet.gzip')
    parser.add_argument("--graph_data_save_path", type=str, default='data/data_with_refs_abs.pt')
    parser.add_argument("--use_edge_weights", action='store_false')
    parser.add_argument("--textual_feature", type=str, default="abstract")
    parser.add_argument("--show_metrics", action='store_true')
    args = parser.parse_args("")

    # for BoW node feature generation later.
    vectorizer_args = {
        "strip_accents": "unicode",
        "lowercase": True,
        "max_df": 0.7,
        "min_df": 5,
        "ngram_range": (1, 1),
        "max_features": None,
        "binary": True,
        "token_pattern": r'(?u)\b[A-Za-z]{3,20}\b',
    }

    df = pd.read_parquet(args.retracted_papers_path)

    # recode the one-hot cluster features into 1 col.
    labels = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        subl = []
        for idx, col in enumerate([col for col in df.columns if "cluster" in col]):
            if not isnan(r[col]):
                subl.append(idx+1)
        if not subl:
            subl.append(0)
        labels.append(subl)
    df["label"] = labels

    SUPERVISED_AUIDS = set(df[["auid"]].explode('auid').auid)

    # extract authors of referenced eIDs that were in the original df. 
    df_refs = pd.read_parquet(args.retracted_papers_refs_path)
    df_refs.dropna(subset=["auid"], inplace=True)
    df_refs["auid"] = [[int(auid) for auid in auids if not isnan(auid)] for auids in df_refs.auid]

    df_refs_retracted_authors_only = df_refs.copy()
    df_refs_retracted_authors_only['auid'] = df_refs['auid'].apply(lambda x: [auid for auid in x if auid in SUPERVISED_AUIDS])
    df_refs_retracted_authors_only = df_refs_retracted_authors_only[df_refs_retracted_authors_only['auid'].apply(len) > 0]

    # prepare df for graph generation.
    input_df = pd.concat([df[["eid", "auid", args.textual_feature]], df_refs_retracted_authors_only[["ref_eid", "auid", args.textual_feature]].rename({"ref_eid":"eid"}, axis=1)])

    assert set(input_df[["auid"]].explode('auid').auid).issubset(SUPERVISED_AUIDS)

    input_df.drop_duplicates(subset=["eid"], inplace=True)
    input_df = input_df[input_df['auid'].apply(len) > 1]
    if args.textual_feature == "title":
        input_df['title'] = [title[0] if type(title) != str else title for title in input_df['title']] # some titles are arrays for some reason. 
    elif args.textual_feature == "abstract":
        input_df['abstract'] = [abs if abs is not None else "" for abs in input_df['abstract']]

    USED_SUP_AUIDS = set(input_df[["auid"]].explode('auid').auid)

    # generate and store graph.
    edge_list, edge_attr = coauthorship_edges(input_df)
    node_map, _, edge_list_remap = remapper(edge_list)
    y = label_vector(df, node_map)

    # features are BoW representation of authors' paper titles. 
    x, _ = node_feats(input_df, vectorizer_args, args.textual_feature)

    edge_attr = edge_attr if args.use_edge_weights else None
    data = store_data(x, edge_list_remap, y, edge_attr)

    print(data.adj_t)
    torch.save(data, args.graph_data_save_path)

    if args.show_metrics:
        # np.mean([i.nonzero().shape[0] for i in data.x])
        n, e, avgcc, degree_freq, dia = data_metrics(data)
        print(f"{n} nodes in LCC, {e} edges in LCC, {avgcc} average clustering coefficient, {dia} diameter of LCC.")

        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size':14})
        plt.loglog(range(len(degree_freq))[1:], degree_freq[1:], alpha=0.7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()
