import torch
import networkx as nx
import numpy as np
from torch_geometric.datasets import TUDataset

# ----------------------------
# Topological feature functions
# ----------------------------
def compute_topological_features(data, features_list):
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T)

    for i in range(data.num_nodes):
        if i not in G:
            G.add_node(i)

    features = []
    nodes = sorted(G.nodes())

    if 'degree' in features_list:
        features.append(
            torch.tensor([G.degree(n) for n in nodes], dtype=torch.float).unsqueeze(1)
        )

    if 'clustering' in features_list:
        c = nx.clustering(G)
        features.append(
            torch.tensor([c[n] for n in nodes], dtype=torch.float).unsqueeze(1)
        )

    if 'betweenness' in features_list:
        b = nx.betweenness_centrality(G)
        features.append(
            torch.tensor([b[n] for n in nodes], dtype=torch.float).unsqueeze(1)
        )

    if 'pagerank' in features_list:
        p = nx.pagerank(G)
        features.append(
            torch.tensor([p[n] for n in nodes], dtype=torch.float).unsqueeze(1)
        )

    if 'core' in features_list:
        core = nx.core_number(G)
        features.append(
            torch.tensor([core[n] for n in nodes], dtype=torch.float).unsqueeze(1)
        )

    return torch.cat(features, dim=1) if features else None


# ----------------------------
# Dataset wrapper
# ----------------------------
class TopologicalDataset:
    def __init__(self, name="MUTAG", topo_config="none"):
        self.dataset = TUDataset(root="../data/TUDataset", name=name)

        self.feature_map = {
            "none": [],
            "degree": ["degree"],
            "local": ["degree", "clustering"],
            "global": ["betweenness", "pagerank", "core"],
            "all": ["degree", "clustering", "betweenness", "pagerank", "core"],
        }

        self.topo_features = []
        for data in self.dataset:
            self.topo_features.append(
                compute_topological_features(data, self.feature_map[topo_config])
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        topo = self.topo_features[idx]

        if topo is not None:
            data.x = topo if data.x is None else torch.cat([data.x, topo], dim=1)

        return data

    @property
    def num_features(self):
        return self[0].x.shape[1]

    @property
    def num_classes(self):
        return self.dataset.num_classes

