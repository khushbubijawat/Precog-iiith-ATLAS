"""
Link prediction for Atlas graphs (unsupervised).

Implemented per task:
- Node2vec embeddings + logistic regression on a proper train/val/test edge split.
- GNN (GCN) in PyTorch Geometric with simple handcrafted node features:
  one-hot first letter, one-hot last letter, and name length.

Usage:
    python -m src.link_prediction --graph-path outputs/country.gpickle --outdir outputs --model node2vec
    python -m src.link_prediction --graph-path outputs/country.gpickle --outdir outputs --model gnn
"""
import argparse
import os
import pickle
import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    LogisticRegression = None


# ---------------------------
# Helpers
# ---------------------------
def to_undirected(G: nx.Graph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from(G.edges())
    return UG


def train_val_test_split_edges(G: nx.Graph, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    edges = list(G.edges())
    random.shuffle(edges)
    n = len(edges)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_edges = edges[:n_val]
    test_edges = edges[n_val : n_val + n_test]
    train_edges = edges[n_val + n_test :]
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    return train_graph, train_edges, val_edges, test_edges


def sample_non_edges(G: nx.Graph, num_samples: int, seed=42) -> List[Tuple[str, str]]:
    random.seed(seed)
    nodes = list(G.nodes())
    existing = set(G.edges()) | set((b, a) for a, b in G.edges())
    neg = []
    while len(neg) < num_samples:
        a, b = random.sample(nodes, 2)
        if (a, b) not in existing:
            neg.append((a, b))
    return neg


def edge_embedding(u_emb, v_emb, mode="hadamard"):
    if mode == "hadamard":
        return u_emb * v_emb
    if mode == "l1":
        return np.abs(u_emb - v_emb)
    if mode == "l2":
        return (u_emb - v_emb) ** 2
    raise ValueError("unknown edge embed mode")


def evaluate_scores(y_true, scores):
    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    return auc, ap


# ---------------------------
# Node2vec baseline
# ---------------------------
def node2vec_features(G: nx.Graph, dimensions=64, walk_length=20, num_walks=200, window=5, seed=42):
    try:
        from gensim.models import Word2Vec
    except ImportError:
        raise ImportError("gensim required for node2vec baseline")

    random.seed(seed)
    np.random.seed(seed)

    def random_walk(start):
        walk = [start]
        for _ in range(walk_length - 1):
            cur = walk[-1]
            nbrs = list(G.neighbors(cur))
            if not nbrs:
                break
            walk.append(random.choice(nbrs))
        return walk

    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for n in nodes:
            walks.append([str(x) for x in random_walk(n)])

    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window,
        min_count=1,
        sg=1,
        workers=4,
        seed=seed,
    )
    embeddings = {str(k): model.wv[str(k)] for k in model.wv.key_to_index}
    return embeddings


def train_node2vec_lp(G: nx.Graph, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    if LogisticRegression is None:
        raise ImportError("scikit-learn required for logistic regression")

    # Edge splits
    train_G, train_edges, val_edges, test_edges = train_val_test_split_edges(G)

    emb = node2vec_features(train_G)

    def build_Xy(pos_edges, neg_edges):
        X, y = [], []
        for u, v in pos_edges:
            if u in emb and v in emb:
                X.append(edge_embedding(emb[u], emb[v]))
                y.append(1)
        for u, v in neg_edges:
            if u in emb and v in emb:
                X.append(edge_embedding(emb[u], emb[v]))
                y.append(0)
        return np.stack(X), np.array(y)

    neg_train = sample_non_edges(train_G, len(train_edges), seed=0)
    neg_val = sample_non_edges(train_G, len(val_edges), seed=1)
    neg_test = sample_non_edges(train_G, len(test_edges), seed=2)

    X_train, y_train = build_Xy(train_edges, neg_train)
    X_val, y_val = build_Xy(val_edges, neg_val)
    X_test, y_test = build_Xy(test_edges, neg_test)

    clf = LogisticRegression(max_iter=400)
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    val_auc, val_ap = evaluate_scores(y_val, val_scores)
    test_auc, test_ap = evaluate_scores(y_test, test_scores)

    with open(os.path.join(outdir, "linkpred_node2vec.txt"), "w", encoding="utf-8") as f:
        f.write(f"VAL AUC: {val_auc:.4f}\nVAL AP: {val_ap:.4f}\n")
        f.write(f"TEST AUC: {test_auc:.4f}\nTEST AP: {test_ap:.4f}\n")
    print(f"[node2vec] val AUC={val_auc:.4f} AP={val_ap:.4f} | test AUC={test_auc:.4f} AP={test_ap:.4f}")


# ---------------------------
# GNN baseline (PyG)
# ---------------------------
def build_node_features(names: List[str]) -> np.ndarray:
    """One-hot first letter (26), one-hot last letter (26), length."""
    feat = []
    for name in names:
        n = name.lower()
        vec = np.zeros(26 + 26 + 1, dtype=np.float32)
        vec[ord(n[0]) - ord("a")] = 1.0
        vec[26 + ord(n[-1]) - ord("a")] = 1.0
        vec[-1] = len(n)
        feat.append(vec)
    return np.vstack(feat)


def train_gnn_lp(G: nx.Graph, outdir: str, epochs=200, lr=0.01, hidden=64, seed=42):
    os.makedirs(outdir, exist_ok=True)
    try:
        import torch
        from torch import nn
        from torch_geometric.utils import from_networkx, negative_sampling
        from torch_geometric.nn import GCNConv
    except ImportError:
        raise ImportError("PyTorch Geometric required for GNN option")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_G, train_edges, val_edges, test_edges = train_val_test_split_edges(G)

    # PyG data
    node_list = list(train_G.nodes())
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(node_list)}
    data = from_networkx(train_G)
    data.x = torch.tensor(build_node_features(node_list), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    class LPModel(nn.Module):
        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        def encode(self, x, edge_index):
            h = self.conv1(x, edge_index).relu()
            h = self.conv2(h, edge_index)
            return h

        def decode(self, z, edge_index):
            src, dst = edge_index
            return (z[src] * z[dst]).sum(dim=1)

    model = LPModel(data.num_features, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    def edges_to_index(edges: List[Tuple[str, str]]):
        src = [name_to_idx[u] for u, v in edges]
        dst = [name_to_idx[v] for u, v in edges]
        return torch.tensor([src, dst], dtype=torch.long, device=device)

    train_pos = edges_to_index(train_edges)
    val_pos = edges_to_index(val_edges)
    test_pos = edges_to_index(test_edges)

    def get_neg(edge_index, num_samples, seed_offset=0):
        torch.manual_seed(seed + seed_offset)
        return negative_sampling(edge_index=edge_index, num_neg_samples=num_samples, force_undirected=True)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pos_out = model.decode(z, train_pos)
        neg_edge = get_neg(train_pos, train_pos.size(1), seed_offset=epoch)
        neg_out = model.decode(z, neg_edge)
        out = torch.cat([pos_out, neg_out], dim=0)
        labels = torch.cat(
            [torch.ones(pos_out.size(0), device=device), torch.zeros(neg_out.size(0), device=device)], dim=0
        )
        loss = bce(out, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"[gnn] epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    @torch.no_grad()
    def eval_edges(pos_edge, neg_edge):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        pos_scores = model.decode(z, pos_edge).cpu().numpy()
        neg_scores = model.decode(z, neg_edge).cpu().numpy()
        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        scores = np.concatenate([pos_scores, neg_scores])
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        return auc, ap

    val_neg = get_neg(val_pos, val_pos.size(1), seed_offset=100)
    test_neg = get_neg(test_pos, test_pos.size(1), seed_offset=200)
    val_auc, val_ap = eval_edges(val_pos, val_neg)
    test_auc, test_ap = eval_edges(test_pos, test_neg)

    with open(os.path.join(outdir, "linkpred_gnn.txt"), "w", encoding="utf-8") as f:
        f.write(f"VAL AUC: {val_auc:.4f}\nVAL AP: {val_ap:.4f}\n")
        f.write(f"TEST AUC: {test_auc:.4f}\nTEST AP: {test_ap:.4f}\n")
    print(f"[gnn] val AUC={val_auc:.4f} AP={val_ap:.4f} | test AUC={test_auc:.4f} AP={test_ap:.4f}")


# ---------------------------
# Entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--model", choices=["node2vec", "gnn"], default="node2vec")
    args = parser.parse_args()

    try:
        G = nx.read_gpickle(args.graph_path)
    except Exception:
        with open(args.graph_path, "rb") as f:
            G = pickle.load(f)
    G = to_undirected(G)

    if args.model == "node2vec":
        train_node2vec_lp(G, args.outdir)
    else:
        train_gnn_lp(G, args.outdir)


if __name__ == "__main__":
    main()
