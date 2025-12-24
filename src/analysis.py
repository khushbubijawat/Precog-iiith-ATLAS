"""
Graph EDA for Atlas graphs.

Computes up to 6 distinct properties and saves metrics/plots:
- size (nodes/edges), density
- in/out-degree distribution (plots)
- PageRank (top-k)
- Betweenness centrality (top-k)
- Clustering coefficient distribution
- Strongly connected components size summary

Usage:
    python -m src.analysis --graph-path outputs/country.gpickle --label country --outdir outputs
"""
import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pickle
import numpy as np
import seaborn as sns


def load_graph(path: str) -> nx.DiGraph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def degree_stats(G: nx.DiGraph):
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    return indeg, outdeg


def pagerank(G: nx.DiGraph, k: int = 10):
    pr = nx.pagerank(G, alpha=0.85)
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]


def betweenness(G: nx.DiGraph, k: int = 10):
    bt = nx.betweenness_centrality(G, normalized=True)
    return sorted(bt.items(), key=lambda x: x[1], reverse=True)[:k]


def clustering(G: nx.DiGraph):
    # Use directed clustering from NetworkX (transitivity-based).
    clust = nx.clustering(G.to_undirected())
    return clust


def scc_summary(G: nx.DiGraph):
    sizes = [len(c) for c in nx.strongly_connected_components(G)]
    return {
        "count": len(sizes),
        "max": max(sizes),
        "min": min(sizes),
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
    }


def plot_hist(data, title, xlabel, outfile):
    plt.figure(figsize=(6, 4))
    sns.histplot(data, bins=30, kde=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_bar(items, title, xlabel, outfile):
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=vals, y=names, orient="h")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def run_eda(graph_path: str, label: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    G = load_graph(graph_path)

    indeg, outdeg = degree_stats(G)
    pr_top = pagerank(G)
    bt_top = betweenness(G)
    clust = clustering(G)
    scc = scc_summary(G)

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_in_degree": float(np.mean(list(indeg.values()))),
        "avg_out_degree": float(np.mean(list(outdeg.values()))),
        "pagerank_top": pr_top,
        "betweenness_top": bt_top,
        "scc": scc,
        "avg_clustering": float(np.mean(list(clust.values()))),
    }

    with open(os.path.join(outdir, f"{label}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_hist(
        list(indeg.values()),
        f"{label} in-degree",
        "in-degree",
        os.path.join(outdir, f"{label}_indegree.png"),
    )
    plot_hist(
        list(outdeg.values()),
        f"{label} out-degree",
        "out-degree",
        os.path.join(outdir, f"{label}_outdegree.png"),
    )
    plot_hist(
        list(clust.values()),
        f"{label} clustering coeff",
        "clustering",
        os.path.join(outdir, f"{label}_clustering.png"),
    )
    plot_bar(
        pr_top,
        f"{label} top PageRank",
        "score",
        os.path.join(outdir, f"{label}_pagerank_top.png"),
    )
    plot_bar(
        bt_top,
        f"{label} top betweenness",
        "score",
        os.path.join(outdir, f"{label}_betweenness_top.png"),
    )
    print(f"[analysis] saved metrics and plots for {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    run_eda(args.graph_path, args.label, args.outdir)


if __name__ == "__main__":
    main()
