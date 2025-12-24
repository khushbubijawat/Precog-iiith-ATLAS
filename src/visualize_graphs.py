"""
Generate node-link visualizations for Atlas graphs.

Usage:
    python -m src.visualize_graphs --graph-path outputs/country.gpickle --label country --outdir outputs

Notes:
- Layout: spring layout (<=200 nodes) else kamada-kawai.
- Node size: out-degree (+1) scaled.
- Node color: PageRank score.
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def load_graph(path: str) -> nx.Graph:
    try:
        return nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def visualize(G: nx.Graph, label: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    H = G.to_undirected()
    n = H.number_of_nodes()
    if n <= 200:
        pos = nx.spring_layout(H, seed=42)
    else:
        pos = nx.kamada_kawai_layout(H)

    out_deg = dict(G.out_degree())
    pr = nx.pagerank(G, alpha=0.85)

    sizes = np.array([out_deg.get(n, 0) + 1 for n in G.nodes()])
    sizes = 100 * sizes / sizes.max()
    colors = np.array([pr.get(n, 0) for n in G.nodes()])
    norm = (colors - colors.min()) / (colors.max() - colors.min() + 1e-9)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=norm, cmap="viridis", alpha=0.8, linewidths=0.2)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.2, arrows=False)
    if n <= 50:
        nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title(f"{label} graph (node size = out-degree, color = PageRank)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=colors.min(), vmax=colors.max()))
    sm.set_array([])
    plt.colorbar(sm, shrink=0.7, label="PageRank")
    plt.axis("off")
    plt.tight_layout()
    outfile = os.path.join(outdir, f"{label}_nodelink.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[viz] saved {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    G = load_graph(args.graph_path)
    visualize(G, args.label, args.outdir)


if __name__ == "__main__":
    main()
