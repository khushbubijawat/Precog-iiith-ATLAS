"""
Community detection on country graph.

Algorithms:
- Louvain (if available via networkx >= 3.2) else skip.
- Girvan–Newman (edge betweenness hierarchical) truncated to first level.

Outputs:
- community assignments CSV
- modularity scores JSON

Usage:
    python -m src.community --graph-path outputs/country.gpickle --outdir outputs
"""
import argparse
import json
import os
from typing import List

import networkx as nx
import pickle
import pandas as pd


def louvain_wrapper(G: nx.Graph):
    if hasattr(nx.algorithms.community, "louvain_communities"):
        comms = list(nx.algorithms.community.louvain_communities(G, seed=42))
        mod = nx.algorithms.community.modularity(G, comms)
        return comms, mod
    return None, None


def girvan_newman_first(G: nx.Graph, top_k: int = 1):
    comp_gen = nx.algorithms.community.girvan_newman(G)
    comms = None
    for i in range(top_k):
        comms = next(comp_gen)
    comms = [set(c) for c in comms]
    mod = nx.algorithms.community.modularity(G, comms)
    return comms, mod


def save_assignments(comms: List[set], outpath: str):
    rows = []
    for idx, comm in enumerate(comms):
        for node in comm:
            rows.append({"node": node, "community": idx})
    pd.DataFrame(rows).to_csv(outpath, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    try:
        G = nx.read_gpickle(args.graph_path)
    except Exception:
        with open(args.graph_path, "rb") as f:
            G = pickle.load(f)
    # Use undirected for community detection.
    H = G.to_undirected()

    results = {}

    comms, mod = louvain_wrapper(H)
    if comms is not None:
        save_assignments(comms, os.path.join(args.outdir, "communities_louvain.csv"))
        results["louvain_modularity"] = mod
        print(f"[community] Louvain communities: {len(comms)}, modularity={mod:.4f}")
    else:
        print("[community] Louvain not available in this NetworkX version")

    comms_gn, mod_gn = girvan_newman_first(H, top_k=1)
    save_assignments(comms_gn, os.path.join(args.outdir, "communities_gn.csv"))
    results["girvan_newman_modularity"] = mod_gn
    print(
        f"[community] Girvan-Newman level-1 communities: {len(comms_gn)}, modularity={mod_gn:.4f}"
    )

    with open(os.path.join(args.outdir, "community_modularity.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
