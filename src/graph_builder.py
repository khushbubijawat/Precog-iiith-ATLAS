"""
Build Atlas graphs from country and city name lists.

Usage:
    python -m src.graph_builder --outdir outputs
"""
import argparse
import os
import re
from typing import Iterable, List, Tuple

import networkx as nx
import pickle
import pandas as pd
import unidecode


def clean_name(name: str) -> str:
    """Normalize names to ASCII alpha words, title-cased."""
    s = unidecode.unidecode(str(name))
    s = s.strip()
    s = re.sub(r"[^A-Za-z ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.title()


def fetch_with_headers(url: str) -> List[pd.DataFrame]:
    """Fetch HTML with headers to avoid 403 and parse tables."""
    import io
    import requests

    resp = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119 Safari/537.36"
        },
        timeout=15,
    )
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def fetch_countries() -> List[str]:
    """Fetch sovereign state names from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_sovereign_states"
    tables = fetch_with_headers(url)
    # Try columns likely to hold country names.
    candidates = []
    for tbl in tables:
        for col in tbl.columns:
            if isinstance(col, str) and any(k in col.lower() for k in ["state", "country", "name"]):
                candidates.extend([clean_name(x) for x in tbl[col].tolist()])
    names = [x for x in candidates if x]
    if not names:
        # Fallback to first column of first table
        raw = tables[0].iloc[:, 0].tolist()
        names = [clean_name(x) for x in raw if isinstance(x, str)]
        names = [x for x in names if x]
    return sorted(set(names))


def fetch_cities(top_n: int = 500) -> List[str]:
    """Fetch top N cities by population (city proper) from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_cities_proper_by_population"
    tables = fetch_with_headers(url)
    candidates = []
    for tbl in tables:
        for col in tbl.columns:
            if isinstance(col, str) and "city" in col.lower():
                candidates.extend([clean_name(x) for x in tbl[col].tolist()])
    names = [x for x in candidates if x]
    names = names[:top_n]
    if not names:
        raise ValueError("City column not found in Wikipedia tables")
    return sorted(set(names))


def load_csv_names(path: str, columns: List[str]) -> List[str]:
    df = pd.read_csv(path)
    for col in columns:
        if col in df.columns:
            names = [clean_name(x) for x in df[col].tolist() if isinstance(x, str)]
            return sorted(set([x for x in names if x]))
    raise ValueError(f"No expected columns {columns} found in {path}")


def build_graph(names: Iterable[str]) -> nx.DiGraph:
    """Create directed graph with edge A->B if last char of A == first char of B."""
    names = sorted(set([n for n in names if n]))
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for a in names:
        last = a[-1].lower()
        for b in names:
            if a == b:
                continue
            if b[0].lower() == last:
                G.add_edge(a, b)
    return G


def save_graph(G: nx.DiGraph, outdir: str, label: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{label}.gpickle"), "wb") as f:
        pickle.dump(G, f)
    nx.write_edgelist(
        G, os.path.join(outdir, f"{label}_edgelist.txt"), data=False, encoding="utf-8"
    )
    pd.Series(list(G.nodes()), name="name").to_csv(
        os.path.join(outdir, f"{label}_nodes.csv"), index=False
    )
    print(f"[saved] {label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs", help="Directory to save graphs")
    parser.add_argument("--top-cities", type=int, default=500, help="Top N cities")
    parser.add_argument("--countries-csv", help="Optional path to CSV of countries")
    parser.add_argument("--cities-csv", help="Optional path to CSV of cities")
    args = parser.parse_args()

    if args.countries_csv:
        countries = load_csv_names(args.countries_csv, ["name", "country", "state"])
    else:
        countries = fetch_countries()

    if args.cities_csv:
        cities = load_csv_names(args.cities_csv, ["city", "name"])
        cities = cities[: args.top_cities]
    else:
        cities = fetch_cities(args.top_cities)

    G_country = build_graph(countries)
    G_city = build_graph(cities)
    G_both = build_graph(countries + cities)

    save_graph(G_country, args.outdir, "country")
    save_graph(G_city, args.outdir, "city")
    save_graph(G_both, args.outdir, "country_city")


if __name__ == "__main__":
    main()
