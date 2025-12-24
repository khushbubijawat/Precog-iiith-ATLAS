# Atlas Graphs Task (Precog Recruitment)

Project skeleton to build and analyze Atlas-style directed graphs for countries and cities, run community detection, and prototype link prediction (node2vec + simple GNN). All code is pure Python; heavy lifting is deferred to runtime scripts so you can re-run easily in Colab/Kaggle or locally.

## Layout
- `src/graph_builder.py`: fetch/clean name lists, build directed graphs (country, city, combined), save nodes/edges.
- `src/analysis.py`: compute up to 6 graph properties/plots for EDA and strategy notes.
- `src/community.py`: community detection (Louvain if available, Girvan–Newman fallback) + modularity scoring.
- `src/link_prediction.py`: node2vec baseline and a minimal PyG GNN scaffold for link prediction on masked edges.
- `requirements.txt`: minimal deps (install in your runtime).
- `outputs/`: place for generated CSVs/plots/metrics (not tracked).

## Quickstart (local or Colab)
```bash
pip install -r requirements.txt
```

Build graphs (downloads data via pandas read_html):
```bash
python -m src.graph_builder --outdir outputs
```

Run EDA (metrics saved to JSON/CSV, plots to PNG):
```bash
python -m src.analysis --graph-path outputs/country.gpickle --label country --outdir outputs
python -m src.analysis --graph-path outputs/city.gpickle --label city --outdir outputs
python -m src.analysis --graph-path outputs/country_city.gpickle --label country_city --outdir outputs
```

Community detection (country graph):
```bash
python -m src.community --graph-path outputs/country.gpickle --outdir outputs
```

Graph visualizations (node-link plot, size = out-degree, color = PageRank):
```bash
python -m src.visualize_graphs --graph-path outputs/country.gpickle --label country --outdir outputs
python -m src.visualize_graphs --graph-path outputs/city.gpickle --label city --outdir outputs
python -m src.visualize_graphs --graph-path outputs/country_city.gpickle --label country_city --outdir outputs
```

Link prediction (baseline node2vec; PyG scaffold included but optional):
```bash
python -m src.link_prediction --graph-path outputs/country.gpickle --outdir outputs --model node2vec
```
Note: node2vec baseline needs `gensim` + `scikit-learn` (included in requirements). PyG GNN requires a manual install following PyG docs.

## Notes
- Sources: countries from Wikipedia “List of sovereign states”; cities from Wikipedia “List of cities proper by population” (top 500). Adjust URLs if you prefer another authoritative list.
- Cleaning: ASCII only via `unidecode`, strip punctuation, collapse whitespace, title-case.
- Edge rule: directed edge A→B if last letter of A == first letter of B, excluding self-loops.
- Plots: matplotlib/seaborn; adjust styles as you like.
- GNN: Provided as a runnable template if PyTorch Geometric is installed; otherwise skip or comment out.

## To-Do (customize)
- Add your own commentary/insights to the report, plug in plots/metrics.
- Tweak features for link prediction (e.g., first/last letter one-hot, length, country/city flag).
- Export visuals for presentation (network diagrams, degree histograms, centrality rankings).
