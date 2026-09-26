"""
Matplotlib Tripartite Graph Visualizer.
Renders 3-column layout visualization (S1 ↔ S2 ↔ S3) for business entity resolution graphs
using NetworkX and Matplotlib. Visualizes nodes, candidate edges, and matched links.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
# Use Agg non-interactive backend by default if no display attached
if "DISPLAY" not in os.environ and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

from src.utils import setup_logger

logger = setup_logger("graph_visualizer")


def build_networkx_tripartite_graph(
    s1_dict: Dict[str, Dict[str, str]],
    target_dict: Dict[str, Dict[str, str]],
    candidate_pairs: Dict[str, List[str]],
    matching_results: Dict[str, List[str]],
    probabilities_map: Optional[Dict[Tuple[str, str], float]] = None
) -> nx.Graph:
    """
    Construct a NetworkX Graph with 3 tripartite partitions:
      - Partition 0: Source 1 (Reference entities)
      - Partition 1: Source 2 (Target entities)
      - Partition 2: Source 3 (Target entities)
    """
    G = nx.Graph()

    # Add S1 nodes
    for s1_id, data in s1_dict.items():
        name = data.get("name") or data.get("business_name") or s1_id
        G.add_node(s1_id, label=f"{s1_id}\n{name[:18]}", layer=0, source="S1", name=name)

    # Add Target nodes (S2 and S3)
    for t_id, data in target_dict.items():
        name = data.get("name") or data.get("business_name") or t_id
        layer = 1 if t_id.startswith("S2") else 2
        source = "S2" if layer == 1 else "S3"
        G.add_node(t_id, label=f"{t_id}\n{name[:18]}", layer=layer, source=source, name=name)

    # Add edges
    prob_map = probabilities_map or {}
    matched_set: Set[Tuple[str, str]] = set()

    for s1_id, matches in matching_results.items():
        for m_id in matches:
            if s1_id in G and m_id in G:
                matched_set.add((s1_id, m_id))
                matched_set.add((m_id, s1_id))
                prob = prob_map.get((s1_id, m_id), 0.95)
                G.add_edge(s1_id, m_id, status="matched", weight=prob, color="#009E73", style="solid", width=2.5)

    # Add candidate non-matched edges
    for s1_id, cands in candidate_pairs.items():
        for c_id in cands:
            if (s1_id, c_id) not in matched_set and s1_id in G and c_id in G:
                prob = prob_map.get((s1_id, c_id), 0.35)
                G.add_edge(s1_id, c_id, status="candidate", weight=prob, color="#D55E00", style="dashed", width=1.0)

    return G


def visualize_tripartite_graph(
    G: nx.Graph,
    output_path: Optional[Path] = None,
    title: str = "Tripartite Business Entity Resolution Graph",
    show: bool = False,
    max_nodes_per_layer: int = 15
) -> Path:
    """
    Render 3-column tripartite graph visualization using NetworkX and Matplotlib.

    Color scheme:
      - S1 Nodes: Deep Blue (#0072B2)
      - S2 Nodes: Emerald Green (#009E73)
      - S3 Nodes: Violet Purple (#CC79A7)
      - Matmatched Edges: Solid Green
      - Candidate Edges: Dashed Amber/Red
    """
    out_file = output_path or Path("output/tripartite_graph.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Subgraph capping for clean readable visual
    s1_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 0][:max_nodes_per_layer]
    s2_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 1][:max_nodes_per_layer]
    s3_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 2][:max_nodes_per_layer]

    sub_nodes = set(s1_nodes + s2_nodes + s3_nodes)
    subG = G.subgraph(sub_nodes).copy()

    pos = {}
    # Layout coordinates: S1 at x=0.0, S2 at x=0.5, S3 at x=1.0
    for idx, node in enumerate(s1_nodes):
        y_val = 1.0 - (idx + 0.5) / max(1, len(s1_nodes))
        pos[node] = (0.0, y_val)

    for idx, node in enumerate(s2_nodes):
        y_val = 1.0 - (idx + 0.5) / max(1, len(s2_nodes))
        pos[node] = (0.5, y_val)

    for idx, node in enumerate(s3_nodes):
        y_val = 1.0 - (idx + 0.5) / max(1, len(s3_nodes))
        pos[node] = (1.0, y_val)

    plt.figure(figsize=(14, 9), dpi=300)
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    # Draw Nodes by layer
    nx.draw_networkx_nodes(subG, pos, nodelist=s1_nodes, node_color="#0072B2", node_size=1200, label="Source 1 (Reference)")
    nx.draw_networkx_nodes(subG, pos, nodelist=s2_nodes, node_color="#009E73", node_size=1200, label="Source 2 (Target)")
    nx.draw_networkx_nodes(subG, pos, nodelist=s3_nodes, node_color="#CC79A7", node_size=1200, label="Source 3 (Target)")

    # Draw Node Labels
    labels = {n: d.get("label", n) for n, d in subG.nodes(data=True)}
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8, font_family="sans-serif", font_color="black")

    # Separate edges by status
    matched_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("status") == "matched"]
    candidate_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("status") == "candidate"]

    # Draw Edges
    if candidate_edges:
        nx.draw_networkx_edges(
            subG, pos, edgelist=candidate_edges, edge_color="#D55E00",
            style="dashed", width=1.0, alpha=0.5, label="Candidate Pair"
        )

    if matched_edges:
        nx.draw_networkx_edges(
            subG, pos, edgelist=matched_edges, edge_color="#009E73",
            style="solid", width=2.5, alpha=0.9, label="Resolved Match"
        )

    # Draw Edge Labels (Probability weights)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in subG.edges(data=True) if "weight" in d}
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=7, font_color="#333333")

    # Annotate column titles
    plt.text(0.0, 1.05, "Source 1 (Reference)", fontsize=12, fontweight="bold", ha="center", color="#0072B2")
    plt.text(0.5, 1.05, "Source 2 (Noisy Target)", fontsize=12, fontweight="bold", ha="center", color="#009E73")
    plt.text(1.0, 1.05, "Source 3 (Noisy Target)", fontsize=12, fontweight="bold", ha="center", color="#CC79A7")

    plt.axis("off")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True)
    plt.tight_layout()

    plt.savefig(out_file, bbox_inches="tight")
    logger.info("Saved Matplotlib Tripartite Graph visualization to %s", out_file)

    if show:
        plt.show()

    plt.close()
    return out_file
