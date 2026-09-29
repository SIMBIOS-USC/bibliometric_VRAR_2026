"""
src/a8_network_graphs.py
========================
Analysis Module 8: Network Analysis & Thematic Clusters
-------------------------------------------------------
Generates high-quality network graphs matching the visual style
requested (VOSviewer / Gephi style).

Includes:
A) Keyword Co-occurrence Network (nodes sized by freq, edges by co-occurrence)
B) Thematic Clusters (modularity class coloring)
C) Author Co-citation/Collaboration Network

Resolves Reviewer 2's request for robustness by making the thresholding
explicit and reproducible.

Outputs:
  - fig_A8_network_keywords.png
  - fig_A8_network_clusters.png
  - fig_A8_network_authors.png
"""

import sys
import itertools
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain # python-louvain
from adjustText import adjust_text
from sklearn.metrics import adjusted_rand_score
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib import patheffects
from utils.plot_style import apply_global_style, save_figure
from utils.institutions import institution_items

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Colors for clusters
CLUSTER_COLORS = [to_hex(colormaps["tab20"](i)) for i in range(20)]


def build_cooccurrence_graph(items_list: list[list[str]], min_freq=5, min_edge=2,
                             max_nodes: int | None = None) -> nx.Graph:
    """Builds a co-occurrence network from a list of item lists."""
    # Count frequencies
    all_items = list(itertools.chain(*items_list))
    counts = Counter(all_items)
    
    # Filter nodes by min_freq
    valid_items = {k for k, v in counts.items() if v >= min_freq}
    if max_nodes is not None:
        valid_items = {k for k, _ in counts.most_common(max_nodes)
                       if k in valid_items}
    
    # Build edges
    edges = Counter()
    for row in items_list:
        valid_row = [x for x in row if x in valid_items]
        # Create all pairs
        for i in range(len(valid_row)):
            for j in range(i + 1, len(valid_row)):
                pair = tuple(sorted([valid_row[i], valid_row[j]]))
                edges[pair] += 1
                
    # Build graph
    G = nx.Graph()
    for (u, v), weight in edges.items():
        if weight >= min_edge:
            G.add_edge(u, v, weight=weight)
            
    # Add node attributes
    for node in G.nodes():
        G.nodes[node]['freq'] = counts[node]
        
    # Remove isolates
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def _compact_layout(G: nx.Graph) -> dict:
    """Lay out each component locally, then pack components into a compact field."""
    if not G:
        return {}
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest = max(map(len, components))
    positions = {}
    satellites = components[1:]
    for index, nodes in enumerate(components):
        sub = G.subgraph(nodes).copy()
        if len(nodes) == 1:
            local = {next(iter(nodes)): np.zeros(2)}
        else:
            local = nx.spring_layout(sub, k=0.55, iterations=500,
                                     seed=42 + index, weight=None, scale=1.0)
        coords = np.asarray(list(local.values()))
        extent = max(float(np.abs(coords).max()), 1e-8)
        local = {node: np.asarray(xy) / extent for node, xy in local.items()}
        if index == 0:
            center = np.zeros(2)
            scale = 1.0
        else:
            count = len(satellites)
            angle = 2 * np.pi * (index - 1) / max(count, 1) - np.pi / 2
            center = 1.25 * np.array([np.cos(angle), 0.78 * np.sin(angle)])
            scale = 0.16 + 0.34 * np.sqrt(len(nodes) / largest)
        positions.update({node: center + scale * xy for node, xy in local.items()})
    return positions


def plot_network(G: nx.Graph, title: str, filename: Path, is_cluster=False,
                 label_limit: int = 16):
    """Draw a static, paper-style force-directed map sized by node degree."""
    apply_global_style(font_size=10)
    fig, ax = plt.subplots(figsize=(12.4, 9.2), facecolor="white")
    ax.set_facecolor("white")
    partition = (community_louvain.best_partition(G, random_state=42, resolution=1.0)
                 if G.number_of_edges() else {n: 0 for n in G})
    pos = _compact_layout(G)
    edges = list(G.edges())
    weights = [G[u][v].get("weight", 1) for u, v in edges]
    max_w = max(weights, default=1)
    edge_widths = [0.18 + 0.72 * np.sqrt(w / max_w) for w in weights]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=edge_widths,
                           edge_color="#687386", alpha=0.20)
    ordered = list(G.nodes())
    degrees = dict(G.degree())
    max_degree = max(degrees.values(), default=1)
    sizes = [28 + 560 * (degrees[n] / max_degree) ** 0.68 for n in ordered]
    colors = [CLUSTER_COLORS[partition.get(n, 0) % len(CLUSTER_COLORS)] for n in ordered]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ordered, node_size=sizes,
                           node_color=colors, edgecolors="#263244", linewidths=0.65,
                           alpha=0.96)
    important = {n for n, _ in sorted(degrees.items(),
                                      key=lambda z: (z[1], G.nodes[z[0]].get("freq", 0)),
                                      reverse=True)[:label_limit]}
    labels = []
    for node in important:
        x, y = pos[node]
        labels.append(ax.text(x, y, str(node), fontsize=8.2, fontweight="semibold",
                              ha="center", va="center", color="#172033",
                              path_effects=[patheffects.withStroke(linewidth=2.8, foreground="white")]))
    if labels:
        adjust_text(labels, x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], ax=ax,
                    force_text=(0.65, 0.85), force_points=(0.18, 0.22),
                    expand=(1.16, 1.30),
                    arrowprops={"arrowstyle": "-", "color": "#526174", "lw": .65, "alpha": .8},
                    iter_lim=260)
    cluster_count = len(set(partition.values())) if partition else 0
    ax.set_title(title, fontsize=17, fontweight="bold", pad=18, color="#172033")
    ax.text(.5, 1.005, f"{G.number_of_nodes():,} nodes  ·  {G.number_of_edges():,} links  ·  {cluster_count} communities",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color="#475569")
    ax.margins(.16)
    ax.axis("off")
    fig.tight_layout(pad=1.4)
    save_figure(fig, str(filename))


_NON_AUTHOR_REFERENCE_STARTS = {
    "journal", "international", "proceedings", "annual", "education",
    "review", "medicine", "medical", "research", "science", "technology",
    "computer", "virtual", "reality", "learning", "conference", "american",
    "european", "world", "springer", "elsevier", "ieee", "acm",
}


def _cited_first_author(reference: str) -> str | None:
    """Return surname + initials from a Scopus cited-reference string.

    This uses first-author co-citation, the conventional author-level projection
    of cited references. Initials reduce (but cannot eliminate) surname collisions.
    """
    parts = [part.strip() for part in str(reference).split(",")]
    if len(parts) < 3:
        return None
    surname, given = parts[0], parts[1]
    if (not surname or len(surname) > 45 or surname.lower() in _NON_AUTHOR_REFERENCE_STARTS
            or not re.search(r"[A-Za-zÀ-ž]", surname) or len(given) > 50):
        return None
    tokens = re.findall(r"[A-Za-zÀ-ž]+", given)
    if not tokens or len(tokens) > 5:
        return None
    initials = "".join(token[0].upper() for token in tokens)
    if len(initials) > 5:
        return None
    label = f"{surname}, {'.'.join(initials)}."
    return re.sub(r"\s+", " ", label)


def _author_cocitation_inputs(df: pd.DataFrame) -> tuple[list[list[str]], Counter]:
    documents: list[list[str]] = []
    cited_counts: Counter = Counter()
    for value in df.get("References", pd.Series("", index=df.index)).fillna(""):
        authors = sorted({author for ref in str(value).split(";")
                          if (author := _cited_first_author(ref.strip()))})
        documents.append(authors)
        cited_counts.update(authors)
    return documents, cited_counts


def _author_cocitation_graph(documents: list[list[str]], cited_counts: Counter,
                             min_citations: int, min_cocitations: int) -> nx.Graph:
    eligible = {author for author, count in cited_counts.items() if count >= min_citations}
    edge_counts: Counter = Counter()
    for authors in documents:
        present = sorted(set(authors) & eligible)
        edge_counts.update(itertools.combinations(present, 2))
    graph = nx.Graph()
    graph.add_nodes_from((author, {"freq": cited_counts[author]}) for author in eligible)
    for (author_a, author_b), weight in edge_counts.items():
        if weight >= min_cocitations:
            graph.add_edge(author_a, author_b, weight=weight)
    return graph


def _plot_author_cocitation(graph: nx.Graph, partition: dict[str, int],
                            output: Path, label_limit: int = 48) -> None:
    # Isolates are reported in the node table; only authors with at least one
    # co-citation link appear in the network map.
    linked_graph = graph.subgraph([n for n in graph if graph.degree(n) > 0]).copy()
    plot_network(linked_graph, "Author Co-citation Network", output,
                 is_cluster=True, label_limit=min(label_limit, 12))


def _run_author_cocitation(df: pd.DataFrame, results_dir: Path) -> None:
    documents, cited_counts = _author_cocitation_inputs(df)
    base_citations, base_links = 15, 2
    graph = _author_cocitation_graph(documents, cited_counts, base_citations, base_links)
    linked_graph = graph.subgraph([n for n in graph if graph.degree(n) > 0]).copy()
    partition = (community_louvain.best_partition(linked_graph, random_state=42, resolution=1.0)
                 if linked_graph.number_of_edges() else {})
    cluster_sizes = Counter(partition.values())
    node_rows = [{"Author": author, "Citing documents": count,
                  "Louvain cluster": partition.get(author, pd.NA),
                  "Connected at plotted threshold": author in graph and graph.degree(author) > 0}
                 for author, count in cited_counts.most_common() if count >= base_citations]
    pd.DataFrame(node_rows).to_csv(results_dir / "table_A8_author_cocitation_nodes.csv", index=False)
    edge_rows = [{"Author 1": a, "Author 2": b, "Co-citing documents": data["weight"],
                  "Cluster": partition.get(a, pd.NA)}
                 for a, b, data in graph.edges(data=True)]
    pd.DataFrame(edge_rows).sort_values("Co-citing documents", ascending=False).to_csv(
        results_dir / "table_A8_author_cocitation_edges.csv", index=False)
    _plot_author_cocitation(graph, partition,
                            results_dir / "fig_A8_author_cocitation.png")

    configs = [(10, 2), (15, 2), (20, 2), (30, 3), (15, 3), (15, 5)]
    partitions = {}
    rows = []
    for min_citations, min_links in configs:
        g = _author_cocitation_graph(documents, cited_counts, min_citations, min_links)
        linked = g.subgraph([n for n in g if g.degree(n) > 0]).copy()
        part = (community_louvain.best_partition(linked, random_state=42, resolution=1.0)
                if linked.number_of_edges() else {})
        partitions[(min_citations, min_links)] = part
        rows.append({"Minimum citing-document frequency": min_citations,
                     "Minimum co-citing documents": min_links,
                     "Authors (including isolates)": g.number_of_nodes(),
                     "Linked authors": linked.number_of_nodes(), "Links": g.number_of_edges(),
                     "Clusters": len(set(part.values())) if part else 0,
                     "Modularity": community_louvain.modularity(part, linked)
                     if part and linked.number_of_edges() else np.nan,
                     "ARI vs baseline": np.nan})
    baseline = partitions[(base_citations, base_links)]
    for row in rows:
        current = partitions[(row["Minimum citing-document frequency"],
                              row["Minimum co-citing documents"])]
        shared = sorted(set(baseline) & set(current))
        if len(shared) >= 2:
            row["ARI vs baseline"] = round(adjusted_rand_score(
                [baseline[n] for n in shared], [current[n] for n in shared]), 4)
    pd.DataFrame(rows).to_csv(results_dir / "table_A8_author_cocitation_sensitivity.csv", index=False)
    linked_count = sum(graph.degree(node) > 0 for node in graph)
    print(f"  Author co-citation: {graph.number_of_nodes():,} eligible authors "
          f"({linked_count:,} linked in map), {graph.number_of_edges():,} links, "
          f"{len(cluster_sizes)} clusters; "
          "full author/edge tables and threshold sensitivity saved.")

def run(df_classified: pd.DataFrame, results_dir: Path) -> None:
    print("\n[A8] Network Graphs (Keywords & Authors)")
    print("-" * 50)
    
    # We use df_classified to be methodologically sound (unlike the original mixed table)
    # 1. Prepare keywords
    papers_kws = []
    for kw_str in df_classified['Author Keywords'].dropna():
        kws = [k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 2]
        if kws:
            papers_kws.append(kws)
            
    # Build Network A (Co-occurrence)
    print("  Building Keyword Co-occurrence Network...")
    G_kws = build_cooccurrence_graph(papers_kws, min_freq=25, min_edge=8,
                                     max_nodes=70)
    G_kws_core = G_kws
    plot_network(G_kws_core, "A) Keyword Co-occurrence Network",
                 results_dir / "fig_A8_network_keywords.png", is_cluster=False,
                 label_limit=18)
    
    # Build Network B (Clusters)
    print("  Building Thematic Clusters Network...")
    plot_network(G_kws_core, "B) Thematic Clusters",
                 results_dir / "fig_A8_network_clusters.png", is_cluster=True,
                 label_limit=18)
    
    # 2. Prepare Authors: this is co-authorship, distinct from cited-author co-citation.
    print("  Building Co-authorship Network...")
    papers_authors = []
    for auth_str in df_classified['Authors'].dropna():
        # Scopus `Authors` is semicolon-delimited; commas belong inside names.
        auths = [a.strip() for a in str(auth_str).split(';') if len(a.strip()) > 2]
        if auths:
            papers_authors.append(auths)
            
    G_auth = build_cooccurrence_graph(papers_authors, min_freq=8, min_edge=3,
                                      max_nodes=70)
    
    # For authors, we use the cluster style (like image C)
    plot_network(G_auth, "Author Collaboration Network", results_dir / "fig_A8_network_authors.png", is_cluster=True)

    # 2b. True author co-citation network from Scopus cited References.
    print("  Building Author Co-citation Network from cited References...")
    _run_author_cocitation(df_classified, results_dir)

    # 3. Institution collaboration network, using the same canonical corpus
    print("  Building Institution Collaboration Network...")
    papers_institutions = []
    for aff in df_classified['Affiliations'].dropna():
        items = institution_items(aff)
        if items:
            papers_institutions.append(items)
    # A single shared document is sufficient evidence of an institutional link;
    # retaining these links produces the full collaboration structure instead
    # of fragmenting the network at an arbitrary frequency cutoff of two.
    G_inst = build_cooccurrence_graph(papers_institutions, min_freq=8, min_edge=1,
                                      max_nodes=60)
    G_inst_core = G_inst
    plot_network(G_inst_core, "Institution Collaboration Network",
                 results_dir / "fig_A8_network_institutions.png",
                 is_cluster=True, label_limit=16)

    # Explicitly report threshold sensitivity for the two co-occurrence graphs.
    sensitivity_rows = []
    configurations = {
        "Keywords": (papers_kws, [(15, 5), (20, 6), (25, 8), (30, 10), (40, 12)]),
        "Authors": (papers_authors, [(5, 2), (8, 3), (12, 4), (15, 5)]),
    }
    reference_partitions = {}
    for graph_name, (paper_items, thresholds) in configurations.items():
        for min_freq, min_edge in thresholds:
            graph = build_cooccurrence_graph(paper_items, min_freq=min_freq,
                                             min_edge=min_edge, max_nodes=70)
            if graph.number_of_nodes() and not nx.is_connected(graph):
                largest_nodes = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_nodes).copy()
            partition = (community_louvain.best_partition(graph, random_state=42, resolution=1.0)
                         if graph.number_of_nodes() else {})
            key = (graph_name, min_freq, min_edge)
            reference_partitions[key] = partition
            sensitivity_rows.append({
                "Network": graph_name,
                "Minimum document frequency": min_freq,
                "Minimum co-occurrence": min_edge,
                "Nodes (largest component)": graph.number_of_nodes(),
                "Edges (largest component)": graph.number_of_edges(),
                "Density": nx.density(graph) if graph.number_of_nodes() > 1 else 0,
                "Communities": len(set(partition.values())) if partition else 0,
                "Modularity": community_louvain.modularity(partition, graph) if partition and graph.number_of_edges() else np.nan,
                "ARI vs baseline": np.nan,
            })
    for row in sensitivity_rows:
        name = row["Network"]
        baseline = (name, 25, 8) if name == "Keywords" else (name, 8, 3)
        current = (name, row["Minimum document frequency"], row["Minimum co-occurrence"])
        p0, p1 = reference_partitions[baseline], reference_partitions[current]
        shared = sorted(set(p0) & set(p1))
        if len(shared) >= 2:
            row["ARI vs baseline"] = round(adjusted_rand_score(
                [p0[n] for n in shared], [p1[n] for n in shared]), 4)
    pd.DataFrame(sensitivity_rows).to_csv(results_dir / "table_A8_network_sensitivity.csv", index=False)
    print(f"  → Saved: {results_dir / 'table_A8_network_sensitivity.csv'}")
    
    print("  → Saved A8 network graphs.")
    print("[A8] Done.")
