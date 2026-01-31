"""
Script Definitivo v5: 'Strong Ties' Visualization
Objetivo: Limpieza agresiva para redes fragmentadas (AR) eliminando ruido de peso=1.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import community as community_louvain 
from tabulate import tabulate
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text
import numpy as np
import seaborn as sns 

# --- CONFIGURACIÓN ---
CSV_PATH = "vr.csv"
TOP_N_INITIAL = 300  # Empezamos con muchos para luego filtrar
SEED = 42
FIGSIZE = (22, 16)
DPI = 300

# Palabras a ignorar
STOP_WORDS = [
    'virtual reality', 'augmented reality', 'vr', 'ar', 'mixed reality',
    'simulation', 'training', 'education', 'learning', 'system', 
    'performance', 'study', 'results', 'based', 'using', 'virtual', 
    'environment', 'environments', 'human', 'user', 'skills', 'assessment',
    'technology', 'application', 'students', 'reality', 'analysis', 'paper', 'review',
    'data', 'research', 'development', 'design', 'effectiveness', 'impact', 'intervention',
    'meta-analysis', 'systematic review'
]

_ws_re = re.compile(r"\s+")
_first_author_re = re.compile(r"^\s*([^,]{1,80})\s*,\s*([^,]{1,80})")

def clean_spaces(s: str) -> str:
    return _ws_re.sub(" ", s).strip()

# --- 1. PARSERS ---
def parse_first_author(ref: str) -> str | None:
    if not isinstance(ref, str): return None
    ref = clean_spaces(ref).strip('"\'')
    if not ref or ref.lower() in {"null", "undefined", "nan"}: return None
    m = _first_author_re.match(ref)
    if not m: return None
    surname = clean_spaces(m.group(1))
    given = clean_spaces(m.group(2))
    if any(ch.isdigit() for ch in surname) or any(ch.isdigit() for ch in given): return None
    parts = re.split(r"[\s\-]+", given)
    initials = [p.strip(".")[0].upper() + "." for p in parts if p.strip(".")]
    if not initials: return None
    return f"{surname}, {''.join(initials)}"

def split_references(refs_field: str) -> list[str]:
    if not isinstance(refs_field, str): return []
    parts = [clean_spaces(x) for x in refs_field.split(";")]
    return [p for p in parts if p]

def classify_paper(row):
    text = f"{str(row.get('Title',''))} {str(row.get('Abstract',''))} {str(row.get('Author Keywords',''))}".lower()
    exclusions = ['agent-mediated', 'multi-agent', 'auction']
    if any(ex in text for ex in exclusions): return None
    if 'augmented reality' in text or ' ar ' in text: return 'Augmented Reality'
    if 'virtual reality' in text or ' vr ' in text: return 'Virtual Reality'
    return None

# --- 2. MOTOR DE PROFILING ---
def analyze_cluster_dna(df, partition, G):
    clusters = sorted(list(set(partition.values())))
    cluster_kws = {c: Counter() for c in clusters}
    cluster_jrn = {c: Counter() for c in clusters}
    
    for _, row in df.iterrows():
        refs = split_references(row.get("References", ""))
        votes = {c: 0 for c in clusters}
        has_votes = False
        
        for r in refs:
            auth = parse_first_author(r)
            if auth and auth in G.nodes():
                cid = partition[auth]
                votes[cid] += 1
                has_votes = True
        
        if not has_votes: continue
        winner = max(votes, key=votes.get)
        if votes[winner] < 1: continue 
        
        raw_kws = str(row.get('Author Keywords', ''))
        journal = str(row.get('Source title', ''))
        
        if raw_kws.lower() != 'nan':
            kws = [k.strip().lower() for k in raw_kws.split(';')]
            clean_kws = [k for k in kws if len(k)>2 and k not in STOP_WORDS and not any(sw in k for sw in ['virtual reality', 'augmented'])]
            cluster_kws[winner].update(clean_kws)
            
        if len(journal) > 3 and journal.lower() != 'nan':
            cluster_jrn[winner].update([journal])
            
    return cluster_kws, cluster_jrn

# --- 3. ANÁLISIS PRINCIPAL ---
def run_cluster_analysis(df, cat_name):
    # A. Construir Grafo Inicial
    author_counts = Counter()
    paper_authors = []
    
    for refs in df.get("References", []):
        refs_list = split_references(refs)
        authors = sorted(set(parse_first_author(r) for r in refs_list if parse_first_author(r)))
        if len(authors) >= 2:
            paper_authors.append(authors)
            author_counts.update(authors)

    # Coger un pool inicial grande
    top_authors = {a for a, _ in author_counts.most_common(TOP_N_INITIAL)}
    G = nx.Graph()
    G.add_nodes_from(top_authors)
    edge_weights = Counter()
    
    for authors in paper_authors:
        authors = [a for a in authors if a in top_authors]
        if len(authors) < 2: continue
        for u, v in combinations(authors, 2):
            edge_weights[tuple(sorted((u, v)))] += 1

    for (u, v), w in edge_weights.items(): G.add_edge(u, v, weight=w)

    # === [LIMPIEZA AGRESIVA PARA AR] ===
    print(f"   > Nodos brutos: {len(G)} | Aristas: {G.number_of_edges()}")
    
    # 1. FILTRO DE PESO: Eliminar conexiones débiles (Ruido)
    # Si es AR, somos muy estrictos: elimina aristas con peso < 2
    # Si es VR, podemos ser más suaves o iguales.
    min_weight = 2 
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_weight]
    G.remove_edges_from(edges_to_remove)
    print(f"   > Tras borrar aristas débiles (<{min_weight}): {len(G)} nodos | {G.number_of_edges()} aristas")

    # 2. FILTRO DE GRADO: Quedarse con la Élite
    # Nos quedamos con los 100 nodos más conectados tras borrar las aristas débiles
    if len(G) > 100:
        deg = dict(G.degree(weight='weight'))
        top_connected = sorted(deg, key=deg.get, reverse=True)[:100]
        G = G.subgraph(top_connected).copy()
        print(f"   > Tras filtrar Top 100 Élite: {len(G)} nodos")

    # 3. K-CORE FINAL: Asegurar estructura sólida
    G = nx.k_core(G, k=2)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # 4. ISLA GIGANTE
    if len(G) > 0 and not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if len(G) < 10: 
        print("⚠️ La red quedó vacía tras limpiar.")
        return

    # B. DETECCIÓN DE COMUNIDADES
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0)
    bet_cent = nx.betweenness_centrality(G, weight='weight')
    clus_kws, clus_jrn = analyze_cluster_dna(df, partition, G)
    
    # C. INFORME
    unique_clusters = sorted(list(set(partition.values())))
    valid_clusters = [c for c in unique_clusters if list(partition.values()).count(c) >= 3]
    
    print(f"\n{'='*80}")
    print(f"🧬 CLEAN REPORT: {cat_name.upper()}")
    print(f"{'='*80}")
    
    table_data = []
    valid_clusters.sort(key=lambda c: list(partition.values()).count(c), reverse=True)
    
    for cid in valid_clusters:
        nodes = [n for n in partition if partition[n] == cid]
        top_auth = sorted(nodes, key=lambda x: bet_cent[x], reverse=True)[:3]
        auth_str = "\n".join([f"{n}" for n in top_auth])
        
        top_kws = [k[0] for k in clus_kws[cid].most_common(5)]
        kw_str = ", ".join(top_kws)
        top_jrn = [j[0] for j in clus_jrn[cid].most_common(2)]
        jrn_str = "\n".join(top_jrn)
        
        kws_blob = " ".join(top_kws)
        jrn_blob = " ".join(top_jrn).lower()
        theme_guess = "TECH / GENERAL"
        
        if any(x in kws_blob for x in ['surg', 'laparo', 'medic', 'clinic']) or 'surg' in jrn_blob: theme_guess = "SURGERY / MEDICINE"
        elif any(x in kws_blob for x in ['rehab', 'stroke', 'motor', 'patient']) or 'rehab' in jrn_blob: theme_guess = "REHABILITATION"
        elif any(x in kws_blob for x in ['anxiety', 'psych', 'emot', 'presenc', 'phobia']) or 'behav' in jrn_blob: theme_guess = "PSYCHOLOGY / MENTAL"
        elif any(x in kws_blob for x in ['educat', 'teach', 'learn', 'classroom', 'instruct', 'stem']) or 'educat' in jrn_blob: theme_guess = "EDUCATION"
        elif any(x in kws_blob for x in ['tourism', 'heritage', 'museum']): theme_guess = "TOURISM / HERITAGE"
        elif any(x in kws_blob for x in ['construct', 'safety', 'engin', 'bim', 'industry']): theme_guess = "ENGINEERING / INDUSTRY"
        elif any(x in kws_blob for x in ['retail', 'consum', 'brand', 'market']): theme_guess = "MARKETING / RETAIL"
        elif any(x in kws_blob for x in ['game', 'play', 'interact']): theme_guess = "GAMING / HCI"
        
        table_data.append([f"C{cid} ({theme_guess})", len(nodes), auth_str, kw_str, jrn_str])

    print(tabulate(table_data, headers=["ID (Type)", "Size", "Leaders", "Topics", "Journals"], tablefmt="grid"))

    # D. VISUALIZACIÓN "CLEAN"
    plt.figure(figsize=(20, 16), facecolor='white')
    ax = plt.gca()
    
    # Layout con alta repulsión y 'weight' para agrupar lo fuerte
    pos = nx.spring_layout(G, k=0.4, iterations=1000, seed=SEED, weight='weight')
    
    colors = sns.color_palette("bright", len(unique_clusters))
    
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='#888888', width=0.8)
    
    for i, cid in enumerate(valid_clusters):
        nodes = [n for n in partition if partition[n] == cid]
        sizes = [bet_cent[n] * 7000 + 200 for n in nodes]
        color_idx = unique_clusters.index(cid)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors[color_idx]], 
                               node_size=sizes, alpha=0.95, edgecolors='white', linewidths=2.0)
        
    texts = []
    for cid in valid_clusters:
        nodes = [n for n in partition if partition[n] == cid]
        # Etiquetar top 30%
        limit = max(3, int(len(nodes)*0.3))
        top = sorted(nodes, key=lambda n: bet_cent[n], reverse=True)[:limit]
        
        for n in top:
            x, y = pos[n]
            t = ax.text(x, y, n, fontsize=10, fontweight='bold', color='#111111', ha='center', va='center')
            t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.85)])
            texts.append(t)
            
    adjust_text(texts, expand_points=(1.4, 1.4), arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))
    
    plt.title(f"Co-Citation Structure: {cat_name}\n(Strong Ties Only: Weight >= 2)", fontsize=22, fontweight='bold')
    plt.axis('off')
    
    fname = f"Clean_Network_{cat_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   > Gráfico guardado: {fname}")

def main():
    print("Cargando datos...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['Category'] = df.apply(classify_paper, axis=1)
    
    for cat in ['Virtual Reality', 'Augmented Reality']:
        df_sub = df[df['Category'] == cat]
        if len(df_sub) > 20:
            print(f"\n--- PROCESANDO {cat} ---")
            run_cluster_analysis(df_sub, cat)

if __name__ == "__main__":
    main()