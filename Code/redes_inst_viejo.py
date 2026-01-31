import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import community as community_louvain 
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
import re
from collections import defaultdict
from tabulate import tabulate

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
archivo = 'vr.csv'

def classify_paper(row):
    text = f"{str(row.get('Title',''))} {str(row.get('Abstract',''))} {str(row.get('Author Keywords',''))}".lower()
    exclusions = ['agent-mediated', 'multi-agent', 'auction', 'negotiation', 'mixed integer']
    hard_xr = ['hololens', 'magic leap', 'oculus', 'htc vive', 'hmd', 'quest']
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr): return None
    
    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', ' xr ', ' mr ']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses']
    vr_terms = ['virtual reality', ' vr ', 'immersive', 'hmd']
    
    if any(t in text for t in mr_terms) or (any(t in text for t in ar_terms) and any(t in text for t in vr_terms)): 
        return 'Mixed_Extended_Reality'
    elif any(t in text for t in ar_terms): return 'Augmented_Reality'
    elif any(t in text for t in vr_terms): return 'Virtual_Reality'
    return None

# =========================================================
# 2. LIMPIEZA DE DATOS (EL "IMÁN" DE INSTITUCIONES)
# =========================================================
def get_clean_institution(aff_string):
    if pd.isna(aff_string) or len(str(aff_string)) < 5: return None
    aff_lower = str(aff_string).lower()
    
    # DICCIONARIO DE CONSOLIDACIÓN (Para arreglar Toronto, Harvard, etc.)
    mappings = {
        # Canadá
        'toronto': 'U. Toronto (CAN)', 'sick children': 'U. Toronto (CAN)', 'sickkids': 'U. Toronto (CAN)',
        'sunnybrook': 'U. Toronto (CAN)', 'uhn': 'U. Toronto (CAN)', 'university health network': 'U. Toronto (CAN)',
        # USA
        'harvard': 'Harvard U. (USA)', 'massachusetts general': 'Harvard U. (USA)', 'brigham': 'Harvard U. (USA)',
        'stanford': 'Stanford U. (USA)', 'mit': 'MIT (USA)', 
        'la jolla': 'UC San Diego (USA)', 'ucsd': 'UC San Diego (USA)',
        'washington': 'U. Washington (USA)', 'johns hopkins': 'Johns Hopkins (USA)',
        # UK/Europa/Asia
        'ucl': 'UCL (UK)', 'college london': 'UCL (UK)', 'imperial college': 'Imperial College (UK)',
        'oxford': 'Oxford U. (UK)', 'cambridge': 'Cambridge U. (UK)',
        'copenhagen': 'U. Copenhagen (DNK)', 'rigshospitalet': 'Rigshospitalet (DNK)',
        'barcelona': 'U. Barcelona (ESP)', 'munich': 'TU Munich (DEU)',
        'tsinghua': 'Tsinghua U. (CHN)', 'nus': 'NUS (SGP)', 'nanyang': 'Nanyang Tech. (SGP)',
        'tokyo': 'U. Tokyo (JPN)', 'seoul national': 'SNU (KOR)',
        'delft': 'TU Delft (NLD)', 'leuven': 'KU Leuven (BEL)'
    }
    
    for k, v in mappings.items():
        if k in aff_lower: return v

    # Limpieza algorítmica si no está en el mapa
    parts = [p.strip() for p in str(aff_string).split(',')]
    target_keywords = ['univ', 'instit', 'polytech', 'college', 'academy', 'technol']
    generic_keywords = ['dept', 'school', 'faculty', 'centre', 'center', 'unit', 'hospital']
    
    for part in parts:
        part_low = part.lower()
        if any(t in part_low for t in target_keywords) and not any(g in part_low for g in generic_keywords):
            clean = re.sub(r'\b(the|of|and)\b', '', part, flags=re.IGNORECASE).strip()
            return clean.replace('University', 'U.').replace('Institute', 'Inst.')
    return None

def get_country(aff_string):
    if pd.isna(aff_string): return None
    parts = str(aff_string).split(',')
    country = parts[-1].strip()
    # Normalización básica
    country = country.replace('USA', 'United States').replace('UK', 'United Kingdom').replace('Peoples R China', 'China')
    if len(country) < 3 or any(c.isdigit() for c in country): return None
    return country

# =========================================================
# 3. ANÁLISIS DE COMUNIDADES (TOP 3 BETWEENNESS)
# =========================================================
def analyze_community_structure(G, partition, centralities, cat_name, node_type):
    """
    Analiza cada clúster y extrae el Top 3 de nodos por Betweenness Centrality.
    Esto permite ver QUIÉN lidera cada grupo (Quantitative Difference).
    """
    communities = defaultdict(list)
    for node, com_id in partition.items():
        communities[com_id].append(node)
    
    # Ordenar comunidades por tamaño
    sorted_coms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\n{'='*80}")
    print(f"🏛️  ANÁLISIS DE CLÚSTERES: {cat_name} ({node_type})")
    print(f"{'='*80}")
    
    table_data = []
    
    # Analizamos los Top 5 clústeres más grandes
    for i, (com_id, nodes) in enumerate(sorted_coms[:5]):
        # Obtener métricas de los nodos de este clúster
        node_stats = []
        for n in nodes:
            node_stats.append({
                'node': n,
                'bt': centralities['betweenness'][n],
                'dg': centralities['degree'][n]
            })
        
        # Ordenar por Betweenness (Influencia de puente)
        top_3 = sorted(node_stats, key=lambda x: x['bt'], reverse=True)[:3]
        
        # Formatear para la tabla
        top_str = "\n".join([f"{x['node'][:30]} (B:{x['bt']:.3f})" for x in top_3])
        size = len(nodes)
        
        # Intentar adivinar la región del clúster (solo para instituciones)
        region_guess = "Global"
        if node_type == 'Institution':
            text_blob = " ".join(nodes).lower()
            if 'usa' in text_blob and 'china' not in text_blob: region_guess = "North America"
            elif 'uk' in text_blob or 'esp' in text_blob or 'deu' in text_blob: region_guess = "Europe"
            elif 'chn' in text_blob or 'sgp' in text_blob: region_guess = "Asia"
            
        table_data.append([f"Cluster {i+1}", size, region_guess, top_str])

    print(tabulate(table_data, headers=["Cluster", "Size", "Region (Est.)", "Top 3 Brokers (Betweenness)"], tablefmt="grid"))
    return sorted_coms # Devolvemos para usar en el plot

# =========================================================
# 4. VISUALIZACIÓN Y MÉTRICAS GLOBALES
# =========================================================
def run_full_analysis(df_subset, cat_name, node_type):
    print(f"\nProcesando red: {cat_name} - {node_type}...")
    
    # 1. Construcción
    G = nx.Graph()
    doc_count = {}
    
    for aff_row in df_subset['Affiliations']:
        if pd.isna(aff_row): continue
        raw_list = str(aff_row).split(';')
        clean_nodes = set()
        for raw in raw_list:
            clean = get_clean_institution(raw) if node_type == 'Institution' else get_country(raw)
            if clean: clean_nodes.add(clean)
        
        nodes_list = list(clean_nodes)
        for n in nodes_list: doc_count[n] = doc_count.get(n, 0) + 1
        
        if len(nodes_list) > 1:
            for u, v in combinations(nodes_list, 2):
                if G.has_edge(u, v): G[u][v]['weight'] += 1
                else: G.add_edge(u, v, weight=1)

    if len(G) < 5: return None

    # 2. Filtros (Limpiar ruido)
    if node_type == 'Institution':
        # Eliminar enlaces débiles (solo 1 colaboración) para limpiar visualización
        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2])
        k_val = 2 # K-Core: nodos con al menos 2 vecinos
    else:
        k_val = 2
        
    G = nx.k_core(G, k=k_val)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G) < 5: return None

    # 3. Métricas Globales
    density = nx.density(G)
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    modularity = community_louvain.modularity(partition, G)
    
    # Centralidades
    cent_deg = nx.degree_centrality(G)
    cent_bet = nx.betweenness_centrality(G, weight='weight')
    
    centralities = {'degree': cent_deg, 'betweenness': cent_bet}
    
    # 4. ANÁLISIS CUANTITATIVO DE CLÚSTERES (Lo que pediste)
    analyze_community_structure(G, partition, centralities, cat_name, node_type)

    # 5. Visualización
    plt.figure(figsize=(20, 16), facecolor='white')
    
    # Layout de "Islas" (k alto = repulsión fuerte)
    pos = nx.spring_layout(G, k=1.8/np.sqrt(len(G)), iterations=200, seed=42, weight='weight')
    
    # Dibujar Aristas
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='#999999', width=[(w/max(weights))*3 for w in weights])
    
    # Dibujar Nodos
    num_coms = len(set(partition.values()))
    colors = sns.color_palette("bright", num_coms)
    
    # Etiquetar solo los importantes (Top 2 por clúster + Top 5 Globales)
    labels_to_show = set(sorted(G.nodes(), key=lambda n: cent_bet[n], reverse=True)[:5])
    
    for com_id in set(partition.values()):
        nodes = [n for n in partition if partition[n] == com_id]
        sizes = [doc_count.get(n, 1) * 100 for n in nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors[com_id]], 
                               node_size=sizes, alpha=0.9, edgecolors='black', linewidths=1)
        
        # Añadir líder del clúster a etiquetas
        leader = max(nodes, key=lambda n: cent_bet[n])
        labels_to_show.add(leader)

    texts = []
    for node in labels_to_show:
        x, y = pos[node]
        t = plt.text(x, y, node, fontsize=11, fontweight='bold', ha='center', va='center', color='#222222')
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.9)])
        texts.append(t)
        
    adjust_text(texts, expand_points=(1.5, 1.5))
    
    # Caja de resumen
    info = f"Nodes: {len(G)} | Edges: {G.number_of_edges()}\nDensity: {density:.4f} | Modularity: {modularity:.3f}"
    plt.gca().text(0.02, 0.98, info, transform=plt.gca().transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f"Network Structure: {cat_name} ({node_type})", fontsize=20, fontweight='bold')
    plt.axis('off')
    plt.savefig(f"FinalNet_{cat_name}_{node_type}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'nodes': len(G), 'density': density, 'modularity': modularity}

# =========================================================
# 5. EJECUCIÓN
# =========================================================
print("Iniciando análisis completo...")
df = pd.read_csv(archivo, low_memory=False)
col_auth = 'Author full names' if 'Author full names' in df.columns else 'Authors'
mask_plaza = (df['Title'].astype(str).str.contains("Competing agents", case=False)) | (df[col_auth].astype(str).str.contains("Plaza, E", case=False))
df = df[~mask_plaza].copy()
df['Category'] = df.apply(classify_paper, axis=1)
df = df.dropna(subset=['Category', 'Affiliations'])

summary = []
for cat in ['Virtual_Reality', 'Augmented_Reality', 'Mixed_Extended_Reality']:
    for n_type in ['Country', 'Institution']:
        res = run_full_analysis(df[df['Category'] == cat], cat, n_type)
        if res:
            res['category'] = cat
            res['type'] = n_type
            summary.append(res)

print("\n" + "="*60)
print("RESUMEN GLOBAL DE MÉTRICAS")
print("="*60)
print(tabulate(summary, headers="keys", tablefmt='pretty'))
print("\n✅ Hecho. Revisa las tablas detalladas arriba para ver los Top 3 por clúster.")