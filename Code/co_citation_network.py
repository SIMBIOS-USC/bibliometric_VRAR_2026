import pandas as pd

archivo = 'vr.csv'

try:
    # Leemos solo la cabecera para que sea instantáneo
    df = pd.read_csv(archivo, nrows=0, engine='python')
    
    print("\n" + "="*50)
    print(f"📂 COLUMNAS ENCONTRADAS EN '{archivo}'")
    print("="*50)
    
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")
        
    print("="*50 + "\n")

except Exception as e:
    print(f"Error al leer el archivo: {e}")

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import networkx.algorithms.community as nx_comm
from adjustText import adjust_text
import re
import matplotlib.patheffects as PathEffects

# --- 1. CONFIGURACIÓN ---
input_file = 'vr.csv'
col_refs = 'References'  # <--- COLUMNA CORRECTA DETECTADA
top_n_authors = 70       # Mantener entre 50-80 para que el gráfico sea legible

# --- 2. FUNCIÓN DE LIMPIEZA DE REFERENCIAS ---
def extract_author_name(ref_string):
    """
    Extrae solo el apellido e inicial del autor de una cadena de referencia larga.
    Ej: "Mayer, R.E., 2010. Multimedia learning..." -> "Mayer, R."
    """
    if not isinstance(ref_string, str) or len(ref_string) < 5:
        return None
        
    ref = ref_string.strip()
    
    # Intento 1: Buscar patrón "Apellido, Inicial." al principio
    # Regex busca: Palabra + Coma + Espacio + Letra + Punto
    match = re.match(r"^([A-Z][a-z\-]+,\s[A-Z]\.)", ref)
    if match:
        return match.group(1)
    
    # Intento 2: Si falla, coger todo lo que hay antes de la primera coma
    # (Funciona para "Mayer R.E., 2005")
    parts = ref.split(',')
    if len(parts) > 1:
        surname = parts[0].strip()
        # Verificar que parece un apellido (sin números)
        if surname and not any(char.isdigit() for char in surname):
            # Intentar sacar la inicial de la segunda parte
            initials = parts[1].strip().split(' ')
            if initials and len(initials[0]) == 1:
                 return f"{surname}, {initials[0]}."
            return surname # Si no hay inicial, devolver solo apellido
            
    return None

# --- 3. PROCESAMIENTO DE DATOS ---
print("1. Extrayendo y limpiando autores citados...")
try:
    df = pd.read_csv(input_file, engine='python')
    df = df.dropna(subset=[col_refs])
    
    all_citations_in_papers = [] # Lista de listas (autores por paper)
    global_author_counts = {}    # Para ver quiénes son los Top

    for refs_row in df[col_refs]:
        # Scopus separa referencias con punto y coma ';'
        refs_list = str(refs_row).split(';')
        authors_in_this_paper = set()
        
        for r in refs_list:
            author_clean = extract_author_name(r)
            if author_clean:
                # Normalizar a Title Case ("mayer, r." -> "Mayer, R.")
                author_clean = author_clean.title()
                authors_in_this_paper.add(author_clean)
                
                # Contar frecuencia global
                global_author_counts[author_clean] = global_author_counts.get(author_clean, 0) + 1
        
        if len(authors_in_this_paper) > 1:
            all_citations_in_papers.append(list(authors_in_this_paper))

except Exception as e:
    print(f"Error leyendo el archivo: {e}")
    exit()

# --- 4. FILTRAR SOLO LOS TOP AUTORES (CRÍTICO) ---
# Ordenamos autores por número de citas recibidas
sorted_authors = sorted(global_author_counts.items(), key=lambda x: x[1], reverse=True)
top_authors_set = set([k for k, v in sorted_authors[:top_n_authors]])

print(f"   -> Autor más citado en bibliografía: {sorted_authors[0][0]} ({sorted_authors[0][1]} citas)")

# --- 5. CONSTRUCCIÓN DE LA RED ---
print("2. Construyendo red de co-citación...")
G = nx.Graph()

for authors_list in all_citations_in_papers:
    # Filtrar: solo nos quedamos con los autores que están en el Top N
    relevant_authors = [a for a in authors_list if a in top_authors_set]
    
    # Crear conexiones (Edges)
    if len(relevant_authors) > 1:
        for u, v in combinations(relevant_authors, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

# Eliminar nodos aislados que hayan quedado tras el filtrado
G.remove_nodes_from(list(nx.isolates(G)))

# --- 6. DETECCIÓN DE COMUNIDADES ---
print("3. Detectando clusters intelectuales...")
communities = list(nx_comm.louvain_communities(G, seed=42, resolution=1.0))
communities.sort(key=len, reverse=True)
palette = sns.color_palette("bright", len(communities))

# --- 7. VISUALIZACIÓN ---
print("4. Generando gráfico...")
plt.figure(figsize=(20, 18), facecolor='white')
ax = plt.gca()
ax.set_axis_off()

# Layout: Spring layout ponderado
pos = nx.spring_layout(G, k=0.5, iterations=150, seed=42, weight='weight')

# A. ARISTAS (Líneas finas y transparentes)
edges = G.edges(data=True)
weights = [d['weight'] for u, v, d in edges]
max_w = max(weights) if weights else 1
widths = [0.1 + (1.5 * w / max_w) for w in weights]
nx.draw_networkx_edges(G, pos, width=widths, alpha=0.1, edge_color='#777777', ax=ax)

# B. NODOS Y ETIQUETAS
texts_to_adjust = []
# Grado Ponderado (Importancia real en la red)
weighted_degree = dict(G.degree(weight='weight'))

for i, community in enumerate(communities):
    nodes = list(community)
    color = palette[i]
    
    # Tamaños basados en importancia
    sizes = [weighted_degree[n] * 3 for n in nodes]
    
    # Dibujar Nodos
    ax.scatter(
        [pos[n][0] for n in nodes],
        [pos[n][1] for n in nodes],
        s=sizes, c=[color], alpha=0.9, edgecolors='white', linewidth=1.5, zorder=2
    )
    
    # Etiquetas (Solo a los nodos más importantes del cluster)
    top_nodes = sorted(nodes, key=lambda x: weighted_degree[x], reverse=True)
    avg_deg = np.mean(list(weighted_degree.values()))
    
    for node in nodes:
        # Etiquetar si es muy importante o Top 3 del cluster
        if weighted_degree[node] > avg_deg or node in top_nodes[:3]:
            x, y = pos[node]
            fontsize = 10 + (10 * weighted_degree[node] / max(weighted_degree.values()))
            
            # Texto
            txt = ax.text(x, y, node, fontsize=fontsize, fontweight='bold', 
                          color='#222222', ha='center', va='center', zorder=10)
            
            # Efecto Halo (Borde blanco para leer mejor)
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white', alpha=0.8)])
            texts_to_adjust.append(txt)

# Leyenda manual
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f"Cluster {i+1}",
                   markerfacecolor=palette[i], markersize=10) for i in range(min(5, len(communities)))]
plt.legend(handles=legend_elements, loc='upper right', title="Schools of Thought", fontsize=12)

# Ajuste final de texto
print("5. Optimizando etiquetas...")
adjust_text(texts_to_adjust, ax=ax, expand_points=(1.2, 1.2), force_text=(0.1, 0.2))

plt.tight_layout()
output_file = 'cocitation_network_final.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ ¡Gráfico guardado!: {output_file}")