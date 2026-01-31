import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter

# --- 1. CONFIGURACIÓN Y CLASIFICACIÓN ---
archivo_entrada = 'vr.csv'
RANGO_INI, RANGO_FIN = 1995, 2025
# Forzamos emergentes clave
emergentes_interes = ['metaverse', 'ai', 'artificial intelligence', 'digital twin', 'xr', 'blockchain', 'chatgpt']
stopwords = ['virtual reality', 'augmented reality', 'mixed reality', 'vr', 'ar', 'mr', 'extended reality', 'system', 'study', 'research', 'article', 'paper', 'analysis']

def classify_technology(row):
    text = f"{str(row.get('Title',''))} {str(row.get('Author Keywords',''))}".lower()
    if any(t in text for t in ['mixed reality', 'extended reality', ' xr ', ' mr ']): return 'Mixed Reality'
    if any(t in text for t in ['augmented reality', ' ar ']): return 'Augmented Reality'
    if any(t in text for t in ['virtual reality', ' vr ']): return 'Virtual Reality'
    return 'Other'

# --- 2. CARGA DE DATOS ---
df = pd.read_csv(archivo_entrada, low_memory=False)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[(df['Year'] >= RANGO_INI) & (df['Year'] <= RANGO_FIN)].dropna(subset=['Author Keywords'])
df['Category'] = df.apply(classify_technology, axis=1)

def get_clean_kws(kw_str):
    return [k.strip().lower() for k in str(kw_str).split(';') if len(k.strip()) > 1 and k.strip().lower() not in stopwords]

df['clean_kws'] = df['Author Keywords'].apply(get_clean_kws)

# --- 3. FUNCIÓN DE RED VISUALMENTE MEJORADA ---
def plot_professional_network(df_sub, title, filename):
    if df_sub.empty: return
    
    # SELECCIÓN ESTRATÉGICA DE NODOS
    kws_pioneras = [kw for sublist in df_sub[df_sub['Year'] < 2010]['clean_kws'] for kw in sublist]
    kws_todas = [kw for sublist in df_sub['clean_kws'] for kw in sublist]
    
    # Selección balanceada: Pioneros + Volumen + Emergentes
    pioneer_top = [k for k, v in Counter(kws_pioneras).most_common(15)]
    volume_top = [k for k, v in Counter(kws_todas).most_common(30)]
    presentes_emergentes = [e for e in emergentes_interes if any(e in k for k in kws_todas)]
    
    top_kws = list(set(pioneer_top + volume_top + presentes_emergentes))

    # Construcción del Grafo
    G = nx.Graph()
    kw_data = {}
    for _, row in df_sub.iterrows():
        kws = [k for k in set(row['clean_kws']) if k in top_kws]
        for kw in kws:
            if kw not in kw_data: kw_data[kw] = {'years': [], 'count': 0}
            kw_data[kw]['years'].append(row['Year'])
            kw_data[kw]['count'] += 1
            
        for p1, p2 in combinations(sorted(kws), 2):
            if G.has_edge(p1, p2): G[p1][p2]['weight'] += 1
            else: G.add_edge(p1, p2, weight=1)

    # Filtrado de aristas débiles para limpiar el ruido
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2])

    # Añadir nodos con atributos
    for kw in top_kws:
        if kw in kw_data and kw_data[kw]['count'] > 0:
            # Usamos el año mínimo para el color (origen)
            G.add_node(kw, first_year=np.min(kw_data[kw]['years']), count=kw_data[kw]['count'])

    # Eliminar nodos que quedaron aislados tras el filtrado de aristas (opcional, pero limpia)
    G.remove_nodes_from(list(nx.isolates(G)))

    # --- VISUALIZACIÓN PROFESIONAL ---
    plt.figure(figsize=(18, 14)) # Lienzo más grande
    
    # LAYOUT MEJORADO: Kamada-Kawai suele separar mejor los clústeres densos
    try:
        pos = nx.kamada_kawai_layout(G, scale=2)
    except:
        # Fallback si Kamada-Kawai falla por nodos desconectados
        pos = nx.spring_layout(G, k=2.5, iterations=200, seed=42)

    # TAMAÑO LOGARÍTMICO: Evita que los nodos gigantes tapen todo
    # Base de 300 + escala logarítmica del conteo
    node_sizes = [300 + np.log(G.nodes[n]['count'] + 1) * 800 for n in G.nodes]
    
    # Dibujar Aristas (muy sutiles)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='#999999', width=0.8)
    
    # Dibujar Nodos
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_size=node_sizes,
                                   node_color=[G.nodes[n]['first_year'] for n in G.nodes],
                                   cmap='plasma', alpha=0.9, edgecolors='white', linewidths=1.5,
                                   vmin=RANGO_INI, vmax=RANGO_FIN)
    
    # Etiquetas con fondo mejorado (más pequeño, más limpio)
    for node, (x, y) in pos.items():
        plt.text(x, y, s=node.upper().replace(' ', '\n'), # Salto de línea para nombres largos
                 fontsize=8, fontweight='bold', color='#333333', ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    # Barra de color y títulos
    cbar = plt.colorbar(nodes, shrink=0.6, aspect=25, pad=0.02)
    cbar.set_label('First Year of Appearance (Origin)', fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.title(title, fontsize=22, fontweight='black', color='#222222', pad=30)
    plt.axis('off')
    
    # Márgenes para asegurar que no se corte nada
    plt.margins(0.15)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

# --- 4. EJECUCIÓN ---
for cat in ['Virtual Reality', 'Augmented Reality', 'Mixed Reality']:
    print(f"Generando red profesional para: {cat}...")
    plot_professional_network(df[df['Category'] == cat], 
                              f"{cat}: Evolutionary Knowledge Map (1995-2025)", 
                              f"net_{cat[:3].lower()}_PRO.png")

print("✅ Redes profesionales generadas. Revisa los archivos *_PRO.png*")