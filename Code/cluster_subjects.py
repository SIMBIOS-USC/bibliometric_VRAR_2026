import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import networkx.algorithms.community as nx_comm
from adjustText import adjust_text

# --- 1. CONFIGURATION ---
input_file = 'vr.csv'
keyword_column = 'Author Keywords'
top_n_words = 75  

# Stopwords: Common terms removed to declutter the graph and separate clusters
ignore_words = [
    'virtual reality', 'vr', 'augmented reality', 'mixed reality', 'simulation',
    'education', 'training', 'learning', 'students', 'paper', 'study', 'results',
    'review', 'system', 'technology', 'analysis', 'teaching', 'application',
    'development', 'design', 'implementation', 'effectiveness'
]

# --- 2. DATA PROCESSING ---
print("Loading and cleaning data...")
try:
    df = pd.read_csv(input_file, engine='python')
    # Clean text: lowercase and remove empty strings
    df[keyword_column] = df[keyword_column].astype(str).str.lower().replace('nan', '')
    # Filter out short keywords that are often noise
    df = df[df[keyword_column].str.len() > 3]
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Vectorization: Convert keywords into a matrix
vectorizer = CountVectorizer(
    tokenizer=lambda x: [w.strip() for w in x.split(';') if w.strip()],
    token_pattern=None,
    max_features=top_n_words,
    stop_words=ignore_words
)
X = vectorizer.fit_transform(df[keyword_column])
terms = vectorizer.get_feature_names_out()

# Build the Co-occurrence Matrix
# (Multiply matrix by its transpose to see which words appear together)
Xc = (X.T * X)
Xc.setdiag(0) # Remove self-loops

# Create the Graph
G = nx.from_scipy_sparse_array(Xc)
mapping = {i: terms[i] for i in range(len(terms))}
G = nx.relabel_nodes(G, mapping)

# --- 3. CLUSTERING & SORTING ---
print("Detecting clusters and sorting by size...")

# A. Detect communities using the Louvain algorithm
communities_generator = nx_comm.louvain_communities(G, weight='weight', resolution=1.0, seed=42)
communities_list = list(communities_generator)

# B. SORTING: Reorder clusters from Largest (Most keywords) to Smallest
# This ensures "Area 1" is the most dominant theme.
communities_list.sort(key=len, reverse=True)

# --- 4. THEMATIC NAMING (SMART TITLES) ---
def get_cluster_title(keywords_list):
    """Assigns a semantic title based on the keywords present in the cluster."""
    text = " ".join(keywords_list)
    
    if any(x in text for x in ['surgery', 'laparoscopy', 'anatomy', 'dental', 'medical']):
        return "Clinical & Surgical Simulation"
    
    elif any(x in text for x in ['nursing', 'clinical', 'patient', 'skills', 'health']):
        return "Nursing Education & Clinical Skills"
    
    elif any(x in text for x in ['gamification', 'game', 'serious games', 'engagement', 'motivation']):
        return "Gamification & Game-Based Learning"
        
    elif any(x in text for x in ['higher education', 'pedagogy', 'classroom', 'stem', 'university']):
        return "Higher Education & Curricular Integration"
        
    elif any(x in text for x in ['autism', 'rehabilitation', 'anxiety', 'special needs', 'therapy']):
        return "Therapeutic & Special Needs Interventions"
        
    elif any(x in text for x in ['distance', 'collaborative', 'virtual worlds', 'metaverse', 'online']):
        return "Virtual Environments & Distance Learning"
    
    elif any(x in text for x in ['haptic', 'interface', 'software', 'unity', 'usability', 'interaction']):
        return "Haptics, Usability & System Design"
        
    else:
        # Fallback title using the top 2 keywords
        return f"{keywords_list[0].title()} & {keywords_list[1].title()}"

# --- 5. VISUALIZATION ---
print("Generating high-resolution visualization...")

# Layout: Physics simulation to position nodes
pos = nx.spring_layout(G, k=0.3/np.sqrt(len(G.nodes())), iterations=100, seed=42, weight='weight')

plt.figure(figsize=(18, 16), facecolor='white')
ax = plt.gca()
ax.set_axis_off()

# Color Palette (one color per cluster)
palette = sns.color_palette("bright", len(communities_list))

# Draw Edges (Background, faint)
edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 1]
nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, width=0.5, alpha=0.15, edge_color='#999999', ax=ax)

texts_to_adjust = []
legend_handles = []

# Loop through Sorted Clusters
for i, community in enumerate(communities_list):
    nodes_cluster = list(community)
    
    # Sort keywords inside the cluster by importance (Degree Centrality)
    nodes_cluster.sort(key=lambda x: G.degree[x], reverse=True)
    
    # Get Title
    cluster_title = get_cluster_title(nodes_cluster)
    final_label = f"Area {i+1}: {cluster_title}" # E.g., "Area 1: Clinical Simulation"
    
    color = palette[i]
    
    # Node Sizes based on frequency
    degrees = [G.degree[n] for n in nodes_cluster]
    node_sizes = [150 + (deg * 40) for deg in degrees]
    
    # Draw Nodes
    ax.scatter(
        [pos[n][0] for n in nodes_cluster],
        [pos[n][1] for n in nodes_cluster],
        s=node_sizes, c=[color], alpha=0.85, edgecolors='white', linewidth=1.5, zorder=2
    )
    
    # Prepare Labels (Only label connected nodes to avoid clutter)
    for node in nodes_cluster:
        if G.degree[node] > 1:
            x, y = pos[node]
            font_size = 10 + (2 if G.degree[node] > np.mean(degrees) else 0)
            texts_to_adjust.append(
                ax.text(x, y, node.capitalize(), fontsize=font_size, fontweight='bold', 
                        color='#333333', ha='center', va='center')
            )
    
    # Add to Legend
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=final_label,
                          markerfacecolor=color, markersize=12))

# Adjust Text Positions (Prevent Overlap)
print("Optimizing label placement (this might take a moment)...")
adjust_text(texts_to_adjust, ax=ax, expand_points=(1.3, 1.3), 
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5))

# Final Legend Configuration
plt.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=12, 
           title="Research Clusters (Ordered by Size)", title_fontsize=14, 
           facecolor='white', framealpha=0.95, edgecolor='#cccccc')

# Save Output
output_filename = 'vr_network_clusters_english.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! Image saved as: {output_filename}")