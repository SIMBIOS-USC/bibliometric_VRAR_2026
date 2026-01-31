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
# 1. CONFIGURATION AND CLASSIFICATION
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
# 2. ENHANCED CLEANING ENGINES (PUBLICATION QUALITY)
# =========================================================
def get_clean_institution_aggressive(aff_string):
    if pd.isna(aff_string) or len(str(aff_string)) < 5: return None
    aff_lower = str(aff_string).lower()
    
    # 1. DIRECT MAPPING OF MAJOR INSTITUTIONS (Expanded)
    mappings = {
        'toronto': 'U. Toronto', 'harvard': 'Harvard U.', 'stanford': 'Stanford U.', 
        'mit': 'MIT', 'ucl': 'UCL', 'imperial college': 'Imperial College',
        'oxford': 'Oxford U.', 'cambridge': 'Cambridge U.', 'eth zurich': 'ETH Zurich',
        'copenhagen': 'U. Copenhagen', 'barcelona': 'U. Barcelona', 'munich': 'TU Munich',
        'tsinghua': 'Tsinghua U.', 'nus': 'NUS', 'nanyang': 'Nanyang Tech.',
        'tokyo': 'U. Tokyo', 'seoul national': 'Seoul Nat. U.', 'johns hopkins': 'Johns Hopkins',
        'mayo clinic': 'Mayo Clinic', 'rigshospitalet': 'Rigshospitalet',
        'kaist': 'KAIST', 'postech': 'POSTECH', 'nthu': 'Nat. Tsing Hua U.',
        'nctu': 'Nat. Chiao Tung U.', 'ntu taiwan': 'Nat. Taiwan U.',
        'delft': 'TU Delft', 'eindhoven': 'TU Eindhoven', 'carnegie mellon': 'Carnegie Mellon',
        'georgia tech': 'Georgia Tech', 'purdue': 'Purdue U.', 'ucla': 'UCLA',
        'uc berkeley': 'UC Berkeley', 'usc': 'USC', 'washington': 'U. Washington'
    }
    for k, v in mappings.items():
        if k in aff_lower: return v

    # 2. INTELLIGENT SEARCH
    parts = [p.strip() for p in str(aff_string).split(',')]
    uni_candidates = []
    
    target_keywords = ['univ', 'instit', 'polytech', 'college', 'academy', 'technol']
    generic_keywords = ['dept', 'school', 'faculty', 'centre', 'center', 'unit', 'hospital', 'service', 'division', 'laboratory']
    
    for part in parts:
        part_low = part.lower()
        if any(t in part_low for t in target_keywords) and not any(g in part_low for g in generic_keywords):
            clean_cand = re.sub(r'\b(the|of|and)\b', '', part, flags=re.IGNORECASE).strip()
            uni_candidates.append(clean_cand)

    if uni_candidates:
        best = max(uni_candidates, key=len)
        return best.replace('University', 'U.').replace('Institute', 'Inst.').replace('Technology', 'Tech.')
        
    return None

def get_country(aff_string):
    """Enhanced country extraction with proper UK, Taiwan and regional detection"""
    if pd.isna(aff_string): return None
    
    aff_str = str(aff_string)
    aff_lower = aff_str.lower()
    
    # PRIORITY MAPPING for common issues
    priority_mappings = {
        # UK and its regions - must be checked before splitting
        'england': 'United Kingdom',
        'scotland': 'United Kingdom', 
        'wales': 'United Kingdom',
        'northern ireland': 'United Kingdom',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        # China and Taiwan - critical distinction
        'taiwan': 'Taiwan',
        'republic of china': 'Taiwan',  # Official name of Taiwan
        'peoples r china': 'China',
        'pr china': 'China',
        'p r china': 'China',
        "people's republic of china": 'China',
        'hong kong': 'Hong Kong',
        'macau': 'Macau',
        # Korea
        'south korea': 'South Korea',
        'republic of korea': 'South Korea',
        'korea': 'South Korea',  # Most papers refer to South Korea
        # USA variants
        'usa': 'United States',
        'u s a': 'United States',
        'united states of america': 'United States',
        # Netherlands
        'holland': 'Netherlands',
        'the netherlands': 'Netherlands',
        # Czech Republic
        'czech republic': 'Czech Republic',
        'czechia': 'Czech Republic',
        # Others
        'russian federation': 'Russia',
        'republic of ireland': 'Ireland',
        'south africa': 'South Africa',
        'new zealand': 'New Zealand',
        'saudi arabia': 'Saudi Arabia'
    }
    
    # Check priority mappings first (BEFORE splitting)
    for key, value in priority_mappings.items():
        if key in aff_lower:
            return value
    
    # Split by comma and take last part (only if priority didn't match)
    parts = aff_str.split(',')
    if not parts: return None
    
    country = parts[-1].strip()
    
    # Clean and validate
    country = country.strip()
    if len(country) < 3: return None
    if any(c.isdigit() for c in country): return None  # Filter postal codes
    if country.lower() in ['email', 'fax', 'tel', 'phone', 'www', 'http']: return None
    
    # Additional normalization for edge cases
    country_map = {
        'USA': 'United States',
        'UK': 'United Kingdom',
        'Peoples R China': 'China',
        'PR China': 'China',
        'P R China': 'China',
        'Republic of Korea': 'South Korea',
        'Russian Federation': 'Russia',
        'The Netherlands': 'Netherlands'
    }
    
    return country_map.get(country, country)

# =========================================================
# 3. ADVANCED NETWORK METRICS CALCULATION
# =========================================================
def calculate_network_metrics(G):
    """Calculate complete set of network metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Connected components
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['num_components'] = nx.number_connected_components(G)
    
    # Centralities
    if G.number_of_nodes() > 0:
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G, weight='weight')
        closeness_cent = nx.closeness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        
        metrics['avg_degree_centrality'] = np.mean(list(degree_cent.values()))
        metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_cent.values()))
        metrics['avg_closeness_centrality'] = np.mean(list(closeness_cent.values()))
        metrics['avg_eigenvector_centrality'] = np.mean(list(eigenvector_cent.values()))
        
        # Top nodes per metric
        metrics['top_degree'] = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_betweenness'] = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['top_eigenvector'] = sorted(eigenvector_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Save all centralities for visualization
        metrics['all_centralities'] = {
            'degree': degree_cent,
            'betweenness': betweenness_cent,
            'closeness': closeness_cent,
            'eigenvector': eigenvector_cent
        }
    
    # Clustering
    if G.number_of_nodes() > 0:
        metrics['avg_clustering'] = nx.average_clustering(G, weight='weight')
        metrics['transitivity'] = nx.transitivity(G)
    
    # Communities
    if G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        metrics['num_communities'] = len(set(partition.values()))
        metrics['modularity'] = community_louvain.modularity(partition, G, weight='weight')
        metrics['partition'] = partition
        
        # Community sizes
        com_sizes = defaultdict(int)
        for node, com in partition.items():
            com_sizes[com] += 1
        metrics['community_sizes'] = dict(com_sizes)
    
    # Average shortest path (only for main component)
    if G.number_of_nodes() > 1:
        try:
            largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            if largest_cc.number_of_nodes() > 1:
                metrics['avg_shortest_path'] = nx.average_shortest_path_length(largest_cc, weight='weight')
                metrics['diameter'] = nx.diameter(largest_cc)
        except:
            metrics['avg_shortest_path'] = None
            metrics['diameter'] = None
    
    return metrics

def identify_community_hubs(G, partition, centralities):
    """
    Identify the most important nodes (hubs) in each community
    Returns detailed information about key nodes per cluster
    """
    communities = {}
    
    # Group nodes by community
    for node, com_id in partition.items():
        if com_id not in communities:
            communities[com_id] = []
        communities[com_id].append(node)
    
    # Analyze each community
    hub_analysis = {}
    
    for com_id, nodes in communities.items():
        if not nodes:
            continue
            
        # Get centralities for nodes in this community
        com_data = []
        for node in nodes:
            com_data.append({
                'node': node,
                'degree': centralities['degree'].get(node, 0),
                'betweenness': centralities['betweenness'].get(node, 0),
                'eigenvector': centralities['eigenvector'].get(node, 0),
                'closeness': centralities['closeness'].get(node, 0)
            })
        
        # Sort by combined score (weighted average of centralities)
        # Betweenness is most important for identifying hubs (bridges between communities)
        for item in com_data:
            item['hub_score'] = (
                0.3 * item['degree'] +
                0.4 * item['betweenness'] +
                0.2 * item['eigenvector'] +
                0.1 * item['closeness']
            )
        
        com_data.sort(key=lambda x: x['hub_score'], reverse=True)
        
        # Identify the main hub (highest betweenness - key bridge)
        main_hub = max(com_data, key=lambda x: x['betweenness'])
        
        hub_analysis[com_id] = {
            'size': len(nodes),
            'nodes': com_data,
            'main_hub': main_hub['node'],
            'hub_betweenness': main_hub['betweenness'],
            'top_5': com_data[:5]  # Top 5 nodes in community
        }
    
    return hub_analysis

def print_community_analysis(hub_analysis, network_name):
    """Print detailed tables of community analysis - MAX 5 COMMUNITIES"""
    
    # Limit to top 5 largest communities
    sorted_communities = sorted(hub_analysis.items(), key=lambda x: x[1]['size'], reverse=True)[:5]
    
    print(f"\n{'='*100}")
    print(f"COMMUNITY HUB ANALYSIS: {network_name}")
    print(f"Showing top 5 largest communities")
    print(f"{'='*100}")
    
    for com_id, com_info in sorted_communities:
        print(f"\n{'─'*100}")
        print(f"📍 COMMUNITY {com_id} | Size: {com_info['size']} nodes | Main Hub: {com_info['main_hub'][:35]}")
        print(f"{'─'*100}")
        
        # Prepare table data - show only top 3 nodes per community to save space
        table_data = []
        for i, node_data in enumerate(com_info['top_5'][:3], 1):
            role = "🌟 HUB" if node_data['node'] == com_info['main_hub'] else ""
            table_data.append([
                i,
                node_data['node'][:30],  # Shorter truncation
                f"{node_data['degree']:.3f}",
                f"{node_data['betweenness']:.3f}",
                f"{node_data['eigenvector']:.3f}",
                f"{node_data['hub_score']:.3f}",
                role
            ])
        
        headers = ['#', 'Node', 'Degree', 'Between.', 'Eigen.', 'Score', 'Role']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
    
    # Summary table of all hubs (compact version)
    print(f"\n{'='*100}")
    print(f"🌟 MAIN HUBS SUMMARY")
    print(f"{'='*100}")
    
    hub_summary = []
    for com_id, com_info in sorted_communities:
        hub_summary.append([
            f"C{com_id}",
            com_info['size'],
            com_info['main_hub'][:40],
            f"{com_info['hub_betweenness']:.3f}"
        ])
    
    print(tabulate(hub_summary, 
                  headers=['Community', 'Size', 'Main Hub', 'Betweenness'],
                  tablefmt='simple'))

# =========================================================
# 4. PUBLICATION-QUALITY VISUALIZATION
# =========================================================
def plot_network_publication_quality(df_subset, cat_name, node_type):
    """Publication-ready network visualization with journal-quality aesthetics"""
    print(f"\n{'='*80}")
    print(f"Generating {node_type} Network for: {cat_name.replace('_', ' ')}")
    print(f"{'='*80}")
    
    G = nx.Graph()
    node_weights = {}

    # Network construction (Full Counting)
    countries_found = defaultdict(int)  # Track what we're finding
    
    for aff_row in df_subset['Affiliations']:
        if pd.isna(aff_row): continue
        raw_list = str(aff_row).split(';')
        clean_nodes = set()
        for raw in raw_list:
            if node_type == 'Institution':
                clean = get_clean_institution_aggressive(raw)
            else:
                clean = get_country(raw)
                if clean:  # Track country detection
                    countries_found[clean] += 1
            if clean: clean_nodes.add(clean)
        
        nodes_list = list(clean_nodes)
        for node in nodes_list:
            node_weights[node] = node_weights.get(node, 0) + 1
            
        if len(nodes_list) > 1:
            for u, v in combinations(nodes_list, 2):
                if G.has_edge(u, v): G[u][v]['weight'] += 1
                else: G.add_edge(u, v, weight=1)

    # Show top countries detected (diagnostic)
    if node_type == 'Country' and countries_found:
        print(f"   Top 10 countries detected:")
        for country, count in sorted(countries_found.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     • {country}: {count} affiliations")
    
    print(f"   Initial network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if G.number_of_nodes() < 5: 
        print("⚠️  Insufficient network (less than 5 nodes)")
        return None

    # ADJUSTED BACKBONE FILTERING FOR BETTER COVERAGE
    # For countries: keep more edges (Taiwan, smaller countries often have fewer collaborations)
    if node_type == 'Country':
        min_weight = 1  # Keep single collaborations for countries
        k_val = 2  # Lower k-core requirement
    else:
        min_weight = 2  # Institutions need at least 2 collaborations
        k_val = 2
    
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_weight])
    print(f"   After weight filter (≥{min_weight}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G = nx.k_core(G, k=k_val)
    print(f"   After k-core ({k_val}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() < 5: 
        print("⚠️  Network too sparse after filtering")
        return None

    # ===== CALCULATE METRICS =====
    metrics = calculate_network_metrics(G)
    
    # Print metrics report
    print(f"\n📊 NETWORK METRICS:")
    print(f"  • Nodes: {metrics['num_nodes']}")
    print(f"  • Edges: {metrics['num_edges']}")
    print(f"  • Density: {metrics['density']:.4f}")
    print(f"  • Components: {metrics['num_components']}")
    print(f"  • Largest component: {metrics['largest_component_size']} nodes")
    print(f"\n🔗 CENTRALITIES (average):")
    print(f"  • Degree: {metrics['avg_degree_centrality']:.4f}")
    print(f"  • Betweenness: {metrics['avg_betweenness_centrality']:.4f}")
    print(f"  • Closeness: {metrics['avg_closeness_centrality']:.4f}")
    print(f"  • Eigenvector: {metrics['avg_eigenvector_centrality']:.4f}")
    print(f"\n🔵 CLUSTERING:")
    print(f"  • Average coefficient: {metrics['avg_clustering']:.4f}")
    print(f"  • Transitivity: {metrics['transitivity']:.4f}")
    print(f"\n🌐 COMMUNITIES:")
    print(f"  • Number: {metrics['num_communities']}")
    print(f"  • Modularity: {metrics['modularity']:.4f}")
    
    # ===== IDENTIFY AND DISPLAY HUBS PER COMMUNITY =====
    hub_analysis = identify_community_hubs(G, metrics['partition'], metrics['all_centralities'])
    network_name = f"{cat_name.replace('_', ' ')} - {node_type}"
    print_community_analysis(hub_analysis, network_name)

    # ===== PUBLICATION-QUALITY FIGURE =====
    # Using Nature/Science style: clean, high contrast, readable
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(26, 16), facecolor='white', dpi=300)
    
    # Main layout with better proportions
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                         height_ratios=[2.5, 1, 1], width_ratios=[2.5, 1, 1])
    ax_main = fig.add_subplot(gs[:, :2])
    
    # ===== MAIN PANEL: NETWORK =====
    plt.sca(ax_main)
    
    # Enhanced layout algorithm with optimal spacing
    pos = nx.spring_layout(G, k=1.8/np.sqrt(len(G)), iterations=400, seed=42, weight='weight')

    # Draw edges with publication-quality style
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    min_w = min(weights) if weights else 1
    
    # Enhanced edge rendering
    for (u, v), weight in zip(G.edges(), weights):
        # Normalize alpha and width based on weight
        alpha = 0.12 + (weight - min_w)/(max_w - min_w + 0.001) * 0.45
        width = 0.6 + (weight - min_w)/(max_w - min_w + 0.001) * 5
        
        nx.draw_networkx_edges(G, pos, [(u, v)], 
                              alpha=alpha, 
                              edge_color='#34495e', 
                              width=width, 
                              style='solid')

    # Nodes with publication-quality colors and sizing
    partition = metrics['partition']
    eigenvector_cent = metrics['all_centralities']['eigenvector']
    betweenness_cent = metrics['all_centralities']['betweenness']
    degree_cent = metrics['all_centralities']['degree']
    num_coms = metrics['num_communities']
    
    # Professional color palette (enhanced for clarity)
    if num_coms <= 5:
        # High-contrast, color-blind friendly palette
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'][:num_coms]
    else:
        colors = sns.color_palette("husl", num_coms)
    
    texts = []
    
    # Label top nodes by combined importance
    node_importance = {}
    for n in G.nodes():
        # Combined importance metric
        importance = (
            eigenvector_cent.get(n, 0) * 0.4 + 
            betweenness_cent.get(n, 0) * 0.3 +
            degree_cent.get(n, 0) * 0.3
        )
        node_importance[n] = importance
    
    sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Label top 35% or at least 12 nodes
    num_to_label = max(int(len(G) * 0.35), min(12, len(G)))
    top_nodes_to_label = [n for n, _ in sorted_nodes[:num_to_label]]
    
    for com_id in range(num_coms):
        nodes = [n for n in partition if partition[n] == com_id]
        if not nodes: continue
        
        # Size nodes by importance with better scaling
        sizes = []
        for n in nodes:
            importance = node_importance.get(n, 0)
            # Scale: 600 base + up to 4500 additional
            node_size = 600 + importance * 4500
            sizes.append(node_size)
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=[colors[com_id]], 
                              node_size=sizes, 
                              alpha=0.92,
                              edgecolors='white', 
                              linewidths=3)
        
        # Add labels with enhanced styling
        for node in nodes:
            if node in top_nodes_to_label:
                x, y = pos[node]
                
                # Dynamic font size based on node type and importance
                if node_type == 'Country':
                    base_size = 13
                    fontweight = 'bold'
                else:
                    base_size = 10
                    fontweight = 'semibold'
                
                # Increase size for most important nodes
                is_top_5 = node in [n for n, _ in sorted_nodes[:5]]
                fontsize = base_size + (2 if is_top_5 else 0)
                
                t = plt.text(x, y, node, fontsize=fontsize, fontweight=fontweight,
                           ha='center', va='center', color='#1a1a1a',
                           family='DejaVu Sans', zorder=1000)
                
                # Enhanced text outline for maximum readability
                t.set_path_effects([
                    PathEffects.withStroke(linewidth=5, foreground='white', alpha=1.0)
                ])
                texts.append(t)

    # Adjust text with optimal parameters
    adjust_text(texts, expand_points=(2.0, 2.0), 
               expand_text=(1.5, 1.5),
               force_text=(0.5, 0.75),
               force_points=(0.3, 0.5),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2, alpha=0.7))
    
    # Title with professional styling
    title_text = f"{cat_name.replace('_', ' ')} — {node_type} Collaboration Network"
    ax_main.set_title(title_text, fontsize=20, fontweight='bold', 
                     pad=25, family='DejaVu Sans', color='#2c3e50')
    ax_main.axis('off')

    # ===== PANEL 1: CENTRALITY DISTRIBUTION =====
    ax1 = fig.add_subplot(gs[0, 2])
    degree_values = list(metrics['all_centralities']['degree'].values())
    betweenness_values = list(metrics['all_centralities']['betweenness'].values())
    
    # Create overlapping histograms with better styling
    ax1.hist(degree_values, bins=18, alpha=0.7, 
            label='Degree', color='#3498db', edgecolor='white', linewidth=1.5)
    ax1.hist(betweenness_values, bins=18, alpha=0.7, 
            label='Betweenness', color='#e74c3c', edgecolor='white', linewidth=1.5)
    
    ax1.set_xlabel('Centrality Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Centrality Distribution', fontsize=13, fontweight='bold', pad=12)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=10)

    # ===== PANEL 2: COMMUNITY SIZES =====
    ax2 = fig.add_subplot(gs[1, 2])
    com_sizes = metrics['community_sizes']
    x_pos = range(len(com_sizes))
    bars = ax2.bar(x_pos, list(com_sizes.values()), 
                   color=colors[:len(com_sizes)], alpha=0.88, 
                   edgecolor='white', linewidth=2.5, width=0.7)
    
    ax2.set_xlabel('Community ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_title('Community Size Distribution', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'C{i}' for i in com_sizes.keys()])
    ax2.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=10)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='#2c3e50')

    # ===== PANEL 3: METRICS SUMMARY =====
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.axis('off')
    
    # Enhanced metrics summary with better formatting
    summary_text = f"""╔═══ NETWORK METRICS ═══╗
║
║  Nodes:        {metrics['num_nodes']:>6}
║  Edges:        {metrics['num_edges']:>6}
║  Density:      {metrics['density']:>6.3f}
║
╠═══ STRUCTURE ═══════╣
║
║  Communities:  {metrics['num_communities']:>6}
║  Modularity:   {metrics['modularity']:>6.3f}
║  Components:   {metrics['num_components']:>6}
║
╠═══ CENTRALITY (μ) ══╣
║
║  Degree:       {metrics['avg_degree_centrality']:>6.3f}
║  Betweenness:  {metrics['avg_betweenness_centrality']:>6.3f}
║  Eigenvector:  {metrics['avg_eigenvector_centrality']:>6.3f}
║
╠═══ CLUSTERING ══════╣
║
║  Avg. Coeff.:  {metrics['avg_clustering']:>6.3f}
║  Transitivity: {metrics['transitivity']:>6.3f}
║
╚═════════════════════╝
"""
    
    ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#f8f9fa', 
                     edgecolor='#7f8c8d', linewidth=2.5, alpha=0.98))

    # Save with maximum publication quality
    filename = f"Network_{node_type}_{cat_name}_Publication.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', pad_inches=0.2)
    plt.close()
    print(f"\n✅ Saved: {filename}")
    
    return metrics

# =========================================================
# 5. MAIN EXECUTION
# =========================================================
def main():
    print("\n" + "="*80)
    print("BIBLIOMETRIC NETWORK ANALYSIS: XR/VR/AR RESEARCH COLLABORATION")
    print("Publication-Quality Network Visualizations")
    print("="*80)
    
    df = pd.read_csv(archivo, low_memory=False)
    col_auth = 'Author full names' if 'Author full names' in df.columns else 'Authors'
    
    # Filters
    mask_plaza = (df['Title'].astype(str).str.contains("Competing agents", case=False)) | \
                 (df[col_auth].astype(str).str.contains("Plaza, E", case=False))
    df = df[~mask_plaza].copy()
    df['Category'] = df.apply(classify_paper, axis=1)
    df = df.dropna(subset=['Category', 'Affiliations'])
    
    print(f"\n📄 Papers per category:")
    print(df['Category'].value_counts())
    
    # Store metrics for comparison
    all_metrics = {}
    
    categories = ['Virtual_Reality', 'Augmented_Reality', 'Mixed_Extended_Reality']
    node_types = ['Country', 'Institution']  # Country first (bibliometric standard)
    
    for cat in categories:
        df_cat = df[df['Category'] == cat]
        print(f"\n{'='*80}")
        print(f"PROCESSING TOPIC: {cat.replace('_', ' ').upper()}")
        print(f"Papers: {len(df_cat)}")
        print(f"{'='*80}")
        
        for node_type in node_types:
            key = f"{cat}_{node_type}"
            metrics = plot_network_publication_quality(df_cat, cat, node_type)
            if metrics:
                all_metrics[key] = metrics
    
    # ===== FINAL COMPARATIVE TABLE =====
    print("\n" + "="*100)
    print("COMPARATIVE NETWORK ANALYSIS SUMMARY")
    print("="*100)
    
    comparison_data = []
    for key, m in all_metrics.items():
        comparison_data.append([
            key.replace('_', ' ')[:35],
            m['num_nodes'],
            m['num_edges'],
            f"{m['density']:.3f}",
            f"{m['modularity']:.3f}",
            m['num_communities'],
            f"{m['avg_clustering']:.3f}",
            f"{m['transitivity']:.3f}"
        ])
    
    headers = ['Network', 'Nodes', 'Edges', 'Density', 'Modular', 'Comm', 'Cluster', 'Transit']
    print(tabulate(comparison_data, headers=headers, tablefmt='simple'))
    
    print("\n" + "="*100)
    print("✅ Analysis completed successfully!")
    print("   📊 Generated 6 publication-quality networks")
    print("   🌍 Country-level: Geographic collaboration patterns")
    print("   🏛️  Institution-level: Organizational collaboration networks")
    print("="*100)

if __name__ == "__main__":
    main()