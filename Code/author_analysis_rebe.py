import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from collections import Counter

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================
archivo = 'vr.csv'
cat_order = ['Virtual Reality', 'Augmented Reality', 'Mixed/Extended Reality']
cat_colors = {
    'Virtual Reality': '#D32F2F',          
    'Augmented Reality': '#1976D2',        
    'Mixed/Extended Reality': '#388E3C'    
}

# =========================================================
# 2. FUNCIONES DE LIMPIEZA
# =========================================================
def classify_paper(row):
    text = f"{str(row.get('Title',''))} {str(row.get('Abstract',''))} {str(row.get('Author Keywords',''))}".lower()
    exclusions = ['agent-mediated', 'multi-agent', 'auction', 'negotiation', 'mixed integer']
    hard_xr = ['hololens', 'magic leap', 'oculus', 'htc vive', 'hmd', 'quest']
    
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr):
        return None

    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', 'hololens', ' xr ', ' mr ']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', ' heads-up ', 'mobile ar']
    vr_terms = ['virtual reality', ' vr ', 'immersive', 'hmd', 'oculus']
    
    has_mr, has_ar, has_vr = any(t in text for t in mr_terms), any(t in text for t in ar_terms), any(t in text for t in vr_terms)
    
    if has_mr or (has_vr and has_ar): return 'Mixed/Extended Reality'
    elif has_ar: return 'Augmented Reality'
    elif has_vr: return 'Virtual Reality'
    return None

def calculate_h_index(citations_list):
    cites = sorted(citations_list, reverse=True)
    h = 0
    for i, c in enumerate(cites):
        if c >= i + 1: h = i + 1
        else: break
    return h

def get_clean_institution_name(aff_string):
    if pd.isna(aff_string) or "unknown" in str(aff_string).lower():
        return None
    
    aff_lower = str(aff_string).lower()
    
    # --- DICCIONARIO DE GIGANTES (MAPPING MANUAL CORREGIDO) ---
    mappings = {
        # CANADA
        'toronto': 'Univ. of Toronto (CAN)',
        # USA - CORRECCIONES DE CIUDAD vs UNI
        'la jolla': 'UC San Diego (USA)',       # <--- AQUÍ ESTÁ LA CORRECCIÓN
        'san diego': 'UC San Diego (USA)',
        'uc san diego': 'UC San Diego (USA)',
        'ucsd': 'UC San Diego (USA)',
        'ann arbor': 'Univ. of Michigan (USA)',
        'ithaca': 'Cornell Univ. (USA)',
        'cambridge, ma': 'Harvard/MIT (USA)', # A veces ambiguo, pero mejor que "Cambridge" a secas
        'harvard': 'Harvard Univ. (USA)',
        'stanford': 'Stanford Univ. (USA)',
        'washington': 'Univ. of Washington (USA)',
        'mit': 'MIT (USA)',
        'massachusetts institute': 'MIT (USA)',
        'johns hopkins': 'Johns Hopkins (USA)',
        'mayo clinic': 'Mayo Clinic (USA)',
        'duke': 'Duke Univ. (USA)',
        # UK
        'college london': 'UCL (UK)',
        'imperial college': 'Imperial College (UK)',
        'oxford': 'Univ. of Oxford (UK)',
        'cambridge': 'Univ. of Cambridge (UK)',
        # EUROPE
        'copenhagen': 'Univ. of Copenhagen (DNK)',
        'københavns': 'Univ. of Copenhagen (DNK)',
        'rigshospitalet': 'Rigshospitalet (DNK)',
        'munich': 'Tech. Univ. Munich (DEU)',
        'barcelona': 'Univ. de Barcelona (ESP)',
        'seville': 'Univ. de Sevilla (ESP)',
        'sevilla': 'Univ. de Sevilla (ESP)',
        # ASIA
        'tsinghua': 'Tsinghua Univ. (CHN)',
        'nanyang': 'Nanyang Tech. Univ. (SGP)',
        'nus': 'National Univ. Singapore (SGP)',
        'hong kong polytechnic': 'HK PolyU (HKG)',
        'tokyo': 'Univ. of Tokyo (JPN)'
    }

    # 1. Búsqueda prioritaria en diccionario
    for key, val in mappings.items():
        if key in aff_lower: return val

    # 2. Extracción Algorítmica (Fallback)
    first_full_aff = str(aff_string).split(';')[0].strip()
    parts = [p.strip() for p in first_full_aff.split(',')]
    pais = parts[-1] if len(parts) > 0 else ""
    
    skip_terms = ['department', 'dept', 'faculty', 'school', 'college', 'division', 'unit', 'center', 'lab', 'service', 'ministry']
    target_terms = ['university', 'universidad', 'institute', 'hospital', 'clinic', 'foundation', 'system']
    
    best = None
    for part in parts:
        if len(part) < 3: continue
        if any(bad in part.lower() for bad in skip_terms) and not any(good in part.lower() for good in target_terms): continue
        if any(good in part.lower() for good in target_terms):
            best = part
            break
    
    if not best:
        best = parts[1] if len(parts) > 1 and len(parts[1]) > 3 else parts[0]
    
    # Limpieza visual
    if len(best) > 28: best = best[:26] + "."
    return f"{best} ({pais})"

# =========================================================
# 3. PROCESAMIENTO
# =========================================================
print("Procesando datos con corrección de La Jolla...")
df = pd.read_csv(archivo, low_memory=False)
df.columns = df.columns.str.strip()
col_authors = 'Author full names' if 'Author full names' in df.columns else 'Authors'
col_cit, col_aff = 'Cited by', 'Affiliations'

# Guillotina
mask_plaza = (df['Title'].astype(str).str.contains("Competing agents", case=False)) | \
             (df[col_authors].astype(str).str.contains("Plaza, E", case=False))
df = df[~mask_plaza].copy()

df['Category'] = df.apply(classify_paper, axis=1)
df = df.dropna(subset=['Category', col_authors])
df[col_cit] = pd.to_numeric(df[col_cit], errors='coerce').fillna(0)

# --- Votación de Afiliación ---
df['Paper_Inst_Clean'] = df[col_aff].apply(get_clean_institution_name)
author_aff_history = {}

for _, row in df.iterrows():
    if pd.isna(row['Paper_Inst_Clean']): continue
    paper_auths = [a.split(' (')[0].strip() for a in str(row[col_authors]).split(';')]
    inst = row['Paper_Inst_Clean']
    for auth in paper_auths:
        if auth not in author_aff_history: author_aff_history[auth] = []
        author_aff_history[auth].append(inst)

final_author_aff_map = {}
for auth, affs in author_aff_history.items():
    if affs:
        final_author_aff_map[auth] = Counter(affs).most_common(1)[0][0]
    else:
        final_author_aff_map[auth] = "Unknown"

# =========================================================
# 4. EXTRACCIÓN Y GRÁFICOS
# =========================================================
stats_by_cat = {}
global_max = {'Pubs': 0, 'Citations': 0, 'h-index': 0}

for cat in cat_order:
    df_sub = df[df['Category'] == cat]
    auth_data = {}
    
    for _, row in df_sub.iterrows():
        paper_auths = [a.split(' (')[0].strip() for a in str(row[col_authors]).split(';')]
        for auth in paper_auths:
            if auth not in auth_data: auth_data[auth] = []
            auth_data[auth].append(row[col_cit])
    
    res = []
    for auth, cites in auth_data.items():
        res.append({
            'Author': auth, 
            'Affiliation': final_author_aff_map.get(auth, 'Unknown'),
            'Pubs': len(cites), 
            'Citations': sum(cites), 
            'h-index': calculate_h_index(cites)
        })
    
    df_res = pd.DataFrame(res)
    stats_by_cat[cat] = df_res
    
    if not df_res.empty:
        global_max['Pubs'] = max(global_max['Pubs'], df_res['Pubs'].max())
        global_max['Citations'] = max(global_max['Citations'], df_res['Citations'].max())
        global_max['h-index'] = max(global_max['h-index'], df_res['h-index'].max())

# Función de graficado
def plot_comparison_metric(metric_key, metric_label):
    print(f"Generando gráfico corregido para {metric_label}...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    x_limit = global_max[metric_key] * 1.2

    for i, cat in enumerate(cat_order):
        ax = axes[i]
        color = cat_colors[cat]
        data = stats_by_cat[cat].sort_values(metric_key, ascending=False).head(10)
        
        # Etiqueta corregida
        labels = [f"{r['Author']}\n{r['Affiliation']}" for _, r in data.iterrows()]
        
        bars = ax.barh(range(10), data[metric_key], color=color, alpha=0.9, edgecolor='black', height=0.7)
        
        ax.set_yticks(range(10))
        ax.set_yticklabels(labels, fontsize=11, fontweight='semibold')
        ax.set_title(f"{cat}", fontsize=18, fontweight='bold', loc='left', color=color, pad=10)
        ax.invert_yaxis()
        ax.set_xlim(0, x_limit)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + (x_limit*0.01), bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', va='center', fontweight='bold', fontsize=11,
                    path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])

    axes[-1].set_xlabel(metric_label, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"Comparison_{metric_key}_Final.png", dpi=300, bbox_inches='tight')
    plt.close()

# Ejecutar
plot_comparison_metric('Pubs', 'Number of Publications')
plot_comparison_metric('Citations', 'Total Citations')
plot_comparison_metric('h-index', 'h-index')

print("\n✅ Corrección aplicada: 'La Jolla' ahora es 'UC San Diego'.")