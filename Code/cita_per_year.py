import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURACIÓN ---
archivo = 'vr.csv'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# --- 2. CLASIFICACIÓN QUIRÚRGICA ---
def classify_paper(row):
    text = f"{str(row.get('Title',''))} {str(row.get('Abstract',''))} {str(row.get('Author Keywords',''))}".lower()
    
    # Exclusiones Anti-Plaza / Agentes
    exclusions = ['agent-mediated', 'multi-agent', 'auction', 'negotiation', 'market mechanism']
    hard_xr_hardware = ['hololens', 'magic leap', 'oculus', 'htc vive', 'hmd', 'quest']
    
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr_hardware):
        return None

    # Diccionarios
    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', 'hololens', ' xr ', ' mr ', 'metaverse']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', 'google glass', ' heads-up ', ' hud ']
    vr_terms = ['virtual reality', ' vr ', 'immersive', 'immersion', 'hmd', 'oculus', 'vive']
    
    has_mr = any(term in text for term in mr_terms)
    has_ar = any(term in text for term in ar_terms)
    has_vr = any(term in text for term in vr_terms)
    
    # Lógica jerárquica
    if has_mr or (has_vr and has_ar): return 'Mixed/Extended Reality'
    elif has_ar: return 'Augmented Reality'
    elif has_vr: return 'Virtual Reality'
    return None

# --- 3. PROCESAMIENTO ---
def generar_grafico_citas_final(csv_path):
    print("Iniciando procesamiento con filtros avanzados...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    # Limpieza inicial y Guillotina Plaza
    col_year = 'Year'
    col_cit = 'Cited by' if 'Cited by' in df.columns else 'Citations'
    col_authors = 'Authors' if 'Authors' in df.columns else 'Author full names'
    
    df[col_year] = pd.to_numeric(df[col_year], errors='coerce')
    df[col_cit] = pd.to_numeric(df[col_cit], errors='coerce').fillna(0)
    
    mask_plaza = df['Title'].astype(str).str.contains("Competing agents", case=False, na=False) | \
                 df[col_authors].astype(str).str.contains("Plaza, E", case=False, na=False)
    df = df[~mask_plaza]

    # Aplicar clasificación
    df['Category'] = df.apply(classify_paper, axis=1)
    df = df.dropna(subset=['Category', col_year])
    
    # Rango 1995-2025
    df = df[(df[col_year] >= 1995) & (df[col_year] <= 2025)].copy()
    years = np.arange(1995, 2026)

    # --- 4. GRAFICADO ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 22), sharex=True)
    fig.subplots_adjust(hspace=0.2, right=0.8)

    grp_names = ['Virtual Reality', 'Augmented Reality', 'Mixed/Extended Reality']
    colors = {'bar': '#CFD8DC', 'cit': '#1565C0', 'imp': '#C62828'}

    for i, grp in enumerate(grp_names):
        ax = axes[i]
        df_grp = df[df['Category'] == grp]
        
        # Agrupar métricas
        stats = df_grp.groupby(col_year).agg(
            N=('Title', 'count'),
            TC=(col_cit, 'sum'),
            Impact=(col_cit, 'mean')
        ).reindex(years, fill_value=0)

        # Eje triple
        par1 = ax.twinx()
        par2 = ax.twinx()
        par2.spines["right"].set_position(("axes", 1.15))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        # Plots
        b = ax.bar(years, stats['N'], color=colors['bar'], alpha=0.7, label="Annual Production (N)")
        l1, = par1.plot(years, stats['TC'], color=colors['cit'], lw=3, marker='o', label="Total Citations (TC)")
        l2, = par2.plot(years, stats['Impact'], color=colors['imp'], lw=2, ls='--', marker='s', label="Impact (TC/N)")

        # Configuración de ejes
        ax.set_ylabel("Production (N)", fontweight='bold', color='#455A64')
        par1.set_ylabel("Total Citations", fontweight='bold', color=colors['cit'])
        par2.set_ylabel("Impact (TC/N)", fontweight='bold', color=colors['imp'])
        
        ax.set_title(f"{i+1}. {grp} Evolution (1995-2025)", fontsize=18, fontweight='bold', loc='left', pad=15)
        ax.grid(True, axis='y', ls=':', alpha=0.6)

    # Leyenda global
    fig.legend([b, l1, l2], ["Production (N)", "Total Citations", "Impact (TC/N)"], 
               loc='upper center', ncol=3, fontsize=14, frameon=False, bbox_to_anchor=(0.5, 0.96))

    plt.xlabel("Publication Year", fontsize=16, fontweight='bold')
    plt.xticks(years[::2], rotation=45)
    
    plt.savefig('citations_evolution_1995_2025.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    generar_grafico_citas_final(archivo)