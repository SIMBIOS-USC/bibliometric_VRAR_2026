import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Patch

# --- 1. CONFIGURACIÓN ---
archivo_entrada = 'vr.csv'
columna_afiliacion = 'Affiliations'

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Verdana']
mpl.rcParams['font.size'] = 11

# Colores: [SCP (Vibrante), MCP (Pastel)]
colors = {
    'Virtual Reality':         ['#e53935', '#ffcdd2'], 
    'Augmented Reality':       ['#1e88e5', '#bbdefb'], 
    'Mixed/Extended Reality':  ['#43a047', '#c8e6c9']  
}

keywords_paises = {
    "USA": ["united states", "usa", "u.s.a.", " ca ", " ny ", "california", "harvard", "mit", "stanford"],
    "China": ["china", "p.r. china", "hong kong", "beijing", "shanghai", "wuhan"],
    "UK": ["united kingdom", " uk ", "england", "london", "oxford", "cambridge"],
    "Germany": ["germany", "deutschland", "berlin", "munich"],
    "South Korea": ["south korea", "korea", "seoul"],
    "Italy": ["italy", "italia", "rome", "milan"],
    "Australia": ["australia", "sydney", "melbourne"],
    "Canada": ["canada", "toronto", "vancouver"],
    "Spain": ["spain", "españa", "madrid", "barcelona"],
    "Japan": ["japan", "tokyo", "kyoto"],
    "France": ["france", "paris", "lyon"],
    "Taiwan": ["taiwan", "taipei"]
}

# --- 2. FILTROS QUIRÚRGICOS ---
def classify_paper(row):
    title, abstract, auth_kw = str(row.get('Title','')), str(row.get('Abstract','')), str(row.get('Author Keywords',''))
    text = f"{title} {abstract} {auth_kw}".lower()
    
    exclusions = ['agent-mediated', 'multi-agent', 'auction', 'negotiation', 'market mechanism', 'mixed integer']
    hard_xr = ['hololens', 'magic leap', 'oculus', 'htc vive', 'head-mounted', ' hmd ']
    
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr):
        return None

    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', 'hololens', 'magic leap', ' xr ', ' mr ']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', ' hud ', 'mobile ar']
    vr_terms = ['virtual reality', ' vr ', 'immersive', 'hmd', 'oculus', 'vive']
    
    has_mr, has_ar, has_vr = any(t in text for t in mr_terms), any(t in text for t in ar_terms), any(t in text for t in vr_terms)
    
    if has_mr or (has_vr and has_ar): return 'Mixed/Extended Reality'
    elif has_ar: return 'Augmented Reality'
    elif has_vr: return 'Virtual Reality'
    return None

def procesar_datos(df):
    data_struct = {p: {f"{cat}_{type}": 0 for cat in colors for type in ['SCP', 'MCP']} for p in keywords_paises}
    for _, row in df.iterrows():
        aff = str(row[columna_afiliacion]).lower()
        if len(aff) < 3: continue
        leader = next((p for p, k in keywords_paises.items() if any(kw in aff.split(';')[0] for kw in k)), None)
        if not leader: continue
        cat = classify_paper(row)
        if not cat: continue
        collab = 'MCP' if len({p for p, k in keywords_paises.items() if any(kw in aff for kw in k)}) > 1 else 'SCP'
        data_struct[leader][f"{cat}_{collab}"] += 1
    return pd.DataFrame(data_struct).T

# --- 3. EJECUCIÓN ---
print("Cargando datos y ejecutando Guillotina...")
try:
    df = pd.read_csv(archivo_entrada, low_memory=False)
    df.columns = df.columns.str.strip()
    col_authors = 'Authors' if 'Authors' in df.columns else 'Author full names'
    
    # 🔪 GUILLOTINA ANTI-PLAZA
    mask_plaza = (df['Title'].astype(str).str.contains("Competing agents", case=False)) | \
                 (df[col_authors].astype(str).str.contains("Plaza, E", case=False))
    df = df[~mask_plaza].copy()
    
    df_plot = procesar_datos(df)
    df_plot['Total_SCP'] = df_plot[[c for c in df_plot.columns if 'SCP' in c]].sum(axis=1)
    df_plot['Total_MCP'] = df_plot[[c for c in df_plot.columns if 'MCP' in c]].sum(axis=1)
    df_plot['Total'] = df_plot['Total_SCP'] + df_plot['Total_MCP']
    df_plot = df_plot.sort_values('Total', ascending=False).head(12)
except Exception as e:
    print(f"Error: {e}"); exit()

# --- 4. GRAFICADO ---
fig, ax = plt.subplots(figsize=(16, 12))
x = np.arange(len(df_plot))
width = 0.6
bottom = np.zeros(len(df_plot))

# Bloques: SCP (Base) -> MCP (Cima)
for tipo, idx in [('SCP', 0), ('MCP', 1)]:
    for cat in colors:
        col = f"{cat}_{tipo}"
        vals = df_plot[col].values
        ax.bar(x, vals, width, bottom=bottom, color=colors[cat][idx], edgecolor='white', linewidth=0.5)
        bottom += vals

# --- INDICADORES DE "LLAVE" Y LÍNEA DE SEPARACIÓN ---
for i, pais in enumerate(df_plot.index):
    scp_h = df_plot.loc[pais, 'Total_SCP']
    total_h = df_plot.loc[pais, 'Total']
    
    # A. Línea divisoria horizontal en la propia barra
    ax.hlines(y=scp_h, xmin=i-width/2, xmax=i+width/2, color='black', linestyle='--', linewidth=1, zorder=5)
    
    # B. Llave SCP (Izquierda de la barra)
    ax.annotate('', xy=(i - width/2 - 0.05, 0), xytext=(i - width/2 - 0.05, scp_h),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax.text(i - width/2 - 0.1, scp_h/2, f"SCP: {int(scp_h)}", ha='right', va='center', 
            fontsize=9, fontweight='bold', color='#333333', rotation=90)

    # C. Llave MCP (Derecha de la barra)
    ax.annotate('', xy=(i + width/2 + 0.05, scp_h), xytext=(i + width/2 + 0.05, total_h),
                arrowprops=dict(arrowstyle='<->', color='#d32f2f', lw=1.5))
    ax.text(i + width/2 + 0.1, scp_h + (total_h-scp_h)/2, f"MCP: {int(total_h-scp_h)}", 
            ha='left', va='center', fontsize=9, fontweight='bold', color='#d32f2f', rotation=270)

    # D. Valor Total arriba
    ax.text(i, total_h + (df_plot['Total'].max()*0.02), f"TOTAL\n{int(total_h)}", 
            ha='center', va='bottom', fontsize=10, fontweight='black', color='black')

# --- ESTÉTICA FINAL ---
ax.set_title("Scientific Production Structure: Domestic (SCP) vs. International (MCP)", fontsize=20, fontweight='bold', pad=30)
ax.set_ylabel("Documents", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=13, fontweight='bold')
ax.yaxis.grid(True, linestyle=':', alpha=0.5)

# Leyenda
legend_elements = [
    Patch(facecolor='white', alpha=0, label=r'$\bf{International\ (MCP)}$'),
    *[Patch(facecolor=colors[cat][1], label=cat) for cat in colors],
    Patch(facecolor='white', alpha=0, label=r'$\bf{Domestic\ (SCP)}$'),
    *[Patch(facecolor=colors[cat][0], label=cat) for cat in colors]
]
ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=10, frameon=True, shadow=True, title="Category & Type")

plt.tight_layout()
plt.savefig('scp_mcp_with_braces.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico generado con llaves de separación SCP/MCP.")
plt.show()