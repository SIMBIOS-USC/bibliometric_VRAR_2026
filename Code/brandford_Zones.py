import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

# --- 1. CONFIGURACIÓN ---
input_file = 'vr.csv'
max_year = 2025  # Filtro temporal
colors_bradford = ['#1f4e79', '#2e75b6', '#89bceb']

# --- 2. FUNCIÓN DE CLASIFICACIÓN (La misma de antes) ---
def classify_paper(row):
    # Recopilación de texto
    title = str(row.get('Title', ''))
    abstract = str(row.get('Abstract', ''))
    auth_kw = str(row.get('Author Keywords', ''))
    kw_plus = str(row.get('Keywords Plus', '')) 
    idx_kw = str(row.get('Index Keywords', ''))
    source = str(row.get('Source title', ''))

    text = f"{title} {abstract} {auth_kw} {kw_plus} {idx_kw}".lower()
    
    # Términos
    mr_xr_terms = ['mixed reality', 'extended reality', 'mediated reality', ' mr ', ' xr ', 'spatial computing', 'hololens', 'magic leap', 'passthrough', 'digital twin', 'metaverse']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', 'google glass', 'epson moverio', 'heads-up display', ' hud ', 'marker-based', 'markerless', 'mobile ar']
    vr_terms = ['virtual reality', 'virtual environment', 'virtual world', ' vr ', ' ive ', ' cve ', 'immersive', 'immersion', 'head-mounted display', ' hmd ', 'oculus', 'vive', 'rift', 'quest', 'cardboard', 'cave', 'stereoscopic', '360 video']
    
    has_mr = any(term in text for term in mr_xr_terms)
    has_ar = any(term in text for term in ar_terms)
    has_vr = any(term in text for term in vr_terms)
    
    # Lógica de 3 grupos
    if has_mr or (has_vr and has_ar) or (not has_vr and not has_ar and not has_mr):
        return 'Mixed_Other'
    elif has_ar:
        return 'AR_Only'
    elif has_vr:
        return 'VR_Only'
    return 'Mixed_Other'

# --- 3. FUNCIÓN DE ANÁLISIS Y GRAFICADO ---
def analizar_y_graficar(df_subset, nombre_grupo, color_palette_bars="tab10"):
    print(f"\n--- Generando gráficas para: {nombre_grupo} ({len(df_subset)} docs) ---")
    
    if len(df_subset) < 10:
        print("   [!] Muy pocos datos para graficar. Saltando.")
        return

    col_source = 'Source title'
    col_year = 'Year'

    # --- A. LEY DE BRADFORD ---
    source_counts = df_subset[col_source].value_counts()
    total_articles = source_counts.sum()
    cumsum_articles = source_counts.cumsum()

    # Definir límites (tercios)
    limit_1 = total_articles / 3
    limit_2 = limit_1 * 2

    # Identificar zonas
    zone1_journals = source_counts[cumsum_articles <= limit_1]
    zone2_journals = source_counts[(cumsum_articles > limit_1) & (cumsum_articles <= limit_2)]
    zone3_journals = source_counts[cumsum_articles > limit_2]

    # Datos para visualización
    zones_data = [
        {"zone": "Zone 1 (Core)", "journals": len(zone1_journals), "articles": zone1_journals.sum()},
        {"zone": "Zone 2 (Middle)", "journals": len(zone2_journals), "articles": zone2_journals.sum()},
        {"zone": "Zone 3 (Minor)", "journals": len(zone3_journals), "articles": zone3_journals.sum()}
    ]

    # --- B. EVOLUCIÓN TOP 10 FUENTES ---
    top_10_names = source_counts.head(10).index.tolist()
    df_top = df_subset[df_subset[col_source].isin(top_10_names)]

    # Crear tabla cruzada y rellenar ceros
    pivot_sources = pd.crosstab(df_top[col_year], df_top[col_source])
    # Asegurar que están todas las columnas del top 10 (aunque sean 0 en algunos años)
    for col in top_10_names:
        if col not in pivot_sources.columns:
            pivot_sources[col] = 0
    
    # Reordenar columnas para coincidir con el ranking
    pivot_sources = pivot_sources[top_10_names] 

    # --- C. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # === PLOT 1: BRADFORD ZONES ===
    widths = [3, 5, 7] 
    heights = [1, 1, 1] 
    bottoms = [2, 1, 0] 

    for i, zone in enumerate(zones_data):
        rect = patches.Rectangle(
            (-widths[i]/2, bottoms[i]),
            widths[i], heights[i], 
            linewidth=1, edgecolor='white', facecolor=colors_bradford[i]
        )
        ax1.add_patch(rect)
        
        # Texto dentro de las cajas
        # Manejo seguro si la zona está vacía (raro pero posible)
        art_count = int(zone['articles']) if not pd.isna(zone['articles']) else 0
        label_text = (f"{zone['zone']}\n"
                      f"{zone['journals']} journals\n"
                      f"{art_count} articles")
        ax1.text(0, bottoms[i] + 0.5, label_text, 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    ax1.set_xlim(-5, 5)
    ax1.set_ylim(0, 3.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f"Bradford's Law Zones\n({nombre_grupo})", fontsize=16, fontweight='bold', pad=20)

    # === PLOT 2: EVOLUTION TOP 10 ===
    colors_top = sns.color_palette(color_palette_bars, n_colors=10)
    
    pivot_sources.plot(kind='bar', stacked=True, ax=ax2, color=colors_top, width=0.8)

    ax2.set_title(f"Annual Evolution of Top 10 Sources\n({nombre_grupo})", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel("Year", fontsize=14)
    ax2.set_ylabel("Number of Publications", fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Leyenda
    handles, labels = ax2.get_legend_handles_labels()
    new_labels = [f"{label} ({source_counts[label]})" for label in labels]
    ax2.legend(handles, new_labels, title="Source (Total Docs)", 
               bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    
    # Guardar
    filename = f"Bradford_Sources_{nombre_grupo.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   -> Guardado: {filename}")
    plt.close() # Cerrar para liberar memoria

# --- 4. EJECUCIÓN ---
print("Cargando y procesando datos...")
try:
    df = pd.read_csv(input_file, low_memory=False)
    # Limpieza
    df.columns = df.columns.str.strip()
    col_year = 'Year'
    col_source = 'Source title'
    
    # Limpiar Source y Año
    df[col_source] = df[col_source].astype(str).str.strip()
    df[col_year] = pd.to_numeric(df[col_year], errors='coerce')
    df = df.dropna(subset=[col_year])
    df[col_year] = df[col_year].astype(int)
    
    # Filtro temporal
    df = df[df[col_year] <= max_year]
    
except Exception as e:
    print(f"Error cargando datos: {e}")
    exit()

# Clasificar
print("Clasificando documentos...")
df['Category'] = df.apply(classify_paper, axis=1)

# Dividir datasets
df_vr = df[df['Category'] == 'VR_Only']
df_ar = df[df['Category'] == 'AR_Only']
df_mixed = df[df['Category'] == 'Mixed_Other']

# Generar gráficos (Usamos paletas distintas para diferenciarlos visualmente si quieres)
analizar_y_graficar(df_vr, "VR Only", "Blues_r")
analizar_y_graficar(df_ar, "AR Only", "Greens_r")
analizar_y_graficar(df_mixed, "Mixed & Other", "Purples_r")

print("\n¡Proceso finalizado! Se han generado 3 imágenes.")