import pandas as pd
import numpy as np
import math

# --- 1. CONFIGURACIÓN ---
archivo = 'vr.csv'
columna_citas = 'Cited by' 
columna_anio = 'Year'      

# --- 2. FUNCIÓN DE CLASIFICACIÓN (La misma de antes) ---
def classify_paper(row):
    title = str(row.get('Title', ''))
    abstract = str(row.get('Abstract', ''))
    auth_kw = str(row.get('Author Keywords', ''))
    kw_plus = str(row.get('Keywords Plus', '')) 
    idx_kw = str(row.get('Index Keywords', ''))
    source = str(row.get('Source title', ''))
    text = f"{title} {abstract} {auth_kw} {kw_plus} {idx_kw}".lower()
    
    mr_xr_terms = ['mixed reality', 'extended reality', 'mediated reality', ' mr ', ' xr ', 'spatial computing', 'hololens', 'magic leap', 'passthrough', 'digital twin', 'metaverse']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', 'google glass', 'epson moverio', 'heads-up display', ' hud ', 'marker-based', 'mobile ar']
    vr_terms = ['virtual reality', 'virtual environment', 'virtual world', ' vr ', ' ive ', ' cve ', 'immersive', 'immersion', 'head-mounted display', ' hmd ', 'oculus', 'vive', 'rift', 'quest', 'cardboard', 'cave', 'stereoscopic', '360 video']
    
    has_mr = any(term in text for term in mr_xr_terms)
    has_ar = any(term in text for term in ar_terms)
    has_vr = any(term in text for term in vr_terms)
    
    if has_mr or (has_vr and has_ar) or (not has_vr and not has_ar and not has_mr):
        return 'Mixed_Other'
    elif has_ar:
        return 'AR_Only'
    elif has_vr:
        return 'VR_Only'
    return 'Mixed_Other'

# --- 3. FUNCIÓN DE CÁLCULO DE MÉTRICAS ---
def calcular_main_info(df_subset, nombre_grupo):
    # Si el grupo está vacío, devolvemos ceros
    if len(df_subset) == 0:
        return {k: 0 for k in ["Total Citations", "Documents", "Growth Rate", "Avg Citations", "NCP", "CCP", "Avg Cit/Year", "H-index"]}

    # Limpieza local
    df_calc = df_subset.copy()
    df_calc[columna_citas] = pd.to_numeric(df_calc[columna_citas], errors='coerce').fillna(0)
    df_calc[columna_anio] = pd.to_numeric(df_calc[columna_anio], errors='coerce')
    df_calc = df_calc.dropna(subset=[columna_anio])

    # A. Métricas Básicas
    total_citations = df_calc[columna_citas].sum()
    num_documents = len(df_calc)
    
    if num_documents == 0: return {} # Seguridad extra

    min_year = int(df_calc[columna_anio].min())
    max_year = int(df_calc[columna_anio].max())
    time_span = max_year - min_year

    # B. Annual Growth Rate (CAGR)
    count_first_year = len(df_calc[df_calc[columna_anio] == min_year])
    count_last_year = len(df_calc[df_calc[columna_anio] == max_year])

    if count_first_year > 0 and time_span > 0:
        annual_growth_rate = (math.pow(count_last_year / count_first_year, 1 / time_span) - 1) * 100
    else:
        annual_growth_rate = 0

    # C. Promedios de Citas
    avg_citations_per_doc = total_citations / num_documents

    # D. Documentos Citados (NCP y CCP)
    cited_docs = df_calc[df_calc[columna_citas] > 0]
    ncp = len(cited_docs)
    ncp_percent = (ncp / num_documents) * 100
    citations_per_cited_doc = total_citations / ncp if ncp > 0 else 0

    # E. Average Citations per Year per Doc
    current_year_ref = 2025 + 1 # Fijo o dinámico según max_year global
    df_calc['Paper_Age'] = current_year_ref - df_calc[columna_anio]
    # Evitar división por cero si Paper_Age es 0 (mismo año)
    df_calc['Paper_Age'] = df_calc['Paper_Age'].replace(0, 1)
    
    df_calc['Citations_Per_Year'] = df_calc[columna_citas] / df_calc['Paper_Age']
    avg_citations_per_year = df_calc['Citations_Per_Year'].mean()

    # F. H-INDEX
    sorted_citations = sorted(df_calc[columna_citas].tolist(), reverse=True)
    h_index = 0
    for i, citation_count in enumerate(sorted_citations):
        rank = i + 1
        if citation_count >= rank:
            h_index = rank
        else:
            break
            
    # Devolver diccionario formateado
    return {
        "Metric": [
            "Documents (N)",
            "Total Citations (TC)",
            "H-index",
            "Annual Growth Rate %",
            "Avg Citations per Doc (AC)",
            "Cited Publications (NCP)",
            "% Cited Publications",
            "Cit. per Cited Doc (CCP)",
            "Avg Citations per Year"
        ],
        nombre_grupo: [
            f"{num_documents}",
            f"{int(total_citations)}",
            f"{h_index}",
            f"{annual_growth_rate:.2f}",
            f"{avg_citations_per_doc:.2f}",
            f"{ncp}",
            f"{ncp_percent:.2f}%",
            f"{citations_per_cited_doc:.2f}",
            f"{avg_citations_per_year:.2f}"
        ]
    }

# --- 4. EJECUCIÓN PRINCIPAL ---
print("Cargando archivo...")
try:
    df = pd.read_csv(archivo, low_memory=False)
    # Limpieza de nombres de columna
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error: {e}")
    exit()

print("Clasificando documentos...")
df['Category'] = df.apply(classify_paper, axis=1)

# Separar grupos
df_vr = df[df['Category'] == 'VR_Only']
df_ar = df[df['Category'] == 'AR_Only']
df_mixed = df[df['Category'] == 'Mixed_Other']

# Calcular métricas para cada uno
print("Calculando métricas...")
res_vr = calcular_main_info(df_vr, "VR Only")
res_ar = calcular_main_info(df_ar, "AR Only")
res_mixed = calcular_main_info(df_mixed, "Mixed/Other")

# --- 5. CONSOLIDAR EN UNA TABLA ---
# Creamos DataFrames parciales
df_res_vr = pd.DataFrame(res_vr)
df_res_ar = pd.DataFrame(res_ar)
df_res_mixed = pd.DataFrame(res_mixed)

# Fusionamos (Merge) usando la columna 'Metric' como clave
df_final = df_res_vr.merge(df_res_ar, on="Metric").merge(df_res_mixed, on="Metric")

print("\n" + "="*80)
print(" COMPARATIVE MAIN INFORMATION (VR vs AR vs MIXED)")
print("="*80)
print(df_final.to_string(index=False))
print("="*80)

# Exportar a CSV para copiar a Excel/Word
df_final.to_csv("main_info_comparative.csv", index=False)
print("\n✅ Tabla guardada como 'main_info_comparative.csv'")