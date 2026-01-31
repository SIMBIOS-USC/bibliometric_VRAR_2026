import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN ---
archivo = 'vr.csv'

# --- 2. FUNCIÓN DE CLASIFICACIÓN (La misma de antes) ---
def classify_paper(row):
    # Recopilamos texto
    title = str(row.get('Title', ''))
    abstract = str(row.get('Abstract', ''))
    auth_kw = str(row.get('Author Keywords', ''))
    kw_plus = str(row.get('Keywords Plus', '')) 
    idx_kw = str(row.get('Index Keywords', ''))
    source = str(row.get('Source title', ''))

    text = f"{title} {abstract} {auth_kw} {kw_plus} {idx_kw}".lower()
    
    # Diccionarios
    mr_xr_terms = ['mixed reality', 'extended reality', 'mediated reality', ' mr ', ' xr ', 'spatial computing', 'hololens', 'magic leap', 'passthrough', 'digital twin', 'metaverse']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', 'google glass', 'epson moverio', 'heads-up display', ' hud ', 'marker-based', 'markerless', 'mobile ar']
    vr_terms = ['virtual reality', 'virtual environment', 'virtual world', ' vr ', ' ive ', ' cve ', 'immersive', 'immersion', 'head-mounted display', ' hmd ', 'oculus', 'vive', 'rift', 'quest', 'cardboard', 'cave', 'pico', 'stereoscopic', '360 video']
    
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

# --- 3. FUNCIÓN DE CÁLCULO DE MÉTRICAS ---
def calcular_metricas_grupo(df_grupo, nombre_grupo):
    if len(df_grupo) == 0:
        return {
            "Grupo": nombre_grupo,
            "Documentos (N)": 0, "NCA": 0, "SA": 0, "CA": 0, "CI": 0.0, "CC": 0.0
        }

    # Detectar columna de autores (Author full names o Authors)
    col_autores = 'Author full names' if 'Author full names' in df_grupo.columns else 'Authors'
    
    # Procesar autores
    def limpiar_autores(cadena):
        if pd.isna(cadena): return []
        # Separar por punto y coma y limpiar espacios
        return [a.strip() for a in str(cadena).split(';') if a.strip() != '']

    # Crear columna temporal de lista de autores
    # Usamos .copy() para evitar SettingWithCopyWarning
    df_calc = df_grupo.copy()
    df_calc['lista_autores_clean'] = df_calc[col_autores].apply(limpiar_autores)
    df_calc['num_autores'] = df_calc['lista_autores_clean'].apply(len)
    
    # Filtrar registros sin autores (num_autores = 0)
    df_calc = df_calc[df_calc['num_autores'] > 0]
    N = len(df_calc)
    
    if N == 0:
        return {"Grupo": nombre_grupo, "N": 0, "NCA": 0, "SA": 0, "CA": 0, "CI": 0, "CC": 0}

    # 1. NCA (Autores únicos)
    todos_autores = set()
    for lista in df_calc['lista_autores_clean']:
        todos_autores.update(lista)
    NCA = len(todos_autores)

    # 2. SA (Sole-authored)
    SA = len(df_calc[df_calc['num_autores'] == 1])

    # 3. CA (Co-authored)
    CA = len(df_calc[df_calc['num_autores'] > 1])

    # 4. CI (Collaboration Index)
    # Promedio de autores SOLO en documentos multi-autor
    df_multi = df_calc[df_calc['num_autores'] > 1]
    if CA > 0:
        total_autores_multi = df_multi['num_autores'].sum()
        CI = total_autores_multi / CA
    else:
        CI = 0

    # 5. CC (Collaboration Coefficient)
    # 1 - ( Sum( (1/j) * fj ) / N )
    sumatoria = 0
    conteo_frecuencias = df_calc['num_autores'].value_counts()
    
    for j, fj in conteo_frecuencias.items():
        if j > 0:
            sumatoria += (1 / j) * fj
            
    CC = 1 - (sumatoria / N)

    return {
        "Grupo": nombre_grupo,
        "Documentos (N)": N,
        "NCA": NCA,
        "SA": SA,
        "CA": CA,
        "CI": round(CI, 2),
        "CC": round(CC, 2)
    }

# --- 4. EJECUCIÓN PRINCIPAL ---
print("Cargando datos...")
try:
    df = pd.read_csv(archivo, low_memory=False)
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error leyendo CSV: {e}")
    exit()

print("Clasificando (VR / AR / Mixed)...")
df['Category'] = df.apply(classify_paper, axis=1)

# Separar DataFrames
df_vr = df[df['Category'] == 'VR_Only']
df_ar = df[df['Category'] == 'AR_Only']
df_mixed = df[df['Category'] == 'Mixed_Other']

print("\nCalculando métricas de colaboración...")
# Calcular para cada grupo
metricas_vr = calcular_metricas_grupo(df_vr, "VR Only")
metricas_ar = calcular_metricas_grupo(df_ar, "AR Only")
metricas_mixed = calcular_metricas_grupo(df_mixed, "Mixed/Other")

# --- 5. RESULTADO FINAL (DataFrame Comparativo) ---
resultados = pd.DataFrame([metricas_vr, metricas_ar, metricas_mixed])

# Reordenar columnas para que quede elegante
orden_cols = ["Grupo", "Documentos (N)", "NCA", "SA", "CA", "CI", "CC"]
resultados = resultados[orden_cols]

# Mostrar en consola
print("\n=== TABLA COMPARATIVA DE COLABORACIÓN ===")
print(resultados.to_string(index=False))
print("=========================================")

# Exportar si quieres
resultados.to_csv("comparativa_colaboracion.csv", index=False)
print("Tabla guardada como 'comparativa_colaboracion.csv'")