import pandas as pd
from thefuzz import process, fuzz # pip install thefuzz

# --- 1. CONFIGURACIÓN ---
archivo = 'vr.csv'
col_de = 'Author Keywords'
col_id = 'Index Keywords' # Scopus suele usar este nombre para Keywords Plus

# MAPEO MANUAL: (Fundamental para que 'e-learning' suba en el ranking)
# Unificamos variaciones comunes en bibliometría educativa
manual_map = {
    'virtual reality (vr)': 'virtual reality',
    'vr': 'virtual reality',
    'augmented reality (ar)': 'augmented reality',
    'ar': 'augmented reality',
    'elearning': 'e-learning',
    'e learning': 'e-learning',
    'electronic learning': 'e-learning',
    'student': 'students',
    'computer aided instruction': 'computer-aided instruction',
    'learning system': 'learning systems'
}

# --- 2. CARGA DE DATOS ---
try:
    df = pd.read_csv(archivo, engine='python')
    print("✅ Archivo cargado.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 3. FUNCIÓN DE PROCESAMIENTO INTELIGENTE ---
def obtener_top_keywords(serie_raw, top_n=20): # Analizamos top 20 para luego recortar a 10
    # A. Limpieza básica
    palabras = serie_raw.dropna().astype(str).str.lower().str.split(';')
    palabras = palabras.explode().str.strip()
    palabras = palabras[palabras.str.len() > 2] # Eliminar basura corta
    
    # B. Aplicar Mapeo Manual (Paso 1)
    palabras = palabras.replace(manual_map)
    
    # C. Contar frecuencias iniciales
    conteo = palabras.value_counts()
    
    # D. Fusión Fuzzy (Paso 2) - Para agrupar lo que se nos haya pasado
    # Trabajamos con los top 200 para no hacer el proceso eterno
    candidates = conteo.head(200).index.tolist()
    merged_counts = {}
    processed = set()
    
    for term in candidates:
        if term in processed:
            continue
            
        # Buscar duplicados similares (ej: "medical education" y "medical  education")
        # Usamos ratio alto (90) para estar seguros
        matches = process.extract(term, candidates, limit=10, scorer=fuzz.ratio)
        
        group_total = 0
        primary_term = term # Por defecto el término actual es el principal
        
        for match_term, score in matches:
            if score >= 90 and match_term not in processed:
                # Sumamos la frecuencia del término similar
                group_total += conteo[match_term]
                processed.add(match_term)
                
                # Preferimos el término más largo o con guiones (e-learning vs elearning)
                if len(match_term) > len(primary_term) or '-' in match_term:
                    primary_term = match_term
        
        # Si ya procesamos este término via un match anterior, no lo sobreescribimos mal
        if group_total > 0:
            merged_counts[primary_term] = merged_counts.get(primary_term, 0) + group_total

    # Crear serie final y ordenar
    final_series = pd.Series(merged_counts).sort_values(ascending=False)
    return final_series, len(conteo) # Devolvemos serie y total único original

# --- 4. EJECUCIÓN ---
print("Procesando Author Keywords (DE)...")
top_de, total_de = obtener_top_keywords(df.get(col_de, pd.Series()))

print("Procesando Keywords Plus (ID)...")
top_id, total_id = obtener_top_keywords(df.get(col_id, pd.Series()))

# --- 5. GENERACIÓN DE TEXTO Y TABLA ---
print("\n" + "="*80)
print("TEXTO GENERADO:")
print(f"En este artículo nos encontramos con {total_de} (DE) y {total_id} (ID).")
if total_id > total_de:
    ratio = total_id / total_de
    print(f"Casi {ratio:.1f} veces más palabras clave generadas a partir de algoritmo (ID) que por autores (DE).")
print("="*80)

# Preparar DataFrame para visualizar
df_tabla = pd.DataFrame({
    'Author Keywords': top_de.head(10).index,
    'Freq. (DE)': top_de.head(10).values,
    '|': ['|']*10,
    'Keywords Plus': top_id.head(10).index,
    'Freq. (ID)': top_id.head(10).values
})

print("\nTable 5. Comparative data between Author Keywords (DE) and Keywords Plus (ID).")
print("-" * 95)
print(f"{'Author Keywords':<35} {'Freq.':<6} | {'Keywords Plus':<35} {'Freq.':<6}")
print("-" * 95)

for _, row in df_tabla.iterrows():
    kw_de = str(row['Author Keywords'])
    fr_de = str(row['Freq. (DE)'])
    kw_id = str(row['Keywords Plus'])
    fr_id = str(row['Freq. (ID)'])
    
    # Truncar si es muy largo
    if len(kw_de) > 33: kw_de = kw_de[:31] + ".."
    if len(kw_id) > 33: kw_id = kw_id[:31] + ".."
    
    print(f"{kw_de:<35} {fr_de:<6} | {kw_id:<35} {fr_id:<6}")
print("-" * 95)