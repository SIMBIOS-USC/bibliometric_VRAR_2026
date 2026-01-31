import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN Y FUNCIÓN DE CLASIFICACIÓN ---
archivo_entrada = 'vr.csv'

def classify_paper(row):
    # Obtener y limpiar los campos (manejando posibles valores nulos/NaN)
    title = str(row.get('Title', '')) if pd.notnull(row.get('Title')) else ""
    abstract = str(row.get('Abstract', '')) if pd.notnull(row.get('Abstract')) else ""
    keywords = str(row.get('Author Keywords', '')) if pd.notnull(row.get('Author Keywords')) else ""
    
    # LÓGICA "OR": Al unir todo en 'text', si el término está en CUALQUIERA de los campos, se encuentra.
    text = f"{title} {abstract} {keywords}".lower()
    
    # 1. Filtros de exclusión (La guillotina para limpiar ruido de agentes/matemáticas)
    exclusions = ['agent-mediated', 'multi-agent', 'auction', 'negotiation', 'market mechanism']
    hard_xr_hardware = ['hololens', 'magic leap', 'oculus', 'htc vive', 'head-mounted', ' hmd ', 'quest']
    
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr_hardware):
        return 'Excluded (Non-XR Agents)'

    # 2. Diccionarios Jerárquicos
    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', 'hololens', ' xr ', ' mr ', 'metaverse']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', 'heads-up display', ' hud ', 'mobile ar']
    vr_terms = ['virtual reality', ' vr ', 'immersive', 'virtual environment', 'head-mounted display', ' hmd ', 'oculus', 'cave']
    
    # 3. Detección de presencia
    has_mr = any(term in text for term in mr_terms)
    has_ar = any(term in text for term in ar_terms)
    has_vr = any(term in text for term in vr_terms)
    
    # 4. Asignación de Categoría (Prioridad: Mixed > AR > VR)
    if has_mr or (has_vr and has_ar): 
        return 'Mixed/Extended Reality'
    elif has_ar: 
        return 'Augmented Reality'
    elif has_vr: 
        return 'Virtual Reality'
    
    return 'Other / Unclassified'

# --- 2. EJECUCIÓN ---
try:
    df = pd.read_csv(archivo_entrada, low_memory=False)
    df.columns = df.columns.str.strip()
    df['Categoria_Final'] = df.apply(classify_paper, axis=1)

    print("=== RECUENTO CON BÚSQUEDA INTEGRAL (Título + Keywords + Abstract) ===")
    print(df['Categoria_Final'].value_counts())
    
    # Guardar resultados
    df.to_csv('papers_clasificados_v2.csv', index=False, encoding='utf-8-sig')
    print("\n✅ Archivo 'papers_clasificados_v2.csv' generado.")

except Exception as e:
    print(f"Error: {e}")