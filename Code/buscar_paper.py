import pandas as pd

# --- CONFIGURACIÓN ---
archivo = 'vr.csv'
busqueda = "Radianti" # Nombre del autor o palabra clave del título
columna_busqueda = 'Authors' # Donde buscar (Authors, Title, etc.)

# --- 1. FUNCIÓN DE CLASIFICACIÓN (La misma que usas en los gráficos) ---
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

# --- 2. BÚSQUEDA Y DIAGNÓSTICO ---
try:
    df = pd.read_csv(archivo, low_memory=False)
    # Limpiar nombres de columnas por si acaso
    df.columns = df.columns.str.strip()
    
    print(f"Buscando '{busqueda}' en la columna '{columna_busqueda}'...")
    
    # Búsqueda insensible a mayúsculas/minúsculas
    # Usamos astype(str) para evitar errores con valores vacíos (NaN)
    mask = df[columna_busqueda].astype(str).str.contains(busqueda, case=False, na=False)
    resultado = df[mask]
    
    print("\n" + "="*70)
    if not resultado.empty:
        print(f"✅ ¡ENCONTRADO! Hay {len(resultado)} coincidencia(s).")
        print("Vamos a ver cómo los clasifica el algoritmo:")
        print("="*70)
        
        for index, row in resultado.iterrows():
            # --- APLICAMOS LA CLASIFICACIÓN EN VIVO ---
            categoria_detectada = classify_paper(row)
            
            # Imprimimos resultados
            print(f"📄 Título: {row['Title'][:100]}...") # Cortamos título si es muy largo
            print(f"👤 Autores: {str(row.get('Authors', 'N/A'))[:80]}...")
            print(f"📅 Año: {row.get('Year', 'N/A')}")
            print(f"🔗 Citas: {row.get('Cited by', 'N/A')}")
            
            # EL VEREDICTO
            print(f"\n🏷️  CLASIFICACIÓN:  >>>  [{categoria_detectada}]  <<<")
            
            # Pequeño análisis de por qué
            text_debug = (str(row.get('Title','')) + " " + str(row.get('Author Keywords',''))).lower()
            if 'virtual' in text_debug: print("   (Detectado término: 'virtual')")
            if 'augmented' in text_debug: print("   (Detectado término: 'augmented')")
            if 'mixed' in text_debug: print("   (Detectado término: 'mixed')")
            
            print("-" * 70)
    else:
        print("❌ NO SE ENCONTRÓ EL PAPER.")
        print(f"Revisa si '{busqueda}' está bien escrito o prueba buscar por Título.")

except Exception as e:
    print(f"Error crítico: {e}")