import pandas as pd
import textwrap

# --- 1. CONFIGURACIÓN ---
archivo_csv = 'vr.csv'
col_citas = 'Cited by'   
col_titulo = 'Title'
col_autores = 'Authors'
col_anio = 'Year'
TOP_N = 5  # Los 5 más citados por grupo

# --- 2. FUNCIÓN DE CLASIFICACIÓN "ANTIVIRUS" (Surgical Filter) ---
def classify_paper(row):
    title = str(row.get(col_titulo, '')).strip()
    abstract = str(row.get('Abstract', ''))
    auth_kw = str(row.get('Author Keywords', ''))
    text = f"{title} {abstract} {auth_kw}".lower()
    
    # EXCLUSIONES RADICALES (IA, Economía, Agentes)
    # Si contiene esto y NO menciona hardware, se descarta.
    exclusions = [
        'agent-mediated', 'multi-agent', 'auction', 'negotiation', 
        'market mechanism', 'mixed integer', 'mixed model', 'mixed method'
    ]
    hard_xr_hardware = ['hololens', 'magic leap', 'oculus', 'htc vive', 'head-mounted', ' hmd ', 'smart glasses']
    
    if any(ex in text for ex in exclusions) and not any(hw in text for hw in hard_xr_hardware):
        return None

    # DICCIONARIOS (Basados en el Continuo de Milgram)
    mr_terms = ['mixed reality', 'extended reality', 'spatial computing', 'hololens', 'magic leap', ' xr ', ' mr ']
    ar_terms = ['augmented reality', ' ar ', 'smart glasses', ' heads-up ', ' mobile ar']
    vr_terms = ['virtual reality', ' virtual environment', ' vr ', 'immersive', 'hmd', 'oculus', 'vive']
    
    has_mr = any(term in text for term in mr_terms)
    has_ar = any(term in text for term in ar_terms)
    has_vr = any(term in text for term in vr_terms)
    
    # LÓGICA DE CATEGORÍAS
    if has_mr or (has_vr and has_ar):
        return 'Mixed/Extended Reality'
    elif has_ar:
        return 'Augmented Reality'
    elif has_vr:
        return 'Virtual Reality'
    
    return None

# --- 3. FUNCIÓN DE RANKING ---
def imprimir_top_citados(df_grupo, nombre_cat):
    print("\n" + "█"*75)
    print(f" 🏆 TOP {TOP_N} MÁS CITADOS: {nombre_cat.upper()}")
    print(f" (Análisis sobre {len(df_grupo)} documentos válidos)")
    print("█"*75)
    
    if df_grupo.empty:
        print(" [!] No se encontraron documentos para esta categoría.")
        return

    # Ordenar y tomar el Top
    top = df_grupo.sort_values(by=col_citas, ascending=False).head(TOP_N)
    
    for i, (_, row) in enumerate(top.iterrows(), 1):
        citas = int(row[col_citas])
        titulo = row[col_titulo]
        anio = int(row[col_anio])
        # Autor principal
        primer_autor = str(row[col_autores]).split(';')[0].split(',')[0].strip()
        
        # Formatear título envuelto
        wrapper = textwrap.TextWrapper(width=70, initial_indent="      ", subsequent_indent="      ")
        titulo_envuelto = wrapper.fill(titulo)

        print(f"\n {i}. 🔥 [{citas} CITAS] - {primer_autor} et al. ({anio})")
        print(f"{titulo_envuelto}")
        print(" " + "—"*70)

# --- 4. EJECUCIÓN ---
try:
    print("Cargando datos y ejecutando limpieza...")
    df = pd.read_csv(archivo_csv, engine='python')
    df.columns = df.columns.str.strip()
    
    # --- LA GUILLOTINA (Borrado físico de Plaza/Agents antes de empezar) ---
    mask_plaza = (df[col_titulo].astype(str).str.contains("Competing agents", case=False, na=False)) | \
                 (df[col_autores].astype(str).str.contains("Plaza, E", case=False, na=False))
    
    if mask_plaza.sum() > 0:
        print(f"✂️  GUILLOTINA: {mask_plaza.sum()} papers de Plaza/Agentes eliminados.")
        df = df[~mask_plaza].copy()

    # Limpieza de nulos en citas y años
    df[col_citas] = pd.to_numeric(df[col_citas], errors='coerce').fillna(0)
    df[col_anio] = pd.to_numeric(df[col_anio], errors='coerce').fillna(0)

    # Clasificación Quirúrgica
    df['Category'] = df.apply(classify_paper, axis=1)
    df_clean = df.dropna(subset=['Category'])

    # Mostrar resultados
    categorias = ['Virtual Reality', 'Augmented Reality', 'Mixed/Extended Reality']
    for cat in categorias:
        df_cat = df_clean[df_clean['Category'] == cat]
        imprimir_top_citados(df_cat, cat)

    print("\n✅ Análisis de citación finalizado.")

except Exception as e:
    print(f"❌ Error: {e}")