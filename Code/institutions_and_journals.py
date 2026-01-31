import pandas as pd

# =========================================================
# 1. CONFIGURACIÓN Y CLASIFICACIÓN
# =========================================================
archivo = 'vr.csv'

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

# =========================================================
# 2. LIMPIADOR DE INSTITUCIÓN INDIVIDUAL
# =========================================================
def clean_single_institution(inst_string):
    """Limpia UNA sola institución (sin ; de por medio)"""
    if not inst_string or len(inst_string) < 3: return None
    
    aff_lower = inst_string.lower()
    
    # --- DICCIONARIO DE GIGANTES (MAPPING MANUAL) ---
    mappings = {
        'toronto': 'Univ. of Toronto (CAN)',
        'harvard': 'Harvard Univ. (USA)',
        'stanford': 'Stanford Univ. (USA)',
        'washington': 'Univ. of Washington (USA)',
        'college london': 'UCL (UK)',
        'imperial college': 'Imperial College (UK)',
        'oxford': 'Univ. of Oxford (UK)',
        'cambridge': 'Univ. of Cambridge (UK)',
        'mit': 'MIT (USA)',
        'massachusetts institute': 'MIT (USA)',
        'johns hopkins': 'Johns Hopkins (USA)',
        'mayo clinic': 'Mayo Clinic (USA)',
        'copenhagen': 'Univ. of Copenhagen (DNK)',
        'københavns': 'Univ. of Copenhagen (DNK)',
        'rigshospitalet': 'Rigshospitalet (DNK)',
        'tsinghua': 'Tsinghua Univ. (CHN)',
        'zhejiang': 'Zhejiang Univ. (CHN)',
        'nanyang': 'Nanyang Tech. Univ. (SGP)',
        'nus': 'National Univ. Singapore (SGP)',
        'tokyo': 'Univ. of Tokyo (JPN)',
        'munich': 'Tech. Univ. Munich (DEU)',
        'barcelona': 'Univ. de Barcelona (ESP)',
        'seville': 'Univ. de Sevilla (ESP)',
        'sevilla': 'Univ. de Sevilla (ESP)',
        'san diego': 'UC San Diego (USA)', # <-- IMPORTANTE PARA AÑÓN
        'la jolla': 'UC San Diego (USA)',  # <-- IMPORTANTE PARA AÑÓN
        'ucsd': 'UC San Diego (USA)',
        'michigan': 'Univ. of Michigan (USA)',
        'ann arbor': 'Univ. of Michigan (USA)',
        'cornell': 'Cornell Univ. (USA)'
    }

    for key, val in mappings.items():
        if key in aff_lower: return val

    # --- ALGORITMO FALLBACK ---
    parts = [p.strip() for p in inst_string.split(',')]
    pais = parts[-1] if len(parts) > 0 else ""
    
    skip_terms = ['department', 'dept', 'faculty', 'school', 'college', 'division', 'unit', 'center', 'lab', 'service']
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
        
    return f"{best} ({pais})".replace(',', '')

# =========================================================
# 3. EXTRACTOR MÚLTIPLE (FULL COUNTING)
# =========================================================
def get_all_institutions(aff_column_value):
    if pd.isna(aff_column_value): return []
    
    # Dividimos por punto y coma para sacar TODAS las instituciones del paper
    raw_institutions = str(aff_column_value).split(';')
    
    clean_set = set()
    for raw in raw_institutions:
        cleaned = clean_single_institution(raw.strip())
        if cleaned:
            clean_set.add(cleaned)
            
    return list(clean_set)

# =========================================================
# 4. PROCESAMIENTO
# =========================================================
print("Procesando datos con técnica 'Full Counting' (Explode)...")
try:
    df = pd.read_csv(archivo, low_memory=False)
    df.columns = df.columns.str.strip()
    col_authors = 'Author full names' if 'Author full names' in df.columns else 'Authors'
    col_aff, col_cit = 'Affiliations', 'Cited by'
    col_source = 'Source title'

    # GUILLOTINA ANTI-PLAZA
    mask_plaza = (df['Title'].astype(str).str.contains("Competing agents", case=False)) | \
                 (df[col_authors].astype(str).str.contains("Plaza, E", case=False))
    df = df[~mask_plaza].copy()

    # Clasificación
    df['Category'] = df.apply(classify_paper, axis=1)
    df = df.dropna(subset=['Category', col_aff, col_source])
    df[col_cit] = pd.to_numeric(df[col_cit], errors='coerce').fillna(0)

    # ---------------------------------------------------------
    # PARTE 1: TABLAS DE INSTITUCIONES (CON EXPLODE)
    # ---------------------------------------------------------
    
    # Generamos una lista de instituciones limpias por fila
    df['Inst_List'] = df[col_aff].apply(get_all_institutions)
    
    # "EXPLOTAMOS" el dataframe: Si un paper tiene 3 instituciones, se convierte en 3 filas
    df_exploded = df.explode('Inst_List')
    
    # Filtramos nulos que hayan podido quedar tras la explosión
    df_exploded = df_exploded.dropna(subset=['Inst_List'])

    categories = ['Virtual Reality', 'Augmented Reality', 'Mixed/Extended Reality']

    print("\n" + "█"*105)
    print(f" 🏫 TOP INSTITUCIONES (CONTABILIDAD COMPLETA / FULL COUNTING)")
    print(" (Nota: Si un paper tiene colaboraciones, cuenta para ambas instituciones)")
    print("█"*105)
    print(f"{'Institution (Country)':<55} | {'Docs':<6} | {'Total Cit':<10} | {'CPP':<6}")
    print("="*105)

    for cat in categories:
        df_cat = df_exploded[df_exploded['Category'] == cat]
        
        # Agrupar por Institución Explotada
        stats = df_cat.groupby('Inst_List').agg(
            Documents=('Title', 'count'),       # Conteo total de apariciones
            Total_Citations=(col_cit, 'sum')    # Suma total de citas de esos papers
        )
        
        stats['CPP'] = (stats['Total_Citations'] / stats['Documents']).round(2)
        
        # Top 10
        top_10 = stats.sort_values(by=['Documents', 'Total_Citations'], ascending=[False, False]).head(10)
        
        print(f"\n🚀 {cat.upper()}")
        print("-" * 105)
        for inst, row in top_10.iterrows():
            inst_name = (inst[:52] + '..') if len(inst) > 52 else inst
            print(f"{inst_name:<55} | {int(row['Documents']):<6} | {int(row['Total_Citations']):<10} | {row['CPP']:<6}")
        print("-" * 105)

    # ---------------------------------------------------------
    # PARTE 2: TABLAS DE JOURNALS (SIN CAMBIOS, YA ESTABA BIEN)
    # ---------------------------------------------------------
    # Función H-index auxiliar
    def calc_h(cites):
        cites = sorted(cites, reverse=True)
        h = 0
        for i, c in enumerate(cites):
            if c >= i + 1: h = i + 1
            else: break
        return h

    print("\n\n" + "█"*105)
    print(f" 📖 TOP JOURNALS (IMPACTO CONTEXTUAL)")
    print("█"*105)
    print(f"{'Source':<65} | {'Arts':<6} | {'H-idx':<6}")
    print("="*105)

    for cat in categories:
        df_cat = df[df['Category'] == cat] # Usamos el DF original (sin explotar) para journals
        
        j_stats = df_cat.groupby(col_source)[col_cit].apply(list).reset_index(name='Cites')
        j_stats['Articles'] = j_stats['Cites'].apply(len)
        j_stats['H-index'] = j_stats['Cites'].apply(calc_h)
        
        top_j = j_stats.sort_values(by=['Articles', 'H-index'], ascending=[False, False]).head(10)
        
        print(f"\n🚀 {cat.upper()}")
        print("-" * 105)
        for _, row in top_j.iterrows():
            src = str(row[col_source])
            if len(src) > 62: src = src[:60] + ".."
            print(f"{src:<65} | {row['Articles']:<6} | {row['H-index']:<6}")
        print("-" * 105)

    print("\n✅ Análisis completado. Ahora las universidades colaboradoras sí reciben crédito.")

except Exception as e:
    print(f"❌ Error: {e}")