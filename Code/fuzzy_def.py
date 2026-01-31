import pandas as pd
from thefuzz import fuzz

# --- 1. CONFIGURACIÓN ---
archivo_entrada = 'vr.csv'
columna_afiliacion = 'Affiliations'
columna_citas = 'Cited by'  # <--- CONFIRMA QUE ESTE ES EL NOMBRE EN TU CSV
umbral_fuzzy = 90 

# LISTA 1: INSTITUCIONES REALES
objetivos_instituciones = {
    "University of Toronto": ["university of toronto", "univ of toronto", "toronto general hospital", "sickchildren", "sunnybrook"],
    "Imperial College London": ["imperial college london", "imperial college", "st mary's hospital"],
    "Harvard Medical School": ["harvard medical school", "harvard university", "massachusetts general hospital", "brigham and women's hospital"],
    "Rigshospitalet": ["rigshospitalet", "copenhagen university hospital", "univ. of copenhagen"],
    "University of Washington": ["university of washington", "univ. of washington", "seattle children"],
    "Københavns Universitet": ["university of copenhagen", "kobenhavns universitet", "faculty of health and medical sciences"],
    "Stanford University": ["stanford university", "stanford school of medicine"]
}

# LISTA 2: TOP 5 PAÍSES
keywords_paises = {
    "USA": ["united states", "usa", "u.s.a.", " ca ", " ny ", " ma ", "california", "massachusetts", "washington", "stanford"],
    "UK": ["united kingdom", " uk ", "england", "scotland", "wales", "london", "oxford", "cambridge"],
    "Canada": ["canada", "toronto", "ontario", "montreal", "vancouver", "british columbia"],
    "Germany": ["germany", "deutschland", "berlin", "munich", "heidelberg"],
    "China": ["china", "p.r. china", "hong kong", "beijing", "shanghai"]
}

# --- 2. CARGA DE DATOS ---
try:
    df = pd.read_csv(archivo_entrada, engine='python')
    
    # Pre-procesamiento de texto
    df[columna_afiliacion] = df[columna_afiliacion].astype(str).str.lower()
    
    # --- NUEVO: LIMPIEZA DE CITAS ---
    # Convertimos a numérico, forzando errores a NaN y luego rellenando con 0
    df[columna_citas] = pd.to_numeric(df[columna_citas], errors='coerce').fillna(0)
    
    print(f"Archivo cargado. Procesando {len(df)} filas...\n")
except Exception as e:
    print(f"Error cargando CSV: {e}")
    exit()

# --- 3. INICIALIZACIÓN DE CONTADORES ---
conteo_instituciones = {k: 0 for k in objetivos_instituciones.keys()}
citas_instituciones = {k: 0 for k in objetivos_instituciones.keys()} # <--- NUEVO CONTADOR

conteo_paises_total = {k: 0 for k in keywords_paises.keys()}
conteo_paises_lider = {k: 0 for k in keywords_paises.keys()}

# --- 4. PROCESAMIENTO ---
for index, row in df.iterrows():
    texto_completo = row[columna_afiliacion]
    num_citas = row[columna_citas] # <--- CAPTURAMOS LAS CITAS DEL PAPER
    
    if texto_completo == "nan": 
        continue

    fragmentos = texto_completo.split(';')

    # --- A. INSTITUCIONES ---
    instituciones_encontradas = set()
    
    for fragmento in fragmentos:
        fragmento_limpio = fragmento.strip()
        for inst_oficial, pistas in objetivos_instituciones.items():
            # 1. Coincidencia exacta
            if any(pista in fragmento_limpio for pista in pistas):
                instituciones_encontradas.add(inst_oficial)
            # 2. Fuzzy match
            elif fuzz.token_set_ratio(inst_oficial.lower(), fragmento_limpio) >= umbral_fuzzy:
                instituciones_encontradas.add(inst_oficial)
    
    # Sumar contadores
    for inst in instituciones_encontradas:
        conteo_instituciones[inst] += 1
        citas_instituciones[inst] += num_citas # <--- SUMAMOS LAS CITAS AQUÍ

    # --- B. PAÍSES ---
    paises_en_paper = set()
    for pais, keywords in keywords_paises.items():
        if any(k in texto_completo for k in keywords):
            paises_en_paper.add(pais)
            
    for pais in paises_en_paper:
        conteo_paises_total[pais] += 1

    # --- C. LIDERAZGO ---
    if len(fragmentos) > 0:
        primera_afiliacion = fragmentos[0].strip()
        for pais, keywords in keywords_paises.items():
            if any(k in primera_afiliacion for k in keywords):
                conteo_paises_lider[pais] += 1
                break

# --- 5. RESULTADOS ---
print("\n" + "="*80)
print(f"{'INSTITUCIÓN':<35} | {'DOCS':<5} | {'CITAS TOTALES':<13} | {'CITAS/DOC':<9}")
print("="*80)

# Ordenamos por número de documentos (o cambia a citas_instituciones[inst] para ordenar por impacto)
ranking_ordenado = sorted(conteo_instituciones.items(), key=lambda x: x[1], reverse=True)

for inst, cuenta in ranking_ordenado:
    total_citas = int(citas_instituciones[inst])
    # Calculamos promedio citas por paper (CPP)
    cpp = total_citas / cuenta if cuenta > 0 else 0
    
    print(f"{inst:<35} | {cuenta:<5} | {total_citas:<13} | {cpp:.2f}")

print("\n" + "="*60)
print("2. PARTICIPACIÓN POR PAÍS (Total Papers)")
print("="*60)
for pais, cuenta in sorted(conteo_paises_total.items(), key=lambda x: x[1], reverse=True):
    print(f"{pais:<15} | {cuenta}")

print("\n" + "="*60)
print("3. LIDERAZGO CIENTÍFICO (Corresponding/1st Author)")
print("="*60)
for pais, cuenta in sorted(conteo_paises_lider.items(), key=lambda x: x[1], reverse=True):
    tasa = (cuenta / conteo_paises_total[pais] * 100) if conteo_paises_total[pais] > 0 else 0
    print(f"{pais:<15} | {cuenta} (Lidera el {tasa:.1f}% de sus papers)")
    
    
    
    
import pandas as pd
from thefuzz import fuzz

# --- 1. CONFIGURACIÓN ---
archivo_entrada = 'vr.csv'
columna_afiliacion = 'Affiliations'
columna_citas = 'Cited by'  # <--- IMPORTANTE: Revisa que este sea el nombre en tu CSV
umbral_fuzzy = 90 

# DICCIONARIO DE BÚSQUEDA (Tus instituciones clave)
objetivos_instituciones = {
    "University of Toronto": ["university of toronto", "univ of toronto", "toronto general hospital", "sickchildren", "sunnybrook"],
    "Imperial College London": ["imperial college london", "imperial college", "st mary's hospital"],
    "Harvard Medical School": ["harvard medical school", "harvard university", "massachusetts general hospital", "brigham and women's hospital"],
    "Rigshospitalet": ["rigshospitalet", "copenhagen university hospital", "univ. of copenhagen"],
    "University of Washington": ["university of washington", "univ. of washington", "seattle children"],
    "Københavns Universitet": ["university of copenhagen", "kobenhavns universitet", "faculty of health and medical sciences"],
    "Stanford University": ["stanford university", "stanford school of medicine"]
}

# --- 2. CARGA Y LIMPIEZA ---
try:
    df = pd.read_csv(archivo_entrada, engine='python')
    
    # Limpieza de Texto
    df[columna_afiliacion] = df[columna_afiliacion].astype(str).str.lower()
    
    # LIMPIEZA DE CITAS (CRÍTICO PARA QUE SUME BIEN)
    # Convierte la columna a números. Si hay errores o vacíos, pone un 0.
    df[columna_citas] = pd.to_numeric(df[columna_citas], errors='coerce').fillna(0)
    
    print(f"✅ Archivo cargado. Analizando {len(df)} documentos...\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# --- 3. INICIALIZAR ACUMULADORES ---
# Usamos un diccionario para guardar 'docs' y 'citas' de cada universidad
stats = {k: {'docs': 0, 'citas': 0} for k in objetivos_instituciones.keys()}

# --- 4. PROCESAMIENTO FILA POR FILA ---
for index, row in df.iterrows():
    texto_afiliacion = row[columna_afiliacion]
    citas_paper = row[columna_citas] # Citas de este paper específico
    
    if texto_afiliacion == "nan": 
        continue

    fragmentos = texto_afiliacion.split(';')

    # Detectar qué instituciones están en este paper
    # Usamos un set() para no contar dos veces la misma uni en el mismo paper
    instituciones_en_este_paper = set()
    
    for fragmento in fragmentos:
        frag_limpio = fragmento.strip()
        for inst_oficial, pistas in objetivos_instituciones.items():
            # 1. Búsqueda exacta
            if any(pista in frag_limpio for pista in pistas):
                instituciones_en_este_paper.add(inst_oficial)
            # 2. Búsqueda aproximada (Fuzzy)
            elif fuzz.token_set_ratio(inst_oficial.lower(), frag_limpio) >= umbral_fuzzy:
                instituciones_en_este_paper.add(inst_oficial)
    
    # Sumar al total de cada institución encontrada
    for inst in instituciones_en_este_paper:
        stats[inst]['docs'] += 1          # Sumamos 1 artículo
        stats[inst]['citas'] += citas_paper # Sumamos las citas de este artículo

# --- 5. RESULTADOS (TABLA FINAL) ---

# Crear una lista para ordenar los datos
tabla_final = []
for inst, data in stats.items():
    docs = data['docs']
    citas = int(data['citas'])
    # Calcular Citas por Paper (CPP)
    cpp = citas / docs if docs > 0 else 0
    tabla_final.append((inst, docs, citas, cpp))

# Ordenar por número de artículos (descendente)
tabla_final.sort(key=lambda x: x[1], reverse=True)

print("="*90)
print(f"{'INSTITUCIÓN':<35} | {'ARTÍCULOS':<10} | {'TOTAL CITAS':<12} | {'CITAS/ART (CPP)':<15}")
print("="*90)

for fila in tabla_final:
    # fila[0] = Nombre, fila[1] = Docs, fila[2] = Citas, fila[3] = CPP
    print(f"{fila[0]:<35} | {fila[1]:<10} | {fila[2]:<12} | {fila[3]:.2f}")
print("="*90)