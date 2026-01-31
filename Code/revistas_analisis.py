import pandas as pd

# --- 1. CONFIGURACIÓN ---
archivo_entrada = 'vr.csv'
# Nombres posibles de columna en Scopus/WoS
col_journal_posibles = ['Source title', 'Source Title', 'Journal', 'Publication Title', 'SO'] 
col_citas = 'Cited by' 

# --- 2. CARGA DE DATOS ---
try:
    df = pd.read_csv(archivo_entrada, engine='python')
    
    # Buscar la columna correcta de la revista
    col_journal = next((c for c in col_journal_posibles if c in df.columns), None)
    
    if not col_journal:
        print(f"❌ Error: No encuentro la columna de Revista. Tus columnas son: {list(df.columns)}")
        exit()
        
    # Limpieza: Convertir citas a números
    df[col_citas] = pd.to_numeric(df[col_citas], errors='coerce').fillna(0)
    # Limpieza: Nombres de revista (quitar espacios, etc.)
    df[col_journal] = df[col_journal].astype(str).str.strip()
    
    print(f"✅ Analizando revistas en: '{col_journal}'...")

except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()

# --- 3. CÁLCULO DE MÉTRICAS POR REVISTA ---
# Agrupamos por revista
grouped = df.groupby(col_journal)

resultados = []

for journal, group in grouped:
    # 1. Artículos
    num_articles = len(group)
    
    # 2. Total Citas
    total_citas = group[col_citas].sum()
    
    # 3. Cálculo de H-Index Local
    # Ordenamos citas de mayor a menor
    citas_ordenadas = sorted(group[col_citas].tolist(), reverse=True)
    h_index = 0
    for i, citas in enumerate(citas_ordenadas):
        if citas >= i + 1:
            h_index = i + 1
        else:
            break
            
    resultados.append({
        'Source': journal,
        'Articles': num_articles,
        'H-index (Local)': h_index,
        'Total Citations': int(total_citas)
    })

# --- 4. CREAR DATAFRAME Y ORDENAR ---
df_journals = pd.DataFrame(resultados)

# Filtramos revistas con "nan" o vacías si las hay
df_journals = df_journals[df_journals['Source'] != 'nan']

# Ordenar: Primero por Artículos, luego por H-index
df_journals = df_journals.sort_values(by=['Articles', 'H-index (Local)'], ascending=[False, False])

# Nos quedamos con el TOP 10 (o 15)
top_journals = df_journals.head(10)

# --- 5. IMPRIMIR TABLA FORMATO PAPER ---
print("\n" + "="*95)
print(f"{'Source (Journal Name)':<50} | {'Articles':<10} | {'H-index':<10} | {'Total Citas':<12}")
print("="*95)

for index, row in top_journals.iterrows():
    # Cortar nombre si es muy largo para que quepa
    nombre = (row['Source'][:47] + '..') if len(row['Source']) > 47 else row['Source']
    print(f"{nombre:<50} | {row['Articles']:<10} | {row['H-index (Local)']:<10} | {row['Total Citations']:<12}")

print("="*95)
print("\n⚠️ NOTA: Las columnas 'Q' (Cuartil) y 'SJR' no vienen en el CSV.")
print("   Debes buscarlas manualmente en https://www.scimagojr.com/ usando el nombre de la revista.")