import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
from thefuzz import process, fuzz  # Importante: pip install thefuzz

# --- CONFIGURACIÓN DE LIMPIEZA ---
# Mapeo manual para términos conocidos que el algoritmo debe unificar sí o sí
MANUAL_MAP = {
    'virtual reality (vr)': 'virtual reality',
    'vr': 'virtual reality',
    'augmented reality (ar)': 'augmented reality',
    'ar': 'augmented reality',
    'elearning': 'e-learning',
    'e learning': 'e-learning',
    'electronic learning': 'e-learning',
    'student': 'students',
    'computer aided instruction': 'computer-aided instruction',
    'learning system': 'learning systems',
    'higher education': 'higher education',
    'simulation': 'simulation'
}

def read_keywords_from_csv(file_path):
    """Lee el CSV y extrae las palabras clave crudas (raw)."""
    try:
        df = pd.read_csv(file_path, engine='python') # Engine python es más robusto
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    # Buscar columna de keywords (Author Keywords)
    keyword_column = None
    for col in df.columns:
        if 'author' in col.lower() and 'keyword' in col.lower():
            keyword_column = col
            break
    
    if not keyword_column:
        print("No se encontró columna de Keywords.")
        return []

    print(f"Using column: {keyword_column}")
    
    all_paper_keywords = []
    
    for _, row in df.iterrows():
        if pd.notna(row[keyword_column]):
            # Separar por punto y coma, limpiar espacios y minúsculas
            keywords = [k.strip().lower() for k in str(row[keyword_column]).split(';')]
            keywords = [k for k in keywords if len(k) > 2] # Ignorar basura corta
            if keywords:
                all_paper_keywords.append(keywords)
    
    return all_paper_keywords

def clean_keywords_list(raw_papers_list):
    """
    Aplica limpieza manual y Fuzzy Matching a toda la lista de papers.
    Devuelve la lista de papers con las palabras corregidas.
    """
    print("\n--- INICIANDO LIMPIEZA Y UNIFICACIÓN DE TÉRMINOS ---")
    
    # 1. Aplanar lista para contar frecuencias globales
    all_flat = list(itertools.chain(*raw_papers_list))
    
    # 2. Aplicar MAPEO MANUAL primero (rápido)
    mapped_flat = [MANUAL_MAP.get(k, k) for k in all_flat]
    
    # 3. Contar frecuencias tras el mapeo manual
    counts = Counter(mapped_flat)
    
    # 4. PREPARAR FUZZY MATCHING
    # Solo intentamos fusionar términos que aparecen en el Top 200 (para eficiencia)
    top_candidates = [word for word, count in counts.most_common(200)]
    
    fuzzy_map = {} # Diccionario final de traducción {sucio: limpio}
    processed = set()
    
    print("Aplicando lógica difusa (esto puede tardar unos segundos)...")
    
    for term in top_candidates:
        if term in processed: continue
            
        # Buscar variaciones (ej: "medical education" vs "medical-education")
        matches = process.extract(term, top_candidates, limit=10, scorer=fuzz.ratio)
        
        primary_term = term
        group = []
        
        for match_term, score in matches:
            if score >= 90: # Umbral alto de similitud
                group.append(match_term)
                # Preferimos términos con guiones o más largos (e-learning > elearning)
                if len(match_term) > len(primary_term) or '-' in match_term:
                    primary_term = match_term
        
        # Registrar en el mapa y marcar como procesados
        for g in group:
            fuzzy_map[g] = primary_term
            processed.add(g)
            
    # 5. APLICAR TRADUCCIÓN FINAL A LA LISTA DE PAPERS
    cleaned_papers = []
    for paper in raw_papers_list:
        new_paper = []
        for k in paper:
            # Paso 1: Manual
            k_step1 = MANUAL_MAP.get(k, k)
            # Paso 2: Fuzzy (si existe en el mapa, si no, se deja igual)
            k_final = fuzzy_map.get(k_step1, k_step1)
            
            # Evitar duplicados en el mismo paper tras la fusión
            if k_final not in new_paper:
                new_paper.append(k_final)
        
        cleaned_papers.append(new_paper)
        
    return cleaned_papers

def create_cooccurrence_matrix(keywords_lists, top_n=10):
    """Crea la matriz basada en los datos YA LIMPIOS."""
    # Contar frecuencias de palabras limpias
    all_keywords = list(itertools.chain(*keywords_lists))
    keyword_counts = Counter(all_keywords)
    
    print(f"\nTop {top_n} Keywords (Unificadas):")
    top_keywords = []
    for k, c in keyword_counts.most_common(top_n):
        print(f"- {k}: {c}")
        top_keywords.append(k)
    
    # Inicializar matriz
    matrix = np.zeros((len(top_keywords), len(top_keywords)))
    
    # Llenar matriz
    for paper in keywords_lists:
        # Solo nos interesan las que están en el Top N
        present = [k for k in paper if k in top_keywords]
        
        # Doble bucle para contar pares
        for i in range(len(present)):
            for j in range(len(present)):
                if i != j: # Opcional: i != j si no quieres diagonal (aquí la queremos para total)
                    k1, k2 = present[i], present[j]
                    idx1 = top_keywords.index(k1)
                    idx2 = top_keywords.index(k2)
                    matrix[idx1, idx2] += 1
    
    # La matriz es simétrica, pero al iterar doble se llena sola, solo hay que ajustar la diagonal
    # Normalmente la diagonal es el conteo total del término
    for i, k in enumerate(top_keywords):
        matrix[i, i] = keyword_counts[k]

    return pd.DataFrame(matrix, index=top_keywords, columns=top_keywords)

def visualize_cooccurrence_matrix(cooccurrence_df):
    plt.figure(figsize=(12, 10))
    
    # Máscara para quitar la diagonal (opcional, aquí la dejamos para ver totales)
    # mask = np.zeros_like(cooccurrence_df); np.fill_diagonal(mask, 1)
    
    ax = sns.heatmap(
        cooccurrence_df,
        annot=True,
        annot_kws={"size": 12, "weight": "bold"}, # Números negrita
        cmap="YlGnBu",
        fmt=".0f",
        linewidths=1,
        linecolor='white',
        cbar_kws={"label": "Frecuencia de Co-ocurrencia"}
    )
    
    plt.xticks(fontsize=13, rotation=45, ha='right') 
    plt.yticks(fontsize=13, rotation=0)
    plt.title("Keyword Co-occurrence Matrix (Top Terms)", fontsize=16, pad=20)
    
    plt.tight_layout()
    return plt

# --- EJECUCIÓN ---
if __name__ == "__main__":
    file_path = 'vr.csv'
    
    # 1. Leer
    raw_keywords = read_keywords_from_csv(file_path)
    
    if raw_keywords:
        # 2. LIMPIAR (Aquí ocurre la magia)
        clean_keywords = clean_keywords_list(raw_keywords)
        
        # 3. Crear Matriz (Top 8 o 10)
        # Sube top_n si quieres ver más (ej. 10 para ver si entra medical education)
        df_matrix = create_cooccurrence_matrix(clean_keywords, top_n=10) 
        
        # 4. Visualizar
        plt = visualize_cooccurrence_matrix(df_matrix)
        plt.savefig('matrix_cooccurrence_clean.png', dpi=300)
        plt.show()
        
        print("\nMatriz generada y guardada como 'matrix_cooccurrence_clean.png'")