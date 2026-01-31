import pandas as pd
import collections

# --- 1. CONFIGURACIÓN ---
archivo_entrada = 'vr.csv'
columna_afiliacion = 'Affiliations'

# Diccionario de búsqueda (el mismo que ya usas)
keywords_paises = {
    "USA": ["united states", "usa", "u.s.a.", " ca ", " ny ", "california", "harvard", "mit", "stanford"],
    "China": ["china", "p.r. china", "hong kong", "beijing", "shanghai", "wuhan"],
    "UK": ["united kingdom", " uk ", "england", "london", "oxford", "cambridge"],
    "Germany": ["germany", "deutschland", "berlin", "munich"],
    "South Korea": ["south korea", "korea", "seoul"],
    "Italy": ["italy", "italia", "rome", "milan"],
    "Australia": ["australia", "sydney", "melbourne"],
    "Canada": ["canada", "toronto", "vancouver"],
    "Spain": ["spain", "españa", "madrid", "barcelona"],
    "Japan": ["japan", "tokyo", "kyoto"],
    "France": ["france", "paris", "lyon"],
    "Taiwan": ["taiwan", "taipei"]
}

def obtener_top_paises(archivo):
    try:
        df = pd.read_csv(archivo, low_memory=False)
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        if columna_afiliacion not in df.columns:
            print(f"Error: No se encontró la columna '{columna_afiliacion}'")
            return

        # Contador global de apariciones
        conteo_paises = collections.Counter()

        print("Analizando afiliaciones de todos los autores...")

        for index, row in df.iterrows():
            aff_string = str(row[columna_afiliacion]).lower()
            
            # Separamos por ';' para obtener cada afiliación individual
            partes_afiliacion = [p.strip() for p in aff_string.split(';')]
            
            # Para cada autor/afiliación en este paper
            paises_en_este_paper = set() # Usamos set si quieres contar 1 vez por país por paper
            # O usa una lista si quieres contar CADA vez que un autor de ese país aparece
            
            for aff in partes_afiliacion:
                for pais, keywords in keywords_paises.items():
                    if any(kw in aff for kw in keywords):
                        # Si quieres contar 1 vez por autor (aunque sean del mismo país):
                        conteo_paises[pais] += 1
                        # Si prefieres contar "Presencia del país en el paper", usa:
                        # paises_en_este_paper.add(pais)
            
            # Si activaste el set, descomenta esto:
            # for p in paises_en_este_paper: conteo_paises[p] += 1

        # --- 2. RESULTADOS ---
        resultado = pd.Series(conteo_paises).sort_values(ascending=False)
        
        print("\n" + "="*30)
        print(" TOP 5 PAÍSES (TODOS LOS AUTORES) ")
        print("="*30)
        for i, (pais, total) in enumerate(resultado.head(5).items(), 1):
            print(f"{i}. {pais}: {total} apariciones de autores")
        print("="*30)

    except Exception as e:
        print(f"Error al procesar: {e}")

# Ejecutar
obtener_top_paises(archivo_entrada)