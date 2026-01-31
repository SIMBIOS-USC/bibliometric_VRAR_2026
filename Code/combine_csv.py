import pandas as pd
import os
from datetime import datetime

def combinar_csv(archivos, output_nombre='vr.csv', axis=0):
    """
    Combina múltiples archivos CSV en uno nuevo
    
    Args:
        archivos (list): Lista con rutas de los archivos CSV
        output_nombre (str): Nombre del archivo de salida
        axis (int): 0 para unión vertical, 1 para horizontal
    
    Returns:
        str: Ruta del archivo combinado
    """
    
    try:
        # Validación inicial
        if len(archivos) < 2:
            raise ValueError("Se necesitan al menos 2 archivos para combinar")
            
        if not all(isinstance(archivo, str) for archivo in archivos):
            raise TypeError("Las rutas deben ser cadenas de texto")

        # Cargar archivos con verificación
        dataframes = []
        for archivo in archivos:
            if not os.path.exists(archivo):
                raise FileNotFoundError(f"Archivo no encontrado: {archivo}")
                
            if os.stat(archivo).st_size == 0:
                print(f"Advertencia: {archivo} está vacío")
                continue
                
            df = pd.read_csv(archivo, engine='python', on_bad_lines='warn')
            dataframes.append(df)
            print(f"✓ {archivo} cargado ({len(df)} registros)")

        # Combinar archivos
        if axis == 0:
            # Unión vertical (por filas)
            df_final = pd.concat(dataframes, axis=axis, ignore_index=True)
            
            # Eliminar duplicados exactos
            inicial = len(df_final)
            df_final = df_final.drop_duplicates()
            final = len(df_final)
            
            print(f"\nDuplicados eliminados: {inicial - final}")
            
        elif axis == 1:
            # Unión horizontal (por columnas)
            df_final = pd.concat(dataframes, axis=axis)
            
            # Eliminar columnas duplicadas
            df_final = df_final.loc[:,~df_final.columns.duplicated()]
            
        else:
            raise ValueError("Axis debe ser 0 (vertical) o 1 (horizontal)")

        # Generar nombre único con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_base, extension = os.path.splitext(output_nombre)
        output_final = f"{nombre_base}_{timestamp}{extension}"
        
        # Guardar archivo
        df_final.to_csv(output_final, index=False, encoding='utf-8-sig')
        print(f"\n✅ Combinación exitosa\nArchivo guardado como: {output_final}")
        print(f"Registros totales: {len(df_final)}")
        print(f"Columnas totales: {len(df_final.columns)}")
        
        return output_final

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    archivos_a_combinar = [
        'vr1.csv',
        'vr2.csv',
    ]
    
    # Combinar verticalmente (apilar registros)
    archivo_final = combinar_csv(
        archivos_a_combinar,
        output_nombre='vr.csv',
        axis=0
    )
    
    # Para combinar horizontalmente (unir columnas)
    # archivo_final = combinar_csv(archivos_a_combinar, axis=1)
