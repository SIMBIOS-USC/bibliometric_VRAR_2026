import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap # Para cortar títulos largos elegantemente

# --- 1. CONFIGURACIÓN ---
archivo = 'vr.csv'
# Revisa que estos nombres coincidan con tu CSV
columna_citas = 'Cited by'  
columna_titulo = 'Title'
columna_autores = 'Authors'
columna_anio = 'Year'

# Configuración Estética
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'

# --- 2. CARGA Y CÁLCULOS ---
try:
    df = pd.read_csv(archivo, engine='python')
    # Limpieza
    df[columna_citas] = pd.to_numeric(df[columna_citas], errors='coerce').fillna(0)
    citations = df[columna_citas]
except Exception as e:
    print(f"Error: {e}")
    exit()

# ESTADÍSTICAS GENERALES
total_docs = len(df)
total_citations = citations.sum()
avg_cit = citations.mean()
max_cit = citations.max()
count_zero = len(df[df[columna_citas] == 0])
pct_zero = (count_zero / total_docs) * 100

# --- 3. DETECTAR EL ARTÍCULO TOP (EL MÁS CITADO) ---
# Encontramos el índice de la fila con el máximo valor
indice_max = df[columna_citas].idxmax()
fila_top = df.loc[indice_max]

# Extraemos los datos (usamos .get por si acaso falta alguna columna)
top_titulo = str(fila_top.get(columna_titulo, 'Unknown Title'))
top_autor = str(fila_top.get(columna_autores, 'Unknown Author')).split(';')[0] # Solo el primer autor
top_anio = str(int(fila_top.get(columna_anio, 0)))
top_citas = int(fila_top.get(columna_citas, 0))

# Imprimimos en consola los detalles completos
print("="*40)
print("ARTÍCULO MÁS CITADO ENCONTRADO:")
print(f"Título: {top_titulo}")
print(f"Autor Principal: {top_autor}")
print(f"Año: {top_anio}")
print(f"Citas: {top_citas}")
print("="*40)

# Preparamos el título corto para la gráfica (para que quepa en el cuadro)
# Si es muy largo, lo cortamos a 50 caracteres y añadimos "..."
titulo_corto = textwrap.shorten(top_titulo, width=50, placeholder="...")

# --- 4. CREAR EL TEXTO PARA LA GRÁFICA ---
stats_text = (
    f"=== Citation Metrics ===\n\n"
    f"• Total Documents: {total_docs}\n"
    f"• Total Citations: {int(total_citations)}\n"
    f"• Avg. Citations/Doc: {avg_cit:.2f}\n"
    f"• Zero Citations: {count_zero} ({pct_zero:.2f}%)\n\n"
    f"=== Most Cited Paper ===\n"
    f"\"{titulo_corto}\"\n"
    f"({top_autor} et al., {top_anio})\n"
    f"Citations: {top_citas}"
)

# --- 5. GENERACIÓN DE LA GRÁFICA ---
plt.figure(figsize=(12, 7))

# Histograma
ax = sns.histplot(
    data=df, 
    x=columna_citas, 
    bins=100,
    color="#6a5acd", # SlateBlue
    edgecolor="white",
    linewidth=0.5,
    alpha=0.8,
    kde=False
)

# Títulos y Etiquetas
ax.set_title("Distribution of Citations & Impact Analysis", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Number of Citations", fontsize=18)
ax.set_ylabel("Frequency (Number of Articles)", fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
# Limpieza visual
sns.despine()

# Insertar el cuadro de texto con el TOP PAPER
plt.text(
    0.95, 0.95, 
    stats_text, 
    transform=ax.transAxes, 
    fontsize=16, 
    verticalalignment='top', 
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="#cccccc", alpha=0.95)
)

# --- 6. GUARDAR ---
nombre_salida = 'citation_distribution_with_top_paper.png'
plt.tight_layout()
plt.savefig(nombre_salida, dpi=300)
print(f"\nGráfica guardada como: {nombre_salida}")
# plt.show()
