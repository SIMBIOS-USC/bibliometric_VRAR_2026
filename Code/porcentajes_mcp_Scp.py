import pandas as pd
import numpy as np

# Datos extraídos manualmente de tu gráfica (Top 10)
data = {
    "Country": ["USA", "China", "UK", "Spain", "Taiwan", "Germany", "South Korea", "Canada", "Australia", "Italy"],
    "SCP": [2589, 1011, 599, 550, 409, 412, 368, 296, 317, 234],
    "MCP": [265, 162, 104, 45, 81, 44, 44, 72, 41, 39]
}

df = pd.DataFrame(data)

# Calcular Total
df['Total'] = df['SCP'] + df['MCP']

# Calcular Porcentajes
df['SCP_Pct'] = (df['SCP'] / df['Total']) * 100
df['MCP_Pct'] = (df['MCP'] / df['Total']) * 100

# Calcular Estadísticas Descriptivas del ratio de Colaboración Internacional (MCP)
media_mcp = df['MCP_Pct'].mean()
desviacion_mcp = df['MCP_Pct'].std() # Desviación típica muestral

# Mostrar Tabla y Estadísticas
print(df[['Country', 'Total', 'SCP_Pct', 'MCP_Pct']].round(2))
print("-" * 30)
print(f"Media de Colaboración Int. (MCP%): {media_mcp:.2f}%")
print(f"Desviación Típica (Std Dev): {desviacion_mcp:.2f}")