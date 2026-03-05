#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- LIBRERÍAS QUE NECESITA EL CÓDIGO ----------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import random
import math
import pickle
import networkx as nx
import ast
import time as tm
import warnings
from datetime import datetime, time
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from sortedcontainers import SortedDict
import matplotlib.patches as mpatches



# PATH_ENTRADA_1  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma300\\'
# PATH_ENTRADA_2  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me300\\'

# PATH_ENTRADA_3  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma330\\'
# PATH_ENTRADA_4  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me330\\'

# PATH_ENTRADA_5  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma360\\'
# PATH_ENTRADA_6  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me360\\'

# PATH_ENTRADA_7  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma390\\'
# PATH_ENTRADA_8  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me390\\'

# PATH_ENTRADA_9  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma420\\'
# PATH_ENTRADA_10 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me420\\'

# PATH_ENTRADA_11 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma450\\'
# PATH_ENTRADA_12 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me450\\'

# # PATH_ENTRADA_13 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\Ma480\\'
# # PATH_ENTRADA_14 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\me480\\'

# Complejidad_celdas_1 = pd.read_pickle(PATH_ENTRADA_1 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_2 = pd.read_pickle(PATH_ENTRADA_2 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_3 = pd.read_pickle(PATH_ENTRADA_3 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_4 = pd.read_pickle(PATH_ENTRADA_4 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_5 = pd.read_pickle(PATH_ENTRADA_5 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_6 = pd.read_pickle(PATH_ENTRADA_6 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_7 = pd.read_pickle(PATH_ENTRADA_7 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_8 = pd.read_pickle(PATH_ENTRADA_8 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_9 = pd.read_pickle(PATH_ENTRADA_9 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_10 = pd.read_pickle(PATH_ENTRADA_10 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_11 = pd.read_pickle(PATH_ENTRADA_11 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# Complejidad_celdas_12 = pd.read_pickle(PATH_ENTRADA_12 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# # Complejidad_celdas_13 = pd.read_pickle(PATH_ENTRADA_13 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')
# # Complejidad_celdas_14 = pd.read_pickle(PATH_ENTRADA_14 + 'Complejidad_por_hora_2022-06-01_06-07.pkl')

# # Nombre común de los ficheros pickle
# FILE_NAME = 'Complejidad_por_hora_2022-06-01_06-07.pkl'

# # Umbrales para etiquetar pares (Ma / me)
# thresholds = [300, 330, 360, 390, 420, 450]  #480

# # 1) Leer todos los DataFrames en una lista
# paths = [
#     PATH_ENTRADA_1, PATH_ENTRADA_2,
#     PATH_ENTRADA_3, PATH_ENTRADA_4,
#     PATH_ENTRADA_5, PATH_ENTRADA_6,
#     PATH_ENTRADA_7, PATH_ENTRADA_8,
#     PATH_ENTRADA_9, PATH_ENTRADA_10,
#     PATH_ENTRADA_11, PATH_ENTRADA_12,
#     # PATH_ENTRADA_13, PATH_ENTRADA_14,
# ]





# # df = pd.read_pickle(PATH_ENTRADA_1 + FILE_NAME)
# # df_sorted = df.sort_values('Suma_Complejidad_total', ascending=False)

# # plt.figure(figsize=(10, 4))
# # plt.plot(df_sorted['Suma_Complejidad_total'].values)
# # plt.title("Distribución de complejidad por celda")
# # plt.xlabel("Celdas ordenadas")
# # plt.ylabel("Suma de complejidad total")
# # plt.grid(True)
# # plt.show()


# dfs = [pd.read_pickle(p + FILE_NAME) for p in paths]

# # 2) Calcular la complejidad total de cada uno
# totals = []
# for i, df in enumerate(dfs, start=1):
#     s = df['Suma_Complejidad_total'].sum()
#     totals.append(s)
#     print(f"Complejidad total en entrada {i}: {s:.2f}")


# # 3) Para cada umbral, comparar el par (MaX vs meX)
# differences = {}
# for idx, th in enumerate(thresholds):
#     above = totals[2*idx]    # índice par   → Ma<th>
#     below = totals[2*idx+1]  # índice impar → me<th>
#     diff = abs(above - below)
#     differences[th] = diff
#     print(f"Umbral {th}: |{above:.2f} - {below:.2f}| = {diff:.2f}")

# # 4) Encontrar el umbral con menor diferencia
# best_th = min(differences, key=differences.get)
# print(f"\n⇒ Umbral óptimo: {best_th} (diferencia mínima = {differences[best_th]:.2f})")





import os
import pandas as pd

# Ruta base
BASE_PATH = 'C:\TFG\Codigos Chema\Datos\3. bloque optimizacion\3. bloque optimizacion\Resultados eCOMMET\Complejidad real\mapas\COMPLETO\\'

# Nombre del archivo a comparar
FILE_NAME = 'Complejidad_por_hora_2022-06-01_06_07.pkl'

# Buscar carpetas MaXXX y meXXX
carpetas = sorted([f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))])

# Filtrar pares MaXXX y meXXX que tengan el archivo deseado
pares_validos = []
for nivel in range(305, 405, 10):
    ma = f'Ma20x20{nivel}'
    me = f'me20x20{nivel}'
    path_ma = os.path.join(BASE_PATH, ma, FILE_NAME)
    path_me = os.path.join(BASE_PATH, me, FILE_NAME)
    if os.path.exists(path_ma) and os.path.exists(path_me):
        pares_validos.append((nivel, path_ma, path_me))

# Calcular la diferencia de complejidad total en cada par
mejor_umbral = None
min_diferencia = float('inf')
for nivel, path_ma, path_me in pares_validos:
    df_ma = pd.read_pickle(path_ma)
    df_me = pd.read_pickle(path_me)
    total_ma = df_ma['Suma_Complejidad_total'].sum()
    total_me = df_me['Suma_Complejidad_total'].sum()
    diferencia = abs(total_ma - total_me)
    print(f"Umbral {nivel}: |{total_ma:.2f} - {total_me:.2f}| = {diferencia:.2f}")
    if diferencia < min_diferencia:
        min_diferencia = diferencia
        mejor_umbral = (nivel, path_ma, path_me)

if mejor_umbral:
    nivel_optimo, ruta_above, ruta_below = mejor_umbral
    print(f"\n✅ Umbral óptimo: {nivel_optimo} (diferencia mínima = {min_diferencia:.2f})")
else:
    raise ValueError("❌ No se encontró ningún par válido MaXXX / meXXX con los archivos necesarios.")





# 1) Configuración de rutas literales tal como las tienes
PATH_MALLADO_DATA = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'
PATH_SECTOR_DATA  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'


# 3) Configuración de sectores disponibles
configuraciones = {
    1: ('CNF1A', ['LECMR1I']),
    2: ('CNF2A', ['LECMDPI', 'LECMSAB']),
    3: ('CNF3A', ['LECMDGI', 'LECMPAI', 'LECMSAB']),
    4: ('CNF4A', ['LECMDGI', 'LECMBLI', 'LECMPAI', 'LECMSAI']),
    5: ('CNF5A', ['LECMASI', 'LECMBLI', 'LECMDGI', 'LECMPAI', 'LECMSAN']),
}
num   = int(input("Elige un número de sectores (1-5): ").strip())
_, lista_sectores = configuraciones[num]
configuracion_estudio, lista_sectores = configuraciones[num]

# 5) Carga de mallado y sectores
DF_MALLADO   = pd.read_pickle(PATH_MALLADO_DATA + 'dataset_flujos_por_celda.pkl')


#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- DATOS NECESARIOS PARA GRAFICAR ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# LECTURA DE LAS CONFIGURACIONES DEL ACC SELECCIONADO
config = pd.read_csv(PATH_SECTOR_DATA + 'config.txt',sep='\t', header=None)
config = config[0].str.split(';', expand=True)
ACC = config[0].iloc[0]
print('El ACC de la base de datos es', ACC)
config = config.rename(columns={1: 'CONFIG', 2: 'SECTORES'})

list_dataframes = [df for df in config.groupby('CONFIG', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    configuracion = df['CONFIG'].iloc[0]
    sectors = list(df['SECTORES'])
    df_temporal = pd.DataFrame({'CONFIG': [configuracion], 'SECTORES': [sectors]})
    dataframes_temporales.append(df_temporal)
CONFIG = pd.concat(dataframes_temporales, ignore_index=True)
del (config)


# LECTURA DE LA COMPOSICION DE LOS ESPACIOS AEREOS
airspaces = pd.read_csv(PATH_SECTOR_DATA + 'airspace.txt',sep='\t', header=None)
airspaces = airspaces.drop(airspaces.index[0])
airspaces = airspaces[0].str.split(';', expand=True)

rows = []
current_id = None
current_nombre = None
for index, row in airspaces.iterrows():
    if 'A' in row.iloc[0]:
        current_id = row.iloc[1]
        current_nombre = row.iloc[2]
        tipo = row.iloc[3]
        number = row.iloc[4]
        rows.append((current_id, current_nombre, tipo, number))
    else:
        rows.append((current_id, current_nombre, tipo, number, row.iloc[1]))

AIRSPACES = pd.DataFrame(rows)
AIRSPACES = AIRSPACES.rename(columns={0: 'AIRSPACE_ID', 1: 'NOMBRE', 2: 'TIPO', 3: 'NUMBER', 4: 'SECTORES'})
AIRSPACES = AIRSPACES.dropna(subset=['SECTORES'])
AIRSPACES = AIRSPACES.reset_index(drop=True)


list_dataframes = [df for df in AIRSPACES.groupby('AIRSPACE_ID', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    id_airspace = df['AIRSPACE_ID'].iloc[0]
    nombre = df['NOMBRE'].iloc[0]
    bloques = list(df['SECTORES'])
    tipo = df['TIPO'].iloc[0]
    number = df['NUMBER'].iloc[0]
    df_temporal = pd.DataFrame({'AIRSPACE_ID': [id_airspace], 'NOMBRE': [nombre], 'TIPO': [tipo], 'NUMBER': [number], 'SECTORES': [bloques]})
    dataframes_temporales.append(df_temporal)

AIRSPACES = pd.concat(dataframes_temporales, ignore_index=True)
del (airspaces)


# LECTURA DE LA COMPOSICION DE LOS SECTORES
sectores = pd.read_csv(PATH_SECTOR_DATA + 'sectors.txt',sep='\t', header=None)
sectores = sectores.drop(sectores.index[0])
sectores = sectores[0].str.split(';', expand=True)

rows = []
current_id = None
current_nombre = None
for index, row in sectores.iterrows():
    if 'S' in row.iloc[0]:
        current_id = row.iloc[1]
        current_nombre = row.iloc[2]
        rows.append((current_id, current_nombre,))
    else:
        rows.append((current_id, current_nombre, row.iloc[1], row.iloc[4]))
SECTORES = pd.DataFrame(rows)
SECTORES = SECTORES.rename(columns={0: 'SECTOR_ID', 1: 'NOMBRE', 2: 'AIR BLOCKS', 3: 'MAX FL'})
SECTORES = SECTORES.dropna(subset=['AIR BLOCKS'])
SECTORES = SECTORES.reset_index(drop=True)

list_dataframes = [df for df in SECTORES.groupby('SECTOR_ID', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    # print(df)
    id_sector = df['SECTOR_ID'].iloc[0]
    nombre = df['NOMBRE'].iloc[0]
    bloques = list(df['AIR BLOCKS'])
    max_FL = list(df['MAX FL'])
    # print(bloques)
    df_temporal = pd.DataFrame({'SECTOR_ID': [id_sector], 'NOMBRE': [nombre], 'AIR BLOCKS': [bloques], 'MAX FL': [max_FL]})
    dataframes_temporales.append(df_temporal)
SECTORES = pd.concat(dataframes_temporales, ignore_index=True)


# LECTURA DE LOS BLOQUES DE ESPACIO AEREO
bloques = pd.read_csv(PATH_SECTOR_DATA + 'bloques.txt',sep='\t', header=None)
bloques = bloques[0].str.split(';', expand=True)
bloques = bloques.drop(bloques.index[0])

rows = []
current_id = None
for index, row in bloques.iterrows():
    if 'A' in row.iloc[0]:
        current_id = row.iloc[1]
        rows.append((current_id, current_id))
    else:
        rows.append((current_id, row.iloc[1], row.iloc[2]))
bloques = pd.DataFrame(rows)
bloques = bloques.rename(columns={0: 'ID_BLOQUE', 1: 'LAT', 2: 'LON'})
bloques = bloques.dropna(subset=['LON'])
bloques = bloques.reset_index(drop=True)

list_dataframes = [df for df in bloques.groupby('ID_BLOQUE', sort=False)]
dataframes_temporales = []
for _, df in list_dataframes:
    id_vuelo = df['ID_BLOQUE'].iloc[0]
    coordenadas = list(zip(df['LAT'], df['LON']))
    df_temporal = pd.DataFrame({'ID_BLOQUE': [id_vuelo], 'Coordenadas': [coordenadas], })
    dataframes_temporales.append(df_temporal)
BLOQUES = pd.concat(dataframes_temporales, ignore_index=True)


# CREAR POLIGONOS CON LOS BLOQUES DE ESPACIO AEREO
BLOQUES['Contorno Bloque'] = None
for index, row in BLOQUES.iterrows():
    coordenadas = row['Coordenadas']
    y_coords = [coord[0] for coord in coordenadas]
    x_coords = [coord[1] for coord in coordenadas]
    poligono = Polygon(zip(x_coords, y_coords))
    BLOQUES.loc[index, 'Contorno Bloque'] = poligono


# CREAR SECTORES ELEMENTALES
SECTORES['Contorno Sector'] = None
SECTORES['TIPO'] = 'EL'
SECTORES['ACC'] = ACC
for index, row in SECTORES.iterrows():
    bloques = row['AIR BLOCKS']
    for bloque in bloques:
        poligono = BLOQUES.loc[BLOQUES['ID_BLOQUE'] == bloque, 'Contorno Bloque'].values[0]
        if row['Contorno Sector'] is None:
            row['Contorno Sector'] = poligono
        else:
            row['Contorno Sector'] = row['Contorno Sector'].union(poligono)
    SECTORES.loc[index, 'Contorno Sector'] = row['Contorno Sector']

SECTORES2 = pd.concat([SECTORES['SECTOR_ID'], SECTORES['Contorno Sector'], SECTORES['TIPO'],
                       SECTORES['ACC']], axis=1)


# CREAR SECTORES COLAPSADOS
AIRSPACES['ACC'] = ACC
AIRSPACES['Contorno Sector Colapsado'] = None
for index, row in AIRSPACES.iterrows():
    sectors = row['SECTORES']
    for sector in sectors:
        # print('Sector:', sector)
        poligono = SECTORES.loc[SECTORES['SECTOR_ID'] == sector, 'Contorno Sector'].values[0]
        if row['Contorno Sector Colapsado'] is None:
            row['Contorno Sector Colapsado'] = poligono
        else:
            row['Contorno Sector Colapsado'] = row['Contorno Sector Colapsado'].union(poligono)
    AIRSPACES.loc[index, 'Contorno Sector Colapsado'] = row['Contorno Sector Colapsado']

AIRSPACES2 = pd.concat([AIRSPACES['AIRSPACE_ID'], AIRSPACES['Contorno Sector Colapsado'], AIRSPACES['TIPO'],
                        AIRSPACES['ACC']], axis=1)
# RENOMBRA COLUMNAS
AIRSPACES2 = AIRSPACES2.rename(columns={'AIRSPACE_ID': 'SECTOR_ID', 'Contorno Sector Colapsado': 'Contorno Sector'})



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------- REPRESENTACIÓN DE LOS SECTORES DEL ACC --------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

print('Configuración de estudio seleccionada:',configuracion_estudio)

# SECTORES DE LA CONFIGURACIÓN
list_sectors = CONFIG.loc[CONFIG['CONFIG'] == configuracion_estudio, 'SECTORES'].iloc[0]
print('Sectores de la configuración:', list_sectors)

# Filtrar datos según los sectores seleccionados
resultado1 = AIRSPACES[AIRSPACES['AIRSPACE_ID'].isin(list_sectors)]
resultado1 = resultado1.rename(columns={'AIRSPACE_ID': 'SECTOR_ID', 'Contorno Sector Colapsado': 'Contorno Sector'})
resultado2 = SECTORES[SECTORES['SECTOR_ID'].isin(list_sectors)]

# Combinar resultados en un solo DataFrame
DF_info_conf = pd.concat([resultado1, resultado2]).reset_index(drop=True)


# =============================================================================
# ASIGNAR LA CONFIGURACIÓN REAL BASADA EN LA INTERSECCIÓN: ASIGNAR A CADA CELDA SU SECTOR REAL
# =============================================================================
gdf_cells = gpd.GeoDataFrame(DF_MALLADO.copy(), geometry='Polygon')
gdf_cells = gdf_cells.set_geometry('Polygon')

gdf_sectors = gpd.GeoDataFrame(DF_info_conf.copy(), geometry='Contorno Sector')
gdf_sectors['sector_geom'] = gdf_sectors['Contorno Sector']
gdf_sectors = gdf_sectors.set_geometry('sector_geom')

join_result = gpd.sjoin(
    gdf_cells[['Cell_Name', 'Polygon']], 
    gdf_sectors[['SECTOR_ID', 'sector_geom']], 
    how='left', 
    predicate='intersects',
    rsuffix='_r'
)

candidate_cols = ['sector_geom', 'sector_geom_r', 'geometry_right']
found_cols = [col for col in candidate_cols if col in join_result.columns]

if found_cols:
    geom_col = found_cols[0]
else:
    join_result = join_result.merge(
        gdf_sectors[['SECTOR_ID', 'sector_geom']], 
        on='SECTOR_ID', 
        how='left', 
        suffixes=('', '_sector')
    )
    geom_col = 'sector_geom'

join_result['intersection_area'] = join_result.apply(
    lambda row: row['Polygon'].intersection(row[geom_col]).area, axis=1
)

join_result_unique = join_result.sort_values('intersection_area', ascending=False)\
                                  .drop_duplicates(subset='Cell_Name', keep='first')

gdf_cells = gdf_cells.merge(
    join_result_unique[['Cell_Name', 'SECTOR_ID']], 
    on='Cell_Name', 
    how='left'
)
gdf_cells = gdf_cells.rename(columns={'SECTOR_ID': 'Sector'})
gdf_cells['Sector'] = gdf_cells['Sector'].fillna('sin_sector')

print("Asignación inicial (configuración real) basada en intersecciones:")
print(gdf_cells[['Cell_Name','Sector']].head())

# =============================================================================
# Carga de complejidad de celdas en ambos cortes
# =============================================================================
ca = pd.read_pickle(ruta_above)
cb = pd.read_pickle(ruta_below)

ca.rename(columns={
    'Celda': 'Cell_Name',
    'Suma_Complejidad_total': 'Valor_Complejidad_Celda'
}, inplace=True)

# Renombrar las columnas de cb
cb.rename(columns={
    'Celda': 'Cell_Name',
    'Suma_Complejidad_total': 'Valor_Complejidad_Celda'
}, inplace=True)


# ---------------------------------------------------------------------------- #
# Crear df_cells_above con complejidad del corte superior
# ---------------------------------------------------------------------------- #
df_cells_above = pd.merge(
    gdf_cells[['Cell_Name', 'Polygon', 'Sector']],  # geometría + asignación
    ca[['Cell_Name', 'Valor_Complejidad_Celda']],   # complejidad del corte superior
    on='Cell_Name', how='left'
)
df_cells_above.rename(columns={'Valor_Complejidad_Celda': 'Complexity'}, inplace=True)
df_cells_above['Complexity'] = df_cells_above['Complexity'].fillna(0)

# ---------------------------------------------------------------------------- #
# Crear df_cells_below con complejidad del corte inferior
# ---------------------------------------------------------------------------- #
df_cells_below = pd.merge(
    gdf_cells[['Cell_Name', 'Polygon', 'Sector']],
    cb[['Cell_Name', 'Valor_Complejidad_Celda']],
    on='Cell_Name', how='left'
)
df_cells_below.rename(columns={'Valor_Complejidad_Celda': 'Complexity'}, inplace=True)
df_cells_below['Complexity'] = df_cells_below['Complexity'].fillna(0)

# (Opcional) Verificación rápida
print("Corte superior - ejemplo:")
print(df_cells_above[['Cell_Name', 'Sector', 'Complexity']].head())

print("Corte inferior - ejemplo:")
print(df_cells_below[['Cell_Name', 'Sector', 'Complexity']].head())

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Reutilizar la misma paleta que en la sectorización inicial
sectors_init = df_cells_above['Sector'].unique()
colors_init = plt.cm.tab20(np.linspace(0, 1, len(sectors_init)))
color_map = dict(zip(sectors_init, colors_init))  # este será el color_map que usamos

# Complejidad por sector
complexity_by_sector_above = df_cells_above.groupby('Sector')['Complexity'].sum()
complexity_by_sector_below = df_cells_below.groupby('Sector')['Complexity'].sum()


# Figura comparativa
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

# Sección superior
for sec in sectors_init:
    celdas_sec = df_cells_above[df_cells_above['Sector'] == sec]
    color = color_map.get(sec, 'lightgray')
    for _, row in celdas_sec.iterrows():
        poly = row['Polygon']
        x, y = poly.exterior.xy
        ax1.fill(x, y, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)

patches_above = [
    mpatches.Patch(
        color=color_map[s],
        label=f'{s}: {complexity_by_sector_above.get(s, 0):.2f}'
    )
    for s in sectors_init if s != 'sin_sector'
]
ax1.legend(
    handles=patches_above,
    loc='upper right',
    title='Sectores (FL > umbral)'
)
ax1.set_title("Corte Superior: Complejidad por Sector")
ax1.set_xlabel("Longitud [º]")
ax1.set_ylabel("Latitud [º]")
ax1.set_aspect('equal')

# Sección inferior
for sec in sectors_init:
    celdas_sec = df_cells_below[df_cells_below['Sector'] == sec]
    color = color_map.get(sec, 'lightgray')
    for _, row in celdas_sec.iterrows():
        poly = row['Polygon']
        x, y = poly.exterior.xy
        ax2.fill(x, y, color=color, alpha=1.0, edgecolor='black', linewidth=0.5)

patches_below = [
    mpatches.Patch(
        color=color_map[s],
        label=f'{s}: {complexity_by_sector_below.get(s, 0):.2f}'
    )
    for s in sectors_init if s != 'sin_sector'
]
ax2.legend(
    handles=patches_below,
    loc='upper right',
    title='Sectores (FL ≤ umbral)'
)
ax2.set_title("Corte Inferior: Complejidad por Sector")
ax2.set_xlabel("Longitud [º]")
ax2.set_ylabel("Latitud [º]")
ax2.set_aspect('equal')

# Mostrar figura
plt.tight_layout()
plt.show()
