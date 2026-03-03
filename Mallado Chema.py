#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- LIBRERIAS Y DIRECTORIOS NECESARIOS ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import itertools
import warnings
import re
import os
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import box, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import itertools
from itertools import product
from matplotlib.lines import Line2D
import geopy.distance
from geopy.distance import geodesic
from shapely.wkt import loads
import pickle
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import nearest_points
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from datetime import datetime
import shap
import time
import seaborn as sns
import gc
import ast
import math

import time
start_time = time.time()


# DIRECTORIOS - ACC Madrid Norte

PATH_TRAFICO = "C:\\TFG\\Codigos Chema\\Datos\\2. bloque complejidad\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\"
PATH_SECTOR_DATA = "C:\\TFG\\Codigos Chema\\Datos\\1. bloque prediccion\\1. bloque prediccion\\datos\\ACC Madrid\\Sector Data\\LECMCTAN\\"
PATH_flujos = "C:\\TFG\\Codigos Chema\\Datos\\2. bloque complejidad\\2. bloque complejidad\\Datos\\MATRIZ DE INTERACCION DE FLUJOS\\"
PATH_resultados = "C:\\TFG\\Codigos Chema\\Datos\\3. bloque optimizacion\\3. bloque optimizacion\\Resultados analisis flujo celda\\"
PATH_TRAFICO_CELDA = "C:\\TFG\\Codigos Chema\\Datos\\3. bloque optimizacion\\3. bloque optimizacion\\Datos de entrada eCOMMET\\"

configuracion_estudio = 'CNF9A2'




#%%
########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------- PRIMERA PARTE DEL CÓDIGO: CARACTERIZACIÓN DEL ESPACIO AÉREO A NIVEL CELDA --------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ IMPORTACIÓN DE BASES DE DATOS ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# IMPORTACIÓN DE LA BASE DE DATOS DE FLUJOS CLUSTERIZADOS
DF_Flujos_ = pd.read_csv(
    PATH_flujos + "flow_trend_DF.csv",
    sep=";",
    encoding="latin1",
    dtype=None,        # intenta inferir tipos
    parse_dates=True,  # intenta convertir fechas
    low_memory=False
)
# Leemos el csv en lugar de un pickle

DF_Flujos = DF_Flujos_.copy()
# DF_Flujos_ = pd.read_pickle(PATH_flujos + 'flow_trend_DF.pkl')
# DF_Flujos = DF_Flujos_.copy() # Como es una porción de otro DataFrame, usar .copy() para evitar un futuro warning con pandas

# Crear objetos geométricos para los flujos -> línea recta que representa el flujo
DF_Flujos.loc[:, 'Line'] = DF_Flujos.apply(lambda row: LineString([(row['lon_f_in'], row['lat_f_in']), (row['lon_f_out'], row['lat_f_out'])]), axis=1)



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------- CREACION DE LOS DATOS NECESARIOS PARA GRAFICAR ---------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# LECTUTRA DE LAS CONFIGURACIONES DEL ACC SELECCIONADO
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
# -------------------------------------- REPRESENTACIÓN DE LOS SECTORES DEL ACC -------------------------------------- #
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

#OBTENER LA MAXIMA LATITUD Y LONGITUD DEL ACC
min_lat = []
max_lat = []
min_lon = []
max_lon = []
for index, row in DF_info_conf.iterrows():
    poligono = row['Contorno Sector']
    x, y = poligono.exterior.xy
    min_lat.append(min(y))
    max_lat.append(max(y))
    min_lon.append(min(x))
    max_lon.append(max(x))

min_lat = min(min_lat) -0.5
max_lat = max(max_lat) +0.5
min_lon = min(min_lon) -0.5
max_lon = max(max_lon) +0.5


#PLOTEAR EL ACC
fig, sects = plt.subplots()
sects.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
sects.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
sects.set_aspect('equal')
plt.xlabel('LONGITUD[º]')
plt.ylabel('LATITUD[º]')
plt.title('REPRESENTACION DE LOS SECTORES DE ESTUDIO')

for index, row in DF_info_conf.iterrows():
    poligono = row['Contorno Sector']
    x, y = poligono.exterior.xy
    sects.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'{row["SECTOR_ID"]}')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- OBTENCIÓN DEL MALLADO DEL ACC ------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DEL ESPACIO AÉREO ASOCIADO AL ACC
# Unir todos los polígonos de los sectores en un único polígono o MultiPolygon
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

# Obtener el contorno exterior del conjunto de sectores
if isinstance(union_poligonos, MultiPolygon):
    poligono_ACC = union_poligonos.convex_hull
else:
    poligono_ACC = union_poligonos.exterior

# Convertir LinearRing a Polygon
poligono_ACC = Polygon(poligono_ACC)


# REPRESENTACIÓN DEL ESPACIO AÉREO ASOCIADO AL ACC
min_lat = []
max_lat = []
min_lon = []
max_lon = []

x, y = poligono_ACC.exterior.xy
min_lat.append(min(y))
max_lat.append(max(y))
min_lon.append(min(x))
max_lon.append(max(x))

min_lat = min(min_lat) -0.5
max_lat = max(max_lat) +0.5
min_lon = min(min_lon) -0.5
max_lon = max(max_lon) +0.5

fig, ACC = plt.subplots()
ACC.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ACC.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ACC.set_aspect('equal')
plt.xlabel('LONGITUD[º]')
plt.ylabel('LATITUD[º]')
plt.title('REPRESENTACION DEL ESPACIO AÉREO DE ESTUDIO - ACC MADRID NORTE')
ACC.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'LECMCTAN')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()


# MALLADO DEL ACC
# Tamaño de celda en millas náuticas
cell_size_nm = 25

# Convertir tamaño de celdas de NM a grados de latitud (constante)
cell_size_lat_deg = cell_size_nm / 60.0

# Obtener los límites del sector
minx, miny, maxx, maxy = poligono_ACC.bounds

# Crear celdas ajustando longitud a partir de latitud promedio del sector
lat_center = (miny + maxy) / 2
cell_size_lon_deg = cell_size_nm / (math.cos(math.radians(lat_center)) * 60)

# Lista para almacenar los datos de las celdas
cell_data = []

# Crear la malla desde la esquina superior izquierda (con soporte para MultiPolygon)
cells = []
cell_id = 1  # Iniciar el contador para las celdas
current_lat = maxy  # Iniciar desde la latitud máxima (norte)

while current_lat > miny:  # Decrecer latitud hacia el sur
    current_lon = minx  # Iniciar desde la longitud mínima (oeste)
    while current_lon < maxx:  # Aumentar longitud hacia el este
        # Crear la celda
        cell = box(current_lon, current_lat - cell_size_lat_deg, current_lon + cell_size_lon_deg, current_lat)
        # Recortar la celda con el espacio aéreo
        intersected_cell = cell.intersection(poligono_ACC)

        if not intersected_cell.is_empty:
            if isinstance(intersected_cell, Polygon):
                coords = list(intersected_cell.exterior.coords)
                cells.append(intersected_cell)
                cell_data.append({
                    'Cell_Name': f'Cell_{cell_id}',
                    'Polygon': intersected_cell,
                    'Coordinates': coords
                })
                cell_id += 1

            elif isinstance(intersected_cell, MultiPolygon):
                for poly in intersected_cell.geoms:
                    coords = list(poly.exterior.coords)
                    cells.append(poly)
                    cell_data.append({
                        'Cell_Name': f'Cell_{cell_id}',
                        'Polygon': poly,
                        'Coordinates': coords
                    })
                    cell_id += 1

        current_lon += cell_size_lon_deg  # Moverse hacia el este
    current_lat -= cell_size_lat_deg  # Moverse hacia el sur


# Crear un DataFrame con los datos
DF_cells = pd.DataFrame(cell_data)

# Graficar el sector con las celdas recortadas
fig, ax_cells = plt.subplots()
ax_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_cells.set_aspect('equal')

# Dibujar el polígono del sector
x, y = poligono_ACC.exterior.xy
ax_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

# Dibujar las celdas recortadas
for cell in cells:
    if isinstance(cell, Polygon):
        x, y = cell.exterior.xy
        ax_cells.plot(x, y, color='gray', alpha=0.5)
    elif isinstance(cell, MultiPolygon):
        for poly in cell.geoms:
            x, y = poly.exterior.xy
            ax_cells.plot(x, y, color='gray', alpha=0.5)

# Configurar la gráfica
ax_cells.set_title("MALLADO DEL ESPACIO AÉREO CON CELDAS 25NM x 25NM")
ax_cells.set_aspect('equal')
ax_cells.set_xlabel('LONGITUD[º]')
ax_cells.set_ylabel('LATITUD[º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')
plt.show()
