#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- LIBRERÍAS QUE NECESITA EL CÓDIGO ----------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import matplotlib.patches as mpatches
from copy import deepcopy
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
import itertools

warnings.filterwarnings("ignore")

##### Registro del inicio del tiempo
inicio_tiempo = tm.time()


# SELECCIÓN DEL TIPO DE DATOS PARA LOS QUE SE QUIERE OBTENER EL MAPA DE COMPLEJIDAD A NIVEL SECTOR O CELDA
print('Datos para realizar el mapa de complejidad: ')
tipo_datos = input('Seleccionar: predicciones o reales')


# DIRECTORIOS
PATH_SECTOR_DATA = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Sector Data\\LECMCTAN\\'
PATH_MALLADO_DATA = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

if tipo_datos == 'predicciones':
    PATH_COMPLEJIDAD_SECTOR = 'C:\\Users\\Larissa Ruiz\\Desktop\\bloque complejidad\\BLOQUE COMPLEJIDAD\\Archivos\\RESULTADOS COMPLEJIDAD\\PREDICCIONES\\complejidad por horas\\'
    PATH_COMPLEJIDAD_CELDA = 'C:\\Users\\Larissa Ruiz\\Desktop\\bloque optimización\\BLOQUE OPTIMIZACION\\Archivos\\Resultados eCOMMET\\complejidad predicha\\complejidad por horas\\'
    PATH_COMPLEJIDAD_OPT = 'C:\\Users\\Larissa Ruiz\\Desktop\\bloque optimización\\BLOQUE OPTIMIZACION\\Archivos\\Resultados eCOMMET\\complejidad predicha\\'
elif tipo_datos == 'reales':
    PATH_COMPLEJIDAD_SECTOR = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\2. bloque complejidad\\Datos\\RESULTADOS COMPLEJIDAD\\REAL\\'
    PATH_COMPLEJIDAD_CELDA = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\'
    PATH_COMPLEJIDAD_OPT = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados eCOMMET\\Complejidad real\\mapas\\'



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ IMPORTACIÓN DE BASE DE DATOS -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# CARGAR BASE DE DATOS DEL MALLADO DEL ACC DE ESTUDIO
DF_MALLADO = pd.read_pickle(PATH_MALLADO_DATA + 'dataset_flujos_por_celda.pkl')


# CARGAR BASE DE DATOS DE COMPLEJIDAD DE LOS SECTORES DE LA CONFIGURACIÓN PARA UNA FRANJA TEMPORAL DE ESTUDIO: 2022-06-01 14:00-15:00
Complejidad_sectores_ = pd.read_pickle(PATH_COMPLEJIDAD_SECTOR + 'Complejidad_por_hora_2022-06-01_06-07.pkl')

# Renombrar la columna
Complejidad_sectores_.rename(columns={'Suma_Complejidad_total': 'Valor_Complejidad_Sector'}, inplace=True)

# Complejar el dataframe asignando una complejidad nula a aquellos sectores no presentes, en caso necesario
# configuracion_estudio = 'CNF9A2'
# lista_sectores = ['LECMASU', 'LECMBLL', 'LECMBLU', 'LECMDGL', 'LECMDGU', 'LECMPAL', 'LECMPAU', 'LECMSAO', 'LECMSAS']
# Definimos un diccionario con las configuraciones disponibles
configuraciones = {
    1: ('CNF1A', ['LECMR1I']),
    2: ('CNF2A', ['LECMDPI', 'LECMSAB']),
    3: ('CNF3A', ['LECMDGI', 'LECMPAI', 'LECMSAB']),
    4: ('CNF4A', ['LECMDGI', 'LECMBLI', 'LECMPAI', 'LECMSAI']),
    5: ('CNF5A', ['LECMASI', 'LECMBLI', 'LECMDGI', 'LECMPAI', 'LECMSAN']),
}

# Pedimos al usuario que elija un número
entrada = input("Elige un número de sectores (1, 2, 3, 4 o 5): ")

try:
    num = int(entrada)
    configuracion_estudio, lista_sectores = configuraciones[num]
    # Mostramos las variables seleccionadas
    print(f"configuracion_estudio = '{configuracion_estudio}'")
    print(f"lista_sectores = {lista_sectores}")
except (ValueError, KeyError):
    print("Número no válido. Por favor, elige 1, 2, 3, 4 o 5.")


df_sectores_completo = pd.DataFrame({"Sector": lista_sectores}) # Crear un DataFrame con todos los sectores
Complejidad_sectores = pd.merge(df_sectores_completo,Complejidad_sectores_, on="Sector", how="left") # Unir con el DataFrame existente
Complejidad_sectores["Valor_Complejidad_Sector"] = Complejidad_sectores["Valor_Complejidad_Sector"].fillna(0) # Rellenar valores NaN con 0 en la columna de complejidad
del(Complejidad_sectores_) # Eliminar dataframe innecesario

# CARGAR BASE DE DATOS DE COMPLEJIDAD DE LAS CELDAS DE LA CONFIGURACIÓN PARA UNA FRANJA TEMPORAL DE ESTUDIO
Complejidad_celdas_ = pd.read_pickle(PATH_COMPLEJIDAD_CELDA + 'Complejidad_por_hora_2022-06-01_06-07.pkl')

# Renombrar columnas
Complejidad_celdas_.rename(columns={'Celda': 'Cell_Name'}, inplace=True)
Complejidad_celdas_.rename(columns={'Suma_Complejidad_total': 'Valor_Complejidad_Celda'}, inplace=True)

# Complejar el dataframe asignando una complejidad nula a aquellas celdas no presentes, en caso necesario
lista_celdas = list(DF_MALLADO['Cell_Name']) # Obtener la lista de celdas
df_celdas_completa = pd.DataFrame({"Cell_Name": lista_celdas}) # Crear un DataFrame con todas las celdas
Complejidad_celdas = pd.merge(df_celdas_completa,Complejidad_celdas_, on="Cell_Name", how="left") # Unir con el DataFrame existente
Complejidad_celdas["Valor_Complejidad_Celda"] = Complejidad_celdas["Valor_Complejidad_Celda"].fillna(0) # Rellenar valores NaN con 0 en la columna de complejidad
del(Complejidad_celdas_) # Eliminar dataframe innecesario




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

# Añadir leyenda
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# Guardar la figura
nombre_figura_ACC = PATH_COMPLEJIDAD_OPT + 'ACC Madrid Norte - conf 9A2.png'
plt.savefig(nombre_figura_ACC, format='png', dpi=300, bbox_inches='tight')

# Mostrar figura
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------- REPRESENTACIÓN DEL ESPACIO AÉREO DE ESTUDIO ACC ---------------------------------- #
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

# Guardar la figura
nombre_figura_espacio_ACC = PATH_COMPLEJIDAD_OPT + 'Espacio aéreo de estudio - ACC Madrid Norte.png'
plt.savefig(nombre_figura_espacio_ACC, format='png', dpi=300, bbox_inches='tight')

# Mostrar figura
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- REPRESENTACIÓN DEL MALLADO DEL ACC ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

fig, ax_cells = plt.subplots()

# Dibujar el polígono del ACC
x, y = poligono_ACC.exterior.xy
ax_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

# Dibujar las celdas del mallado
for _, row in DF_MALLADO.iterrows():
    polygon = row['Polygon']  # Obtener el polígono
    x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno
    ax_cells.plot(x, y, color='gray', alpha=0.5)  # Dibujar el contorno de la celda

# Configurar la gráfica
ax_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_cells.set_aspect('equal')

ax_cells.set_title("MALLADO DEL ESPACIO AÉREO CON CELDAS 30NM x 30NM")
ax_cells.set_aspect('equal')
ax_cells.set_xlabel('LONGITUD[º]')
ax_cells.set_ylabel('LATITUD[º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# Guardar la figura
nombre_figura_mallado_ACC = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo de estudio - ACC Madrid Norte.png'
plt.savefig(nombre_figura_mallado_ACC, format='png', dpi=300, bbox_inches='tight')

# Mostrar figura
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------- REPRESENTACIÓN DE LA COMPLEJIDAD DE LOS SECTORES DEL ACC ------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DE UN DATAFRAME CON LA INFORMACIÓN GEOMÉTRICA DE LOS SECTORES Y SU COMPLEJIDAD ASOCIADA
DF_info_conf_copia = DF_info_conf.copy()
DF_info_conf_copia.rename(columns={'SECTOR_ID': 'Sector'}, inplace=True)
DF_COMPLEJIDAD_SECTORES = pd.merge(DF_info_conf_copia[['Sector','Contorno Sector']], Complejidad_sectores, on="Sector", how="left")

# GRAFICAR UN MAPA DE COLOR A PARTIR DE LA COMPLEJIDAD DE LOS SECTORES
fig, ax_complejidad_sects = plt.subplots(figsize=(12, 8))

# Dibujar el polígono del ACC
x, y = poligono_ACC.exterior.xy
ax_complejidad_sects.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

# Lista para almacenar las posiciones de los centroides
centroid_positions = []

# Dibujar los sectores con colores basados en 'Valor_Complejidad_Sector'
for _, row in DF_COMPLEJIDAD_SECTORES.iterrows():
    polygon = row['Contorno Sector']  # Obtener el polígono
    x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno

    # Seleccionar el color basado en el valor de complejidad
    valor_complejidad = row['Valor_Complejidad_Sector']

    # Escala de colores (ajusta según los datos)
    color = plt.cm.viridis((valor_complejidad - DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min()) /
                        (DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].max() - DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min()))

    # Rellenar el polígono con el color correspondiente
    ax_complejidad_sects.fill(x, y, color=color, alpha=0.7)

    # Dibujar el contorno del sector
    ax_complejidad_sects.plot(x, y, color='black', alpha=0.5)

    # Calcular el centroide del polígono para colocar la etiqueta
    centroid = polygon.centroid
    centroid_x, centroid_y = centroid.x, centroid.y

    # Comprobar si el centroide actual está demasiado cerca de algún centroide ya registrado
    # Función para calcular la distancia entre dos puntos
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    displacement = 0
    new_centroid_x, new_centroid_y = centroid_x, centroid_y
    while any(distance(new_centroid_x, new_centroid_y, cx, cy) < 0.5 for cx, cy in centroid_positions):
        # Si el centroide está demasiado cerca, moverlo un poco (puedes ajustar el valor de 'displacement')
        displacement += 0.5  # Desplazar más si hay centroids cercanos
        new_centroid_x = centroid_x
        new_centroid_y = centroid_y + displacement

    # Añadir la nueva posición del centroide a la lista
    centroid_positions.append((new_centroid_x, new_centroid_y))

    # Colocar el valor de complejidad como etiqueta en el centroide, con desplazamiento
    ax_complejidad_sects.text(new_centroid_x, new_centroid_y, f'{valor_complejidad:.2f}', ha='center', va='center', fontsize=10, color='black', fontweight='bold')


# Crear la barra de colores (colorbar) para la escala de complejidad
sm_sect = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min(),
                                                                vmax=DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].max()))
sm_sect.set_array([])  # Se necesita para que funcione el colorbar
# Ajustar el tamaño de la barra de colores (colorbar)
cbar_sect = fig.colorbar(sm_sect, ax=ax_complejidad_sects, label='Valor Complejidad', shrink=0.8, aspect=10)

# Configurar la gráfica
ax_complejidad_sects.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_complejidad_sects.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_complejidad_sects.set_aspect('equal')

if tipo_datos == 'predicciones':
    ax_complejidad_sects.set_title("COMPLEJIDAD PREDICHA DE LOS SECTORES - MAPA DE COLOR")
elif tipo_datos == 'reales':
    ax_complejidad_sects.set_title("COMPLEJIDAD REAL DE LOS SECTORES - MAPA DE COLOR")

ax_complejidad_sects.set_aspect('equal')
ax_complejidad_sects.set_xlabel('LONGITUD[º]')
ax_complejidad_sects.set_ylabel('LATITUD[º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# Guardar la figura
if tipo_datos == 'predicciones':
        nombre_figura_complejidad_sectores = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad predicha sectores - ACC Madrid Norte.png'
elif tipo_datos == 'reales':
    nombre_figura_complejidad_sectores = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad real sectores - ACC Madrid Norte.png'

plt.savefig(nombre_figura_complejidad_sectores, format='png', dpi=300, bbox_inches='tight')

# Mostrar figura
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------- REPRESENTACIÓN DE LA COMPLEJIDAD DE LAS CELDAS DEL ACC ------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# OBTENCIÓN DE UN DATAFRAME CON LA INFORMACIÓN GEOMÉTRICA DE LAS CELDAS Y SU COMPLEJIDAD ASOCIADA
DF_COMPLEJIDAD_CELDAS = pd.merge(DF_MALLADO[['Cell_Name','Polygon']], Complejidad_celdas, on="Cell_Name", how="left")

# GRAFICAR UN MAPA DE COLOR A PARTIR DE LA COMPLEJIDAD DE LAS CELDAS
fig, ax_complejidad_cells = plt.subplots(figsize=(12, 8))

# Dibujar el polígono del ACC
x, y = poligono_ACC.exterior.xy
ax_complejidad_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

# Dibujar las celdas del mallado con colores basados en 'Valor_Complejidad_Celda'
for _, row in DF_COMPLEJIDAD_CELDAS.iterrows():
    polygon = row['Polygon']  # Obtener el polígono
    x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno

    # Seleccionar el color basado en el valor de complejidad
    valor_complejidad = row['Valor_Complejidad_Celda']

    # Escala de colores (ajusta según tus datos)
    color = plt.cm.viridis((valor_complejidad - DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min()) /
                        (DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].max() - DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min()))

    # Rellenar el polígono con el color correspondiente
    ax_complejidad_cells.fill(x, y, color=color, alpha=0.7)

    # Dibujar el contorno de la celda
    ax_complejidad_cells.plot(x, y, color='black', alpha=0.5)

    # Calcular el centroide del polígono para colocar la etiqueta
    centroid = polygon.centroid
    centroid_x, centroid_y = centroid.x, centroid.y

    # Colocar el valor de complejidad como etiqueta en el centroide
    ax_complejidad_cells.text(centroid_x, centroid_y, f'{valor_complejidad:.2f}', ha='center', va='center',fontsize=10, color='black', fontweight='bold')


# Crear la barra de colores (colorbar) para la escala de complejidad
sm_cell = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min(),
                                                                vmax=DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].max()))
sm_cell.set_array([])  # Se necesita para que funcione el colorbar
# Ajustar el tamaño de la barra de colores (colorbar)
cbar_cell = fig.colorbar(sm_cell, ax=ax_complejidad_cells, label='Complexity Value', shrink=0.8, aspect=10)

# Configurar la gráfica
ax_complejidad_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_complejidad_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_complejidad_cells.set_aspect('equal')

ax_complejidad_cells.set_title("Cell Complexity – Color Map")
ax_complejidad_cells.set_aspect('equal')
ax_complejidad_cells.set_xlabel('Longitude[º]')
ax_complejidad_cells.set_ylabel('Latitude[º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')


# Guardar la figura
if tipo_datos == 'predicciones':
    nombre_figura_complejidad_celdas = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad predicha celdas - ACC Madrid Norte.png'
elif tipo_datos == 'reales':
    nombre_figura_complejidad_celdas = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad real celdas - ACC Madrid Norte.png'

plt.savefig(nombre_figura_complejidad_celdas, format='png', dpi=300, bbox_inches='tight')

# Mostrar figura
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------ GUARDADO DE BASES DE DATOS DE COMPLEJIDAD ------------------------------------  #
# -------------------------------------------------------------------------------------------------------------------- #


if tipo_datos == 'predicciones':
    nombre_archivo_sectores = 'DF_Complejidad_Sectores_predicciones_2022-06-01_14-15.pkl'
    nombre_archivo_celdas = 'DF_Complejidad_Celdas_predicciones_2022-06-01_14-15.pkl'
elif tipo_datos == 'reales':
    nombre_archivo_sectores = 'DF_Complejidad_Sectores_reales_2022-06-01_14-15.pkl'
    nombre_archivo_celdas = 'DF_Complejidad_Celdas_reales_2022-06-01_14-15.pkl'


# FORMATO .pkl
DF_info_conf.to_pickle(PATH_COMPLEJIDAD_OPT + 'DATOS_CONFIGURACIÓN.pkl')
DF_COMPLEJIDAD_SECTORES.to_pickle(PATH_COMPLEJIDAD_OPT + nombre_archivo_sectores + '.pkl')
DF_COMPLEJIDAD_CELDAS.to_pickle(PATH_COMPLEJIDAD_OPT + nombre_archivo_celdas + '.pkl')


# FORMATO .csv
DF_info_conf.to_csv(PATH_COMPLEJIDAD_OPT + 'DATOS_CONFIGURACIÓN.csv', index=False, sep=';')
DF_COMPLEJIDAD_SECTORES.to_csv(PATH_COMPLEJIDAD_OPT + nombre_archivo_sectores + '.csv', index=False, sep=';')
DF_COMPLEJIDAD_CELDAS.to_csv(PATH_COMPLEJIDAD_OPT + nombre_archivo_celdas + '.csv', index=False, sep=';')


# =============================================================================
# 2. ASIGNAR LA CONFIGURACIÓN REAL BASADA EN LA INTERSECCIÓN: ASIGNAR A CADA CELDA SU SECTOR REAL
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
# 3. INCORPORAR LA COMPLEJIDAD DE LAS CELDAS
# =============================================================================
df_cells = pd.merge(gdf_cells, DF_COMPLEJIDAD_CELDAS[['Cell_Name','Valor_Complejidad_Celda']],
                    on='Cell_Name', how='left')
df_cells.rename(columns={'Valor_Complejidad_Celda': 'Complexity'}, inplace=True)
df_cells['Complexity'] = df_cells['Complexity'].fillna(0)

print("Asignación inicial (configuración real):")
print(df_cells[['Cell_Name','Sector']].head())


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 1) Complejidad TOTAL por sector (suma de las celdas)
complexity_by_sector = (
    df_cells
    .groupby('Sector')['Complexity']
    .sum()
    .rename('total_complexity')
)

# 2) Paleta discreta única para TODOS los sectores
sectors = complexity_by_sector.index.tolist()
colors  = plt.cm.viridis(np.linspace(0, 1, len(sectors)))
color_map = {sec: colors[i] for i, sec in enumerate(sectors)}

# 3) Creamos figura con dos ejes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# --------------------------------------------------
# A) Plot 1: sectorización por celdas (simple)
# --------------------------------------------------
for _, row in df_cells.iterrows():
    poly = row['Polygon']
    sec  = row['Sector']
    x, y = poly.exterior.xy
    ax1.fill(x, y,
             color=color_map.get(sec, 'lightgray'),
             edgecolor='black', linewidth=0.5, alpha=0.8)

ax1.set_title("Sectorización por celdas")
ax1.set_xlabel("Longitud [º]")
ax1.set_ylabel("Latitud [º]")
ax1.set_aspect('equal')

patches_simple = [
    mpatches.Patch(color=color_map[s], label=s)
    for s in sectors if s != 'sin_sector'
]
ax1.legend(handles=patches_simple,
           loc='upper right', title="Sector")

# --------------------------------------------------
# B) Plot 2: sectorización + COMPLEJIDAD TOTAL
# --------------------------------------------------
for _, row in df_cells.iterrows():
    poly = row['Polygon']
    sec  = row['Sector']
    x, y = poly.exterior.xy
    ax2.fill(x, y,
             color=color_map.get(sec, 'lightgray'),
             edgecolor='black', linewidth=0.5, alpha=0.8)

ax2.set_title("Sectorización con Complejidad Total")
ax2.set_xlabel("Longitud [º]")
ax2.set_ylabel("Latitud [º]")
ax2.set_aspect('equal')

patches_total = [
    mpatches.Patch(color=color_map[s],
                   label=f"{s}: {complexity_by_sector.loc[s]:.2f}")
    for s in sectors if s != 'sin_sector'
]
ax2.legend(handles=patches_total,
           loc='upper right', title="Sector: complejidad total")

plt.tight_layout()
plt.show()

# 3) Crear solo una figura y eje
fig, ax = plt.subplots(figsize=(8, 8))

# --------------------------------------------------
# Plot ÚNICO: sectorización + COMPLEJIDAD TOTAL
# --------------------------------------------------
for _, row in df_cells.iterrows():
    poly = row['Polygon']
    sec  = row['Sector']
    x, y = poly.exterior.xy
    ax.fill(x, y,
            color=color_map.get(sec, 'lightgray'),
            edgecolor='black', linewidth=0.5, alpha=0.8)

ax.set_title("Sectorization with Total Complexity")
ax.set_xlabel("Longitude [º]")
ax.set_ylabel("Latitude [º]")
ax.set_aspect('equal')

patches_total = [
    mpatches.Patch(color=color_map[s],
                   label=f"{s}: {complexity_by_sector.loc[s]:.2f}")
    for s in sectors if s != 'sin_sector'
]
ax.legend(handles=patches_total,
          loc='upper right', title="Sector: total complexity")

plt.tight_layout()
plt.show()
# --- 5.x Identificar celdas de “borde” entre sectores por arista ---

def find_edge_sharing_cells(graph, assignment):
    """
    Devuelve la lista de nombres de celdas que tienen al menos un vecino
    conectado por arista y asignado a un sector diferente.

    Parámetros:
    - graph: grafo NetworkX donde los nodos son Cell_Name y las aristas
            indican contigüidad por arista.
    - assignment: dict mapping Cell_Name -> sector_id (p.ej. assignment)

    Retorna:
    - List[str]: lista de Cell_Name en el “frontera” inter–sectorial.
    """
    border_cells = []
    for cell in graph.nodes():
        for nbr in graph.neighbors(cell):
            if assignment[cell] != assignment[nbr]:
                border_cells.append(cell)
                break
    return border_cells


# =============================================================================
# 4. CONSTRUIR EL GRAFO DE VECINOS BASADO EN LA CONTIGÜIDAD DE CELDAS (POR ARISTA)
# =============================================================================
def share_edge(poly1, poly2):
    """
    Devuelve True si poly1 y poly2 comparten un segmento de arista (LineString con longitud > 0).
    """
    if not poly1.intersects(poly2):
        return False
    inter = poly1.intersection(poly2)
    # Para que sea un borde compartido, la intersección debe ser un LineString o MultiLineString con longitud > 0.
    if inter.is_empty:
        return False
    if inter.geom_type in ['LineString', 'MultiLineString'] and inter.length > 0:
        return True
    # Si se superponen con área o solo se tocan en un punto, no lo consideramos adyacencia por arista.
    return False

# Creamos un grafo donde cada nodo es una celda.
G = nx.Graph()
for idx, row in df_cells.iterrows():
    G.add_node(row['Cell_Name'])

# Conectamos dos nodos solo si comparten un borde (no vértice).
cells_list = df_cells[['Cell_Name', 'Polygon']].values
n_cells = len(cells_list)
for i in range(n_cells):
    cell_i, poly_i = cells_list[i]
    for j in range(i + 1, n_cells):
        cell_j, poly_j = cells_list[j]
        if share_edge(poly_i, poly_j):
            G.add_edge(cell_i, cell_j)


##############################################
def split_to_k(assignment, cell_complexity, G):

    from collections import Counter  # Import del Counter
    from shapely.ops import unary_union

    from collections import Counter
    import heapq, random, networkx as nx


    # —————————————————————————————————————————
    # 0. Pre-cálculos sobre la malla inicial
    # —————————————————————————————————————————

    print(f"Assignment original (antes de split): {assignment}")
    # 0.1 Contorno real del ACC
    union_all = unary_union(df_cells['Polygon'])
    contorno_real = union_all.boundary
    union_poligonos = unary_union(df_cells['Polygon'])  # Ahora se crea correctamente

    # 0.2 Celdas frontera inter-sectorial (basado en la asignación real inicial)
    initial_border_cells = set(find_edge_sharing_cells(G, assignment))

    # 0.3 Celdas que tocan el contorno del ACC
    initial_acc_cells = set(
        df_cells.loc[
            df_cells['Polygon'].touches(contorno_real),
            'Cell_Name'
        ]
    )

    # --- 1. Parámetro k y peso ideal W dinámico ---
    k = int(entrada) + i
    total_complexity = sum(cell_complexity.values())

    # --- 2. Funciones auxiliares y datos de frontera (igual que antes) ---
    outer_boundary = union_poligonos.boundary
    is_border = {
        row['Cell_Name']: row['Polygon'].touches(outer_boundary)
        for _, row in df_cells.iterrows()
    }

    def is_connected_subset(graph, nodes):
        if not nodes: return False
        if len(nodes) == 1: return True
        return nx.is_connected(graph.subgraph(nodes))

    def objective(assign):
        sums = Counter()
        for cell, sec in assign.items():
            sums[sec] += cell_complexity[cell]
        vals = list(sums.values())
        mean = sum(vals)/len(vals)
        var  = sum((v-mean)**2 for v in vals)/(len(vals)-1) if len(vals)>1 else 0
        diff = max(vals) - min(vals)
        # penalty por celdas interiores de grado 1
        penalty = 0
        for cell, sec in assign.items():
            if not is_border[cell]:
                deg_int = sum(1 for nbr in G.neighbors(cell) if assign[nbr]==sec)
                if deg_int == 1:
                    penalty += 1
        big_w = total_complexity
        return var + diff + big_w * penalty*0.5

    # --- 3. Preparamos el heap para ir spliteando siempre el sector más grande ---
    # Calculamos sumas iniciales
    sector_sums = Counter()
    for cell, sec in assignment.items():
        sector_sums[sec] += cell_complexity[cell]
    num_sectors = len(sector_sums)

    # Heap ordena por -suma para extraer el más pesado
    heap = [(-s, sec) for sec, s in sector_sums.items()]
    heapq.heapify(heap)
    split_counters = Counter()

    # --- 4. Bucle de splits hasta k sectores ---
    while num_sectors < k:
        W = total_complexity / k

        # 4.1. Sacamos el sector más pesado
        neg_s, sec = heapq.heappop(heap)
        cells_in = {c for c,s in assignment.items() if s == sec}
        subG     = G.subgraph(cells_in).copy()
        
        
        
        # 4.2. Region-growing para aproximar A a W
        # 4.2.2 Selección de semilla modificada:
        frontier_acc_candidates = [
            c for c in cells_in
            if c in initial_border_cells and c in initial_acc_cells
        ]

        
        if frontier_acc_candidates:
            seed = max(frontier_acc_candidates, key=lambda c: cell_complexity[c])
        else:
            seed = max(cells_in, key=lambda c: cell_complexity[c])

        # 1.3 Region-growing clásica a partir de seed…
        region_A = {seed}
        sum_A    = cell_complexity[seed]

        # Variable para rastrear si la última celda añadida fue aislada
        last_added_is_isolated = False

        while sum_A < W:
            frontier = {
                nbr for u in region_A
                for nbr in subG.neighbors(u)
                if nbr not in region_A
            }
            if not frontier:
                break
            
            # Filtramos celdas que son aisladas si la última fue aislada
            if last_added_is_isolated:
                frontier = {c for c in frontier if sum(1 for nbr in subG.neighbors(c) if nbr in region_A) > 1}

            # El que mejor acerca a W
            candidates = sorted(
                frontier,
                key=lambda c: abs((sum_A + cell_complexity[c]) - W)
            )

            added = False
            for c in candidates:
                newA = region_A | {c}
                resto = cells_in - newA
                if not resto or not is_connected_subset(subG, resto):
                    continue
                region_A, sum_A = newA, sum_A + cell_complexity[c]
                
                # Verificar si la celda agregada es aislada
                if sum(1 for nbr in subG.neighbors(c) if nbr in region_A) <= 1:
                    last_added_is_isolated = True
                else:
                    last_added_is_isolated = False

                added = True
                break
            if not added:
                break

        # 4.3. Estrategia fallback si region_A quedó muy pequeña
        if sum_A < 0.5 * W and len(region_A) < 3:
            fallback = {
                nbr for u in region_A
                for nbr in subG.neighbors(u)
                if nbr not in region_A
            }
            for c in sorted(fallback, key=lambda c: cell_complexity[c], reverse=True):
                region_A.add(c)
                sum_A += cell_complexity[c]
                if sum_A >= W:
                    break

        # 4.4. Construimos region_B y reparamos su conectividad
        region_B = cells_in - region_A
        comps = list(nx.connected_components(subG.subgraph(region_B)))
        if len(comps) > 1:
            comp_sums = [(sum(cell_complexity[c] for c in comp), comp) for comp in comps]
            main_comp = max(comp_sums)[1]
            for _, comp in comp_sums:
                if comp is not main_comp:
                    for c in comp:
                        region_A.add(c)
            region_B = main_comp
            sum_A = sum(cell_complexity[c] for c in region_A)

        # 4.5. Renombrado y reasignación
        split_counters[sec] += 1
        suf  = split_counters[sec]
        A_id = f"{sec}_A{suf}"
        B_id = f"{sec}_B{suf}"
        for c in region_A: assignment[c] = A_id
        for c in region_B: assignment[c] = B_id

        # 4.6. Actualización de contadores y heap
        num_sectors += 1
        sumA = sum(cell_complexity[c] for c in region_A)
        sumB = sum(cell_complexity[c] for c in region_B)
        heapq.heappush(heap, (-sumA, A_id))
        heapq.heappush(heap, (-sumB, B_id))
        del sector_sums[sec]
        sector_sums[A_id], sector_sums[B_id] = sumA, sumB

    # --- 5. Refinamiento local (tu paso 7, sin cambios) ---
    best_obj = objective(assignment)
    improved = True
    while improved:
        improved = False
        border = [
            c for c,s in assignment.items()
            if any(assignment[nbr] != s for nbr in G.neighbors(c))
        ]
        random.shuffle(border)
        for cell in border:
            sec0 = assignment[cell]
            neigh_secs = {
                assignment[nbr]
                for nbr in G.neighbors(cell)
                if assignment[nbr] != sec0
            }
            for sec1 in neigh_secs:
                A = {n for n,s in assignment.items() if s == sec0} - {cell}
                B = {n for n,s in assignment.items() if s == sec1} | {cell}
                if not A or not B: continue
                if not (is_connected_subset(G, A) and is_connected_subset(G, B)): continue
                new_assign = assignment.copy()
                new_assign[cell] = sec1
                new_obj = objective(new_assign)
                if new_obj < best_obj:
                    assignment = new_assign
                    best_obj   = new_obj
                    improved   = True
                    break
            if improved:
                break

    df_cells['Final_Sector'] = df_cells['Cell_Name'].map(assignment)

    contorno_real     = union_poligonos.boundary   # línea real de contorno

    # Uso:
    border_cell_names = find_edge_sharing_cells(G, assignment)
    print(f"Celdas que comparten arista con otro sector (total {len(border_cell_names)}):")
    print(border_cell_names)


    def find_acc_contour_cells(
        df_cells,
        acc_contour_geom=None,
        cell_col='Cell_Name',
        geom_col='Polygon',
        sector_col='Final_Sector'
    ):
        from shapely.ops import unary_union

        # 1) Si no me pasan nada, construyo la línea real de contorno
        if acc_contour_geom is None:
            union_all = unary_union(df_cells[geom_col])
            acc_boundary = union_all.boundary
        else:
            # si me pasaron una línea (LineString/MultiLineString), la uso directa
            acc_boundary = acc_contour_geom

        # 2) Encuentro todas las celdas que INTERSECTAN esa línea
        mask = df_cells[geom_col].intersects(acc_boundary)

        # 3) Subconjunto resultante
        gdf_border = df_cells[mask].copy()

        # 4) Mapeo sector -> lista de celdas
        sector2cells = (
            gdf_border
            .groupby(sector_col)[cell_col]
            .apply(list)
            .to_dict()
        )

        return sector2cells, gdf_border


    # -------------------------
    # Uso 1: sin contorno externo definido (usa convex hull)
    # --- luego, al identificar celdas en contacto con el ACC ---
    sector2cells, gdf_acc_border = find_acc_contour_cells(
        df_cells,
        acc_contour_geom= contorno_real,   # aquí le pasas la línea real
        cell_col='Cell_Name',
        geom_col='Polygon',
        sector_col='Final_Sector'
    )

    # Uso 2: si tienes un GeoDataFrame gdf_acc con la geometría real del ACC:
    # contorno_real = gdf_acc.geometry.unary_union
    # sector2cells, gdf_acc_border = find_acc_contour_cells(df_cells, contorno_real)

    print("Celdas que tocan el contorno del ACC por sector:")
    for sec, cells in sector2cells.items():
        print(f"  {sec}: {len(cells)} celdas -> {cells}")

    # (Luego continúa tu paso 6)
    # df_cells['Final_Sector'] = df_cells['Cell_Name'].map(assignment)


    # --- 5.x Identificar celdas internas frontera entre sectores (sin contorno ACC) ---

    def find_internal_edge_sharing_cells(
        df_cells,
        graph,
        assignment,
        acc_contour_geom=None,
        cell_col='Cell_Name',
        geom_col='Polygon',
        sector_col='Final_Sector'
    ):
        """
        Identifica las celdas que comparten arista con otro sector pero que no
        delimitan con el contorno del ACC.

        Parámetros
        ----------
        df_cells : GeoDataFrame
            Debe contener al menos las columnas `cell_col` y `geom_col`.
        graph : networkx.Graph
            Grafo de contigüidad por arista.
        assignment : dict
            Mapeo Cell_Name -> sector_id.
        acc_contour_geom : shapely.geometry o GeoSeries, opcional
            Geometría del contorno real del ACC. Si es None, usa el convex hull
            de todas las celdas.
        cell_col, geom_col, sector_col : str
            Nombres de columnas en df_cells.

        Devuelve
        -------
        internal_cells : list[str]
            Celdas que comparten arista con otro sector pero no con ACC.
        gdf_internal : GeoDataFrame
            Subconjunto de df_cells con esas celdas.
        """
        from shapely.ops import unary_union

        # 1) Definir contorno ACC
        if acc_contour_geom is None:
            acc_geom = unary_union(df_cells[geom_col]).convex_hull
        else:
            # si viene en un GeoDataFrame/GeoSeries
            try:
                acc_geom = unary_union(acc_contour_geom)
            except:
                acc_geom = acc_contour_geom

        # 2) Todas las celdas que comparten arista con otro sector
        border_cells = []
        for cell in graph.nodes():
            for nbr in graph.neighbors(cell):
                if assignment[cell] != assignment[nbr]:
                    border_cells.append(cell)
                    break

        # 3) Celdas que tocan el contorno ACC
        mask_acc = df_cells[geom_col].touches(acc_geom)
        acc_cells = set(df_cells.loc[mask_acc, cell_col])

        # 4) Filtrar internas: frontera inter–sectores sin ACC
        internal_cells = [c for c in border_cells if c not in acc_cells]
        gdf_internal = df_cells[df_cells[cell_col].isin(internal_cells)].copy()

        return internal_cells, gdf_internal


    from shapely.ops      import unary_union
    from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
    import math

    def find_rays_from_cells_mixed(
        df_cells,
        cells_list,
        acc_contour_geom,
        boundary_cells,
        cell_col='Cell_Name',
        geom_col='Polygon',
        threshold_internal=60.0,
        threshold_boundary=30.0
    ):
        """
        Dispara rayos desde:
        - centroides de las celdas internas frontera (hasta threshold_internal NM)
        - representative_point() de las celdas en el borde ACC (hasta threshold_boundary NM)
        Retorna crossings[cell][edge_idx] = { distance_nm, cells_crossed }
        """
        # 1) Prepara acc_geom y acc_boundary correctamente
        if acc_contour_geom is None:
            union_all = unary_union(df_cells[geom_col])
            acc_geom  = union_all
        else:
            acc_geom  = acc_contour_geom

        if isinstance(acc_geom, (LineString, MultiLineString)):
            acc_boundary = acc_geom
        elif isinstance(acc_geom, (Polygon, MultiPolygon)):
            acc_boundary = acc_geom.boundary
        else:
            try:
                acc_boundary = acc_geom.boundary
            except:
                acc_boundary = acc_geom

        # 2) Prepara spatial index y resultado
        sindex   = df_cells.sindex
        crossings = {}

        # 3) Filtra sólo las celdas relevantes
        gdf_sub = df_cells[df_cells[cell_col].isin(cells_list)]
        for _, row in gdf_sub.iterrows():
            cell = row[cell_col]
            poly  = row[geom_col]

            # 4) Origen y umbral según tipo de celda
            if cell in boundary_cells:
                origin = poly.representative_point()
                max_nm = threshold_boundary
            else:
                origin = poly.centroid
                max_nm = threshold_internal

            lat0 = origin.y

            # 5) Construye dinámicamente las aristas
            coords = list(poly.exterior.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            n = len(coords)
            if n < 2:
                continue

            edges = [(coords[i], coords[(i+1) % n]) for i in range(n)]
            crossings[cell] = {}

            # 6) Dispara un rayo por cada arista
            for ei, (p1, p2) in enumerate(edges):
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                nx_, ny_ = -dy, dx
                norm = math.hypot(nx_, ny_)
                if norm == 0:
                    continue
                nx_, ny_ = nx_ / norm, ny_ / norm

                # Asegura que apunte hacia fuera
                test_pt = Point(origin.x + nx_ * 1e-6, origin.y + ny_ * 1e-6)
                if poly.contains(test_pt):
                    nx_, ny_ = -nx_, -ny_

                # Escala a grados según latitud
                if abs(nx_) > abs(ny_):
                    deg = max_nm / (60.0 * math.cos(math.radians(lat0)))
                else:
                    deg = max_nm / 60.0

                end_pt = Point(origin.x + nx_ * deg,
                            origin.y + ny_ * deg)
                ray = LineString([origin, end_pt])

                # 7) Intersección contra la línea real del ACC
                inter = ray.intersection(acc_boundary)
                if inter.is_empty:
                    continue

                # Normaliza a lista de puntos
                if   inter.geom_type == 'Point':
                    pts = [inter]
                elif inter.geom_type == 'MultiPoint':
                    pts = list(inter.geoms)
                else:
                    try:
                        pts = [g for g in inter.geoms if g.geom_type == 'Point']
                    except:
                        continue
                if not pts:
                    continue

                # 8) Elige el punto más cercano dentro del umbral
                dists = []
                for pt in pts:
                    ddeg = origin.distance(pt)
                    if abs(pt.x - origin.x) > abs(pt.y - origin.y):
                        nm = ddeg * 60.0 * math.cos(math.radians(lat0))
                    else:
                        nm = ddeg * 60.0
                    dists.append(nm)

                i0 = min(range(len(dists)), key=lambda i: dists[i])
                dist_nm = dists[i0]
                if dist_nm > max_nm:
                    continue

                hit = pts[i0]
                seg = LineString([origin, hit])

                # 9) Qué celdas cruza el segmento
                cand = df_cells.iloc[list(sindex.intersection(seg.bounds))]
                crossed = cand[cand[geom_col].intersects(seg)][cell_col].tolist()
                crossed = [c for c in crossed if c != cell]

                crossings[cell][ei] = {
                    'distance_nm': dist_nm,
                    'cells_crossed': crossed
                }

        return crossings


    def reassign_crossed_cells_strict(graph, ray_crossings, assignment):
        """
        Reasigna iterativamente las celdas atravesadas por rayos al sector origen,
        siempre que **ambos** sectores (origen sin la celda y destino con la celda)
        permanezcan conectados. Repite hasta estabilizarse.
        """
        new_assign = deepcopy(assignment)

        # 1) Construir mapping cell -> {sectores destino}
        candidates = {}
        for src, edges in ray_crossings.items():
            sec_src = new_assign[src]
            for info in edges.values():
                for c in info['cells_crossed']:
                    if c not in new_assign or new_assign[c] == sec_src:
                        continue
                    candidates.setdefault(c, set()).add(sec_src)

        moved = True
        while moved:
            moved = False
            # Recorremos snapshot de candidatas para poder modificar en el loop
            for cell in list(candidates.keys()):
                sec0 = new_assign[cell]
                dests = candidates[cell]

                for sec1 in dests:
                    # Conjunto A: sector de origen sin la celda
                    A_nodes = {n for n,s in new_assign.items() if s==sec0} - {cell}
                    # Conjunto B: sector destino + la celda
                    B_nodes = {n for n,s in new_assign.items() if s==sec1} | {cell}

                    # Compruebo conectividad de ambos subconjuntos
                    ok_A = (not A_nodes) or nx.is_connected(graph.subgraph(A_nodes))
                    ok_B = (not B_nodes) or nx.is_connected(graph.subgraph(B_nodes))

                    if ok_A and ok_B:
                        # ¡Movemos definitivamente!
                        new_assign[cell] = sec1
                        moved = True
                        # Ya no lo consideramos candidato
                        del candidates[cell]
                        break
                # si lo movimos, pasamos al siguiente cell
            # vuelve a iterar si hubo algún movimiento
        return new_assign


    # --- 0) Asegúrate de haber calculado contorno_real al inicio ---
    from shapely.ops import unary_union
    poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
    union_poligonos = unary_union(poligonos_sectores)
    contorno_real = union_poligonos.boundary  # línea real de contorno

    # --- 1) Extrae las celdas frontera interiores y exteriores ---
    # 1.1. Frontera interior (inter–sectores sin tocar el ACC)
    internal_cells, gdf_internal = find_internal_edge_sharing_cells(
        df_cells, G, assignment,
        acc_contour_geom=contorno_real,
        cell_col='Cell_Name',
        geom_col='Polygon',
        sector_col='Final_Sector'
    )
    internal_cells = set(internal_cells)

    # 1.2. Frontera exterior (tocan el ACC)
    sector2acc, gdf_acc_border = find_acc_contour_cells(
        df_cells,
        acc_contour_geom=contorno_real,
        cell_col='Cell_Name',
        geom_col='Polygon',
        sector_col='Final_Sector'
    )
    acc_border_cells = {c for cells in sector2acc.values() for c in cells}

    # 1.3. Pool completo de celdas desde las que disparar rayos
    # Filtrar las celdas de los sectores que terminan en '_A1'
    cells_for_rays = {c for c, sec in assignment.items() if sec.endswith('_A1')}
    cells_for_rays = cells_for_rays.intersection(internal_cells | acc_border_cells)

    # --- 2) Lanza los rayos con distintos umbrales según tipo ---
    ray_crossings = find_rays_from_cells_mixed(
        df_cells,
        cells_list=list(cells_for_rays),  # Usar solo las celdas filtradas
        acc_contour_geom=contorno_real,      # es un LineString/MultiLineString
        boundary_cells=acc_border_cells,
        threshold_internal=60.0,
        threshold_boundary=30.0
    )

    # (opcional) Imprime los cruces detectados
    for src, edges in ray_crossings.items():
        for ei, info in edges.items():
            if info['cells_crossed']:
                print(f"Celda {src}, arista {ei}, "
                    f"dist={info['distance_nm']:.1f} NM → "
                    f"atraviesa {info['cells_crossed']}")

    # --- 3) Reasignación estricta manteniendo conectividad ---
    # Reasignar celdas cruzadas
    assignment = reassign_crossed_cells_strict(G, ray_crossings, assignment)

    # 4) Vuelca al GeoDataFrame
    df_cells['Final_Sector'] = df_cells['Cell_Name'].map(assignment)

    # --- 6. Volcar resultado final ---
    df_cells['Final_Sector'] = df_cells['Cell_Name'].map(assignment)
    print("Asignación final grabada en df_cells['Final_Sector'] (",
        len(set(assignment.values())), "sectores )")


    import matplotlib.patches as mpatches


    # Calcular complejidad total por sector final
    final_complexities = df_cells.groupby('Final_Sector')['Complexity'].sum()

    # Obtener lista de sectores y asignar colores
    sectors = list(final_complexities.index)
    colors = plt.cm.tab20.colors  # paleta tab20
    color_map = {sec: colors[i % len(colors)] for i, sec in enumerate(sectors)}

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujar cada celda coloreada por su sector final
    for _, row in df_cells.iterrows():
        poly = row['Polygon']
        sec = row['Final_Sector']
        color = color_map[sec]
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Preparar leyenda con valor de complejidad
    patches = [
        mpatches.Patch(color=color_map[sec],
                    label=f"{sec}: {final_complexities[sec]:.2f}")
        for sec in sectors
    ]
    ax.legend(handles=patches,
            title='Sector (Complejidad total)',
            loc='upper right',
            bbox_to_anchor=(1.3, 1))

    # Ajustes finales
    ax.set_title("Sectorización Final con Complejidad por Sector")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    df_cells['Sector'] = df_cells['Final_Sector']
    # Borro Final_Sector para evitar duplicar el nombre:
    df_cells.drop(columns=['Final_Sector'], inplace=True)


    return assignment


def optimize_pairs(original_assignment, df_cells, G, max_change=40, n_runs=5):

    # =============================================================================
    # 5. DEFINICIÓN DE FUNCIONES AUXILIARES PARA LA OPTIMIZACIÓN
    # =============================================================================
    def get_border_cells(assignment):
        
        border = []
        for cell in assignment:
            current_sec = assignment[cell]
            for neighbor in G.neighbors(cell):
                if assignment.get(neighbor) != current_sec:
                    border.append(cell)
                    break
        return border


    def is_sector_connected(assignment, sector):
        """
        Verifica si todas las celdas asignadas a 'sector' forman una única componente conexa
        en el grafo G, considerando adyacencia por arista.
        """
        cells_sector = [c for c, s in assignment.items() if s == sector]
        if len(cells_sector) <= 1:
            return True
        subG = G.subgraph(cells_sector)
        return nx.is_connected(subG)

    def check_move_connectivity(assignment, old_sec, new_sec, cell):
        """
        Verifica que, tras mover 'cell' de old_sec a new_sec, tanto old_sec como new_sec queden conexos.
        """
        new_assignment = assignment.copy()
        new_assignment[cell] = new_sec
        return is_sector_connected(new_assignment, old_sec) and is_sector_connected(new_assignment, new_sec)


    # =============================================================================
    # 5. DEFINICIÓN DE FUNCIONES AUXILIARES PARA LA OPTIMIZACIÓN
    # =============================================================================
    def get_border_cells(assignment):
        """
        Retorna la lista de celdas de borde: aquellas que tienen al menos un vecino en otro sector.
        """
        border = []
        for cell in assignment:
            current_sec = assignment[cell]
            for neighbor in G.neighbors(cell):
                if assignment.get(neighbor) != current_sec:
                    border.append(cell)
                    break
        return border

    def is_sector_connected(assignment, sector):
        """
        Verifica si todas las celdas asignadas a 'sector' forman una única componente conexa.
        """
        cells_sector = [c for c, s in assignment.items() if s == sector]
        if len(cells_sector) <= 1:
            return True
        subG = G.subgraph(cells_sector)
        return nx.is_connected(subG)

    # def check_move_connectivity(assignment, old_sec, new_sec, cell):
    #     """
    #     Verifica que, tras mover 'cell', ambos sectores queden conexos.
    #     """
    #     new_assignment = assignment.copy()
    #     new_assignment[cell] = new_sec
    #     return is_sector_connected(new_assignment, old_sec) and is_sector_connected(new_assignment, new_sec)

    def get_border_cell_pairs(assignment):
        """
        Retorna lista de tuplas (cell1, cell2, sector)
        tales que ambas celdas son frontera del mismo sector y se tocan por arista.
        """
        pairs = []
        border = get_border_cells(assignment)
        border_by_sector = {}
        for cell in border:
            sec = assignment[cell]
            border_by_sector.setdefault(sec, []).append(cell)
        for sec, cells in border_by_sector.items():
            for c1, c2 in itertools.combinations(cells, 2):
                poly1 = df_cells.loc[df_cells['Cell_Name']==c1, 'Polygon'].iloc[0]
                poly2 = df_cells.loc[df_cells['Cell_Name']==c2, 'Polygon'].iloc[0]
                if share_edge(poly1, poly2):
                    pairs.append((c1, c2, sec))
        return pairs

    def check_move_pair_connectivity(assignment, old_sec, new_sec, cell1, cell2):
        """
        Verifica que, tras mover cell1 y cell2 de old_sec a new_sec, ambos sectores queden conexos.
        """
        new_assignment = assignment.copy()
        new_assignment[cell1] = new_sec
        new_assignment[cell2] = new_sec
        return (is_sector_connected(new_assignment, old_sec)
                and is_sector_connected(new_assignment, new_sec))

    def changed_cells_count(assignment, original_assignment):
        """
        Cuenta cuántas celdas cambian de sector respecto a la asignación original.
        """
        counts = {}
        for cell, new_sec in assignment.items():
            orig_sec = original_assignment[cell]
            if new_sec != orig_sec:
                counts.setdefault(orig_sec, 0)
                counts[orig_sec] += 1
        return counts

    # =============================================================================
    # 6. NUEVA FUNCIÓN OBJETIVO CON REPARTO DE COMPLEJIDAD ENTRE SECTORES
    # =============================================================================
    def improved_objective(assignment):
        """
        Minimiza varianza muestral de complejidad total + penalización por diferencia extrema.
        """
        comp_by_sector = {}
        for cell, sec in assignment.items():
            comp = df_cells.loc[df_cells['Cell_Name'] == cell, 'Complexity'].values[0]
            comp_by_sector.setdefault(sec, 0)
            comp_by_sector[sec] += comp

        values = list(comp_by_sector.values())
        n_sec = len(values)
        overall_avg = sum(values) / n_sec if n_sec else 0

        if n_sec > 1:
            variance = sum((v - overall_avg) ** 2 for v in values) / (n_sec - 1)
        else:
            variance = 0

        diff_penalty = max(values) - min(values) if values else 0
        return variance + diff_penalty

    # =============================================================================
    # 7. PARÁMETROS INICIALES Y RESTRICCIONES
    # =============================================================================
    initial_counts = df_cells.groupby('Sector').size().to_dict()
    max_change = 40
    original_assignment = df_cells.set_index('Cell_Name')['Sector'].to_dict()

    from shapely.ops import unary_union

    # =============================================================================
    # Detectar celdas en el borde EXTERIOR del dominio (exentas de la regla de aislamiento)
    # =============================================================================
    union_poly = unary_union(df_cells['Polygon'].tolist())
    boundary_line = union_poly.boundary

    outer_border_cells = set()
    for _, row in df_cells.iterrows():
        poly = row['Polygon']
        inter = poly.intersection(boundary_line)
        if (not inter.is_empty
            and inter.geom_type in ('LineString','MultiLineString')
            and inter.length > 1e-8):
            outer_border_cells.add(row['Cell_Name'])


    # =============================================================================
    # 8. OPTIMIZACIÓN ITERATIVA: MOVER SÓLO PARES DE CELDAS - 5 CORRIDAS
    # =============================================================================
    n_runs = 5
    best_assignment = None
    best_obj = float('inf')

    for run in range(1, n_runs + 1):
        print(f"\n========== INICIO DE LA CORRIDA {run} ==========")
        assignment_run = original_assignment.copy()
        current_counts_run = initial_counts.copy()
        improved = True
        iteration = 0

        while improved:
            improved = False
            iteration += 1
            current_obj = improved_objective(assignment_run)
            print(f"Iteración {iteration}: Objetivo = {current_obj:.2f}")

            # 1) Generar todos los pares frontera contiguos y del mismo sector
            border_pairs = get_border_cell_pairs(assignment_run)
            random.shuffle(border_pairs)

            # 2) Complejidad acumulada por sector (para priorizar destinos)
            comp_by_sector = {}
            for cell, sec in assignment_run.items():
                comp_by_sector.setdefault(sec, 0)
                comp_by_sector[sec] += df_cells.loc[
                    df_cells['Cell_Name'] == cell, 'Complexity'
                ].iloc[0]

            # 3) Intentar mover cada par de celdas
            for cell1, cell2, old_sec in border_pairs:
                # Sectores candidatos: intersección de vecinos de ambas celdas, sin incluir el actual
                neigh1 = {assignment_run[n] for n in G.neighbors(cell1)}
                neigh2 = {assignment_run[n] for n in G.neighbors(cell2)}
                candidate_sectors = sorted(
                    (neigh1 & neigh2) - {old_sec},
                    key=lambda s: comp_by_sector.get(s, 0)
                )

                for cand in candidate_sectors:
                    # a) Ambas celdas deben tener al menos un vecino en cand
                    if not any(assignment_run[n] == cand for n in G.neighbors(cell1)):
                        continue
                    if not any(assignment_run[n] == cand for n in G.neighbors(cell2)):
                        continue

                    # b) Conectividad tras mover el par
                    if not check_move_pair_connectivity(
                        assignment_run, old_sec, cand, cell1, cell2
                    ):
                        continue

                    # c) Restricción neta de cambios (por sector) no excede max_change
                    new_orig = current_counts_run[old_sec] - 2
                    new_cand = current_counts_run[cand] + 2
                    if (abs(new_orig - initial_counts[old_sec]) > max_change or
                        abs(new_cand - initial_counts[cand]) > max_change):
                        continue

                    # d) Cada sector original no pierde más de max_change desde la original
                    temp_assign = assignment_run.copy()
                    temp_assign[cell1] = cand
                    temp_assign[cell2] = cand
                    cambios = changed_cells_count(temp_assign, original_assignment)
                    if any(v > max_change for v in cambios.values()):
                        continue

                    # e) Impedir que cualquier otra celda (no exterior) quede "aislada"
                    #    (conectada por menos de 2 aristas en su sector)
                    aisla = False
                    for c, sec_c in temp_assign.items():
                        if c in outer_border_cells:
                            continue
                        same_sec_neighbors = sum(
                            1 for n in G.neighbors(c)
                            if temp_assign[n] == sec_c
                        )
                        if same_sec_neighbors < 2:
                            aisla = True
                            break
                    if aisla:
                        continue

                    # f) Evaluar mejora en la función objetivo
                    new_obj = improved_objective(temp_assign)
                    if new_obj < current_obj:
                        print(
                            f"Moviendo par ({cell1},{cell2}) de {old_sec} → {cand}: "
                            f"{current_obj:.2f} → {new_obj:.2f}"
                        )
                        assignment_run = temp_assign
                        current_counts_run[old_sec] = new_orig
                        current_counts_run[cand] = new_cand
                        improved = True
                        break

                if improved:
                    break

        final_obj = improved_objective(assignment_run)
        print(f"Corrida {run} finalizada. Objetivo final: {final_obj:.2f}")
        if final_obj < best_obj:
            best_obj = final_obj
            best_assignment = assignment_run.copy()

    print("\n========== OPTIMIZACIÓN FINALIZADA ==========")
    print("Mejor objetivo obtenido:", best_obj)

    # =============================================================================
    # 8. GUARDAR Y/O VISUALIZAR LA NUEVA ASIGNACIÓN OPTIMIZADA (MEJOR CORRIDA)
    # =============================================================================
    df_cells['Optimized_Sector'] = df_cells['Cell_Name'].map(best_assignment)

    print("Ejemplo de asignación optimizada:")
    print(df_cells[['Cell_Name', 'Sector', 'Optimized_Sector']].head())


    # =============================================================================
    # 9. CALCULAR LA COMPLEJIDAD POR SECTOR ANTES Y DESPUES DE LA OPTIMIZACIÓN
    # =============================================================================

    # 9.1 Complejidad por sector antes de la optimización (configuración inicial)
    initial_complexity_by_sector = df_cells.groupby('Sector')['Complexity'].agg(['sum', 'count'])
    initial_complexity_by_sector['avg_complexity'] = initial_complexity_by_sector['sum']
    # initial_complexity_by_sector['avg_complexity'] = initial_complexity_by_sector['sum'] / initial_complexity_by_sector['count']
    # 9.2 Complejidad por sector después de la optimización (configuración optimizada)
    optimized_complexity_by_sector = df_cells.groupby('Optimized_Sector')['Complexity'].agg(['sum', 'count'])
    optimized_complexity_by_sector['avg_complexity'] = optimized_complexity_by_sector['sum']
    # optimized_complexity_by_sector['avg_complexity'] = optimized_complexity_by_sector['sum'] / optimized_complexity_by_sector['count']
    # ---------------------------------------------------------------------
    # Visualización comparativa: Configuración Inicial vs Optimizada
    # ---------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    # ---------------------------
    # Visualización de la configuración inicial
    # ---------------------------
    # Obtener los sectores únicos de la asignación inicial (columna 'Sector')
    sectors_init = df_cells['Sector'].unique()
    colors_init = plt.cm.tab20(np.linspace(0, 1, len(sectors_init)))
    color_map_init = dict(zip(sectors_init, colors_init))

    for _, row in df_cells.iterrows():
        poly = row['Polygon']
        sec = row['Sector']
        color = color_map_init.get(sec, 'lightgray')
        x, y = poly.exterior.xy
        ax1.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    patches_init = [mpatches.Patch(color=color_map_init[s], label=s) for s in sectors_init if s != 'sin_sector']
    ax1.legend(handles=patches_init, loc='upper right', title='Sectores iniciales')
    ax1.set_title("Sectorización Inicial")
    ax1.set_xlabel("Longitud [º]")
    ax1.set_ylabel("Latitud [º]")
    ax1.set_aspect('equal')

    # ---------------------------
    # Visualización de la configuración optimizada
    # ---------------------------
    # Obtener los sectores únicos de la asignación optimizada (columna 'Optimized_Sector')
    sectors_opt = df_cells['Optimized_Sector'].unique()
    colors_opt = plt.cm.tab20(np.linspace(0, 1, len(sectors_opt)))
    color_map_opt = dict(zip(sectors_opt, colors_opt))

    for _, row in df_cells.iterrows():
        poly = row['Polygon']
        sec = row['Optimized_Sector']
        color = color_map_opt.get(sec, 'lightgray')
        x, y = poly.exterior.xy
        ax2.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    patches_opt = [mpatches.Patch(color=color_map_opt[s], label=s) for s in sectors_opt if s != 'sin_sector']
    ax2.legend(handles=patches_opt, loc='upper right', title='Sectores optimizados')
    ax2.set_title("Sectorización Optimizada")
    ax2.set_xlabel("Longitud [º]")
    ax2.set_ylabel("Latitud [º]")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()


    # =============================================================================
    # 10. VISUALIZACIÓN DE LA SECTORIZACIÓN INICIAL Y OPTIMIZADA CON LA COMPLEJIDAD POR SECTOR
    # =============================================================================
    # Visualización de la sectorización inicial y optimizada con complejidad
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    # ---------------------------
    # Visualización de la sectorización inicial con complejidad
    # ---------------------------
    sectors_init = df_cells['Sector'].unique()
    colors_init = plt.cm.viridis(np.linspace(0, 1, len(sectors_init)))
    color_map_init = {sector: colors_init[i] for i, sector in enumerate(sectors_init)}

    for idx, row in df_cells.iterrows():
        poly = row['Polygon']
        sec = row['Sector']
        color = color_map_init.get(sec, 'lightgray')
        x, y = poly.exterior.xy
        # Dibuja la celda con el color correspondiente
        ax1.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        # Si la celda cambió de sector, se dibuja un contorno rojo sobre ella
        if row['Sector'] != row['Optimized_Sector']:
            ax1.plot(x, y, color='red', linewidth=2)

    # Se agrega una leyenda para indicar las celdas modificadas
    red_patch = mpatches.Patch(edgecolor='red', facecolor='none', label='Celdas cambiadas', linewidth=2)
    patches_init = [mpatches.Patch(color=color_map_init[s], label=f'{s}: {initial_complexity_by_sector.loc[s, "avg_complexity"]:.2f}')
                    for s in sectors_init if s != 'sin_sector']
    patches_init.append(red_patch)
    ax1.legend(handles=patches_init, loc='upper right', title='Sectores iniciales')

    ax1.set_title("Sectorización Inicial con Complejidad")
    ax1.set_xlabel("Longitud [º]")
    ax1.set_ylabel("Latitud [º]")
    ax1.set_aspect('equal')

    # ---------------------------
    # Visualización de la sectorización optimizada con complejidad
    # ---------------------------
    sectors_opt = df_cells['Optimized_Sector'].unique()
    colors_opt = plt.cm.viridis(np.linspace(0, 1, len(sectors_opt)))
    color_map_opt = {sector: colors_opt[i] for i, sector in enumerate(sectors_opt)}

    for idx, row in df_cells.iterrows():
        poly = row['Polygon']
        sec = row['Optimized_Sector']
        color = color_map_opt.get(sec, 'lightgray')
        x, y = poly.exterior.xy
        ax2.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        # Se marca en rojo si la celda cambió de sector
        if row['Sector'] != row['Optimized_Sector']:
            ax2.plot(x, y, color='red', linewidth=2)

    red_patch_opt = mpatches.Patch(edgecolor='red', facecolor='none', label='Celdas cambiadas', linewidth=2)
    patches_opt = [mpatches.Patch(color=color_map_opt[s], label=f'{s}: {optimized_complexity_by_sector.loc[s, "avg_complexity"]:.2f}')
                for s in sectors_opt if s != 'sin_sector']
    patches_opt.append(red_patch_opt)
    ax2.legend(handles=patches_opt, loc='upper right', title='Sectores optimizados')

    ax2.set_title("Sectorización Optimizada con Complejidad")
    ax2.set_xlabel("Longitud [º]")
    ax2.set_ylabel("Latitud [º]")
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()




    ruta_pkl = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\DF_T_REAL_CELDA.pkl'
    df_flujos = pd.read_pickle(ruta_pkl)

    # -----------------------------------------------------------------------------
    # 2. CREAR LA GEOMETRÍA DE CADA FLUJO
    # -----------------------------------------------------------------------------
   
    df_flujos['geometry'] = df_flujos.apply(
        lambda row: LineString([
            (row['lon_cell_in'], row['lat_cell_in']),
            (row['lon_cell_out'], row['lat_cell_out'])
        ]),
        axis=1
    )
    gdf_flujos = gpd.GeoDataFrame(df_flujos, geometry='geometry')

    # -----------------------------------------------------------------------------
    # 3. FILTRAR LAS CELDAS QUE CAMBIARON DE SECTOR
    # -----------------------------------------------------------------------------

    changed_cells = df_cells[df_cells['Sector'] != df_cells['Optimized_Sector']]
    changed_cell_names = changed_cells['Cell_Name'].unique()

    # -----------------------------------------------------------------------------
    # 4. FILTRAR LOS FLUJOS QUE PASAN POR CELDAS CAMBIADAS
    # -----------------------------------------------------------------------------

    changed_flujos = gdf_flujos[gdf_flujos['Cell_Name'].isin(changed_cell_names)]

    # -----------------------------------------------------------------------------
    # 5. REPRESENTAR EL MALLADO Y SUPERPONER LOS FLUJOS CAMBIADOS
    # -----------------------------------------------------------------------------
    fig, ax_cells = plt.subplots()

    # Dibujar el polígono del ACC
    x_acc, y_acc = poligono_ACC.exterior.xy
    ax_cells.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')

    # Dibujar las celdas del mallado
    for _, row in DF_MALLADO.iterrows():
        polygon = row['Polygon']  # Obtener el polígono de la celda
        x, y = polygon.exterior.xy  # Coordenadas del contorno
        ax_cells.plot(x, y, color='gray', alpha=0.5)

    # Dibujar los flujos (en rojo) correspondientes a las celdas cambiadas
    primer_flujo = True
    for _, row in changed_flujos.iterrows():
        linea = row['geometry']
        if linea is not None:
            x_line, y_line = linea.xy
            if primer_flujo:
                ax_cells.plot(x_line, y_line, color='red', linewidth=1.5, label='Flujos en celdas cambiadas')
                primer_flujo = False
            else:
                ax_cells.plot(x_line, y_line, color='red', linewidth=1.5)

    # Configurar la gráfica
    ax_cells.set_xlim(min_lon, max_lon)
    ax_cells.set_ylim(min_lat, max_lat)
    ax_cells.set_aspect('equal')
    ax_cells.set_title("MALLADO DEL ESPACIO AÉREO CON FLUJOS EN CELDAS CAMBIADAS")
    ax_cells.set_xlabel('LONGITUD [º]')
    ax_cells.set_ylabel('LATITUD [º]')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

    # Guardar la figura
    nombre_figura = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo de estudio con flujos en celdas cambiadas.png'
    plt.savefig(nombre_figura, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------------------
    # PROCESAMIENTO DEL DATASET DE TRÁFICO PARA IDENTIFICAR VUELOS EN CELDAS CAMBIADAS
    # -----------------------------------------------------------------------------

    # Definir la ruta del directorio de tráfico
    PATH_TRAFICO = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\2. bloque complejidad\\Datos\\DATASET ENTRADA PREDICCIONES\\'
    # Cargar el dataset de tráfico desde un archivo PKL
    df_trafico = pd.read_pickle(PATH_TRAFICO + 'dataset_vuelos_reales_2022-06-01.pkl')

    # Crear un mapeo (sin eliminar duplicados) entre 'Flujo_Clusterizado' y 'Cell_Name' a partir de los flujos cambiados.
    # Esto permite que, si un mismo flujo pasa por varias celdas cambiadas, se generen varias filas.
    mapping = changed_flujos[['Flujo_Clusterizado', 'Cell_Name']]

    # Realizar el merge para añadir la columna 'Cell_Name' a df_trafico según 'Flujo_Clusterizado'
    df_trafico_updated = df_trafico.merge(mapping, on='Flujo_Clusterizado', how='left')

    # Filtrar los registros donde se asignó una celda modificada (es decir, donde 'Cell_Name' no es nulo)
    df_trafico_changed = df_trafico_updated[df_trafico_updated['Cell_Name'].notna()]

    # Seleccionar las columnas de interés: 'flightkey', 'Flujo_Clusterizado' y 'Cell_Name'
    # Si un mismo vuelo aparece varias veces en la misma celda y flujo, se eliminan duplicados exactos.
    result_df = df_trafico_changed[['flightKey', 'Flujo_Clusterizado', 'Cell_Name']].drop_duplicates()

    # -----------------------------------------------------------------------------
    # GUARDAR LOS RESULTADOS DE LOS VUELOS EN CELDAS CAMBIADAS
    # -----------------------------------------------------------------------------

    PATH_CAEHO = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\CAEHO\\'
    # GUARDAR LOS RESULTADOS DE LOS VUELOS EN CELDAS CAMBIADAS
    result_df.to_csv(PATH_CAEHO + 'Vuelos_en_celdas_cambiadas.csv', index=False, sep=';')
    result_df.to_pickle(PATH_CAEHO + 'Vuelos_en_celdas_cambiadas.pkl')

    print("finalizado caeho")



    return best_assignment


# Empezamos con la configuración inicial de sectores
initial_assignment = df_cells.set_index('Cell_Name')['Sector'].to_dict()
initial_num_sectors = len(set(initial_assignment.values()))  # Número inicial de sectores

# Definir cell_complexity (Diccionario de complejidad de las celdas)
cell_complexity = df_cells.set_index('Cell_Name')['Complexity'].to_dict()

# Número de iteraciones que se quieren hacer
num_runs = int(input("¿Cuántas veces quieres crear+optimizar un nuevo sector? "))

# Variable para llevar el seguimiento de la asignación de sectores
assignment = initial_assignment.copy()  # Copia de la asignación inicial

# Iterar sobre el número de ejecuciones que quieres hacer
for i in range(1, num_runs + 1):
    target_k = initial_num_sectors + i  # Nuevo número de sectores (se incrementa con cada iteración)
    print(f"\n=== Iteración {i}: objetivo {target_k} sectores ===")

    # 1) Crear nuevos sectores hasta el objetivo 'target_k', pero usando la asignación de la iteración anterior
    assignment = split_to_k(assignment, cell_complexity, G)

    # 2) Optimizar la asignación de los sectores
    assignment = optimize_pairs(assignment, df_cells, G, max_change=40, n_runs=5)

    # Al final de cada iteración, actualizamos la asignación
    df_cells[f'Sector_run_{i}'] = df_cells['Cell_Name'].map(assignment)

# Al final de todas las iteraciones, puedes tener la asignación final
df_cells['Final_Sector_MultiRun'] = df_cells['Cell_Name'].map(assignment)
print("Proceso completo. Total de sectores finales:", len(set(assignment.values())))
