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


warnings.filterwarnings("ignore")

##### Registro del inicio del tiempo
inicio_tiempo = tm.time()


##############################################


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

ax_cells.set_title("AIRSPACE MESHING WITH 15NM x 15NM CELLS")
ax_cells.set_aspect('equal')
ax_cells.set_xlabel('LONGITUDE[º]')
ax_cells.set_ylabel('LATITUDE[º]')
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
cbar_cell = fig.colorbar(sm_cell, ax=ax_complejidad_cells, label='Valor Complejidad', shrink=0.8, aspect=10)

# Configurar la gráfica
ax_complejidad_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
ax_complejidad_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
ax_complejidad_cells.set_aspect('equal')

ax_complejidad_cells.set_title("COMPLEJIDAD DE LAS CELDAS - MAPA DE COLOR")
ax_complejidad_cells.set_aspect('equal')
ax_complejidad_cells.set_xlabel('LONGITUD[º]')
ax_complejidad_cells.set_ylabel('LATITUD[º]')
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
# 1. IMPORTAR RESULTADOS PREVIOS: MALLADO Y CONFIGURACIÓN REAL
# =============================================================================
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from shapely.geometry import Polygon
import matplotlib.patches as mpatches
import itertools

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
max_change = 60
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
n_runs = 1
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
ax1.legend(handles=patches_init, loc='upper right', title='Initial Sectors')
ax1.set_title("Initial Sectorization")
ax1.set_xlabel("Longitude [º]")
ax1.set_ylabel("Latitude [º]")
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
ax2.legend(handles=patches_opt, loc='upper right', title='Optimized sectors')
ax2.set_title("Optimized Sectors")
ax2.set_xlabel("Longitude [º]")
ax2.set_ylabel("Latitude [º]")
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
red_patch = mpatches.Patch(edgecolor='red', facecolor='none', label='Changed cells', linewidth=2)
patches_init = [mpatches.Patch(color=color_map_init[s], label=f'{s}: {initial_complexity_by_sector.loc[s, "avg_complexity"]:.2f}')
                for s in sectors_init if s != 'sin_sector']
patches_init.append(red_patch)
ax1.legend(handles=patches_init, loc='upper right', title='Initial Sectors')

ax1.set_title("Initial Sectorization with Complexity")
ax1.set_xlabel("Longitude [º]")
ax1.set_ylabel("Latitude [º]")
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

red_patch_opt = mpatches.Patch(edgecolor='red', facecolor='none', label='Changed cells', linewidth=2)
patches_opt = [mpatches.Patch(color=color_map_opt[s], label=f'{s}: {optimized_complexity_by_sector.loc[s, "avg_complexity"]:.2f}')
               for s in sectors_opt if s != 'sin_sector']
patches_opt.append(red_patch_opt)
ax2.legend(handles=patches_opt, loc='upper right', title='Optimized Sectors')

ax2.set_title("Optimized Sectorization with Complexity")
ax2.set_xlabel("Longitude [º]")
ax2.set_ylabel("Latitude [º]")
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- TIEMPO TRANSCURRIDO -----------------------------------------------  #
# -------------------------------------------------------------------------------------------------------------------- #

# Registrar el tiempo de finalización
fin_tiempo = tm.time()
# Calcular la diferencia de tiempo
tiempo_transcurrido = fin_tiempo - inicio_tiempo
print('--------------------------------------------------------------------------------------------')
print('')
print("Tiempo transcurrido:", tiempo_transcurrido, "segundos")


# -----------------------------------------------------------------------------
# 1. CARGAR EL DATASET COMPLETO DE FLUJOS DESDE EL ARCHIVO PKL
# -----------------------------------------------------------------------------
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


import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

# Asumimos que tienes un GeoDataFrame 'df_cells' con las celdas y su asignación de sectores optimizada.

# Primero, obtenemos la geometría de las celdas y sus sectores
gdf_cells = gpd.GeoDataFrame(df_cells, geometry='Polygon')

# Agregamos la columna de los sectores optimizados
gdf_cells['Sector'] = df_cells['Optimized_Sector']

# Creamos un grafo para las celdas y las fronteras compartidas
def get_shared_borders(cells):
    """
    Devuelve las aristas compartidas entre las celdas que tienen sectores diferentes.
    """
    shared_borders = []
    n_cells = len(cells)
    for i in range(n_cells):
        cell_i, poly_i, sector_i = cells[i]
        for j in range(i + 1, n_cells):
            cell_j, poly_j, sector_j = cells[j]
            # Si las celdas comparten un borde y están en sectores diferentes
            if poly_i.intersects(poly_j) and sector_i != sector_j:
                inter = poly_i.intersection(poly_j)
                if inter.geom_type == 'LineString' and inter.length > 0:
                    shared_borders.append(inter)
    return shared_borders

# Obtener las celdas y sus geometrías junto con los sectores
cells_info = [(row['Cell_Name'], row['Polygon'], row['Sector']) for idx, row in gdf_cells.iterrows()]

# Obtener las fronteras compartidas
shared_borders = get_shared_borders(cells_info)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(12, 8))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Dibujar las celdas con los sectores optimizados
for _, row in gdf_cells.iterrows():
    poly = row['Polygon']
    sector = row['Sector']
    
    # Aquí se usa un mapa de colores para cada sector
    color_map = {
        'LECMSAN': 'green',
        'LECMASI': 'blue',
        'LECMBLI': 'purple',
        'LECMPAI': 'yellow',
        'LECMDGI': 'gray',
        'LECMR1I': 'cyan',  
        'LECMDPI': 'magenta',  
        'LECMSAB': 'orange',  
        'LECMSAI': 'red'  
    }
    color = color_map.get(sector, 'lightgray')
    
    # Añadir el sector a la lista de sectores visibles
    visible_sectors.add(sector)
    
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# Dibujar las fronteras compartidas con líneas oscuras
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Títulos y etiquetas
ax.set_title("Sectorización Optimizada")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')

# Crear la leyenda solo con los sectores visibles
legend_labels = {
    'LECMSAN': 'LECMSAN ',
    'LECMASI': 'LECMASI ',
    'LECMBLI': 'LECMBLI ',
    'LECMPAI': 'LECMPAI ',
    'LECMDGI': 'LECMDGI ',
    'LECMR1I': 'LECMR1I ',
    'LECMDPI': 'LECMDPI ',
    'LECMSAB': 'LECMSAB ',
    'LECMSAI': 'LECMSAI '
}

# Filtrar los sectores visibles en el gráfico
visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# Crear los patches para la leyenda solo de los sectores visibles
patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------- 
# 1. IDENTIFICAR LAS CELDAS DE LA FRONTERA ENTRE SECTORES EN LA SECTORIZACIÓN OPTIMIZADA
# ----------------------------------------------------------------------------- 

# Usamos la asignación optimizada para identificar las celdas fronterizas
optimized_assignment = df_cells.set_index('Cell_Name')['Optimized_Sector'].to_dict()

# Identificar las celdas fronterizas en la sectorización optimizada
border_cells_optimized = get_border_cells(optimized_assignment)
print(f"Celdas fronterizas optimizadas encontradas: {border_cells_optimized}")

# ----------------------------------------------------------------------------- 
# 2. FILTRAR LOS FLUJOS QUE PERTENECEN A CELDAS FRONTERIZAS DE LA SECTORIZACIÓN OPTIMIZADA
# ----------------------------------------------------------------------------- 

# Filtrar los flujos cuyos nombres de celda están en la lista de celdas fronterizas optimizadas
border_flujos_optimized = gdf_flujos[gdf_flujos['Cell_Name'].isin(border_cells_optimized)]

# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS DE CELDAS FRONTERIZAS DE LA SECTORIZACIÓN OPTIMIZADA
# # ----------------------------------------------------------------------------- 
# fig, ax_cells = plt.subplots()

# # Dibujar el polígono del ACC
# x_acc, y_acc = poligono_ACC.exterior.xy
# ax_cells.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')

# # Dibujar las celdas del mallado (sectorización optimizada)
# for _, row in DF_MALLADO.iterrows():
#     polygon = row['Polygon']  # Obtener el polígono de la celda
#     x, y = polygon.exterior.xy  # Coordenadas del contorno
#     sec = optimized_assignment.get(row['Cell_Name'], 'sin_sector')  # Asignación optimizada
#     # Usamos un color distinto para las celdas fronterizas
#     if row['Cell_Name'] in border_cells_optimized:
#         ax_cells.fill(x, y, color='orange', alpha=0.7)  # Color para las celdas frontera
#     else:
#         ax_cells.fill(x, y, color='gray', alpha=0.5)

# # Dibujar los flujos (en rojo) correspondientes a las celdas fronterizas optimizadas
# primer_flujo = True
# for _, row in border_flujos_optimized.iterrows():
#     linea = row['geometry']
#     if linea is not None:
#         x_line, y_line = linea.xy
#         if primer_flujo:
#             ax_cells.plot(x_line, y_line, color='red', linewidth=1.5, label='Flujos en celdas fronterizas optimizadas')
#             primer_flujo = False
#         else:
#             ax_cells.plot(x_line, y_line, color='red', linewidth=1.5)

# # Configurar la gráfica
# ax_cells.set_xlim(min_lon, max_lon)
# ax_cells.set_ylim(min_lat, max_lat)
# ax_cells.set_aspect('equal')
# ax_cells.set_title("MALLADO DEL ESPACIO AÉREO OPTIMIZADO CON FLUJOS EN CELDAS FRONTERIZAS")
# ax_cells.set_xlabel('LONGITUD [º]')
# ax_cells.set_ylabel('LATITUD [º]')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# # Guardar la figura
# nombre_figura = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo con flujos en celdas fronterizas optimizadas.png'
# plt.savefig(nombre_figura, format='png', dpi=300, bbox_inches='tight')
# plt.show()



# import math

# # ----------------------------------------------------------------------------- 
# # 1. FUNCIONES PARA CALCULAR LA DISTANCIA ENTRE LAS COORDENADAS DE ENTRADA Y SALIDA
# # ----------------------------------------------------------------------------- 
# def calculate_distance(lat_in, lon_in, lat_out, lon_out):
#     """
#     Calcula la distancia entre dos puntos geográficos en millas náuticas.
#     Utiliza la fórmula de distancia euclidiana simplificada (dada la pequeña escala de la distancia en grados).
#     """
#     delta_lat = lat_in - lat_out
#     delta_lon = lon_in - lon_out
#     distance = math.sqrt(delta_lat**2 + delta_lon**2) * 60  # La constante 60 convierte grados a millas náuticas
#     return distance

# # ----------------------------------------------------------------------------- 
# # 2. FILTRAR LOS FLUJOS QUE PERTENECEN A CELDAS FRONTERIZAS Y CUMPLEN LA CONDICIÓN DE DISTANCIA
# # ----------------------------------------------------------------------------- 
# # Calcular la distancia para cada flujo y filtrar aquellos menores a 10 millas náuticas
# max_distance_nautical = 80  # 20 millas náuticas

# # Lista para almacenar los flujos filtrados
# filtered_flujos = []

# for _, row in border_flujos_optimized.iterrows():
#     lat_in = row['lat_cell_in']
#     lon_in = row['lon_cell_in']
#     lat_out = row['lat_cell_out']
#     lon_out = row['lon_cell_out']
    
#     # Calcular la distancia del flujo
#     distance = calculate_distance(lat_in, lon_in, lat_out, lon_out)
    
#     # Si la distancia es menor a 10 millas náuticas, agregarlo a la lista
#     if distance < max_distance_nautical:
#         filtered_flujos.append(row)

# # Crear un GeoDataFrame con los flujos filtrados
# gdf_filtered_flujos = gpd.GeoDataFrame(filtered_flujos, geometry='geometry')

# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS FILTRADOS
# # ----------------------------------------------------------------------------- 
# fig, ax_cells = plt.subplots()

# # Dibujar el polígono del ACC
# x_acc, y_acc = poligono_ACC.exterior.xy
# ax_cells.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')

# # Dibujar las celdas del mallado (sectorización optimizada)
# for _, row in DF_MALLADO.iterrows():
#     polygon = row['Polygon']  # Obtener el polígono de la celda
#     x, y = polygon.exterior.xy  # Coordenadas del contorno
#     sec = optimized_assignment.get(row['Cell_Name'], 'sin_sector')  # Asignación optimizada
#     # Usamos un color distinto para las celdas frontera
#     if row['Cell_Name'] in border_cells_optimized:
#         ax_cells.fill(x, y, color='orange', alpha=0.7)  # Color para las celdas frontera
#     else:
#         ax_cells.fill(x, y, color='gray', alpha=0.5)

# # Dibujar los flujos filtrados (en rojo) correspondientes a las celdas fronterizas optimizadas
# for _, row in gdf_filtered_flujos.iterrows():
#     linea = row['geometry']
#     if linea is not None:
#         x_line, y_line = linea.xy
#         ax_cells.plot(x_line, y_line, color='red', linewidth=1.5)

# # Configurar la gráfica
# ax_cells.set_xlim(min_lon, max_lon)
# ax_cells.set_ylim(min_lat, max_lat)
# ax_cells.set_aspect('equal')
# ax_cells.set_title("MALLADO DEL ESPACIO AÉREO OPTIMIZADO CON FLUJOS MENORES A 80 MILLAS NÁUTICAS EN CELDAS FRONTERIZAS")
# ax_cells.set_xlabel('LONGITUD [º]')
# ax_cells.set_ylabel('LATITUD [º]')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# # Guardar la figura
# nombre_figura = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo con flujos menores a 10 millas náuticas.png'
# plt.savefig(nombre_figura, format='png', dpi=300, bbox_inches='tight')
# plt.show()




# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS FILTRADOS SOBRE LA SECTORIZACIÓN
# # ----------------------------------------------------------------------------- 
# fig, ax = plt.subplots(figsize=(12, 8))

# # Crear una lista para los sectores que realmente aparecen en el gráfico
# visible_sectors = set()

# # Crear un mapa de colores para los sectores
# color_map = {
#     'LECMSAN': 'green',
#     'LECMASI': 'blue',
#     'LECMBLI': 'purple',
#     'LECMPAI': 'yellow',
#     'LECMDGI': 'gray',
#     'LECMR1I': 'cyan',  
#     'LECMDPI': 'magenta',  
#     'LECMSAB': 'orange',  
#     'LECMSAI': 'red'  
# }

# # Dibujar las celdas con los sectores optimizados
# for _, row in gdf_cells.iterrows():
#     poly = row['Polygon']
#     sector = row['Sector']
    
#     color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
#     # Añadir el sector a la lista de sectores visibles
#     visible_sectors.add(sector)
    
#     # Dibujar la celda
#     x, y = poly.exterior.xy
#     ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# # Dibujar las fronteras compartidas con líneas oscuras
# for border in shared_borders:
#     x, y = border.xy
#     ax.plot(x, y, color='black', linewidth=2)

# # Dibujar los flujos filtrados (en rojo) correspondientes a las celdas fronterizas optimizadas
# for _, row in gdf_filtered_flujos.iterrows():
#     linea = row['geometry']
#     if linea is not None:
#         x_line, y_line = linea.xy
#         ax.plot(x_line, y_line, color='red', linewidth=1.5, label='Flujos < 20 millas')

# # Títulos y etiquetas
# ax.set_title("Sectorización Optimizada con Flujos Menores a 80 Millas Náuticas")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')

# # Crear la leyenda solo con los sectores visibles
# legend_labels = {
#     'LECMSAN': 'LECMSAN ',
#     'LECMASI': 'LECMASI ',
#     'LECMBLI': 'LECMBLI ',
#     'LECMPAI': 'LECMPAI ',
#     'LECMDGI': 'LECMDGI ',
#     'LECMR1I': 'LECMR1I ',
#     'LECMDPI': 'LECMDPI ',
#     'LECMSAB': 'LECMSAB ',
#     'LECMSAI': 'LECMSAI '
# }

# # Filtrar los sectores visibles en el gráfico
# visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# # Crear los patches para la leyenda solo de los sectores visibles
# patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# # Añadir leyenda para los flujos
# patches.append(mpatches.Patch(color='red', label='Flujos < 20 millas náuticas'))

# # Mostrar la leyenda
# ax.legend(handles=patches, loc='upper right')

# # Ajuste y visualización final
# plt.tight_layout()
# plt.show()



# # ----------------------------------------------------------------------------- 
# # 1. FUNCIONES PARA VERIFICAR LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# # ----------------------------------------------------------------------------- 

# def count_intersections(flow, shared_borders):
#     """
#     Cuenta cuántas veces un flujo corta las fronteras compartidas entre sectores.
#     """
#     intersection_count = 0
#     for border in shared_borders:
#         if flow.intersects(border):  # Si el flujo corta la frontera
#             intersection_count += 1
#     return intersection_count

# # ----------------------------------------------------------------------------- 
# # 2. FILTRAR LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# # ----------------------------------------------------------------------------- 
# filtered_flows = []

# for _, row in gdf_filtered_flujos.iterrows():
#     flow = row['geometry']
    
#     # Verificar cuántas veces el flujo corta las fronteras
#     intersections = count_intersections(flow, shared_borders)
    
#     # Si corta 2 veces las fronteras, lo mantenemos
#     if intersections >= 2:
#         filtered_flows.append(row)

# # Crear un GeoDataFrame con los flujos filtrados
# gdf_filtered_flows_2_cross = gpd.GeoDataFrame(filtered_flows, geometry='geometry')

# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# # ----------------------------------------------------------------------------- 
# fig, ax = plt.subplots(figsize=(12, 8))

# # Crear una lista para los sectores que realmente aparecen en el gráfico
# visible_sectors = set()

# # Dibujar las celdas con los sectores optimizados
# for _, row in gdf_cells.iterrows():
#     poly = row['Polygon']
#     sector = row['Sector']
    
#     color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
#     # Añadir el sector a la lista de sectores visibles
#     visible_sectors.add(sector)
    
#     # Dibujar la celda
#     x, y = poly.exterior.xy
#     ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# # Dibujar las fronteras compartidas con líneas oscuras
# for border in shared_borders:
#     x, y = border.xy
#     ax.plot(x, y, color='black', linewidth=2)

# # Dibujar los flujos que cortan dos veces las fronteras (en rojo)
# for _, row in gdf_filtered_flows_2_cross.iterrows():
#     flow = row['geometry']
#     if flow is not None:
#         x_flow, y_flow = flow.xy
#         ax.plot(x_flow, y_flow, color='red', linewidth=1.5, label='Flujos que cortan 2 veces las fronteras')

# # Títulos y etiquetas
# ax.set_title("Sectorización Optimizada con Flujos que Atraviesan 2 Veces las Fronteras")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')

# # Crear la leyenda solo con los sectores visibles
# visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# # Crear los patches para la leyenda solo de los sectores visibles
# patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# # Añadir leyenda para los flujos
# patches.append(mpatches.Patch(color='red', label='Flujos que cortan 2 veces las fronteras'))

# # Mostrar la leyenda
# ax.legend(handles=patches, loc='upper right')

# # Ajuste y visualización final
# plt.tight_layout()
# plt.show()



# # ----------------------------------------------------------------------------- 
# # 2. FILTRAR LOS FLUJOS QUE INTERSECTAN LAS FRONTERAS Y SELECCIONAR EL DE MAYOR DISTANCIA
# # ----------------------------------------------------------------------------- 

# # Diccionario para almacenar el flujo con la mayor distancia por cada celda
# max_distance_flows = {}

# # Recorrer todos los flujos filtrados
# for _, row in gdf_filtered_flujos.iterrows():
#     lat_in = row['lat_cell_in']
#     lon_in = row['lon_cell_in']
#     lat_out = row['lat_cell_out']
#     lon_out = row['lon_cell_out']
#     flow = row['geometry']
    
#     # Verificar cuántas veces el flujo corta las fronteras
#     intersections = count_intersections(flow, shared_borders)
    
#     # Si corta las fronteras dos veces (o más)
#     if intersections >= 2:
#         # Calcular la distancia del flujo
#         distance = calculate_distance(lat_in, lon_in, lat_out, lon_out)
        
#         # Obtener el nombre de la celda (si existe)
#         cell_name = row['Cell_Name']
        
#         # Verificar si ya tenemos un flujo con mayor distancia para esa celda
#         if cell_name not in max_distance_flows or distance > max_distance_flows[cell_name]['distance']:
#             # Si es el mayor flujo, lo almacenamos
#             max_distance_flows[cell_name] = {'flow': flow, 'distance': distance}

# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS CON MAYOR DISTANCIA POR CELDA
# # ----------------------------------------------------------------------------- 
# fig, ax = plt.subplots(figsize=(12, 8))

# # Crear una lista para los sectores que realmente aparecen en el gráfico
# visible_sectors = set()

# # Crear un mapa de colores para los sectores
# color_map = {
#     'LECMSAN': 'green',
#     'LECMASI': 'blue',
#     'LECMBLI': 'purple',
#     'LECMPAI': 'yellow',
#     'LECMDGI': 'gray',
#     'LECMR1I': 'cyan',  
#     'LECMDPI': 'magenta',  
#     'LECMSAB': 'orange',  
#     'LECMSAI': 'red'  
# }

# # Dibujar las celdas con los sectores optimizados
# for _, row in gdf_cells.iterrows():
#     poly = row['Polygon']
#     sector = row['Sector']
    
#     color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
#     # Añadir el sector a la lista de sectores visibles
#     visible_sectors.add(sector)
    
#     # Dibujar la celda
#     x, y = poly.exterior.xy
#     ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# # Dibujar las fronteras compartidas con líneas oscuras
# for border in shared_borders:
#     x, y = border.xy
#     ax.plot(x, y, color='black', linewidth=2)

# # Dibujar solo los flujos con mayor distancia por celda que intersecten dos veces las fronteras
# for cell_name, flow_data in max_distance_flows.items():
#     flow = flow_data['flow']
#     if flow is not None:
#         x_flow, y_flow = flow.xy
#         ax.plot(x_flow, y_flow, color='red', linewidth=1.5, label=f'Flujo mayor distancia {cell_name}')

# # Títulos y etiquetas
# ax.set_title("Sectorización Optimizada con Flujo con Mayor Distancia por Celda")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')

# # Crear la leyenda solo con los sectores visibles
# visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# # Crear los patches para la leyenda solo de los sectores visibles
# patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# # Añadir leyenda para los flujos
# patches.append(mpatches.Patch(color='red', label='Flujos con mayor distancia'))

# # Mostrar la leyenda
# ax.legend(handles=patches, loc='upper right')

# # Ajuste y visualización final
# plt.tight_layout()
# plt.show()


# # # ----------------------------------------------------------------------------- 
# # # 2. FUNCIONES PARA DETECTAR INTERSECCIONES Y MODIFICAR LAS FRONTERAS
# # # ----------------------------------------------------------------------------- 

# # def modify_border_color(border, flow):
# #     """
# #     Modifica el color de la frontera según su intersección con el flujo.
# #     Si hay intersección, desvanece la frontera.
# #     """
# #     # Si la frontera interseca el flujo, reducimos la intensidad del color
# #     if border.intersects(flow):
# #         intersection = border.intersection(flow)
# #         if isinstance(intersection, LineString):  # Si la intersección es una línea
# #             # Dividir la frontera en segmentos y modificamos la parte que cruza el flujo
# #             return 'gray', intersection  # Cambiar el color a gris y marcar la intersección
# #     return 'black', None  # Si no hay intersección, la frontera permanece negra

# # # ----------------------------------------------------------------------------- 
# # # 3. REPRESENTAR EL MALLADO Y LOS FLUJOS CON COLORES MODIFICADOS
# # # ----------------------------------------------------------------------------- 
# # fig, ax = plt.subplots(figsize=(12, 8))

# # # Crear una lista para los sectores que realmente aparecen en el gráfico
# # visible_sectors = set()

# # # Colorear el flujo de negro
# # for _, row in gdf_filtered_flows_2_cross.iterrows():
# #     flow = row['geometry']
# #     if flow is not None:
# #         x_flow, y_flow = flow.xy
# #         ax.plot(x_flow, y_flow, color='black', linewidth=2, label='Flujos (Negros)')

# # # Dibujar las celdas con los sectores optimizados
# # for _, row in gdf_cells.iterrows():
# #     poly = row['Polygon']
# #     sector = row['Sector']
    
# #     color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
# #     # Añadir el sector a la lista de sectores visibles
# #     visible_sectors.add(sector)
    
# #     # Dibujar la celda
# #     x, y = poly.exterior.xy
# #     ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# # # Dibujar las fronteras compartidas con líneas modificadas
# # for border in shared_borders:
# #     for _, row in gdf_filtered_flows_2_cross.iterrows():
# #         flow = row['geometry']
# #         color, intersection = modify_border_color(border, flow)
        
# #         # Si hay intersección, modificamos la parte de la frontera que intersecta
# #         if color == 'gray' and intersection:
# #             # Dibujar la frontera en gris en la sección intersectada
# #             x_intersect, y_intersect = intersection.xy
# #             ax.plot(x_intersect, y_intersect, color=color, linewidth=2)
        
# #         # Dibujar el resto de la frontera que no intersecta
# #         if color == 'black':
# #             x_border, y_border = border.xy
# #             ax.plot(x_border, y_border, color=color, linewidth=2)

# # # Títulos y etiquetas
# # ax.set_title("Sectorización Optimizada con Fronteras Compartidas y Flujos Modificados")
# # ax.set_xlabel("Longitud [º]")
# # ax.set_ylabel("Latitud [º]")
# # ax.set_aspect('equal')

# # # Crear la leyenda solo con los sectores visibles
# # visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# # # Crear los patches para la leyenda solo de los sectores visibles
# # patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# # # Añadir leyenda para los flujos
# # patches.append(mpatches.Patch(color='black', label='Flujos (Negros)'))

# # # Mostrar la leyenda
# # ax.legend(handles=patches, loc='upper right')

# # # Ajuste y visualización final
# # plt.tight_layout()
# # plt.show()


# # ----------------------------------------------------------------------------- 
# # 3. REPRESENTAR SOLO LAS FRONTERAS DE LOS SECTORES, LOS FLUJOS DE MAYOR DISTANCIA Y EL ACC
# # ----------------------------------------------------------------------------- 
# fig, ax = plt.subplots(figsize=(12, 8))

# # Dibujar el polígono del ACC
# x_acc, y_acc = poligono_ACC.exterior.xy  # Asegúrate de que `poligono_ACC` esté definido
# ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')  # Delimitación del ACC

# # Crear una lista para los sectores que realmente aparecen en el gráfico
# visible_sectors = set()

# # Dibujar las fronteras compartidas entre los sectores
# for border in shared_borders:
#     x, y = border.xy
#     ax.plot(x, y, color='black', linewidth=2)

# # Dibujar solo los flujos con mayor distancia por celda que intersectan las fronteras
# for cell_name, flow_data in max_distance_flows.items():
#     flow = flow_data['flow']
#     if flow is not None:
#         # Dibujar el flujo de mayor distancia en rojo
#         x_flow, y_flow = flow.xy
#         ax.plot(x_flow, y_flow, color='red', linewidth=2, label=f'Flujo mayor distancia {cell_name}')

# # Títulos y etiquetas
# ax.set_title("Fronteras de Sectores con Flujos de Mayor Distancia Intersecando las Fronteras")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')

# # Crear la leyenda con flujos intersecando fronteras
# patches = [mpatches.Patch(color='red', label='Flujos de mayor distancia')]
# patches.append(mpatches.Patch(color='black', label='LECMCTAN (ACC)'))

# # Mostrar la leyenda
# ax.legend(handles=patches, loc='upper right')

# # Ajuste y visualización final
# plt.tight_layout()
# plt.show()

# from shapely.ops import unary_union
# from shapely.geometry import Point, Polygon, LineString
# from shapely.ops import unary_union
# from shapely.geometry import Point, Polygon

# import pandas as pd
# from shapely.geometry import Point, MultiPoint

# from shapely.geometry import Point, MultiPoint, LineString
# from shapely.ops import snap
# import math
# import pandas as pd


# import math
# import pandas as pd
# from shapely.geometry import Point, MultiPoint, LineString
# from shapely.ops import snap
# import pandas as pd
# import numpy as np
# from shapely.geometry import Point, MultiPoint, GeometryCollection
# from shapely.geometry.base import BaseGeometry



# def compute_flow_border_intersections(flows, border_lines):
#     """
#     Calcula los puntos de intersección entre flujos y aristas de frontera.
#     Siempre devuelve un DataFrame con las columnas esperadas, incluso si está vacío.
#     """
#     records = []
#     for flow_id, flow_line in flows:
#         if not isinstance(flow_line, BaseGeometry):
#             continue
#         for border_id, border_line in border_lines:
#             if not isinstance(border_line, BaseGeometry):
#                 continue
#             try:
#                 inter = flow_line.intersection(border_line)
#             except Exception:
#                 continue
#             if inter.is_empty:
#                 continue

#             pts = []
#             if isinstance(inter, Point):
#                 pts = [inter]
#             elif isinstance(inter, MultiPoint):
#                 pts = list(inter)
#             elif isinstance(inter, GeometryCollection):
#                 pts = [g for g in inter.geoms if isinstance(g, Point)]
#             else:
#                 for geom in getattr(inter, 'geoms', [inter]):
#                     if hasattr(geom, 'coords'):
#                         pts.extend(Point(c) for c in geom.coords)

#             for pt in pts:
#                 records.append({
#                     'flow_id': flow_id,
#                     'border_id': border_id,
#                     'latitude': round(pt.y, 2),
#                     'longitude': round(pt.x, 2)
#                 })

#     # Aseguramos siempre estas columnas
#     return pd.DataFrame(records, columns=['flow_id','border_id','latitude','longitude'])

# # Ejemplo de uso:
# # --- 1) Calcula las intersecciones, pasando enumerate(shared_borders) ---
# flows_max = [(cell, data['flow']) for cell, data in max_distance_flows.items()]
# intersection_df = compute_flow_border_intersections(flows_max,
#                                                     list(enumerate(shared_borders)))


# from shapely.geometry import Polygon
# import matplotlib.pyplot as plt

# def draw_rectangle_between_points(ax, p1, p2, tolerance=0,
#                                   edgecolor='blue', linewidth=2, linestyle='--'):
#     """
#     Dibuja en `ax` un rectángulo definido por dos puntos p1 y p2 (shapely Points),
#     usando también los dos puntos formados por (latitud de uno, longitud del otro)
#     y añade una pequeña tolerancia para expandir el rectángulo.

#     Parámetros:
#     -----------
#     ax : matplotlib.axes.Axes
#         Eje donde se dibujará el rectángulo.
#     p1, p2 : shapely.geometry.Point
#         Puntos de intersección del flujo de máxima distancia.
#     tolerance : float, opcional
#         Margen (en grados) para expandir el rectángulo en todas direcciones.
#     edgecolor : str, opcional
#         Color del borde del rectángulo.
#     linewidth : float, opcional
#         Grosor de la línea del rectángulo.
#     linestyle : str, opcional
#         Estilo de línea (p.ej. '-', '--', '-.', ':').
    
#     Devuelve:
#     ---------
#     polygon : shapely.geometry.Polygon
#         Objeto Polygon con las coordenadas del rectángulo.
#     """
#     # Extraer longitudes (x) y latitudes (y)
#     lon1, lat1 = p1.x, p1.y
#     lon2, lat2 = p2.x, p2.y

#     # Calcular mínimos y máximos + tolerancia
#     min_lon = min(lon1, lon2) - tolerance
#     max_lon = max(lon1, lon2) + tolerance
#     min_lat = min(lat1, lat2) - tolerance
#     max_lat = max(lat1, lat2) + tolerance

#     # Definir las 5 esquinas (cerrando el polígono)
#     corners = [
#         (min_lon, min_lat),  # (lon1, lat1) ajustado a min-min
#         (min_lon, max_lat),  # (lon1, lat2)
#         (max_lon, max_lat),  # (lon2, lat2)
#         (max_lon, min_lat),  # (lon2, lat1)
#         (min_lon, min_lat)   # cerrar
#     ]

#     # Extraer coordenadas y dibujar
#     xs, ys = zip(*corners)
#     ax.plot(xs, ys, color=edgecolor, linewidth=linewidth, linestyle=linestyle)

#     # Devolver el polígono por si quieres reutilizarlo
#     return Polygon(corners)


# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from shapely.ops import unary_union
# from shapely.geometry import Point, LineString, MultiLineString

# # --- 1) Prepara el lienzo ---
# fig, ax = plt.subplots(figsize=(12, 8))

# # Dibuja el polígono del ACC
# x_acc, y_acc = poligono_ACC.exterior.xy
# ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# # --- 2) Calcula intersecciones (ya lo tienes) ---
# # flows_max = [(cell, data['flow']) for cell, data in max_distance_flows.items()]
# # intersection_df = compute_flow_border_intersections(flows_max, list(enumerate(shared_borders)))

# # --- 3) Genera cada rectángulo a partir de p1, p2 extraídos ---
# rectangles = []
# for flow_id in intersection_df['flow_id'].unique():
#     df_flow = intersection_df[intersection_df['flow_id'] == flow_id]
#     if len(df_flow) >= 2:
#         # Coge los dos primeros puntos de la tabla
#         lon1, lat1 = df_flow.iloc[0][['longitude','latitude']]
#         lon2, lat2 = df_flow.iloc[1][['longitude','latitude']]
#         p1 = Point(lon1, lat1)
#         p2 = Point(lon2, lat2)
#         # Dibuja el rectángulo (añade al ax) y guarda el polígono
#         rect = draw_rectangle_between_points(
#             ax, p1, p2,
#             tolerance=0.01,
#             edgecolor='blue', linewidth=2, linestyle='--'
#         )
#         rectangles.append(rect)

# # --- 4) Une todos los rectángulos en un solo geometry ---
# area_to_remove = unary_union(rectangles)

# # --- 5) Dibuja las fronteras recortadas ---
# for border in shared_borders:
#     clipped = border.difference(area_to_remove)
#     if clipped.is_empty:
#         continue

#     # Puede ser LineString o MultiLineString
#     if isinstance(clipped, LineString):
#         segments = [clipped]
#     elif isinstance(clipped, MultiLineString):
#         segments = list(clipped.geoms)
#     else:
#         continue

#     for seg in segments:
#         x, y = seg.xy
#         ax.plot(x, y, color='black', linewidth=2)

# # --- 6) Dibuja los flujos de mayor distancia en rojo ---
# for cell_name, data in max_distance_flows.items():
#     flow = data.get('flow')
#     if flow:
#         x_flow, y_flow = flow.xy
#         ax.plot(x_flow, y_flow, color='red', linewidth=2)

# # --- 7) Leyenda, títulos y ajuste final ---
# from matplotlib.lines import Line2D
# legend_handles = [
#     Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
#     Line2D([0],[0], color='black', lw=2, label='Frontera recortada'),
#     Line2D([0],[0], color='red',   lw=2, label='Flujos mayor distancia'),
#     Line2D([0],[0], color='blue',  lw=2, linestyle='--', label='Rectángulos de intersección')
# ]
# ax.legend(handles=legend_handles, loc='upper right')
# ax.set_title("Fronteras de Sectores recortadas por Rectángulos de Intersección")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from shapely.ops import unary_union
# from shapely.geometry import Point, LineString, MultiLineString
# import math

# # # — asume poligono_ACC, shared_borders, max_distance_flows,
# # #   compute_flow_border_intersections(), draw_rectangle_between_points() —

# # # 1) Dibuja el contorno del ACC
# # fig, ax = plt.subplots(figsize=(12, 8))
# # x_acc, y_acc = poligono_ACC.exterior.xy
# # ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# # # 2) Genera y une los rectángulos de recorte (igual que antes)
# # rectangles = []
# # for flow_id in intersection_df['flow_id'].unique():
# #     df = intersection_df[intersection_df['flow_id']==flow_id]
# #     if len(df) >= 2:
# #         lon1, lat1 = df.iloc[0][['longitude','latitude']]
# #         lon2, lat2 = df.iloc[1][['longitude','latitude']]
# #         p1, p2 = Point(lon1, lat1), Point(lon2, lat2)
# #         rect = draw_rectangle_between_points(ax, p1, p2,
# #                                              tolerance=0.01,
# #                                              edgecolor='blue',
# #                                              linewidth=2,
# #                                              linestyle='--')
# #         rectangles.append(rect)
# # area_to_remove = unary_union(rectangles)

# # # 3) Guarda todos los fragmentos resultantes de recortar cada frontera
# # clipped_borders = []
# # for border in shared_borders:
# #     clipped = border.difference(area_to_remove)
# #     if clipped.is_empty:
# #         continue
# #     if isinstance(clipped, LineString):
# #         clipped_borders.append(clipped)
# #     elif isinstance(clipped, MultiLineString):
# #         clipped_borders.extend(clipped.geoms)

# # # 4) Prepara target_geom = fronteras recortadas + flujos
# # flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow') is not None]
# # target_geom = unary_union(clipped_borders + flows_lines)

# # # 5) Función auxiliar para extender un extremo hasta target_geom
# # from shapely.ops import unary_union
# # from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
# # import math

# # # flows_lines: lista de todos los flujos rojos
# # flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow') is not None]

# # def extend_to_targets(pt_from, pt_dir, target_geom, ax, max_dist=100):
# #     dx, dy = pt_from.x - pt_dir.x, pt_from.y - pt_dir.y
# #     L = math.hypot(dx, dy)
# #     if L == 0:
# #         return
# #     ux, uy = dx/L, dy/L
# #     far = Point(pt_from.x + ux*max_dist, pt_from.y + uy*max_dist)
# #     ray = LineString([pt_from, far])
# #     inter = ray.intersection(target_geom)
# #     if inter.is_empty:
# #         return

# #     # recogemos puntos de intersección, incluso si vienen en un segmento
# #     pts = []
# #     def collect(geom):
# #         if isinstance(geom, Point):
# #             pts.append(geom)
# #         elif isinstance(geom, LineString):
# #             for c in geom.coords:
# #                 pts.append(Point(c))
# #         elif isinstance(geom, (MultiLineString, GeometryCollection)):
# #             for g in geom.geoms:
# #                 collect(g)

# #     collect(inter)
# #     if not pts:
# #         return
# #     pts.sort(key=lambda p: p.distance(pt_from))
# #     nearest = pts[0]
# #     ax.plot([pt_from.x, nearest.x], [pt_from.y, nearest.y],
# #             color='black', linewidth=2)

# # # --- 5) Reemplaza tu bucle actual por este ---
# # clipped_borders = []
# # for border in shared_borders:
# #     clipped = border.difference(area_to_remove)
# #     if clipped.is_empty:
# #         continue
# #     if isinstance(clipped, LineString):
# #         clipped_borders.append(clipped)
# #     else:  # MultiLineString
# #         clipped_borders.extend(clipped.geoms)

# # # Dibuja + extiende
# # for i, seg in enumerate(clipped_borders):
# #     # 5.a) traza el propio segmento
# #     x, y = seg.xy
# #     ax.plot(x, y, color='black', linewidth=2)

# #     coords = list(seg.coords)
# #     if len(coords) < 2:
# #         continue

# #     # construye target_geom **sin** este seg
# #     other = clipped_borders[:i] + clipped_borders[i+1:]
# #     target = unary_union(other + flows_lines)

# #     # extremos
# #     p0, p1 = Point(coords[0]), Point(coords[1])
# #     pe, pp = Point(coords[-1]), Point(coords[-2])
# #     tol = 1e-6

# #     if target.distance(p0) > tol:
# #         extend_to_targets(p0, p1, target, ax)
# #     if target.distance(pe) > tol:
# #         extend_to_targets(pe, pp, target, ax)

# # # 6) Dibuja y extiende cada fragmento recortado
# # for seg in clipped_borders:
# #     x, y = seg.xy
# #     ax.plot(x, y, color='black', linewidth=2)

# #     coords = list(seg.coords)
# #     if len(coords) < 2:
# #         continue

# #     # extremos
# #     p0, p1 = Point(coords[0]), Point(coords[1])
# #     pe, pp = Point(coords[-1]), Point(coords[-2])
# #     tol = 1e-6

# #     # sólo extiende si el extremo NO ya toca target_geom
# #     if target_geom.distance(p0) > tol:
# #         extend_to_targets(p0, p1, target_geom, ax)
# #     if target_geom.distance(pe) > tol:
# #         extend_to_targets(pe, pp, target_geom, ax)

# # # 7) Sobrepone los flujos de máxima distancia en rojo
# # for data in max_distance_flows.values():
# #     flow = data.get('flow')
# #     if flow:
# #         xf, yf = flow.xy
# #         ax.plot(xf, yf, color='red', linewidth=2)

# # # 8) Leyenda y acabado
# # handles = [
# #     Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
# #     Line2D([0],[0], color='black', lw=2, label='Frontera recortada/extendida'),
# #     Line2D([0],[0], color='red',   lw=2, label='Flujos de mayor distancia')
# # ]
# # ax.legend(handles=handles, loc='upper right')

# # ax.set_title("Fronteras recortadas y prolongadas hasta otras fronteras o flujos")
# # ax.set_xlabel("Longitud [º]")
# # ax.set_ylabel("Latitud [º]")
# # ax.set_aspect('equal')
# # plt.tight_layout()
# # plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from shapely.ops import unary_union
# from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
# import math

# # — asume poligono_ACC, shared_borders, max_distance_flows,
# #   intersection_df, draw_rectangle_between_points() ya definidos —

# fig, ax = plt.subplots(figsize=(12, 8))
# # 1) dibujamos el ACC
# x_acc, y_acc = poligono_ACC.exterior.xy
# ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# # 2) creamos los rectángulos y los unimos
# rects = []
# for fid in intersection_df['flow_id'].unique():
#     df = intersection_df[intersection_df['flow_id'] == fid]
#     if len(df) >= 2:
#         lon1, lat1 = df.iloc[0][['longitude','latitude']]
#         lon2, lat2 = df.iloc[1][['longitude','latitude']]
#         p1, p2 = Point(lon1, lat1), Point(lon2, lat2)
#         rect = draw_rectangle_between_points(
#             ax, p1, p2, tolerance=0.01,
#             edgecolor='blue', linewidth=2, linestyle='--'
#         )
#         rects.append(rect)
# area_to_remove = unary_union(rects)

# # 3) extraemos todos los fragmentos recortados
# clipped_borders = []
# for b in shared_borders:
#     c = b.difference(area_to_remove)
#     if c.is_empty:
#         continue
#     if isinstance(c, LineString):
#         clipped_borders.append(c)
#     else:
#         clipped_borders.extend(c.geoms)

# # 4) preparamos la lista de flujos y la frontera del ACC como LineString
# flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow') is not None]
# acc_boundary = LineString(poligono_ACC.exterior.coords)

# # 5) función que dispara el rayo y dibuja hasta target
# def extend_to(pt_from, pt_dir, target, ax, max_dist=100):
#     dx, dy = pt_from.x - pt_dir.x, pt_from.y - pt_dir.y
#     L = math.hypot(dx, dy)
#     if L == 0: return
#     ux, uy = dx/L, dy/L
#     far = Point(pt_from.x + ux*max_dist, pt_from.y + uy*max_dist)
#     ray = LineString([pt_from, far])
#     inter = ray.intersection(target)
#     if inter.is_empty: return

#     pts = []
#     def collect(g):
#         if isinstance(g, Point):
#             pts.append(g)
#         elif isinstance(g, LineString):
#             pts.extend(Point(c) for c in g.coords)
#         elif hasattr(g, 'geoms'):
#             for gg in g.geoms:
#                 collect(gg)
#     collect(inter)
#     if not pts: return
#     nearest = min(pts, key=lambda p: p.distance(pt_from))
#     ax.plot([pt_from.x, nearest.x], [pt_from.y, nearest.y],
#             color='black', linewidth=2)

# # 6) dibujamos y extendemos cada fragmento
# for i, seg in enumerate(clipped_borders):
#     # trazamos el propio trozo
#     x, y = seg.xy
#     ax.plot(x, y, color='black', linewidth=2)

#     coords = list(seg.coords)
#     if len(coords) < 2:
#         continue

#     # target = todos los demás trozos  + flujos + ACC
#     others = clipped_borders[:i] + clipped_borders[i+1:]
#     target = unary_union(others + flows_lines + [acc_boundary])

#     p0, p1 = Point(coords[0]), Point(coords[1])
#     pe, pp = Point(coords[-1]), Point(coords[-2])
#     tol = 1e-6

#     if target.distance(p0) > tol:
#         extend_to(p0, p1, target, ax)
#     if target.distance(pe) > tol:
#         extend_to(pe, pp, target, ax)

# # 7) sobreponemos los flujos
# for d in max_distance_flows.values():
#     f = d.get('flow')
#     if f:
#         xf, yf = f.xy
#         ax.plot(xf, yf, color='red', linewidth=2)

# # leyenda y ajustes finales
# handles = [
#     Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
#     Line2D([0],[0], color='black', lw=2, label='Fronteras recortadas/extendidas'),
#     Line2D([0],[0], color='red',   lw=2, label='Flujos máx. distancia')
# ]
# ax.legend(handles=handles, loc='upper right')
# ax.set_title("Fronteras recortadas y Prolongadas")
# ax.set_xlabel("Longitud [º]"); ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal'); plt.tight_layout(); plt.show()


# import matplotlib.pyplot as plt
# from shapely.ops import unary_union
# from shapely.geometry import Point, LineString, MultiLineString
# import math

# import matplotlib.pyplot as plt
# from shapely.ops import unary_union
# from shapely.geometry import Point, LineString, MultiLineString
# import math

# # --- Prepara figura y dibuja ACC como antes ---
# fig, ax = plt.subplots(figsize=(12,8))
# x_acc, y_acc = poligono_ACC.exterior.xy
# ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# # --- 1) Reúne fronteras recortadas y flujos ---
# # clipped_borders: ya los tienes de tu paso 3-4
# flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow')]

# # --- 2) Define una única función de extensión ---
# def compute_extension(pt_from, pt_dir, target, max_dist=200):
#     dx, dy = pt_from.x - pt_dir.x, pt_from.y - pt_dir.y
#     L = math.hypot(dx, dy)
#     if L == 0:
#         return None
#     ux, uy = dx/L, dy/L
#     far = Point(pt_from.x + ux*max_dist, pt_from.y + uy*max_dist)
#     ray = LineString([pt_from, far])
#     inter = ray.intersection(target)
#     if inter.is_empty:
#         return None

#     # recogen todos los posibles puntos de choque
#     pts = []
#     if isinstance(inter, Point):
#         pts = [inter]
#     else:
#         for g in getattr(inter, 'geoms', []):
#             if isinstance(g, Point):
#                 pts.append(g)
#             elif isinstance(g, LineString):
#                 pts.extend(Point(c) for c in g.coords)
#     if not pts:
#         return None

#     # quedamos con el más cercano
#     nearest = min(pts, key=lambda p: p.distance(pt_from))
#     return LineString([(pt_from.x, pt_from.y), (nearest.x, nearest.y)])

# # --- 3) Calcula todas las extensiones ---
# extension_lines = []
# tol = 1e-6

# for i, seg in enumerate(clipped_borders):
#     coords = list(seg.coords)
#     if len(coords) < 2:
#         continue

#     # target: todo excepto este segmento
#     others = clipped_borders[:i] + clipped_borders[i+1:]
#     target = unary_union(others + flows_lines + [acc_boundary])

#     p0, p1 = Point(coords[0]), Point(coords[1])
#     pe, pp = Point(coords[-1]), Point(coords[-2])

#     # extiende desde cada extremo si está separado
#     if target.distance(p0) > tol:
#         ext0 = compute_extension(p0, p1, target)
#         if ext0 is not None:
#             extension_lines.append(ext0)

#     if target.distance(pe) > tol:
#         ext1 = compute_extension(pe, pp, target)
#         if ext1 is not None:
#             extension_lines.append(ext1)

# # --- 4) Une todo y dibuja de una vez ---
# all_pieces = clipped_borders + extension_lines + flows_lines
# combined  = unary_union(all_pieces)

# # extrae las LineString para plotear
# to_plot = []
# if isinstance(combined, LineString):
#     to_plot = [combined]
# elif isinstance(combined, MultiLineString):
#     to_plot = list(combined.geoms)
# else:
#     for g in getattr(combined, 'geoms', []):
#         if isinstance(g, LineString):
#             to_plot.append(g)

# for line in to_plot:
#     x, y = line.xy
#     ax.plot(x, y, color='black', linewidth=2)

# # --- 5) Leyenda y acabado ---
# from matplotlib.lines import Line2D
# handles = [
#     Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
#     Line2D([0],[0], color='black', lw=2, label='Fronteras Finales'),
# ]
# ax.legend(handles=handles, loc='upper right')

# ax.set_title("Fronteras Finales")
# ax.set_xlabel("Longitud [º]")
# ax.set_ylabel("Latitud [º]")
# ax.set_aspect('equal')
# plt.tight_layout()
# plt.show()
















import geopandas as gpd
from shapely.geometry import LineString

import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon, MultiPoint, GeometryCollection, MultiLineString
from shapely.ops import unary_union
from itertools import combinations
from collections import defaultdict
from numpy.linalg import eigh

# ------------------------------------------------------------------------------
def get_shared_borders_and_nodes(cells, crs, tol=1e-9):
    """
    Devuelve:
      • gdf_borders: GeoDataFrame con las líneas compartidas (sector_1, sector_2, geometry)
      • gdf_nodes: GeoDataFrame con los nodos (puntos) donde confluyen ≥3 sectores,
                   con columnas ['sectors','geometry'].
    """
    # 1) extraer fronteras pareadas
    records_b = []
    n = len(cells)
    for i in range(n):
        _, poly_i, sec_i = cells[i]
        for j in range(i+1, n):
            _, poly_j, sec_j = cells[j]
            if sec_i == sec_j or not poly_i.intersects(poly_j):
                continue
            inter = poly_i.intersection(poly_j)
            if isinstance(inter, LineString) and inter.length > 0:
                records_b.append({
                    'sector_1': sec_i,
                    'sector_2': sec_j,
                    'geometry': inter
                })
    gdf_borders = gpd.GeoDataFrame(records_b, crs=crs)

    # 2) agrupar extremos en nodos multígrado
    node_sectors = defaultdict(set)
    for _, row in gdf_borders.iterrows():
        s1, s2 = row.sector_1, row.sector_2
        for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
            key = (round(x/tol)*tol, round(y/tol)*tol)
            node_sectors[key].update([s1, s2])

    records_n = []
    for (x, y), secs in node_sectors.items():
        if len(secs) >= 3:
            records_n.append({
                'sectors': sorted(secs),
                'geometry': Point(x, y)
            })
    gdf_nodes = gpd.GeoDataFrame(records_n, crs=crs)
    return gdf_borders, gdf_nodes

# ------------------------------------------------------------------------------
def extract_corner_points(gdf_borders, tol=1e-9):
    """
    Extrae los cruces horizontal-vertical de gdf_borders.
    Devuelve gdf_corners con columnas:
      ['h_sector_1','h_sector_2','v_sector_1','v_sector_2','geometry'].
    """
    is_h = gdf_borders.geometry.apply(lambda g: abs(g.coords[0][1] - g.coords[-1][1]) < tol)
    is_v = gdf_borders.geometry.apply(lambda g: abs(g.coords[0][0] - g.coords[-1][0]) < tol)
    gdf_h = gdf_borders[is_h]
    gdf_v = gdf_borders[is_v]

    records = []
    for _, h in gdf_h.iterrows():
        for _, v in gdf_v.iterrows():
            pt = h.geometry.intersection(v.geometry)
            if not pt.is_empty and isinstance(pt, Point):
                records.append({
                    'h_sector_1': h.sector_1,
                    'h_sector_2': h.sector_2,
                    'v_sector_1': v.sector_1,
                    'v_sector_2': v.sector_2,
                    'geometry': pt
                })
    gdf_corners = gpd.GeoDataFrame(records, crs=gdf_borders.crs)
    return gdf_corners.drop_duplicates(subset='geometry').reset_index(drop=True)


# 0) Prepara cells_info
cells_info = [
    (row['Cell_Name'], row['Polygon'], row['Sector'])
    for _, row in gdf_cells.iterrows()
]

# 1) get_shared_borders_and_nodes
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) polígono ACC y su contorno
polygons = DF_info_conf['Contorno Sector'].tolist()
union_poly = unary_union(polygons)
if union_poly.geom_type == 'MultiPolygon':
    poligono_ACC = union_poly.convex_hull
else:
    poligono_ACC = Polygon(union_poly.exterior)
acc_boundary = poligono_ACC.boundary

# 3) puntos de contacto ACC
records = []
for _, row in gdf_borders.iterrows():
    s1, s2 = row.sector_1, row.sector_2
    for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
        pt = Point(x, y)
        if pt.distance(acc_boundary) < 1e-9:
            records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
                     .drop_duplicates('geometry') \
                     .reset_index(drop=True)

# 4) cruces H-V
gdf_corners = extract_corner_points(gdf_borders)


from shapely.geometry import LineString
from shapely.ops import unary_union
import geopandas as gpd

def extract_constituent_const_segments(gdf_borders, tol=1e-9):
    """
    Para cada frontera (misma combinación sector_1–sector_2) de gdf_borders,
    descompone su LineString en segmentos horizontales y verticales,
    agrupa en cadenas consecutivas (que se tocan), y une cada cadena
    en un LineString (o MultiLineString).

    Devuelve un GeoDataFrame con columnas:
      - sector_1, sector_2
      - orientation: 'h' ó 'v'
      - geometry: LineString (o MultiLineString) resultante
    """
    records = []

    # Asegurarnos de tener un identificador por fila
    gdf = gdf_borders.reset_index().rename(columns={'index':'__orig_idx'}).copy()

    for _, row in gdf.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        coords = list(row.geometry.coords)
        # 1) Descomponer en segmentos orientados
        segs = []
        for i in range(len(coords)-1):
            x1,y1 = coords[i]
            x2,y2 = coords[i+1]
            if abs(y1-y2) < tol:
                orient = 'h'
            elif abs(x1-x2) < tol:
                orient = 'v'
            else:
                continue
            seg = LineString([(x1,y1),(x2,y2)])
            segs.append({'orient':orient, 'geom':seg})

        # 2) Para cada orientación, construir componentes conexas
        for orient in ('h','v'):
            # filtrar segmentos de esta orientación
            s_or = [s for s in segs if s['orient']==orient]
            n = len(s_or)
            if n==0:
                continue

            # construir grafo implícito por matriz de adyacencia touches()
            visited = [False]*n
            for i in range(n):
                if visited[i]:
                    continue
                # empezar una nueva cadena
                stack = [i]
                comp = []
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(s_or[u]['geom'])
                    for v in range(n):
                        if not visited[v] and s_or[u]['geom'].touches(s_or[v]['geom']):
                            visited[v] = True
                            stack.append(v)
                # 3) unir la cadena
                merged = unary_union(comp)
                records.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'orientation': orient,
                    'geometry': merged
                })

    gdf_const = gpd.GeoDataFrame(records, geometry='geometry', crs=gdf_borders.crs)
    return gdf_const



import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

def midpoints_of_constituent_segments(gdf_borders, tol=1e-9):
    """
    Para cada par (sector_1, sector_2) en gdf_borders, agrupa los segmentos
    puramente horizontales o verticales que se tocan y comparten la misma
    latitud/longitud, y calcula un único punto medio por cada grupo.

    Devuelve un GeoDataFrame con columnas:
      - sector_1, sector_2
      - geometry: Point (punto medio del tramo unido)
    """
    records = []
    # 1) Agrupar por frontera (sector_1, sector_2)
    for (s1, s2), grp in gdf_borders.groupby(['sector_1','sector_2']):
        # 2) separar horizontales y verticales
        horizontales = []
        verticales   = []
        for seg in grp.geometry:
            x1,y1 = seg.coords[0]
            x2,y2 = seg.coords[-1]
            if abs(y1 - y2) < tol:
                # horizontal: guardamos (y0, LineString)
                y0 = (y1 + y2)/2
                horizontales.append((y0, seg))
            elif abs(x1 - x2) < tol:
                # vertical: guardamos (x0, LineString)
                x0 = (x1 + x2)/2
                verticales.append((x0, seg))
            # else: ignoramos segmentos inclinados

        # 3) función auxiliar para procesar cada orientación
        def _process(groups, is_horizontal):
            # agrupar por coordenada constante (redondeada a tol)
            buckets = defaultdict(list)
            for const, seg in groups:
                key = round(const/tol)*tol
                buckets[key].append(seg)

            # para cada cubeta, extraer componentes conexas
            for const, segs in buckets.items():
                n = len(segs)
                visited = [False]*n
                for i in range(n):
                    if visited[i]:
                        continue
                    # DFS para agrupar segs que se tocan
                    stack = [i]
                    comp = []
                    visited[i] = True
                    while stack:
                        u = stack.pop()
                        comp.append(segs[u])
                        for v in range(n):
                            if not visited[v] and segs[u].touches(segs[v]):
                                visited[v] = True
                                stack.append(v)
                    # 4) calcular extremo-a-extremo
                    xs = []
                    ys = []
                    for s in comp:
                        for x,y in s.coords:
                            xs.append(x)
                            ys.append(y)
                    if is_horizontal:
                        y0    = const
                        x_min = min(xs); x_max = max(xs)
                        mx, my = (x_min + x_max)/2, y0
                    else:
                        x0    = const
                        y_min = min(ys); y_max = max(ys)
                        mx, my = x0, (y_min + y_max)/2

                    records.append({
                        'sector_1': s1,
                        'sector_2': s2,
                        'geometry': Point(mx, my)
                    })

        _process(horizontales, is_horizontal=True)
        _process(verticales,   is_horizontal=False)

    gdf_mid = gpd.GeoDataFrame(records, crs=gdf_borders.crs)
    return gdf_mid


# 1) Ya tienes tu gdf_borders
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) Obtienes los puntos medios de los tramos horizontales/verticales consecutivos
gdf_mid_const = midpoints_of_constituent_segments(gdf_borders)

# 3) Inspecciona
print(gdf_mid_const)
gdf_mid_const.plot(markersize=50)



# # Función para obtener punto medio en tramos con longitud o latitud constante
# def get_midpoints_const(gdf, tol=1e-9):
#     records = []
#     for _, row in gdf.iterrows():
#         s1 = row['sector_1']
#         s2 = row['sector_2']
#         coords = list(row.geometry.coords)
#         x1, y1 = coords[0]
#         x2, y2 = coords[-1]

#         if abs(x1 - x2) < tol:  # Longitud constante (segmento vertical)
#             mid_x = x1
#             mid_y = (y1 + y2) / 2.0
#         elif abs(y1 - y2) < tol:  # Latitud constante (segmento horizontal)
#             mid_x = (x1 + x2) / 2.0
#             mid_y = y1
#         else:
#             # Omitir segmentos no puramente horizontales/verticales
#             continue

#         records.append({
#             'sector_1': s1,
#             'sector_2': s2,
#             'geometry': Point(mid_x, mid_y)
#         })

#     return gpd.GeoDataFrame(records, geometry='geometry')

# # Aplicamos la función a gdf_borders
# gdf_midpoints = get_midpoints_const(gdf_borders)



# from shapely.geometry import Point
# import geopandas as gpd
# import numpy as np

# import numpy as np
# from shapely.geometry import Point
# import geopandas as gpd

# def get_midpoints_same_sector_pairs(gdf, tol=0.1):
#     """
#     Para cada par (sector_1, sector_2) en gdf, agrupa los puntos que tengan
#     la misma latitud (|y_i − y_j| < tol) o la misma longitud (|x_i − x_j| < tol),
#     calcula su punto medio (tomando el de menor x y el de mayor x en agrupación por latitud,
#     o el de menor y y el de mayor y en agrupación por longitud),
#     y genera un GeoDataFrame con columnas ['sector_1', 'sector_2', 'geometry'].
#     Si dentro de ese par queda un punto que no pudo agruparse con ningún otro
#     (bajo el criterio tol), se conserva tal cual como su “propio punto medio”.

#     Parámetros
#     ----------
#     gdf : GeoDataFrame
#         Debe tener columnas:
#           - 'sector_1'
#           - 'sector_2'
#           - 'geometry' (de tipo Point)
#     tol : float, opcional
#         Tolerancia para considerar “misma” latitud o longitud (en grados).
#         Ej. tol=0.1 equivale a que 44.50 ≈ 44.59, pero 44.50 no ≈ 44.60.

#     Devuelve
#     -------
#     GeoDataFrame con columnas:
#       - 'sector_1' (string)
#       - 'sector_2' (string)
#       - 'geometry' (Point)
#     Cada fila es un punto medio calculado **dentro del mismo par** (sector_1, sector_2),
#     o bien un punto “aislado” si no encontró par dentro de tol.
#     """
#     records = []

#     # Agrupar por cada par de sectores (sector_1, sector_2)
#     for (s1, s2), sub in gdf.groupby(['sector_1', 'sector_2']):
#         # obtener vectores de coordenadas x e y
#         xs = np.array([pt.x for pt in sub.geometry])
#         ys = np.array([pt.y for pt in sub.geometry])
#         m = len(sub)
#         used = np.zeros(m, dtype=bool)

#         # 1) Dentro de este par, agrupar por latitud aproximada
#         for i in range(m):
#             if used[i]:
#                 continue
#             # buscar todos los índices j no usados con |y[j] - y[i]| < tol
#             grupo_lat = [j for j in range(m) if (not used[j]) and abs(ys[j] - ys[i]) < tol]
#             if len(grupo_lat) >= 2:
#                 # de ese grupo, elegir los de menor x y de mayor x
#                 xs_grupo = xs[grupo_lat]
#                 j_min = grupo_lat[int(np.argmin(xs_grupo))]
#                 j_max = grupo_lat[int(np.argmax(xs_grupo))]
#                 mid_x = (xs[j_min] + xs[j_max]) / 2.0
#                 mid_y = (ys[j_min] + ys[j_max]) / 2.0
#                 records.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': Point(mid_x, mid_y)
#                 })
#                 for j in grupo_lat:
#                     used[j] = True

#         # 2) Dentro de este par, agrupar los que queden por longitud aproximada
#         for i in range(m):
#             if used[i]:
#                 continue
#             # buscar todos los índices j no usados con |x[j] - x[i]| < tol
#             grupo_lon = [j for j in range(m) if (not used[j]) and abs(xs[j] - xs[i]) < tol]
#             if len(grupo_lon) >= 2:
#                 # de ese grupo, elegir los de menor y y de mayor y
#                 ys_grupo = ys[grupo_lon]
#                 j_min = grupo_lon[int(np.argmin(ys_grupo))]
#                 j_max = grupo_lon[int(np.argmax(ys_grupo))]
#                 mid_x = (xs[j_min] + xs[j_max]) / 2.0
#                 mid_y = (ys[j_min] + ys[j_max]) / 2.0
#                 records.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': Point(mid_x, mid_y)
#                 })
#                 for j in grupo_lon:
#                     used[j] = True

#         # 3) Los puntos que queden sin usar se añaden tal cual (punto “aislado”)
#         for i in range(m):
#             if not used[i]:
#                 records.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': Point(xs[i], ys[i])
#                 })
#                 used[i] = True

#     return gpd.GeoDataFrame(records, geometry='geometry')


# gdf_midpoints = get_midpoints_same_sector_pairs(gdf_midpoints, tol=0.1)














# # 1) Asegúrate de que gdf_borders tenga un ID por fila
# gdf_borders = gdf_borders.reset_index().rename(columns={'index':'border_id'})


# def get_midpoints_per_segment(gdf_borders, tol=1e-9):
#     """
#     Para cada frontera (LineString) en gdf_borders, descompone sus segmentos
#     consecutivos y genera un punto medio si el tramo es puramente horizontal
#     o vertical. Conserva también el border_id y los sectores.
#     """
#     records = []
#     for _, row in gdf_borders.iterrows():
#         bid    = row['border_id']
#         s1, s2 = row['sector_1'], row['sector_2']
#         coords = list(row.geometry.coords)
#         # recorre cada segmento (coords[i] → coords[i+1])
#         for i in range(len(coords)-1):
#             x1, y1 = coords[i]
#             x2, y2 = coords[i+1]
#             # vertical
#             if abs(x1 - x2) < tol:
#                 mx, my = x1,        (y1 + y2)/2.0
#             # horizontal
#             elif abs(y1 - y2) < tol:
#                 mx, my = (x1 + x2)/2.0, y1
#             else:
#                 continue
#             records.append({
#                 'border_id': bid,
#                 'sector_1' : s1,
#                 'sector_2' : s2,
#                 'geometry' : Point(mx, my)
#             })

#     return gpd.GeoDataFrame(records, crs=gdf_borders.crs)

# # 2) Llamada reemplazando ambos pasos anteriores:
# gdf_midpoints = get_midpoints_per_segment(gdf_borders)











# gdf_midpoints = gdf_mid_const

# from shapely.geometry import LineString
# import geopandas as gpd

# # 0) Prepara cells_info
# cells_info = [
#     (row['Cell_Name'], row['Polygon'], row['Sector'])
#     for _, row in gdf_cells.iterrows()
# ]

# # 1) get_shared_borders_and_nodes
# gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# # 2) polígono ACC y su contorno
# polygons = DF_info_conf['Contorno Sector'].tolist()
# union_poly = unary_union(polygons)
# if union_poly.geom_type == 'MultiPolygon':
#     poligono_ACC = union_poly.convex_hull
# else:
#     poligono_ACC = Polygon(union_poly.exterior)
# acc_boundary = poligono_ACC.boundary


# # 3) puntos de contacto ACC
# records = []
# for _, row in gdf_borders.iterrows():
#     s1, s2 = row.sector_1, row.sector_2
#     for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
#         pt = Point(x, y)
#         if pt.distance(acc_boundary) < 1e-9:
#             records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
# gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
#                      .drop_duplicates('geometry') \
#                      .reset_index(drop=True)


# # 4) cruces H-V
# gdf_corners = extract_corner_points(gdf_borders)


# from pyproj import Geod

# def filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=10):
#     """
#     Elimina los puntos de gdf_midpoints que estén a menos de `max_nm` millas náuticas
#     de cualquiera de los puntos en gdf_acc_touch.

#     Si los GeoDataFrames no tienen CRS definido, asume WGS84 (grados).
#     """
#     # Convertir millas náuticas a metros
#     threshold_m = max_nm * 1852

#     # Determinar si es geográfico (lat/lon) o proyectado
#     is_geographic = True
#     if hasattr(gdf_midpoints, 'crs') and gdf_midpoints.crs is not None:
#         try:
#             is_geographic = gdf_midpoints.crs.is_geographic
#         except AttributeError:
#             is_geographic = True

#     # Obtener coordenadas de puntos ACC
#     acc_coords = [(pt.x, pt.y) for pt in gdf_acc_touch.geometry]

#     if is_geographic:
#         # Usar cálculo geodésico en WGS84
#         geod = Geod(ellps="WGS84")
#         keep_indices = []
#         for idx, mid_pt in enumerate(gdf_midpoints.geometry):
#             lon, lat = mid_pt.x, mid_pt.y
#             # Calcular distancia a cada punto ACC
#             distances = [geod.inv(lon, lat, acc_lon, acc_lat)[2] for acc_lon, acc_lat in acc_coords]
#             # Conservar si todas las distancias > umbral
#             if all(d > threshold_m for d in distances):
#                 keep_indices.append(idx)
#         return gdf_midpoints.loc[keep_indices].reset_index(drop=True)
#     else:
#         # CRS proyectado (unidades en metros): usar buffer único
#         acc_buffer = gdf_acc_touch.buffer(threshold_m).unary_union
#         filtered = gdf_midpoints[~gdf_midpoints.geometry.within(acc_buffer)].copy()
#         return filtered.reset_index(drop=True)

# # Ejemplo de uso:
# gdf_midpoints = filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=5)




# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import LineString, Point

# def connect_all_points(gdf_midpoints, gdf_nodes, gdf_acc_touch):
#     """
#     Conecta:
#       - gdf_midpoints: puntos medios con columnas ['sector_1','sector_2','geometry']
#       - gdf_acc_touch: puntos ACC con columnas ['sector_1','sector_2','geometry']
#       - gdf_nodes: puntos nodales con columnas ['sectors','geometry'],
#                    donde 'sectors' es una lista de sectores (strings)
#     Siguiendo estas reglas:
#       • Cada punto ACC (tiene dos sectores [s1,s2]) se conecta a SU vecino MÁS CERCANO
#         que comparta ambos sectores s1 y s2 (puede ser midpoint o node).
#       • Cada punto medio (tiene dos sectores [s1,s2]) se conecta a SUS DOS vecinos MÁS CERCANOS
#         que compartan el par {s1,s2}. Si solo hay uno, se conecta a ese.
#       • Cada nodo (tiene lista de sectores [s1,s2,…,sK]) se conecta, para cada sector si en su lista,
#         a SU vecino MÁS CERCANO que contenga ESE sector en su lista de sectores (puede ser midpoint,
#         ACC o node distinto). De ese modo genera un número de aristas igual al número de sectores.
#       • No se crean duplicados: si A se conecta a B, no se vuelve a crear B→A.

#     Devuelve:
#       GeoDataFrame con columnas ['sector_1','sector_2','geometry'], donde cada fila
#       es una arista (LineString) que une dos puntos conforme a las reglas.
#     """

#     # 1) Construir lista unificada de “puntos” con:
#     #     - uid: índice entero único
#     #     - coords: (x,y) tupla de coordenadas
#     #     - sectors: lista de sectores (para mid/acc, dos; para node, lista)
#     #     - type: 'mid', 'acc' o 'node'
#     puntos = []
#     uid = 0

#     # a) Midpoints
#     for _, row in gdf_midpoints.iterrows():
#         s1, s2 = row['sector_1'], row['sector_2']
#         pts = row.geometry
#         puntos.append({
#             'uid': uid,
#             'coords': (pts.x, pts.y),
#             'sectors': [s1, s2],
#             'type': 'mid'
#         })
#         uid += 1

#     # b) ACC Touch
#     for _, row in gdf_acc_touch.iterrows():
#         s1, s2 = row['sector_1'], row['sector_2']
#         pts = row.geometry
#         puntos.append({
#             'uid': uid,
#             'coords': (pts.x, pts.y),
#             'sectors': [s1, s2],
#             'type': 'acc'
#         })
#         uid += 1

#     # c) Nodes
#     for _, row in gdf_nodes.iterrows():
#         sec_list = row['sectors']  # asume lista de strings
#         pts = row.geometry
#         puntos.append({
#             'uid': uid,
#             'coords': (pts.x, pts.y),
#             'sectors': list(sec_list),  # ya es lista
#             'type': 'node'
#         })
#         uid += 1

#     if not puntos:
#         return gpd.GeoDataFrame(columns=['sector_1','sector_2','geometry'], geometry='geometry')

#     # 2) Preparar arrays para cálculos de distancia
#     n = len(puntos)
#     coords = np.array([p['coords'] for p in puntos])  # shape (n,2)
#     # Distancia euclídea² entre cada par (i,j)
#     diffs = coords[:, None, :] - coords[None, :, :]    # shape (n,n,2)
#     dist2 = np.sum(diffs**2, axis=2)                   # shape (n,n)
#     np.fill_diagonal(dist2, np.inf)

#     # 3) Función auxiliar: para un índice i, devuelve lista de índices de candidatos j donde:
#     #      - La lista de sectores de j contiene TODOS los sectores de i (para mid y acc)
#     #      - Para node: la lista de sectores de j contiene el sector sc que estamos procesando
#     #    y j != i.
#     def candidatos_para(i, modo, sector_obj=None):
#         """
#         i: índice del punto en 'puntos'
#         modo: 'pair' si queremos pares exactos (mid/acc),
#               'single' si es búsqueda por un solo sector (para node)
#         sector_obj: en modo 'single', el sector en cuestión (string)
#         Retorna lista de índices j válidos.
#         """
#         lista_j = []
#         if modo == 'pair':
#             set_i = set(puntos[i]['sectors'])  # debe tener exactamente 2
#             for j in range(n):
#                 if j == i:
#                     continue
#                 if set_i.issubset(set(puntos[j]['sectors'])):
#                     lista_j.append(j)
#         else:  # modo == 'single'
#             # sector_obj es un string. Buscamos todos j != i con sector_obj in puntos[j]['sectors']
#             for j in range(n):
#                 if j == i:
#                     continue
#                 if sector_obj in puntos[j]['sectors']:
#                     lista_j.append(j)
#         return lista_j

#     # 4) Construir aristas sin duplicados
#     added = set()  # almacenará frozenset({i,j}) para evitar duplicados
#     records = []

#     for idx in range(n):
#         punto = puntos[idx]
#         tipo = punto['type']

#         if tipo == 'acc':
#             # buscar candidatos que contengan el mismo par de sectores
#             cands = candidatos_para(idx, modo='pair')
#             if not cands:
#                 continue
#             # elegir el más cercano según dist2
#             j_min = int(np.argmin(dist2[idx, cands]))
#             vecino = cands[j_min]
#             key = frozenset({idx, vecino})
#             if key not in added:
#                 p1 = coords[idx]
#                 p2 = coords[vecino]
#                 records.append({
#                     'sector_1': punto['sectors'][0],
#                     'sector_2': punto['sectors'][1],
#                     'geometry': LineString([Point(*p1), Point(*p2)])
#                 })
#                 added.add(key)

#         elif tipo == 'mid':
#             # buscar candidatos que contengan el par exacto
#             cands = candidatos_para(idx, modo='pair')
#             if not cands:
#                 continue
#             # necesitamos hasta dos vecinos más cercanos
#             k = min(2, len(cands))
#             idxs_k = np.argsort(dist2[idx, cands])[:k]
#             for pos in idxs_k:
#                 vecino = cands[pos]
#                 key = frozenset({idx, vecino})
#                 if key in added:
#                     continue
#                 p1 = coords[idx]
#                 p2 = coords[vecino]
#                 records.append({
#                     'sector_1': punto['sectors'][0],
#                     'sector_2': punto['sectors'][1],
#                     'geometry': LineString([Point(*p1), Point(*p2)])
#                 })
#                 added.add(key)

#         else:  # tipo == 'node'
#             # para cada sector en su lista, conectar al vecino más cercano que contenga ese sector
#             for sector in punto['sectors']:
#                 cands = candidatos_para(idx, modo='single', sector_obj=sector)
#                 if not cands:
#                     continue
#                 j_min = int(np.argmin(dist2[idx, cands]))
#                 vecino = cands[j_min]
#                 key = frozenset({idx, vecino})
#                 if key in added:
#                     continue
#                 # La arista pertenece a la frontera entre el sector “sector”
#                 # y algún otro. Pero debemos decidir qué par guardamos en 'sector_1','sector_2'.
#                 # Tomamos el par formado por ese 'sector' y el otro sector compartido.
#                 # Encontrar intersección de listas de sectores:
#                 set_i = set(puntos[idx]['sectors'])
#                 set_j = set(puntos[vecino]['sectors'])
#                 comunes = set_i.intersection(set_j)
#                 # Debería contener al menos 'sector'. Ahora elegimos:
#                 #   - si hay exactamente 2 comunes, ese par es la frontera
#                 #   - si hay más de 2, tomamos arbitrariamente los dos primeros
#                 if len(comunes) >= 2:
#                     s1, s2 = sorted(list(comunes))[:2]
#                 else:
#                     # Si solo 'sector' está en común, buscamos en punto j un sector diferente:
#                     otros_j = set_j - {sector}
#                     if otros_j:
#                         s1, s2 = sorted([sector, list(otros_j)[0]])
#                     else:
#                         # cae aquí si ambos puntos comparten exactamente ese sector;
#                         # en ese caso, no hay "par" claro, we still put sector twice
#                         s1, s2 = sector, sector
#                 p1 = coords[idx]
#                 p2 = coords[vecino]
#                 records.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': LineString([Point(*p1), Point(*p2)])
#                 })
#                 added.add(key)

#     # 5) Devolver GeoDataFrame final
#     gdf_edges = gpd.GeoDataFrame(records, geometry='geometry')
#     # Asignar CRS heredado (si alguno de los inputs lo tiene)
#     for df_in in (gdf_midpoints, gdf_acc_touch, gdf_nodes):
#         if df_in is not None and hasattr(df_in, 'crs') and df_in.crs is not None:
#             gdf_edges.set_crs(df_in.crs, inplace=True)
#             break

#     return gdf_edges


# # Asumiendo que ya tienes:
# #   gdf_midpoints con ['sector_1','sector_2','geometry']
# #   gdf_acc_touch con ['sector_1','sector_2','geometry']
# #   gdf_nodes con ['sectors','geometry']  (donde 'sectors' es lista de strings)

# gdf_edges = connect_all_points(
#     gdf_midpoints=gdf_midpoints,
#     gdf_nodes=gdf_nodes,
#     gdf_acc_touch=gdf_acc_touch
# )

# # El GeoDataFrame resultante 'gdf_edges' tendrá columnas:
# #   - 'sector_1', 'sector_2'  (indicando la frontera asociada a cada arista)
# #   - 'geometry' (LineString entre los dos puntos unidos)
# print(gdf_edges)



# import geopandas as gpd

# def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
#     """
#     Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
#     aparece. Si aparece en más de una, elimina la línea de mayor longitud
#     entre las que comparten ese punto.

#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
#     gdf_acc_touch : GeoDataFrame
#         GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
#         (son los puntos ACC).

#     Retorna
#     -------
#     GeoDataFrame
#         Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
#         más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
#     """
#     # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
#     #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
#     #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
#     occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
#     # Crear un set de coordenadas de ACC para comparación rápida:
#     acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

#     # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
#     #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
#     #    exactamente con uno de sus endpoints (primera o última coordenada).
#     for idx_edge, linea in gdf_edges.geometry.items():
#         # Obtener los endpoints (coordenadas) del LINESTRING
#         coords = list(linea.coords)
#         extremo1 = tuple(coords[0])
#         extremo2 = tuple(coords[-1])

#         # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
#         if extremo1 in acc_coords:
#             occ.setdefault(extremo1, []).append(idx_edge)
#         # Si extremo2 corresponde a un punto ACC, apuntamos también
#         if extremo2 in acc_coords:
#             occ.setdefault(extremo2, []).append(idx_edge)

#     # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
#     #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
#     #    para eliminación la que resulte más larga.
#     a_eliminar = set()
#     for punto_acc, lista_indices in occ.items():
#         if len(lista_indices) > 1:
#             # Calcular longitudes de todas las líneas que comparten este punto ACC
#             longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
#             # Ordenar por longitud
#             longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
#             # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
#             idx_linea_mas_larga, _ = longitudes[0]
#             a_eliminar.add(idx_linea_mas_larga)

#     # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
#     if a_eliminar:
#         # Drop por índices y resetear índice (opcional)
#         gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
#     else:
#         # Si no hay nada que eliminar, devolvemos una copia idéntica
#         gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

#     # Conservamos el CRS original
#     if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
#         gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

#     return gdf_filtrado


# gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
#     gdf_edges=gdf_edges,
#     gdf_acc_touch=gdf_acc_touch
# )

# gdf_edges = gdf_edges_sin_duplicados



# import geopandas as gpd
# from shapely.geometry import Point, LineString
# from collections import defaultdict


# import geopandas as gpd
# from shapely.geometry import Point, LineString
# from collections import defaultdict

# def eliminar_aristas_prefer_node_mid(gdf_edges, gdf_midpoints, gdf_nodes, tol=1e-9):
#     """
#     Para cada punto medio con grado > 2, elimina primero las aristas que conectan
#     con un nodo, y solo si faltan por eliminar, las que conectan con otro midpoint,
#     hasta dejar grado = 2.
    
#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         Debe contener geometrías LINESTRING y un índice único.
#     gdf_midpoints : GeoDataFrame
#         Puntos medios (geom POINT).
#     gdf_nodes : GeoDataFrame
#         Nodos (geom POINT).
#     tol : float
#         Tolerancia para comparar coordenadas.
    
#     Retorna
#     -------
#     GeoDataFrame
#         Copia de gdf_edges sin las aristas eliminadas.
#     """
#     edges = gdf_edges.copy()
    
#     # Función para obtener clave redondeada de un punto
#     def key(pt):
#         return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
#     # Conjuntos de claves de midpoints y nodos
#     mid_keys  = { key(p) for p in gdf_midpoints.geometry }
#     node_keys = { key(p) for p in gdf_nodes.geometry }
    
#     # Incidencia: midpoint -> lista de índices de edges conectados
#     incidence = defaultdict(list)
#     for idx, line in edges.geometry.items():
#         coords = list(line.coords)
#         for extremo in (coords[0], coords[-1]):
#             k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
#             if k in mid_keys:
#                 incidence[k].append(idx)
    
#     to_drop = set()
#     # Procesar cada midpoint con más de 2 conexiones
#     for mid_k, edge_idxs in incidence.items():
#         degree = len(edge_idxs)
#         if degree <= 2:
#             continue
#         eliminar = degree - 2
        
#         # Clasificar candidatos según destino node o midpoint
#         node_conns = []
#         mid_conns  = []
#         for idx in edge_idxs:
#             coords = list(edges.geometry.loc[idx].coords)
#             e0 = (round(coords[0][0]/tol)*tol, round(coords[0][1]/tol)*tol)
#             e1 = (round(coords[-1][0]/tol)*tol, round(coords[-1][1]/tol)*tol)
#             other = e1 if e0 == mid_k else e0
#             if other in node_keys:
#                 node_conns.append(idx)
#             elif other in mid_keys:
#                 mid_conns.append(idx)
        
#         # Primero eliminar conexiones a nodos
#         for idx in node_conns[:eliminar]:
#             to_drop.add(idx)
#         faltan = eliminar - min(len(node_conns), eliminar)
#         # # Si todavía faltan, eliminar conexiones a otros midpoints
#         # for idx in mid_conns[:faltan]:
#         #     to_drop.add(idx)
    
#     # Eliminar y resetear índice
#     if to_drop:
#         edges = edges.drop(index=list(to_drop)).reset_index(drop=True)
#     # Mantener CRS
#     if hasattr(gdf_edges, 'crs'):
#         edges.set_crs(gdf_edges.crs, inplace=True)
#     return edges

# # Ejemplo:
# # gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
# #     gdf_edges, gdf_midpoints, gdf_nodes
# # )

# gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
#     gdf_edges,
#     gdf_midpoints,
#     gdf_nodes
# )



# gdf_edges = gdf_edges_filtrado


# def conectar_midpoints_unicos(gdf_edges, gdf_midpoints, tol=1e-9):
#     """
#     Para cada punto medio con solo una conexión, conecta este punto con los dos midpoints
#     más cercanos en su frontera, si es posible.
    
#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         Debe contener geometrías LINESTRING y un índice único.
#     gdf_midpoints : GeoDataFrame
#         Puntos medios (geom POINT).
#     tol : float
#         Tolerancia para comparar coordenadas.
    
#     Retorna
#     -------
#     GeoDataFrame
#         Copia de gdf_edges con las nuevas aristas agregadas.
#     """
#     edges = gdf_edges.copy()
    
#     # Función para obtener clave redondeada de un punto
#     def key(pt):
#         return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
#     # Conjuntos de claves de midpoints
#     mid_keys = { key(p) for p in gdf_midpoints.geometry }
    
#     # Incidencia: midpoint -> lista de índices de edges conectados
#     incidence = defaultdict(list)
#     for idx, line in edges.geometry.items():
#         coords = list(line.coords)
#         for extremo in (coords[0], coords[-1]):
#             k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
#             if k in mid_keys:
#                 incidence[k].append(idx)
    
#     # Nueva lista de aristas
#     new_edges = []

#     # 1) Identificar midpoints con una única conexión
#     for mid_k, edge_idxs in incidence.items():
#         if len(edge_idxs) == 1:
#             # Identificar el sector de este midpoint
#             midpoint = gdf_midpoints[gdf_midpoints.geometry.apply(key) == mid_k]
#             if midpoint.empty:
#                 continue
#             s1, s2 = midpoint.iloc[0]['sector_1'], midpoint.iloc[0]['sector_2']
            
#             # Buscar otros midpoints con la misma frontera
#             vecinos_potenciales = gdf_midpoints[(gdf_midpoints['sector_1'] == s1) & (gdf_midpoints['sector_2'] == s2)]
            
#             # Eliminar el punto medio actual de la lista de vecinos
#             vecinos_potenciales = vecinos_potenciales[vecinos_potenciales.geometry.apply(key) != mid_k]
            
#             # Si hay más de uno, encontrar los dos más cercanos
#             if len(vecinos_potenciales) > 1:
#                 coords_mid = midpoint.iloc[0].geometry.coords[0]
#                 vecinos_coords = [(v.geometry.coords[0], idx) for idx, v in vecinos_potenciales.iterrows()]
                
#                 # Calcular distancias
#                 # distancias = [(idx, np.sqrt((coords_mid[0] - v[0][0])**2 + (coords_mid[1] - v[0][1])**2)) for v, idx in vecinos_coords]
#                 distancias = [(idx,np.sqrt((coords_mid[0] - v[0])**2 +(coords_mid[1] - v[1])**2))for v, idx in vecinos_coords]

#                 # Ordenar por distancia
#                 distancias.sort(key=lambda x: x[1])
                
#                 # Seleccionar los dos más cercanos
#                 closest_idx = distancias[0][0]
#                 second_closest_idx = distancias[1][0]
                
#                 # Crear nuevas aristas entre el midpoint y los dos más cercanos
#                 new_edges.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[closest_idx].geometry])
#                 })
#                 new_edges.append({
#                     'sector_1': s1,
#                     'sector_2': s2,
#                     'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[second_closest_idx].geometry])
#                 })
    
#     # 2) Agregar las nuevas aristas al GeoDataFrame
#     if new_edges:
#         new_edges_gdf = gpd.GeoDataFrame(new_edges, geometry='geometry')
#         edges = pd.concat([edges, new_edges_gdf], ignore_index=True)
    
#     # Mantener CRS
#     if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
#         edges.set_crs(gdf_edges.crs, inplace=True)
    
#     return edges


# gdf_edges_nuevo = conectar_midpoints_unicos(gdf_edges, gdf_midpoints)
# gdf_edges=gdf_edges_nuevo


# def eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints, tol=1e-9):
#     """
#     Elimina solo las aristas que conectan entre sí dos midpoints,
#     cuando ambos tienen grado > 2.

#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         Aristas del grafo con geometrías LineString.
#     gdf_midpoints : GeoDataFrame
#         Midpoints con geometría Point.
#     tol : float
#         Tolerancia para comparación de coordenadas.

#     Retorna
#     -------
#     GeoDataFrame
#         gdf_edges sin las aristas que unen midpoints con grado > 2 entre sí.
#     """
#     from collections import defaultdict

#     def key(pt):
#         return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

#     # Crear set de claves de midpoints
#     mid_keys = {key(p) for p in gdf_midpoints.geometry}

#     # Calcular grado de cada midpoint
#     degree_map = defaultdict(int)
#     edge_extremos = {}

#     for idx, line in gdf_edges.geometry.items():
#         coords = list(line.coords)
#         k1 = key(Point(*coords[0]))
#         k2 = key(Point(*coords[-1]))
#         edge_extremos[idx] = (k1, k2)

#         if k1 in mid_keys:
#             degree_map[k1] += 1
#         if k2 in mid_keys:
#             degree_map[k2] += 1

#     # Identificar aristas entre midpoints de grado > 2
#     to_drop = set()
#     for idx, (k1, k2) in edge_extremos.items():
#         if k1 in mid_keys and k2 in mid_keys:
#             if degree_map[k1] > 2 and degree_map[k2] > 2:
#                 to_drop.add(idx)

#     # Mostrar resumen
#     print(f"Aristas eliminadas entre midpoints de grado > 2: {len(to_drop)}")

#     # Eliminar aristas identificadas
#     gdf_filtrado = gdf_edges.drop(index=list(to_drop)).reset_index(drop=True) if to_drop else gdf_edges.copy()

#     # Conservar CRS
#     if hasattr(gdf_edges, 'crs') and gdf_edges.crs:
#         gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

#     return gdf_filtrado


# gdf_edges = eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints)



# def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
#     """
#     Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
#     aparece. Si aparece en más de una, elimina la línea de mayor longitud
#     entre las que comparten ese punto.

#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
#     gdf_acc_touch : GeoDataFrame
#         GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
#         (son los puntos ACC).

#     Retorna
#     -------
#     GeoDataFrame
#         Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
#         más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
#     """
#     # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
#     #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
#     #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
#     occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
#     # Crear un set de coordenadas de ACC para comparación rápida:
#     acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

#     # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
#     #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
#     #    exactamente con uno de sus endpoints (primera o última coordenada).
#     for idx_edge, linea in gdf_edges.geometry.items():
#         # Obtener los endpoints (coordenadas) del LINESTRING
#         coords = list(linea.coords)
#         extremo1 = tuple(coords[0])
#         extremo2 = tuple(coords[-1])

#         # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
#         if extremo1 in acc_coords:
#             occ.setdefault(extremo1, []).append(idx_edge)
#         # Si extremo2 corresponde a un punto ACC, apuntamos también
#         if extremo2 in acc_coords:
#             occ.setdefault(extremo2, []).append(idx_edge)

#     # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
#     #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
#     #    para eliminación la que resulte más larga.
#     a_eliminar = set()
#     for punto_acc, lista_indices in occ.items():
#         if len(lista_indices) > 1:
#             # Calcular longitudes de todas las líneas que comparten este punto ACC
#             longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
#             # Ordenar por longitud
#             longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
#             # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
#             idx_linea_mas_larga, _ = longitudes[0]
#             a_eliminar.add(idx_linea_mas_larga)

#     # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
#     if a_eliminar:
#         # Drop por índices y resetear índice (opcional)
#         gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
#     else:
#         # Si no hay nada que eliminar, devolvemos una copia idéntica
#         gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

#     # Conservamos el CRS original
#     if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
#         gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

#     return gdf_filtrado


# gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
#     gdf_edges=gdf_edges,
#     gdf_acc_touch=gdf_acc_touch
# )

# gdf_edges = gdf_edges_sin_duplicados

# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import LineString
# from collections import defaultdict

# def conectar_midpoints_aislados(gdf_edges, gdf_midpoints, tol=1e-9):
#     """
#     Para cada frontera (sector_1, sector_2), identifica los midpoints
#     que solo tienen una conexión (grado==1). Si en esa misma frontera
#     hay exactamente dos midpoints de grado 1, añade una arista que los una.

#     Parámetros
#     ----------
#     gdf_edges : GeoDataFrame
#         Aristas existentes con geometrías LINESTRING.
#     gdf_midpoints : GeoDataFrame
#         Midpoints con columnas ['sector_1','sector_2','geometry'].
#     tol : float
#         Tolerancia para comparar coordenadas (por defecto 1e-9).

#     Retorna
#     -------
#     GeoDataFrame
#         Copia de gdf_edges con las nuevas aristas agregadas.
#     """
#     # 1) Función para obtener clave “redondeada” de un punto
#     def key(pt):
#         return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

#     # 2) Preparamos el conteo de grado para cada midpoint (clave->grado)
#     #    y almacenamos la geometría de cada clave
#     mid_keys = {}      # key -> Point geom
#     for _, row in gdf_midpoints.iterrows():
#         k = key(row.geometry)
#         mid_keys[k] = row.geometry

#     grado = defaultdict(int)
#     # Recorremos cada arista y contamos sus endpoints si son midpoints
#     for line in gdf_edges.geometry:
#         x0, y0 = line.coords[0]
#         x1, y1 = line.coords[-1]
#         k0, k1 = key(type(line)(x0, y0)) if False else (None, None), (None, None)
#         # mejor extraer directamente
#         k0 = key(Point(x0, y0))
#         k1 = key(Point(x1, y1))
#         if k0 in mid_keys:
#             grado[k0] += 1
#         if k1 in mid_keys:
#             grado[k1] += 1

#     new_edges = []
#     # 3) Agrupamos los midpoints por frontera
#     for (s1, s2), group in gdf_midpoints.groupby(['sector_1', 'sector_2']):
#         # Mapeamos clave->fila para este grupo
#         key_to_row = { key(row.geometry): row for _, row in group.iterrows() }
#         # Filtramos los que tengan grado == 1
#         aislados = [k for k in key_to_row if grado.get(k, 0) == 1]
#         # Si hay exactamente dos, los conectamos
#         if len(aislados) == 2:
#             geom1 = key_to_row[aislados[0]].geometry
#             geom2 = key_to_row[aislados[1]].geometry
#             new_edges.append({
#                 'sector_1': s1,
#                 'sector_2': s2,
#                 'geometry': LineString([geom1, geom2])
#             })

#     # 4) Si hay nuevas aristas, las concatenamos
#     if new_edges:
#         gdf_nuevas = gpd.GeoDataFrame(new_edges, geometry='geometry', crs=gdf_edges.crs)
#         return pd.concat([gdf_edges, gdf_nuevas], ignore_index=True)
#     else:
#         return gdf_edges.copy()


# gdf_edges = conectar_midpoints_aislados(gdf_edges, gdf_midpoints)


# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon, MultiPolygon
# from shapely.ops import unary_union

# # 1) Reconstruir el contorno real del ACC (sin usar convex_hull)
# poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
# union_poligonos = unary_union(poligonos_sectores)

# if isinstance(union_poligonos, MultiPolygon):
#     # Si la unión da MultiPolygon, tomamos cada polígono por separado
#     lista_polys = list(union_poligonos.geoms)
# else:
#     # Si es un único Polygon
#     lista_polys = [union_poligonos]

# # 2) Extraer todas las coordenadas de los exteriores para ajustar límites
# all_x = []
# all_y = []
# for poly in lista_polys:
#     x_poly, y_poly = poly.exterior.xy
#     all_x.extend(x_poly)
#     all_y.extend(y_poly)

# min_lon = min(all_x) - 0.5
# max_lon = max(all_x) + 0.5
# min_lat = min(all_y) - 0.5
# max_lat = max(all_y) + 0.5

# # 3) Crear figura y ejes
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(min_lon, max_lon)
# ax.set_ylim(min_lat, max_lat)
# ax.set_aspect('equal')
# ax.set_xlabel('LONGITUD [º]')
# ax.set_ylabel('LATITUD [º]')
# ax.set_title('Sectorización Final')

# # 4) Dibujar solo el contorno (exterior) de cada polígono de ACC, sin rellenar
# for poly in lista_polys:
#     x_poly, y_poly = poly.exterior.xy
#     ax.plot(x_poly, y_poly, color='black', linewidth=1, label='_nolegend_')

# # 5) Ahora superponemos cada LineString de gdf_edges en rojo
# for linea in gdf_edges.geometry:
#     xs, ys = linea.xy
#     ax.plot(xs, ys, color='red', linewidth=1.2)

# # Opcional: añadir una leyenda manual para gdf_edges
# ax.plot([], [], color='red', linewidth=1.2, label='Fronteras nuevas')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')

# plt.tight_layout()
# plt.show()

# # 1) Filtrar puntos que están dentro de la delimitación del ACC
# gdf_midpoints_in = gdf_midpoints[gdf_midpoints.geometry.within(union_poligonos)]
# gdf_acc_touch_in = gdf_acc_touch
# # gdf_acc_touch_in = gdf_acc_touch[gdf_acc_touch.geometry.within(union_poligonos)]
# gdf_nodes_in = gdf_nodes[gdf_nodes.geometry.within(union_poligonos)]

# # 2) Crear figura y ejes
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.set_xlim(min_lon, max_lon)
# ax.set_ylim(min_lat, max_lat)
# ax.set_aspect('equal')
# ax.set_xlabel('LONGITUD [º]')
# ax.set_ylabel('LATITUD [º]')
# ax.set_title('Puntos Utilizados para Realizar la Representación')

# # 3) Dibujar contorno ACC
# for poly in lista_polys:
#     x_poly, y_poly = poly.exterior.xy
#     ax.plot(x_poly, y_poly, color='black', linewidth=1, label='Contorno ACC')

# # 4) Dibujar puntos
# gdf_midpoints_in.plot(ax=ax, color='blue', markersize=20, label='Puntos medios')
# gdf_acc_touch_in.plot(ax=ax, color='green', markersize=30, marker='^', label='Puntos ACC')
# gdf_nodes_in.plot(ax=ax, color='purple', markersize=40, marker='s', label='Nodos')

# # 5) Leyenda y presentación
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
# plt.tight_layout()
# plt.show()































# 1) Asegúrate de que gdf_borders tenga un ID por fila
gdf_borders = gdf_borders.reset_index().rename(columns={'index':'border_id'})






def get_midpoints_per_segment(gdf_borders, tol=1e-9):
    """
    Para cada frontera (LineString) en gdf_borders, descompone sus segmentos
    consecutivos y genera un punto medio si el tramo es puramente horizontal
    o vertical. Conserva también el border_id y los sectores.
    """
    records = []
    for _, row in gdf_borders.iterrows():
        bid    = row['border_id']
        s1, s2 = row['sector_1'], row['sector_2']
        coords = list(row.geometry.coords)
        # recorre cada segmento (coords[i] → coords[i+1])
        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            # vertical
            if abs(x1 - x2) < tol:
                mx, my = x1,        (y1 + y2)/2.0
            # horizontal
            elif abs(y1 - y2) < tol:
                mx, my = (x1 + x2)/2.0, y1
            else:
                continue
            records.append({
                'border_id': bid,
                'sector_1' : s1,
                'sector_2' : s2,
                'geometry' : Point(mx, my)
            })

    return gpd.GeoDataFrame(records, crs=gdf_borders.crs)

# 2) Llamada reemplazando ambos pasos anteriores:
gdf_midpoints = get_midpoints_per_segment(gdf_borders)



from shapely.geometry import LineString
import geopandas as gpd

# 0) Prepara cells_info
cells_info = [
    (row['Cell_Name'], row['Polygon'], row['Sector'])
    for _, row in gdf_cells.iterrows()
]

# 1) get_shared_borders_and_nodes
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) polígono ACC y su contorno
polygons = DF_info_conf['Contorno Sector'].tolist()
union_poly = unary_union(polygons)
if union_poly.geom_type == 'MultiPolygon':
    poligono_ACC = union_poly.convex_hull
else:
    poligono_ACC = Polygon(union_poly.exterior)
acc_boundary = poligono_ACC.boundary


# 3) puntos de contacto ACC
records = []
for _, row in gdf_borders.iterrows():
    s1, s2 = row.sector_1, row.sector_2
    for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
        pt = Point(x, y)
        if pt.distance(acc_boundary) < 1e-9:
            records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
                     .drop_duplicates('geometry') \
                     .reset_index(drop=True)


# 4) cruces H-V
gdf_corners = extract_corner_points(gdf_borders)


from pyproj import Geod

def filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=10):
    """
    Elimina los puntos de gdf_midpoints que estén a menos de `max_nm` millas náuticas
    de cualquiera de los puntos en gdf_acc_touch.

    Si los GeoDataFrames no tienen CRS definido, asume WGS84 (grados).
    """
    # Convertir millas náuticas a metros
    threshold_m = max_nm * 1852

    # Determinar si es geográfico (lat/lon) o proyectado
    is_geographic = True
    if hasattr(gdf_midpoints, 'crs') and gdf_midpoints.crs is not None:
        try:
            is_geographic = gdf_midpoints.crs.is_geographic
        except AttributeError:
            is_geographic = True

    # Obtener coordenadas de puntos ACC
    acc_coords = [(pt.x, pt.y) for pt in gdf_acc_touch.geometry]

    if is_geographic:
        # Usar cálculo geodésico en WGS84
        geod = Geod(ellps="WGS84")
        keep_indices = []
        for idx, mid_pt in enumerate(gdf_midpoints.geometry):
            lon, lat = mid_pt.x, mid_pt.y
            # Calcular distancia a cada punto ACC
            distances = [geod.inv(lon, lat, acc_lon, acc_lat)[2] for acc_lon, acc_lat in acc_coords]
            # Conservar si todas las distancias > umbral
            if all(d > threshold_m for d in distances):
                keep_indices.append(idx)
        return gdf_midpoints.loc[keep_indices].reset_index(drop=True)
    else:
        # CRS proyectado (unidades en metros): usar buffer único
        acc_buffer = gdf_acc_touch.buffer(threshold_m).unary_union
        filtered = gdf_midpoints[~gdf_midpoints.geometry.within(acc_buffer)].copy()
        return filtered.reset_index(drop=True)

# Ejemplo de uso:
gdf_midpoints = filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=5)




import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

def connect_all_points(gdf_midpoints, gdf_nodes, gdf_acc_touch):
    """
    Conecta:
      - gdf_midpoints: puntos medios con columnas ['sector_1','sector_2','geometry']
      - gdf_acc_touch: puntos ACC con columnas ['sector_1','sector_2','geometry']
      - gdf_nodes: puntos nodales con columnas ['sectors','geometry'],
                   donde 'sectors' es una lista de sectores (strings)
    Siguiendo estas reglas:
      • Cada punto ACC (tiene dos sectores [s1,s2]) se conecta a SU vecino MÁS CERCANO
        que comparta ambos sectores s1 y s2 (puede ser midpoint o node).
      • Cada punto medio (tiene dos sectores [s1,s2]) se conecta a SUS DOS vecinos MÁS CERCANOS
        que compartan el par {s1,s2}. Si solo hay uno, se conecta a ese.
      • Cada nodo (tiene lista de sectores [s1,s2,…,sK]) se conecta, para cada sector si en su lista,
        a SU vecino MÁS CERCANO que contenga ESE sector en su lista de sectores (puede ser midpoint,
        ACC o node distinto). De ese modo genera un número de aristas igual al número de sectores.
      • No se crean duplicados: si A se conecta a B, no se vuelve a crear B→A.

    Devuelve:
      GeoDataFrame con columnas ['sector_1','sector_2','geometry'], donde cada fila
      es una arista (LineString) que une dos puntos conforme a las reglas.
    """

    # 1) Construir lista unificada de “puntos” con:
    #     - uid: índice entero único
    #     - coords: (x,y) tupla de coordenadas
    #     - sectors: lista de sectores (para mid/acc, dos; para node, lista)
    #     - type: 'mid', 'acc' o 'node'
    puntos = []
    uid = 0

    # a) Midpoints
    for _, row in gdf_midpoints.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'mid'
        })
        uid += 1

    # b) ACC Touch
    for _, row in gdf_acc_touch.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'acc'
        })
        uid += 1

    # c) Nodes
    for _, row in gdf_nodes.iterrows():
        sec_list = row['sectors']  # asume lista de strings
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': list(sec_list),  # ya es lista
            'type': 'node'
        })
        uid += 1

    if not puntos:
        return gpd.GeoDataFrame(columns=['sector_1','sector_2','geometry'], geometry='geometry')

    # 2) Preparar arrays para cálculos de distancia
    n = len(puntos)
    coords = np.array([p['coords'] for p in puntos])  # shape (n,2)
    # Distancia euclídea² entre cada par (i,j)
    diffs = coords[:, None, :] - coords[None, :, :]    # shape (n,n,2)
    dist2 = np.sum(diffs**2, axis=2)                   # shape (n,n)
    np.fill_diagonal(dist2, np.inf)

    # 3) Función auxiliar: para un índice i, devuelve lista de índices de candidatos j donde:
    #      - La lista de sectores de j contiene TODOS los sectores de i (para mid y acc)
    #      - Para node: la lista de sectores de j contiene el sector sc que estamos procesando
    #    y j != i.
    def candidatos_para(i, modo, sector_obj=None):
        """
        i: índice del punto en 'puntos'
        modo: 'pair' si queremos pares exactos (mid/acc),
              'single' si es búsqueda por un solo sector (para node)
        sector_obj: en modo 'single', el sector en cuestión (string)
        Retorna lista de índices j válidos.
        """
        lista_j = []
        if modo == 'pair':
            set_i = set(puntos[i]['sectors'])  # debe tener exactamente 2
            for j in range(n):
                if j == i:
                    continue
                if set_i.issubset(set(puntos[j]['sectors'])):
                    lista_j.append(j)
        else:  # modo == 'single'
            # sector_obj es un string. Buscamos todos j != i con sector_obj in puntos[j]['sectors']
            for j in range(n):
                if j == i:
                    continue
                if sector_obj in puntos[j]['sectors']:
                    lista_j.append(j)
        return lista_j

    # 4) Construir aristas sin duplicados
    added = set()  # almacenará frozenset({i,j}) para evitar duplicados
    records = []

    for idx in range(n):
        punto = puntos[idx]
        tipo = punto['type']

        if tipo == 'acc':
            # buscar candidatos que contengan el mismo par de sectores
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # elegir el más cercano según dist2
            j_min = int(np.argmin(dist2[idx, cands]))
            vecino = cands[j_min]
            key = frozenset({idx, vecino})
            if key not in added:
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        elif tipo == 'mid':
            # buscar candidatos que contengan el par exacto
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # necesitamos hasta dos vecinos más cercanos
            k = min(2, len(cands))
            idxs_k = np.argsort(dist2[idx, cands])[:k]
            for pos in idxs_k:
                vecino = cands[pos]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        else:  # tipo == 'node'
            # para cada sector en su lista, conectar al vecino más cercano que contenga ese sector
            for sector in punto['sectors']:
                cands = candidatos_para(idx, modo='single', sector_obj=sector)
                if not cands:
                    continue
                j_min = int(np.argmin(dist2[idx, cands]))
                vecino = cands[j_min]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                # La arista pertenece a la frontera entre el sector “sector”
                # y algún otro. Pero debemos decidir qué par guardamos en 'sector_1','sector_2'.
                # Tomamos el par formado por ese 'sector' y el otro sector compartido.
                # Encontrar intersección de listas de sectores:
                set_i = set(puntos[idx]['sectors'])
                set_j = set(puntos[vecino]['sectors'])
                comunes = set_i.intersection(set_j)
                # Debería contener al menos 'sector'. Ahora elegimos:
                #   - si hay exactamente 2 comunes, ese par es la frontera
                #   - si hay más de 2, tomamos arbitrariamente los dos primeros
                if len(comunes) >= 2:
                    s1, s2 = sorted(list(comunes))[:2]
                else:
                    # Si solo 'sector' está en común, buscamos en punto j un sector diferente:
                    otros_j = set_j - {sector}
                    if otros_j:
                        s1, s2 = sorted([sector, list(otros_j)[0]])
                    else:
                        # cae aquí si ambos puntos comparten exactamente ese sector;
                        # en ese caso, no hay "par" claro, we still put sector twice
                        s1, s2 = sector, sector
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

    # 5) Devolver GeoDataFrame final
    gdf_edges = gpd.GeoDataFrame(records, geometry='geometry')
    # Asignar CRS heredado (si alguno de los inputs lo tiene)
    for df_in in (gdf_midpoints, gdf_acc_touch, gdf_nodes):
        if df_in is not None and hasattr(df_in, 'crs') and df_in.crs is not None:
            gdf_edges.set_crs(df_in.crs, inplace=True)
            break

    return gdf_edges


# Asumiendo que ya tienes:
#   gdf_midpoints con ['sector_1','sector_2','geometry']
#   gdf_acc_touch con ['sector_1','sector_2','geometry']
#   gdf_nodes con ['sectors','geometry']  (donde 'sectors' es lista de strings)

gdf_edges = connect_all_points(
    gdf_midpoints=gdf_midpoints,
    gdf_nodes=gdf_nodes,
    gdf_acc_touch=gdf_acc_touch
)

# El GeoDataFrame resultante 'gdf_edges' tendrá columnas:
#   - 'sector_1', 'sector_2'  (indicando la frontera asociada a cada arista)
#   - 'geometry' (LineString entre los dos puntos unidos)
print(gdf_edges)



import geopandas as gpd

def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados



import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict


import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

def eliminar_aristas_prefer_node_mid(gdf_edges, gdf_midpoints, gdf_nodes, tol=1e-9):
    """
    Para cada punto medio con grado > 2, elimina primero las aristas que conectan
    con un nodo, y solo si faltan por eliminar, las que conectan con otro midpoint,
    hasta dejar grado = 2.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    gdf_nodes : GeoDataFrame
        Nodos (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges sin las aristas eliminadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints y nodos
    mid_keys  = { key(p) for p in gdf_midpoints.geometry }
    node_keys = { key(p) for p in gdf_nodes.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    to_drop = set()
    # Procesar cada midpoint con más de 2 conexiones
    for mid_k, edge_idxs in incidence.items():
        degree = len(edge_idxs)
        if degree <= 2:
            continue
        eliminar = degree - 2
        
        # Clasificar candidatos según destino node o midpoint
        node_conns = []
        mid_conns  = []
        for idx in edge_idxs:
            coords = list(edges.geometry.loc[idx].coords)
            e0 = (round(coords[0][0]/tol)*tol, round(coords[0][1]/tol)*tol)
            e1 = (round(coords[-1][0]/tol)*tol, round(coords[-1][1]/tol)*tol)
            other = e1 if e0 == mid_k else e0
            if other in node_keys:
                node_conns.append(idx)
            elif other in mid_keys:
                mid_conns.append(idx)
        
        # Primero eliminar conexiones a nodos
        for idx in node_conns[:eliminar]:
            to_drop.add(idx)
        faltan = eliminar - min(len(node_conns), eliminar)
        # # Si todavía faltan, eliminar conexiones a otros midpoints
        # for idx in mid_conns[:faltan]:
        #     to_drop.add(idx)
    
    # Eliminar y resetear índice
    if to_drop:
        edges = edges.drop(index=list(to_drop)).reset_index(drop=True)
    # Mantener CRS
    if hasattr(gdf_edges, 'crs'):
        edges.set_crs(gdf_edges.crs, inplace=True)
    return edges

# Ejemplo:
# gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
#     gdf_edges, gdf_midpoints, gdf_nodes
# )

gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
    gdf_edges,
    gdf_midpoints,
    gdf_nodes
)



gdf_edges = gdf_edges_filtrado


def conectar_midpoints_unicos(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada punto medio con solo una conexión, conecta este punto con los dos midpoints
    más cercanos en su frontera, si es posible.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints
    mid_keys = { key(p) for p in gdf_midpoints.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    # Nueva lista de aristas
    new_edges = []

    # 1) Identificar midpoints con una única conexión
    for mid_k, edge_idxs in incidence.items():
        if len(edge_idxs) == 1:
            # Identificar el sector de este midpoint
            midpoint = gdf_midpoints[gdf_midpoints.geometry.apply(key) == mid_k]
            if midpoint.empty:
                continue
            s1, s2 = midpoint.iloc[0]['sector_1'], midpoint.iloc[0]['sector_2']
            
            # Buscar otros midpoints con la misma frontera
            vecinos_potenciales = gdf_midpoints[(gdf_midpoints['sector_1'] == s1) & (gdf_midpoints['sector_2'] == s2)]
            
            # Eliminar el punto medio actual de la lista de vecinos
            vecinos_potenciales = vecinos_potenciales[vecinos_potenciales.geometry.apply(key) != mid_k]
            
            # Si hay más de uno, encontrar los dos más cercanos
            if len(vecinos_potenciales) > 1:
                coords_mid = midpoint.iloc[0].geometry.coords[0]
                vecinos_coords = [(v.geometry.coords[0], idx) for idx, v in vecinos_potenciales.iterrows()]
                
                # Calcular distancias
                # distancias = [(idx, np.sqrt((coords_mid[0] - v[0][0])**2 + (coords_mid[1] - v[0][1])**2)) for v, idx in vecinos_coords]
                distancias = [(idx,np.sqrt((coords_mid[0] - v[0])**2 +(coords_mid[1] - v[1])**2))for v, idx in vecinos_coords]

                # Ordenar por distancia
                distancias.sort(key=lambda x: x[1])
                
                # Seleccionar los dos más cercanos
                closest_idx = distancias[0][0]
                second_closest_idx = distancias[1][0]
                
                # Crear nuevas aristas entre el midpoint y los dos más cercanos
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[closest_idx].geometry])
                })
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[second_closest_idx].geometry])
                })
    
    # 2) Agregar las nuevas aristas al GeoDataFrame
    if new_edges:
        new_edges_gdf = gpd.GeoDataFrame(new_edges, geometry='geometry')
        edges = pd.concat([edges, new_edges_gdf], ignore_index=True)
    
    # Mantener CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        edges.set_crs(gdf_edges.crs, inplace=True)
    
    return edges


gdf_edges_nuevo = conectar_midpoints_unicos(gdf_edges, gdf_midpoints)
gdf_edges=gdf_edges_nuevo


def eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Elimina solo las aristas que conectan entre sí dos midpoints,
    cuando ambos tienen grado > 2.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas del grafo con geometrías LineString.
    gdf_midpoints : GeoDataFrame
        Midpoints con geometría Point.
    tol : float
        Tolerancia para comparación de coordenadas.

    Retorna
    -------
    GeoDataFrame
        gdf_edges sin las aristas que unen midpoints con grado > 2 entre sí.
    """
    from collections import defaultdict

    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # Crear set de claves de midpoints
    mid_keys = {key(p) for p in gdf_midpoints.geometry}

    # Calcular grado de cada midpoint
    degree_map = defaultdict(int)
    edge_extremos = {}

    for idx, line in gdf_edges.geometry.items():
        coords = list(line.coords)
        k1 = key(Point(*coords[0]))
        k2 = key(Point(*coords[-1]))
        edge_extremos[idx] = (k1, k2)

        if k1 in mid_keys:
            degree_map[k1] += 1
        if k2 in mid_keys:
            degree_map[k2] += 1

    # Identificar aristas entre midpoints de grado > 2
    to_drop = set()
    for idx, (k1, k2) in edge_extremos.items():
        if k1 in mid_keys and k2 in mid_keys:
            if degree_map[k1] > 2 and degree_map[k2] > 2:
                to_drop.add(idx)

    # Mostrar resumen
    print(f"Aristas eliminadas entre midpoints de grado > 2: {len(to_drop)}")

    # Eliminar aristas identificadas
    gdf_filtrado = gdf_edges.drop(index=list(to_drop)).reset_index(drop=True) if to_drop else gdf_edges.copy()

    # Conservar CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges = eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints)



def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from collections import defaultdict

def conectar_midpoints_aislados(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada frontera (sector_1, sector_2), identifica los midpoints
    que solo tienen una conexión (grado==1). Si en esa misma frontera
    hay exactamente dos midpoints de grado 1, añade una arista que los una.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas existentes con geometrías LINESTRING.
    gdf_midpoints : GeoDataFrame
        Midpoints con columnas ['sector_1','sector_2','geometry'].
    tol : float
        Tolerancia para comparar coordenadas (por defecto 1e-9).

    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    # 1) Función para obtener clave “redondeada” de un punto
    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # 2) Preparamos el conteo de grado para cada midpoint (clave->grado)
    #    y almacenamos la geometría de cada clave
    mid_keys = {}      # key -> Point geom
    for _, row in gdf_midpoints.iterrows():
        k = key(row.geometry)
        mid_keys[k] = row.geometry

    grado = defaultdict(int)
    # Recorremos cada arista y contamos sus endpoints si son midpoints
    for line in gdf_edges.geometry:
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        k0, k1 = key(type(line)(x0, y0)) if False else (None, None), (None, None)
        # mejor extraer directamente
        k0 = key(Point(x0, y0))
        k1 = key(Point(x1, y1))
        if k0 in mid_keys:
            grado[k0] += 1
        if k1 in mid_keys:
            grado[k1] += 1

    new_edges = []
    # 3) Agrupamos los midpoints por frontera
    for (s1, s2), group in gdf_midpoints.groupby(['sector_1', 'sector_2']):
        # Mapeamos clave->fila para este grupo
        key_to_row = { key(row.geometry): row for _, row in group.iterrows() }
        # Filtramos los que tengan grado == 1
        aislados = [k for k in key_to_row if grado.get(k, 0) == 1]
        # Si hay exactamente dos, los conectamos
        if len(aislados) == 2:
            geom1 = key_to_row[aislados[0]].geometry
            geom2 = key_to_row[aislados[1]].geometry
            new_edges.append({
                'sector_1': s1,
                'sector_2': s2,
                'geometry': LineString([geom1, geom2])
            })

    # 4) Si hay nuevas aristas, las concatenamos
    if new_edges:
        gdf_nuevas = gpd.GeoDataFrame(new_edges, geometry='geometry', crs=gdf_edges.crs)
        return pd.concat([gdf_edges, gdf_nuevas], ignore_index=True)
    else:
        return gdf_edges.copy()


gdf_edges = conectar_midpoints_aislados(gdf_edges, gdf_midpoints)


import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# 1) Reconstruir el contorno real del ACC (sin usar convex_hull)
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

if isinstance(union_poligonos, MultiPolygon):
    # Si la unión da MultiPolygon, tomamos cada polígono por separado
    lista_polys = list(union_poligonos.geoms)
else:
    # Si es un único Polygon
    lista_polys = [union_poligonos]

# 2) Extraer todas las coordenadas de los exteriores para ajustar límites
all_x = []
all_y = []
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    all_x.extend(x_poly)
    all_y.extend(y_poly)

min_lon = min(all_x) - 0.5
max_lon = max(all_x) + 0.5
min_lat = min(all_y) - 0.5
max_lat = max(all_y) + 0.5

# 3) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUD [º]')
ax.set_ylabel('LATITUD [º]')
ax.set_title('Sectorización Final')

# 4) Dibujar solo el contorno (exterior) de cada polígono de ACC, sin rellenar
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='_nolegend_')

# 5) Ahora superponemos cada LineString de gdf_edges en rojo
for linea in gdf_edges.geometry:
    xs, ys = linea.xy
    ax.plot(xs, ys, color='red', linewidth=1.2)

# Opcional: añadir una leyenda manual para gdf_edges
ax.plot([], [], color='red', linewidth=1.2, label='Fronteras nuevas')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')

plt.tight_layout()
plt.show()

# 1) Filtrar puntos que están dentro de la delimitación del ACC
gdf_midpoints_in = gdf_midpoints[gdf_midpoints.geometry.within(union_poligonos)]
gdf_acc_touch_in = gdf_acc_touch
# gdf_acc_touch_in = gdf_acc_touch[gdf_acc_touch.geometry.within(union_poligonos)]
gdf_nodes_in = gdf_nodes[gdf_nodes.geometry.within(union_poligonos)]

# 2) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUD [º]')
ax.set_ylabel('LATITUD [º]')
ax.set_title('Puntos Utilizados para Realizar la Representación')

# 3) Dibujar contorno ACC
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='Contorno ACC')

# 4) Dibujar puntos
gdf_midpoints_in.plot(ax=ax, color='blue', markersize=20, label='Puntos medios')
gdf_acc_touch_in.plot(ax=ax, color='green', markersize=30, marker='^', label='Puntos ACC')
gdf_nodes_in.plot(ax=ax, color='purple', markersize=40, marker='s', label='Nodos')

# 5) Leyenda y presentación
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

# Asumimos que tienes un GeoDataFrame 'df_cells' con las celdas y su asignación de sectores optimizada.

# Primero, obtenemos la geometría de las celdas y sus sectores
gdf_cells = gpd.GeoDataFrame(df_cells, geometry='Polygon')

# Agregamos la columna de los sectores optimizados
gdf_cells['Sector'] = df_cells['Optimized_Sector']


## A PARTIR DE AQUI SE PUEDE COMENTAR
# Creamos un grafo para las celdas y las fronteras compartidas
def get_shared_borders(cells):
    """
    Devuelve las aristas compartidas entre las celdas que tienen sectores diferentes.
    """
    shared_borders = []
    n_cells = len(cells)
    for i in range(n_cells):
        cell_i, poly_i, sector_i = cells[i]
        for j in range(i + 1, n_cells):
            cell_j, poly_j, sector_j = cells[j]
            # Si las celdas comparten un borde y están en sectores diferentes
            if poly_i.intersects(poly_j) and sector_i != sector_j:
                inter = poly_i.intersection(poly_j)
                if inter.geom_type == 'LineString' and inter.length > 0:
                    shared_borders.append(inter)
    return shared_borders

# Obtener las celdas y sus geometrías junto con los sectores
cells_info = [(row['Cell_Name'], row['Polygon'], row['Sector']) for idx, row in gdf_cells.iterrows()]

# Obtener las fronteras compartidas
shared_borders = get_shared_borders(cells_info)

# Crear el gráfico
fig, ax = plt.subplots(figsize=(12, 8))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Dibujar las celdas con los sectores optimizados
for _, row in gdf_cells.iterrows():
    poly = row['Polygon']
    sector = row['Sector']
    
    # Aquí se usa un mapa de colores para cada sector
    color_map = {
        'LECMSAN': 'green',
        'LECMASI': 'blue',
        'LECMBLI': 'purple',
        'LECMPAI': 'yellow',
        'LECMDGI': 'gray',
        'LECMR1I': 'cyan',  
        'LECMDPI': 'magenta',  
        'LECMSAB': 'orange',  
        'LECMSAI': 'red'  
    }
    color = color_map.get(sector, 'lightgray')
    
    # Añadir el sector a la lista de sectores visibles
    visible_sectors.add(sector)
    
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# Dibujar las fronteras compartidas con líneas oscuras
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Títulos y etiquetas
ax.set_title("Optimized Sectorization")
ax.set_xlabel("Longitude [º]")
ax.set_ylabel("Latitude [º]")
ax.set_aspect('equal')

# Crear la leyenda solo con los sectores visibles
legend_labels = {
    'LECMSAN': 'LECMSAN ',
    'LECMASI': 'LECMASI ',
    'LECMBLI': 'LECMBLI ',
    'LECMPAI': 'LECMPAI ',
    'LECMDGI': 'LECMDGI ',
    'LECMR1I': 'LECMR1I ',
    'LECMDPI': 'LECMDPI ',
    'LECMSAB': 'LECMSAB ',
    'LECMSAI': 'LECMSAI '
}

# Filtrar los sectores visibles en el gráfico
visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# Crear los patches para la leyenda solo de los sectores visibles
patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------- 
# 1. IDENTIFICAR LAS CELDAS DE LA FRONTERA ENTRE SECTORES EN LA SECTORIZACIÓN OPTIMIZADA
# ----------------------------------------------------------------------------- 

# Usamos la asignación optimizada para identificar las celdas fronterizas
optimized_assignment = df_cells.set_index('Cell_Name')['Optimized_Sector'].to_dict()

# Identificar las celdas fronterizas en la sectorización optimizada
border_cells_optimized = get_border_cells(optimized_assignment)
print(f"Celdas fronterizas optimizadas encontradas: {border_cells_optimized}")

# ----------------------------------------------------------------------------- 
# 2. FILTRAR LOS FLUJOS QUE PERTENECEN A CELDAS FRONTERIZAS DE LA SECTORIZACIÓN OPTIMIZADA
# ----------------------------------------------------------------------------- 

# Filtrar los flujos cuyos nombres de celda están en la lista de celdas fronterizas optimizadas
border_flujos_optimized = gdf_flujos[gdf_flujos['Cell_Name'].isin(border_cells_optimized)]

# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR EL MALLADO Y LOS FLUJOS DE CELDAS FRONTERIZAS DE LA SECTORIZACIÓN OPTIMIZADA
# ----------------------------------------------------------------------------- 
fig, ax_cells = plt.subplots()

# Dibujar el polígono del ACC
x_acc, y_acc = poligono_ACC.exterior.xy
ax_cells.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')

# Dibujar las celdas del mallado (sectorización optimizada)
for _, row in DF_MALLADO.iterrows():
    polygon = row['Polygon']  # Obtener el polígono de la celda
    x, y = polygon.exterior.xy  # Coordenadas del contorno
    sec = optimized_assignment.get(row['Cell_Name'], 'sin_sector')  # Asignación optimizada
    # Usamos un color distinto para las celdas fronterizas
    if row['Cell_Name'] in border_cells_optimized:
        ax_cells.fill(x, y, color='orange', alpha=0.7)  # Color para las celdas frontera
    else:
        ax_cells.fill(x, y, color='gray', alpha=0.5)

# Dibujar los flujos (en rojo) correspondientes a las celdas fronterizas optimizadas
primer_flujo = True
for _, row in border_flujos_optimized.iterrows():
    linea = row['geometry']
    if linea is not None:
        x_line, y_line = linea.xy
        if primer_flujo:
            ax_cells.plot(x_line, y_line, color='red', linewidth=1.5, label='Flujos en celdas fronterizas optimizadas')
            primer_flujo = False
        else:
            ax_cells.plot(x_line, y_line, color='red', linewidth=1.5)

# Configurar la gráfica
ax_cells.set_xlim(min_lon, max_lon)
ax_cells.set_ylim(min_lat, max_lat)
ax_cells.set_aspect('equal')
ax_cells.set_title("MALLADO DEL ESPACIO AÉREO OPTIMIZADO CON FLUJOS EN CELDAS FRONTERIZAS")
ax_cells.set_xlabel('LONGITUD [º]')
ax_cells.set_ylabel('LATITUD [º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# Guardar la figura
nombre_figura = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo con flujos en celdas fronterizas optimizadas.png'
plt.savefig(nombre_figura, format='png', dpi=300, bbox_inches='tight')
plt.show()



import math

# ----------------------------------------------------------------------------- 
# 1. FUNCIONES PARA CALCULAR LA DISTANCIA ENTRE LAS COORDENADAS DE ENTRADA Y SALIDA
# ----------------------------------------------------------------------------- 
def calculate_distance(lat_in, lon_in, lat_out, lon_out):
    """
    Calcula la distancia entre dos puntos geográficos en millas náuticas.
    Utiliza la fórmula de distancia euclidiana simplificada (dada la pequeña escala de la distancia en grados).
    """
    delta_lat = lat_in - lat_out
    delta_lon = lon_in - lon_out
    distance = math.sqrt(delta_lat**2 + delta_lon**2) * 60  # La constante 60 convierte grados a millas náuticas
    return distance

# ----------------------------------------------------------------------------- 
# 2. FILTRAR LOS FLUJOS QUE PERTENECEN A CELDAS FRONTERIZAS Y CUMPLEN LA CONDICIÓN DE DISTANCIA
# ----------------------------------------------------------------------------- 
# Calcular la distancia para cada flujo y filtrar aquellos menores a 10 millas náuticas
max_distance_nautical = 20  # 20 millas náuticas

# Lista para almacenar los flujos filtrados
filtered_flujos = []

for _, row in border_flujos_optimized.iterrows():
    lat_in = row['lat_cell_in']
    lon_in = row['lon_cell_in']
    lat_out = row['lat_cell_out']
    lon_out = row['lon_cell_out']
    
    # Calcular la distancia del flujo
    distance = calculate_distance(lat_in, lon_in, lat_out, lon_out)
    
    # Si la distancia es menor a 10 millas náuticas, agregarlo a la lista
    if distance < max_distance_nautical:
        filtered_flujos.append(row)

# Crear un GeoDataFrame con los flujos filtrados
gdf_filtered_flujos = gpd.GeoDataFrame(filtered_flujos, geometry='geometry')

# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR EL MALLADO Y LOS FLUJOS FILTRADOS
# ----------------------------------------------------------------------------- 
fig, ax_cells = plt.subplots()

# Dibujar el polígono del ACC
x_acc, y_acc = poligono_ACC.exterior.xy
ax_cells.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')

# Dibujar las celdas del mallado (sectorización optimizada)
for _, row in DF_MALLADO.iterrows():
    polygon = row['Polygon']  # Obtener el polígono de la celda
    x, y = polygon.exterior.xy  # Coordenadas del contorno
    sec = optimized_assignment.get(row['Cell_Name'], 'sin_sector')  # Asignación optimizada
    # Usamos un color distinto para las celdas frontera
    if row['Cell_Name'] in border_cells_optimized:
        ax_cells.fill(x, y, color='orange', alpha=0.7)  # Color para las celdas frontera
    else:
        ax_cells.fill(x, y, color='gray', alpha=0.5)

# Dibujar los flujos filtrados (en rojo) correspondientes a las celdas fronterizas optimizadas
for _, row in gdf_filtered_flujos.iterrows():
    linea = row['geometry']
    if linea is not None:
        x_line, y_line = linea.xy
        ax_cells.plot(x_line, y_line, color='red', linewidth=1.5)

# Configurar la gráfica
ax_cells.set_xlim(min_lon, max_lon)
ax_cells.set_ylim(min_lat, max_lat)
ax_cells.set_aspect('equal')
ax_cells.set_title("MALLADO DEL ESPACIO AÉREO OPTIMIZADO CON FLUJOS MENORES A 20 MILLAS NÁUTICAS EN CELDAS FRONTERIZAS")
ax_cells.set_xlabel('LONGITUD [º]')
ax_cells.set_ylabel('LATITUD [º]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

# Guardar la figura
nombre_figura = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo con flujos menores a 10 millas náuticas.png'
plt.savefig(nombre_figura, format='png', dpi=300, bbox_inches='tight')
plt.show()




# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR EL MALLADO Y LOS FLUJOS FILTRADOS SOBRE LA SECTORIZACIÓN
# ----------------------------------------------------------------------------- 
fig, ax = plt.subplots(figsize=(12, 8))

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Crear un mapa de colores para los sectores
color_map = {
    'LECMSAN': 'green',
    'LECMASI': 'blue',
    'LECMBLI': 'purple',
    'LECMPAI': 'yellow',
    'LECMDGI': 'gray',
    'LECMR1I': 'cyan',  
    'LECMDPI': 'magenta',  
    'LECMSAB': 'orange',  
    'LECMSAI': 'red'  
}

# Dibujar las celdas con los sectores optimizados
for _, row in gdf_cells.iterrows():
    poly = row['Polygon']
    sector = row['Sector']
    
    color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
    # Añadir el sector a la lista de sectores visibles
    visible_sectors.add(sector)
    
    # Dibujar la celda
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# Dibujar las fronteras compartidas con líneas oscuras
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Dibujar los flujos filtrados (en rojo) correspondientes a las celdas fronterizas optimizadas
for _, row in gdf_filtered_flujos.iterrows():
    linea = row['geometry']
    if linea is not None:
        x_line, y_line = linea.xy
        ax.plot(x_line, y_line, color='red', linewidth=1.5, label='Flujos < 20 millas')

# Títulos y etiquetas
ax.set_title("Sectorización Optimizada con Flujos Menores a 20 Millas Náuticas")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')

# Crear la leyenda solo con los sectores visibles
legend_labels = {
    'LECMSAN': 'LECMSAN ',
    'LECMASI': 'LECMASI ',
    'LECMBLI': 'LECMBLI ',
    'LECMPAI': 'LECMPAI ',
    'LECMDGI': 'LECMDGI ',
    'LECMR1I': 'LECMR1I ',
    'LECMDPI': 'LECMDPI ',
    'LECMSAB': 'LECMSAB ',
    'LECMSAI': 'LECMSAI '
}

# Filtrar los sectores visibles en el gráfico
visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# Crear los patches para la leyenda solo de los sectores visibles
patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# Añadir leyenda para los flujos
patches.append(mpatches.Patch(color='red', label='Flujos < 20 millas náuticas'))

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------- 
# 1. FUNCIONES PARA VERIFICAR LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# ----------------------------------------------------------------------------- 

def count_intersections(flow, shared_borders):
    """
    Cuenta cuántas veces un flujo corta las fronteras compartidas entre sectores.
    """
    intersection_count = 0
    for border in shared_borders:
        if flow.intersects(border):  # Si el flujo corta la frontera
            intersection_count += 1
    return intersection_count

# ----------------------------------------------------------------------------- 
# 2. FILTRAR LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# ----------------------------------------------------------------------------- 
filtered_flows = []

for _, row in gdf_filtered_flujos.iterrows():
    flow = row['geometry']
    
    # Verificar cuántas veces el flujo corta las fronteras
    intersections = count_intersections(flow, shared_borders)
    
    # Si corta 2 veces las fronteras, lo mantenemos
    if intersections >= 2:
        filtered_flows.append(row)

# Crear un GeoDataFrame con los flujos filtrados
gdf_filtered_flows_2_cross = gpd.GeoDataFrame(filtered_flows, geometry='geometry')

# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR EL MALLADO Y LOS FLUJOS QUE CORTAN 2 VECES LAS FRONTERAS
# ----------------------------------------------------------------------------- 
fig, ax = plt.subplots(figsize=(12, 8))

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Dibujar las celdas con los sectores optimizados
for _, row in gdf_cells.iterrows():
    poly = row['Polygon']
    sector = row['Sector']
    
    color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
    # Añadir el sector a la lista de sectores visibles
    visible_sectors.add(sector)
    
    # Dibujar la celda
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# Dibujar las fronteras compartidas con líneas oscuras
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Dibujar los flujos que cortan dos veces las fronteras (en rojo)
for _, row in gdf_filtered_flows_2_cross.iterrows():
    flow = row['geometry']
    if flow is not None:
        x_flow, y_flow = flow.xy
        ax.plot(x_flow, y_flow, color='red', linewidth=1.5, label='Flujos que cortan 2 veces las fronteras')

# Títulos y etiquetas
ax.set_title("Sectorización Optimizada con Flujos que Atraviesan 2 Veces las Fronteras")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')

# Crear la leyenda solo con los sectores visibles
visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# Crear los patches para la leyenda solo de los sectores visibles
patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# Añadir leyenda para los flujos
patches.append(mpatches.Patch(color='red', label='Flujos que cortan 2 veces las fronteras'))

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------------- 
# 2. FILTRAR LOS FLUJOS QUE INTERSECTAN LAS FRONTERAS Y SELECCIONAR EL DE MAYOR DISTANCIA
# ----------------------------------------------------------------------------- 

# Diccionario para almacenar el flujo con la mayor distancia por cada celda
max_distance_flows = {}

# Recorrer todos los flujos filtrados
for _, row in gdf_filtered_flujos.iterrows():
    lat_in = row['lat_cell_in']
    lon_in = row['lon_cell_in']
    lat_out = row['lat_cell_out']
    lon_out = row['lon_cell_out']
    flow = row['geometry']
    
    # Verificar cuántas veces el flujo corta las fronteras
    intersections = count_intersections(flow, shared_borders)
    
    # Si corta las fronteras dos veces (o más)
    if intersections >= 2:
        # Calcular la distancia del flujo
        distance = calculate_distance(lat_in, lon_in, lat_out, lon_out)
        
        # Obtener el nombre de la celda (si existe)
        cell_name = row['Cell_Name']
        
        # Verificar si ya tenemos un flujo con mayor distancia para esa celda
        if cell_name not in max_distance_flows or distance > max_distance_flows[cell_name]['distance']:
            # Si es el mayor flujo, lo almacenamos
            max_distance_flows[cell_name] = {'flow': flow, 'distance': distance}

# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR EL MALLADO Y LOS FLUJOS CON MAYOR DISTANCIA POR CELDA
# ----------------------------------------------------------------------------- 
fig, ax = plt.subplots(figsize=(12, 8))

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Crear un mapa de colores para los sectores
color_map = {
    'LECMSAN': 'green',
    'LECMASI': 'blue',
    'LECMBLI': 'purple',
    'LECMPAI': 'yellow',
    'LECMDGI': 'gray',
    'LECMR1I': 'cyan',  
    'LECMDPI': 'magenta',  
    'LECMSAB': 'orange',  
    'LECMSAI': 'red'  
}

# Dibujar las celdas con los sectores optimizados
for _, row in gdf_cells.iterrows():
    poly = row['Polygon']
    sector = row['Sector']
    
    color = color_map.get(sector, 'lightgray')  # Asignar color según sector
    
    # Añadir el sector a la lista de sectores visibles
    visible_sectors.add(sector)
    
    # Dibujar la celda
    x, y = poly.exterior.xy
    ax.fill(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

# Dibujar las fronteras compartidas con líneas oscuras
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Dibujar solo los flujos con mayor distancia por celda que intersecten dos veces las fronteras
for cell_name, flow_data in max_distance_flows.items():
    flow = flow_data['flow']
    if flow is not None:
        x_flow, y_flow = flow.xy
        ax.plot(x_flow, y_flow, color='red', linewidth=1.5, label=f'Flujo mayor distancia {cell_name}')

# Títulos y etiquetas
ax.set_title("Sectorización Optimizada con Flujo con Mayor Distancia por Celda")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')

# Crear la leyenda solo con los sectores visibles
visible_legend_labels = {sector: label for sector, label in legend_labels.items() if sector in visible_sectors}

# Crear los patches para la leyenda solo de los sectores visibles
patches = [mpatches.Patch(color=color_map[sector], label=label) for sector, label in visible_legend_labels.items()]

# Añadir leyenda para los flujos
patches.append(mpatches.Patch(color='red', label='Flujos con mayor distancia'))

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------------- 
# 3. REPRESENTAR SOLO LAS FRONTERAS DE LOS SECTORES, LOS FLUJOS DE MAYOR DISTANCIA Y EL ACC
# ----------------------------------------------------------------------------- 
fig, ax = plt.subplots(figsize=(12, 8))

# Dibujar el polígono del ACC
x_acc, y_acc = poligono_ACC.exterior.xy  # Asegúrate de que `poligono_ACC` esté definido
ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN')  # Delimitación del ACC

# Crear una lista para los sectores que realmente aparecen en el gráfico
visible_sectors = set()

# Dibujar las fronteras compartidas entre los sectores
for border in shared_borders:
    x, y = border.xy
    ax.plot(x, y, color='black', linewidth=2)

# Dibujar solo los flujos con mayor distancia por celda que intersectan las fronteras
for cell_name, flow_data in max_distance_flows.items():
    flow = flow_data['flow']
    if flow is not None:
        # Dibujar el flujo de mayor distancia en rojo
        x_flow, y_flow = flow.xy
        ax.plot(x_flow, y_flow, color='red', linewidth=2, label=f'Flujo mayor distancia {cell_name}')

# Títulos y etiquetas
ax.set_title("Fronteras de Sectores con Flujos de Mayor Distancia Intersecando las Fronteras")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')

# Crear la leyenda con flujos intersecando fronteras
patches = [mpatches.Patch(color='red', label='Flujos de mayor distancia')]
patches.append(mpatches.Patch(color='black', label='LECMCTAN (ACC)'))

# Mostrar la leyenda
ax.legend(handles=patches, loc='upper right')

# Ajuste y visualización final
plt.tight_layout()
plt.show()

from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon

import pandas as pd
from shapely.geometry import Point, MultiPoint

from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import snap
import math
import pandas as pd


import math
import pandas as pd
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import snap
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint, GeometryCollection
from shapely.geometry.base import BaseGeometry



def compute_flow_border_intersections(flows, border_lines):
    """
    Calcula los puntos de intersección entre flujos y aristas de frontera.
    Siempre devuelve un DataFrame con las columnas esperadas, incluso si está vacío.
    """
    records = []
    for flow_id, flow_line in flows:
        if not isinstance(flow_line, BaseGeometry):
            continue
        for border_id, border_line in border_lines:
            if not isinstance(border_line, BaseGeometry):
                continue
            try:
                inter = flow_line.intersection(border_line)
            except Exception:
                continue
            if inter.is_empty:
                continue

            pts = []
            if isinstance(inter, Point):
                pts = [inter]
            elif isinstance(inter, MultiPoint):
                pts = list(inter)
            elif isinstance(inter, GeometryCollection):
                pts = [g for g in inter.geoms if isinstance(g, Point)]
            else:
                for geom in getattr(inter, 'geoms', [inter]):
                    if hasattr(geom, 'coords'):
                        pts.extend(Point(c) for c in geom.coords)

            for pt in pts:
                records.append({
                    'flow_id': flow_id,
                    'border_id': border_id,
                    'latitude': round(pt.y, 2),
                    'longitude': round(pt.x, 2)
                })

    # Aseguramos siempre estas columnas
    return pd.DataFrame(records, columns=['flow_id','border_id','latitude','longitude'])

# Ejemplo de uso:
# --- 1) Calcula las intersecciones, pasando enumerate(shared_borders) ---
flows_max = [(cell, data['flow']) for cell, data in max_distance_flows.items()]
intersection_df = compute_flow_border_intersections(flows_max,
                                                    list(enumerate(shared_borders)))


from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def draw_rectangle_between_points(ax, p1, p2, tolerance=0,
                                  edgecolor='blue', linewidth=2, linestyle='--'):
    """
    Dibuja en `ax` un rectángulo definido por dos puntos p1 y p2 (shapely Points),
    usando también los dos puntos formados por (latitud de uno, longitud del otro)
    y añade una pequeña tolerancia para expandir el rectángulo.

    Parámetros:
    -----------
    ax : matplotlib.axes.Axes
        Eje donde se dibujará el rectángulo.
    p1, p2 : shapely.geometry.Point
        Puntos de intersección del flujo de máxima distancia.
    tolerance : float, opcional
        Margen (en grados) para expandir el rectángulo en todas direcciones.
    edgecolor : str, opcional
        Color del borde del rectángulo.
    linewidth : float, opcional
        Grosor de la línea del rectángulo.
    linestyle : str, opcional
        Estilo de línea (p.ej. '-', '--', '-.', ':').
    
    Devuelve:
    ---------
    polygon : shapely.geometry.Polygon
        Objeto Polygon con las coordenadas del rectángulo.
    """
    # Extraer longitudes (x) y latitudes (y)
    lon1, lat1 = p1.x, p1.y
    lon2, lat2 = p2.x, p2.y

    # Calcular mínimos y máximos + tolerancia
    min_lon = min(lon1, lon2) - tolerance
    max_lon = max(lon1, lon2) + tolerance
    min_lat = min(lat1, lat2) - tolerance
    max_lat = max(lat1, lat2) + tolerance

    # Definir las 5 esquinas (cerrando el polígono)
    corners = [
        (min_lon, min_lat),  # (lon1, lat1) ajustado a min-min
        (min_lon, max_lat),  # (lon1, lat2)
        (max_lon, max_lat),  # (lon2, lat2)
        (max_lon, min_lat),  # (lon2, lat1)
        (min_lon, min_lat)   # cerrar
    ]

    # Extraer coordenadas y dibujar
    xs, ys = zip(*corners)
    ax.plot(xs, ys, color=edgecolor, linewidth=linewidth, linestyle=linestyle)

    # Devolver el polígono por si quieres reutilizarlo
    return Polygon(corners)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, MultiLineString

# --- 1) Prepara el lienzo ---
fig, ax = plt.subplots(figsize=(12, 8))

# Dibuja el polígono del ACC
x_acc, y_acc = poligono_ACC.exterior.xy
ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# --- 2) Calcula intersecciones (ya lo tienes) ---
# flows_max = [(cell, data['flow']) for cell, data in max_distance_flows.items()]
# intersection_df = compute_flow_border_intersections(flows_max, list(enumerate(shared_borders)))

# --- 3) Genera cada rectángulo a partir de p1, p2 extraídos ---
rectangles = []
for flow_id in intersection_df['flow_id'].unique():
    df_flow = intersection_df[intersection_df['flow_id'] == flow_id]
    if len(df_flow) >= 2:
        # Coge los dos primeros puntos de la tabla
        lon1, lat1 = df_flow.iloc[0][['longitude','latitude']]
        lon2, lat2 = df_flow.iloc[1][['longitude','latitude']]
        p1 = Point(lon1, lat1)
        p2 = Point(lon2, lat2)
        # Dibuja el rectángulo (añade al ax) y guarda el polígono
        rect = draw_rectangle_between_points(
            ax, p1, p2,
            tolerance=0.01,
            edgecolor='blue', linewidth=2, linestyle='--'
        )
        rectangles.append(rect)

# --- 4) Une todos los rectángulos en un solo geometry ---
area_to_remove = unary_union(rectangles)

# --- 5) Dibuja las fronteras recortadas ---
for border in shared_borders:
    clipped = border.difference(area_to_remove)
    if clipped.is_empty:
        continue

    # Puede ser LineString o MultiLineString
    if isinstance(clipped, LineString):
        segments = [clipped]
    elif isinstance(clipped, MultiLineString):
        segments = list(clipped.geoms)
    else:
        continue

    for seg in segments:
        x, y = seg.xy
        ax.plot(x, y, color='black', linewidth=2)

# --- 6) Dibuja los flujos de mayor distancia en rojo ---
for cell_name, data in max_distance_flows.items():
    flow = data.get('flow')
    if flow:
        x_flow, y_flow = flow.xy
        ax.plot(x_flow, y_flow, color='red', linewidth=2)

# --- 7) Leyenda, títulos y ajuste final ---
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
    Line2D([0],[0], color='black', lw=2, label='Frontera recortada'),
    Line2D([0],[0], color='red',   lw=2, label='Flujos mayor distancia'),
    Line2D([0],[0], color='blue',  lw=2, linestyle='--', label='Rectángulos de intersección')
]
ax.legend(handles=legend_handles, loc='upper right')
ax.set_title("Fronteras de Sectores recortadas por Rectángulos de Intersección")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
import math

# — asume poligono_ACC, shared_borders, max_distance_flows,
#   intersection_df, draw_rectangle_between_points() ya definidos —

fig, ax = plt.subplots(figsize=(12, 8))
# 1) dibujamos el ACC
x_acc, y_acc = poligono_ACC.exterior.xy
ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# 2) creamos los rectángulos y los unimos
rects = []
for fid in intersection_df['flow_id'].unique():
    df = intersection_df[intersection_df['flow_id'] == fid]
    if len(df) >= 2:
        lon1, lat1 = df.iloc[0][['longitude','latitude']]
        lon2, lat2 = df.iloc[1][['longitude','latitude']]
        p1, p2 = Point(lon1, lat1), Point(lon2, lat2)
        rect = draw_rectangle_between_points(
            ax, p1, p2, tolerance=0.01,
            edgecolor='blue', linewidth=2, linestyle='--'
        )
        rects.append(rect)
area_to_remove = unary_union(rects)

# 3) extraemos todos los fragmentos recortados
clipped_borders = []
for b in shared_borders:
    c = b.difference(area_to_remove)
    if c.is_empty:
        continue
    if isinstance(c, LineString):
        clipped_borders.append(c)
    else:
        clipped_borders.extend(c.geoms)

# 4) preparamos la lista de flujos y la frontera del ACC como LineString
flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow') is not None]
acc_boundary = LineString(poligono_ACC.exterior.coords)

# 5) función que dispara el rayo y dibuja hasta target
def extend_to(pt_from, pt_dir, target, ax, max_dist=100):
    dx, dy = pt_from.x - pt_dir.x, pt_from.y - pt_dir.y
    L = math.hypot(dx, dy)
    if L == 0: return
    ux, uy = dx/L, dy/L
    far = Point(pt_from.x + ux*max_dist, pt_from.y + uy*max_dist)
    ray = LineString([pt_from, far])
    inter = ray.intersection(target)
    if inter.is_empty: return

    pts = []
    def collect(g):
        if isinstance(g, Point):
            pts.append(g)
        elif isinstance(g, LineString):
            pts.extend(Point(c) for c in g.coords)
        elif hasattr(g, 'geoms'):
            for gg in g.geoms:
                collect(gg)
    collect(inter)
    if not pts: return
    nearest = min(pts, key=lambda p: p.distance(pt_from))
    ax.plot([pt_from.x, nearest.x], [pt_from.y, nearest.y],
            color='black', linewidth=2)

# 6) dibujamos y extendemos cada fragmento
for i, seg in enumerate(clipped_borders):
    # trazamos el propio trozo
    x, y = seg.xy
    ax.plot(x, y, color='black', linewidth=2)

    coords = list(seg.coords)
    if len(coords) < 2:
        continue

    # target = todos los demás trozos  + flujos + ACC
    others = clipped_borders[:i] + clipped_borders[i+1:]
    target = unary_union(others + flows_lines + [acc_boundary])

    p0, p1 = Point(coords[0]), Point(coords[1])
    pe, pp = Point(coords[-1]), Point(coords[-2])
    tol = 1e-6

    if target.distance(p0) > tol:
        extend_to(p0, p1, target, ax)
    if target.distance(pe) > tol:
        extend_to(pe, pp, target, ax)

# 7) sobreponemos los flujos
for d in max_distance_flows.values():
    f = d.get('flow')
    if f:
        xf, yf = f.xy
        ax.plot(xf, yf, color='red', linewidth=2)

# leyenda y ajustes finales
handles = [
    Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
    Line2D([0],[0], color='black', lw=2, label='Fronteras recortadas/extendidas'),
    Line2D([0],[0], color='red',   lw=2, label='Flujos máx. distancia')
]
ax.legend(handles=handles, loc='upper right')
ax.set_title("Fronteras recortadas y Prolongadas")
ax.set_xlabel("Longitud [º]"); ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal'); plt.tight_layout(); plt.show()


import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, MultiLineString
import math

import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, MultiLineString
import math

# --- Prepara figura y dibuja ACC como antes ---
fig, ax = plt.subplots(figsize=(12,8))
x_acc, y_acc = poligono_ACC.exterior.xy
ax.plot(x_acc, y_acc, color='black', linewidth=1, label='LECMCTAN (ACC)')

# --- 1) Reúne fronteras recortadas y flujos ---
# clipped_borders: ya los tienes de tu paso 3-4
flows_lines = [d['flow'] for d in max_distance_flows.values() if d.get('flow')]

# --- 2) Define una única función de extensión ---
def compute_extension(pt_from, pt_dir, target, max_dist=200):
    dx, dy = pt_from.x - pt_dir.x, pt_from.y - pt_dir.y
    L = math.hypot(dx, dy)
    if L == 0:
        return None
    ux, uy = dx/L, dy/L
    far = Point(pt_from.x + ux*max_dist, pt_from.y + uy*max_dist)
    ray = LineString([pt_from, far])
    inter = ray.intersection(target)
    if inter.is_empty:
        return None

    # recogen todos los posibles puntos de choque
    pts = []
    if isinstance(inter, Point):
        pts = [inter]
    else:
        for g in getattr(inter, 'geoms', []):
            if isinstance(g, Point):
                pts.append(g)
            elif isinstance(g, LineString):
                pts.extend(Point(c) for c in g.coords)
    if not pts:
        return None

    # quedamos con el más cercano
    nearest = min(pts, key=lambda p: p.distance(pt_from))
    return LineString([(pt_from.x, pt_from.y), (nearest.x, nearest.y)])

# --- 3) Calcula todas las extensiones ---
extension_lines = []
tol = 1e-6

for i, seg in enumerate(clipped_borders):
    coords = list(seg.coords)
    if len(coords) < 2:
        continue

    # target: todo excepto este segmento
    others = clipped_borders[:i] + clipped_borders[i+1:]
    target = unary_union(others + flows_lines + [acc_boundary])

    p0, p1 = Point(coords[0]), Point(coords[1])
    pe, pp = Point(coords[-1]), Point(coords[-2])

    # extiende desde cada extremo si está separado
    if target.distance(p0) > tol:
        ext0 = compute_extension(p0, p1, target)
        if ext0 is not None:
            extension_lines.append(ext0)

    if target.distance(pe) > tol:
        ext1 = compute_extension(pe, pp, target)
        if ext1 is not None:
            extension_lines.append(ext1)

# --- 4) Une todo y dibuja de una vez ---
all_pieces = clipped_borders + extension_lines + flows_lines
combined  = unary_union(all_pieces)

# extrae las LineString para plotear
to_plot = []
if isinstance(combined, LineString):
    to_plot = [combined]
elif isinstance(combined, MultiLineString):
    to_plot = list(combined.geoms)
else:
    for g in getattr(combined, 'geoms', []):
        if isinstance(g, LineString):
            to_plot.append(g)

for line in to_plot:
    x, y = line.xy
    ax.plot(x, y, color='black', linewidth=2)

# --- 5) Leyenda y acabado ---
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color='black', lw=1, label='LECMCTAN (ACC)'),
    Line2D([0],[0], color='black', lw=2, label='Fronteras Finales'),
]
ax.legend(handles=handles, loc='upper right')

ax.set_title("Fronteras Finales")
ax.set_xlabel("Longitud [º]")
ax.set_ylabel("Latitud [º]")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()


# ESTO ES PARA UNIR LOS PUNTOS MEDIOS DE LOS TRAMOS



import geopandas as gpd
from shapely.geometry import LineString

import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon, MultiPoint, GeometryCollection, MultiLineString
from shapely.ops import unary_union
from itertools import combinations
from collections import defaultdict
from numpy.linalg import eigh

# ------------------------------------------------------------------------------
def get_shared_borders_and_nodes(cells, crs, tol=1e-9):
    """
    Devuelve:
      • gdf_borders: GeoDataFrame con las líneas compartidas (sector_1, sector_2, geometry)
      • gdf_nodes: GeoDataFrame con los nodos (puntos) donde confluyen ≥3 sectores,
                   con columnas ['sectors','geometry'].
    """
    # 1) extraer fronteras pareadas
    records_b = []
    n = len(cells)
    for i in range(n):
        _, poly_i, sec_i = cells[i]
        for j in range(i+1, n):
            _, poly_j, sec_j = cells[j]
            if sec_i == sec_j or not poly_i.intersects(poly_j):
                continue
            inter = poly_i.intersection(poly_j)
            if isinstance(inter, LineString) and inter.length > 0:
                records_b.append({
                    'sector_1': sec_i,
                    'sector_2': sec_j,
                    'geometry': inter
                })
    gdf_borders = gpd.GeoDataFrame(records_b, crs=crs)

    # 2) agrupar extremos en nodos multígrado
    node_sectors = defaultdict(set)
    for _, row in gdf_borders.iterrows():
        s1, s2 = row.sector_1, row.sector_2
        for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
            key = (round(x/tol)*tol, round(y/tol)*tol)
            node_sectors[key].update([s1, s2])

    records_n = []
    for (x, y), secs in node_sectors.items():
        if len(secs) >= 3:
            records_n.append({
                'sectors': sorted(secs),
                'geometry': Point(x, y)
            })
    gdf_nodes = gpd.GeoDataFrame(records_n, crs=crs)
    return gdf_borders, gdf_nodes

# ------------------------------------------------------------------------------
def extract_corner_points(gdf_borders, tol=1e-9):
    """
    Extrae los cruces horizontal-vertical de gdf_borders.
    Devuelve gdf_corners con columnas:
      ['h_sector_1','h_sector_2','v_sector_1','v_sector_2','geometry'].
    """
    is_h = gdf_borders.geometry.apply(lambda g: abs(g.coords[0][1] - g.coords[-1][1]) < tol)
    is_v = gdf_borders.geometry.apply(lambda g: abs(g.coords[0][0] - g.coords[-1][0]) < tol)
    gdf_h = gdf_borders[is_h]
    gdf_v = gdf_borders[is_v]

    records = []
    for _, h in gdf_h.iterrows():
        for _, v in gdf_v.iterrows():
            pt = h.geometry.intersection(v.geometry)
            if not pt.is_empty and isinstance(pt, Point):
                records.append({
                    'h_sector_1': h.sector_1,
                    'h_sector_2': h.sector_2,
                    'v_sector_1': v.sector_1,
                    'v_sector_2': v.sector_2,
                    'geometry': pt
                })
    gdf_corners = gpd.GeoDataFrame(records, crs=gdf_borders.crs)
    return gdf_corners.drop_duplicates(subset='geometry').reset_index(drop=True)


# 0) Prepara cells_info
cells_info = [
    (row['Cell_Name'], row['Polygon'], row['Sector'])
    for _, row in gdf_cells.iterrows()
]

# 1) get_shared_borders_and_nodes
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) polígono ACC y su contorno
polygons = DF_info_conf['Contorno Sector'].tolist()
union_poly = unary_union(polygons)
if union_poly.geom_type == 'MultiPolygon':
    poligono_ACC = union_poly.convex_hull
else:
    poligono_ACC = Polygon(union_poly.exterior)
acc_boundary = poligono_ACC.boundary

# 3) puntos de contacto ACC
records = []
for _, row in gdf_borders.iterrows():
    s1, s2 = row.sector_1, row.sector_2
    for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
        pt = Point(x, y)
        if pt.distance(acc_boundary) < 1e-9:
            records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
                     .drop_duplicates('geometry') \
                     .reset_index(drop=True)

# 4) cruces H-V
gdf_corners = extract_corner_points(gdf_borders)


from shapely.geometry import LineString
from shapely.ops import unary_union
import geopandas as gpd

def extract_constituent_const_segments(gdf_borders, tol=1e-9):
    """
    Para cada frontera (misma combinación sector_1–sector_2) de gdf_borders,
    descompone su LineString en segmentos horizontales y verticales,
    agrupa en cadenas consecutivas (que se tocan), y une cada cadena
    en un LineString (o MultiLineString).

    Devuelve un GeoDataFrame con columnas:
      - sector_1, sector_2
      - orientation: 'h' ó 'v'
      - geometry: LineString (o MultiLineString) resultante
    """
    records = []

    # Asegurarnos de tener un identificador por fila
    gdf = gdf_borders.reset_index().rename(columns={'index':'__orig_idx'}).copy()

    for _, row in gdf.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        coords = list(row.geometry.coords)
        # 1) Descomponer en segmentos orientados
        segs = []
        for i in range(len(coords)-1):
            x1,y1 = coords[i]
            x2,y2 = coords[i+1]
            if abs(y1-y2) < tol:
                orient = 'h'
            elif abs(x1-x2) < tol:
                orient = 'v'
            else:
                continue
            seg = LineString([(x1,y1),(x2,y2)])
            segs.append({'orient':orient, 'geom':seg})

        # 2) Para cada orientación, construir componentes conexas
        for orient in ('h','v'):
            # filtrar segmentos de esta orientación
            s_or = [s for s in segs if s['orient']==orient]
            n = len(s_or)
            if n==0:
                continue

            # construir grafo implícito por matriz de adyacencia touches()
            visited = [False]*n
            for i in range(n):
                if visited[i]:
                    continue
                # empezar una nueva cadena
                stack = [i]
                comp = []
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(s_or[u]['geom'])
                    for v in range(n):
                        if not visited[v] and s_or[u]['geom'].touches(s_or[v]['geom']):
                            visited[v] = True
                            stack.append(v)
                # 3) unir la cadena
                merged = unary_union(comp)
                records.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'orientation': orient,
                    'geometry': merged
                })

    gdf_const = gpd.GeoDataFrame(records, geometry='geometry', crs=gdf_borders.crs)
    return gdf_const



import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

def midpoints_of_constituent_segments(gdf_borders, tol=1e-9):
    """
    Para cada par (sector_1, sector_2) en gdf_borders, agrupa los segmentos
    puramente horizontales o verticales que se tocan y comparten la misma
    latitud/longitud, y calcula un único punto medio por cada grupo.

    Devuelve un GeoDataFrame con columnas:
      - sector_1, sector_2
      - geometry: Point (punto medio del tramo unido)
    """
    records = []
    # 1) Agrupar por frontera (sector_1, sector_2)
    for (s1, s2), grp in gdf_borders.groupby(['sector_1','sector_2']):
        # 2) separar horizontales y verticales
        horizontales = []
        verticales   = []
        for seg in grp.geometry:
            x1,y1 = seg.coords[0]
            x2,y2 = seg.coords[-1]
            if abs(y1 - y2) < tol:
                # horizontal: guardamos (y0, LineString)
                y0 = (y1 + y2)/2
                horizontales.append((y0, seg))
            elif abs(x1 - x2) < tol:
                # vertical: guardamos (x0, LineString)
                x0 = (x1 + x2)/2
                verticales.append((x0, seg))
            # else: ignoramos segmentos inclinados

        # 3) función auxiliar para procesar cada orientación
        def _process(groups, is_horizontal):
            # agrupar por coordenada constante (redondeada a tol)
            buckets = defaultdict(list)
            for const, seg in groups:
                key = round(const/tol)*tol
                buckets[key].append(seg)

            # para cada cubeta, extraer componentes conexas
            for const, segs in buckets.items():
                n = len(segs)
                visited = [False]*n
                for i in range(n):
                    if visited[i]:
                        continue
                    # DFS para agrupar segs que se tocan
                    stack = [i]
                    comp = []
                    visited[i] = True
                    while stack:
                        u = stack.pop()
                        comp.append(segs[u])
                        for v in range(n):
                            if not visited[v] and segs[u].touches(segs[v]):
                                visited[v] = True
                                stack.append(v)
                    # 4) calcular extremo-a-extremo
                    xs = []
                    ys = []
                    for s in comp:
                        for x,y in s.coords:
                            xs.append(x)
                            ys.append(y)
                    if is_horizontal:
                        y0    = const
                        x_min = min(xs); x_max = max(xs)
                        mx, my = (x_min + x_max)/2, y0
                    else:
                        x0    = const
                        y_min = min(ys); y_max = max(ys)
                        mx, my = x0, (y_min + y_max)/2

                    records.append({
                        'sector_1': s1,
                        'sector_2': s2,
                        'geometry': Point(mx, my)
                    })

        _process(horizontales, is_horizontal=True)
        _process(verticales,   is_horizontal=False)

    gdf_mid = gpd.GeoDataFrame(records, crs=gdf_borders.crs)
    return gdf_mid


# 1) Ya tienes tu gdf_borders
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) Obtienes los puntos medios de los tramos horizontales/verticales consecutivos
gdf_mid_const = midpoints_of_constituent_segments(gdf_borders)

# 3) Inspecciona
print(gdf_mid_const)
gdf_mid_const.plot(markersize=50)


gdf_midpoints = gdf_mid_const

from shapely.geometry import LineString
import geopandas as gpd

# 0) Prepara cells_info
cells_info = [
    (row['Cell_Name'], row['Polygon'], row['Sector'])
    for _, row in gdf_cells.iterrows()
]

# 1) get_shared_borders_and_nodes
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) polígono ACC y su contorno
polygons = DF_info_conf['Contorno Sector'].tolist()
union_poly = unary_union(polygons)
if union_poly.geom_type == 'MultiPolygon':
    poligono_ACC = union_poly.convex_hull
else:
    poligono_ACC = Polygon(union_poly.exterior)
acc_boundary = poligono_ACC.boundary


# 3) puntos de contacto ACC
records = []
for _, row in gdf_borders.iterrows():
    s1, s2 = row.sector_1, row.sector_2
    for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
        pt = Point(x, y)
        if pt.distance(acc_boundary) < 1e-9:
            records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
                     .drop_duplicates('geometry') \
                     .reset_index(drop=True)


# 4) cruces H-V
gdf_corners = extract_corner_points(gdf_borders)

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

from pyproj import Geod

def filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=10):
    """
    Elimina los puntos de gdf_midpoints que estén a menos de `max_nm` millas náuticas
    de cualquiera de los puntos en gdf_acc_touch.

    Si los GeoDataFrames no tienen CRS definido, asume WGS84 (grados).
    """
    # Convertir millas náuticas a metros
    threshold_m = max_nm * 1852

    # Determinar si es geográfico (lat/lon) o proyectado
    is_geographic = True
    if hasattr(gdf_midpoints, 'crs') and gdf_midpoints.crs is not None:
        try:
            is_geographic = gdf_midpoints.crs.is_geographic
        except AttributeError:
            is_geographic = True

    # Obtener coordenadas de puntos ACC
    acc_coords = [(pt.x, pt.y) for pt in gdf_acc_touch.geometry]

    if is_geographic:
        # Usar cálculo geodésico en WGS84
        geod = Geod(ellps="WGS84")
        keep_indices = []
        for idx, mid_pt in enumerate(gdf_midpoints.geometry):
            lon, lat = mid_pt.x, mid_pt.y
            # Calcular distancia a cada punto ACC
            distances = [geod.inv(lon, lat, acc_lon, acc_lat)[2] for acc_lon, acc_lat in acc_coords]
            # Conservar si todas las distancias > umbral
            if all(d > threshold_m for d in distances):
                keep_indices.append(idx)
        return gdf_midpoints.loc[keep_indices].reset_index(drop=True)
    else:
        # CRS proyectado (unidades en metros): usar buffer único
        acc_buffer = gdf_acc_touch.buffer(threshold_m).unary_union
        filtered = gdf_midpoints[~gdf_midpoints.geometry.within(acc_buffer)].copy()
        return filtered.reset_index(drop=True)

# Ejemplo de uso:
gdf_midpoints = filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=5)




import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

def connect_all_points(gdf_midpoints, gdf_nodes, gdf_acc_touch):
    """
    Conecta:
      - gdf_midpoints: puntos medios con columnas ['sector_1','sector_2','geometry']
      - gdf_acc_touch: puntos ACC con columnas ['sector_1','sector_2','geometry']
      - gdf_nodes: puntos nodales con columnas ['sectors','geometry'],
                   donde 'sectors' es una lista de sectores (strings)
    Siguiendo estas reglas:
      • Cada punto ACC (tiene dos sectores [s1,s2]) se conecta a SU vecino MÁS CERCANO
        que comparta ambos sectores s1 y s2 (puede ser midpoint o node).
      • Cada punto medio (tiene dos sectores [s1,s2]) se conecta a SUS DOS vecinos MÁS CERCANOS
        que compartan el par {s1,s2}. Si solo hay uno, se conecta a ese.
      • Cada nodo (tiene lista de sectores [s1,s2,…,sK]) se conecta, para cada sector si en su lista,
        a SU vecino MÁS CERCANO que contenga ESE sector en su lista de sectores (puede ser midpoint,
        ACC o node distinto). De ese modo genera un número de aristas igual al número de sectores.
      • No se crean duplicados: si A se conecta a B, no se vuelve a crear B→A.

    Devuelve:
      GeoDataFrame con columnas ['sector_1','sector_2','geometry'], donde cada fila
      es una arista (LineString) que une dos puntos conforme a las reglas.
    """

    # 1) Construir lista unificada de “puntos” con:
    #     - uid: índice entero único
    #     - coords: (x,y) tupla de coordenadas
    #     - sectors: lista de sectores (para mid/acc, dos; para node, lista)
    #     - type: 'mid', 'acc' o 'node'
    puntos = []
    uid = 0

    # a) Midpoints
    for _, row in gdf_midpoints.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'mid'
        })
        uid += 1

    # b) ACC Touch
    for _, row in gdf_acc_touch.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'acc'
        })
        uid += 1

    # c) Nodes
    for _, row in gdf_nodes.iterrows():
        sec_list = row['sectors']  # asume lista de strings
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': list(sec_list),  # ya es lista
            'type': 'node'
        })
        uid += 1

    if not puntos:
        return gpd.GeoDataFrame(columns=['sector_1','sector_2','geometry'], geometry='geometry')

    # 2) Preparar arrays para cálculos de distancia
    n = len(puntos)
    coords = np.array([p['coords'] for p in puntos])  # shape (n,2)
    # Distancia euclídea² entre cada par (i,j)
    diffs = coords[:, None, :] - coords[None, :, :]    # shape (n,n,2)
    dist2 = np.sum(diffs**2, axis=2)                   # shape (n,n)
    np.fill_diagonal(dist2, np.inf)

    # 3) Función auxiliar: para un índice i, devuelve lista de índices de candidatos j donde:
    #      - La lista de sectores de j contiene TODOS los sectores de i (para mid y acc)
    #      - Para node: la lista de sectores de j contiene el sector sc que estamos procesando
    #    y j != i.
    def candidatos_para(i, modo, sector_obj=None):
        """
        i: índice del punto en 'puntos'
        modo: 'pair' si queremos pares exactos (mid/acc),
              'single' si es búsqueda por un solo sector (para node)
        sector_obj: en modo 'single', el sector en cuestión (string)
        Retorna lista de índices j válidos.
        """
        lista_j = []
        if modo == 'pair':
            set_i = set(puntos[i]['sectors'])  # debe tener exactamente 2
            for j in range(n):
                if j == i:
                    continue
                if set_i.issubset(set(puntos[j]['sectors'])):
                    lista_j.append(j)
        else:  # modo == 'single'
            # sector_obj es un string. Buscamos todos j != i con sector_obj in puntos[j]['sectors']
            for j in range(n):
                if j == i:
                    continue
                if sector_obj in puntos[j]['sectors']:
                    lista_j.append(j)
        return lista_j

    # 4) Construir aristas sin duplicados
    added = set()  # almacenará frozenset({i,j}) para evitar duplicados
    records = []

    for idx in range(n):
        punto = puntos[idx]
        tipo = punto['type']

        if tipo == 'acc':
            # buscar candidatos que contengan el mismo par de sectores
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # elegir el más cercano según dist2
            j_min = int(np.argmin(dist2[idx, cands]))
            vecino = cands[j_min]
            key = frozenset({idx, vecino})
            if key not in added:
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        elif tipo == 'mid':
            # buscar candidatos que contengan el par exacto
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # necesitamos hasta dos vecinos más cercanos
            k = min(2, len(cands))
            idxs_k = np.argsort(dist2[idx, cands])[:k]
            for pos in idxs_k:
                vecino = cands[pos]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        else:  # tipo == 'node'
            # para cada sector en su lista, conectar al vecino más cercano que contenga ese sector
            for sector in punto['sectors']:
                cands = candidatos_para(idx, modo='single', sector_obj=sector)
                if not cands:
                    continue
                j_min = int(np.argmin(dist2[idx, cands]))
                vecino = cands[j_min]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                # La arista pertenece a la frontera entre el sector “sector”
                # y algún otro. Pero debemos decidir qué par guardamos en 'sector_1','sector_2'.
                # Tomamos el par formado por ese 'sector' y el otro sector compartido.
                # Encontrar intersección de listas de sectores:
                set_i = set(puntos[idx]['sectors'])
                set_j = set(puntos[vecino]['sectors'])
                comunes = set_i.intersection(set_j)
                # Debería contener al menos 'sector'. Ahora elegimos:
                #   - si hay exactamente 2 comunes, ese par es la frontera
                #   - si hay más de 2, tomamos arbitrariamente los dos primeros
                if len(comunes) >= 2:
                    s1, s2 = sorted(list(comunes))[:2]
                else:
                    # Si solo 'sector' está en común, buscamos en punto j un sector diferente:
                    otros_j = set_j - {sector}
                    if otros_j:
                        s1, s2 = sorted([sector, list(otros_j)[0]])
                    else:
                        # cae aquí si ambos puntos comparten exactamente ese sector;
                        # en ese caso, no hay "par" claro, we still put sector twice
                        s1, s2 = sector, sector
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

    # 5) Devolver GeoDataFrame final
    gdf_edges = gpd.GeoDataFrame(records, geometry='geometry')
    # Asignar CRS heredado (si alguno de los inputs lo tiene)
    for df_in in (gdf_midpoints, gdf_acc_touch, gdf_nodes):
        if df_in is not None and hasattr(df_in, 'crs') and df_in.crs is not None:
            gdf_edges.set_crs(df_in.crs, inplace=True)
            break

    return gdf_edges


# Asumiendo que ya tienes:
#   gdf_midpoints con ['sector_1','sector_2','geometry']
#   gdf_acc_touch con ['sector_1','sector_2','geometry']
#   gdf_nodes con ['sectors','geometry']  (donde 'sectors' es lista de strings)

gdf_edges = connect_all_points(
    gdf_midpoints=gdf_midpoints,
    gdf_nodes=gdf_nodes,
    gdf_acc_touch=gdf_acc_touch
)

# El GeoDataFrame resultante 'gdf_edges' tendrá columnas:
#   - 'sector_1', 'sector_2'  (indicando la frontera asociada a cada arista)
#   - 'geometry' (LineString entre los dos puntos unidos)
print(gdf_edges)



import geopandas as gpd

def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados



import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict


import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

def eliminar_aristas_prefer_node_mid(gdf_edges, gdf_midpoints, gdf_nodes, tol=1e-9):
    """
    Para cada punto medio con grado > 2, elimina primero las aristas que conectan
    con un nodo, y solo si faltan por eliminar, las que conectan con otro midpoint,
    hasta dejar grado = 2.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    gdf_nodes : GeoDataFrame
        Nodos (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges sin las aristas eliminadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints y nodos
    mid_keys  = { key(p) for p in gdf_midpoints.geometry }
    node_keys = { key(p) for p in gdf_nodes.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    to_drop = set()
    # Procesar cada midpoint con más de 2 conexiones
    for mid_k, edge_idxs in incidence.items():
        degree = len(edge_idxs)
        if degree <= 2:
            continue
        eliminar = degree - 2
        
        # Clasificar candidatos según destino node o midpoint
        node_conns = []
        mid_conns  = []
        for idx in edge_idxs:
            coords = list(edges.geometry.loc[idx].coords)
            e0 = (round(coords[0][0]/tol)*tol, round(coords[0][1]/tol)*tol)
            e1 = (round(coords[-1][0]/tol)*tol, round(coords[-1][1]/tol)*tol)
            other = e1 if e0 == mid_k else e0
            if other in node_keys:
                node_conns.append(idx)
            elif other in mid_keys:
                mid_conns.append(idx)
        
        # Primero eliminar conexiones a nodos
        for idx in node_conns[:eliminar]:
            to_drop.add(idx)
        faltan = eliminar - min(len(node_conns), eliminar)
        # # Si todavía faltan, eliminar conexiones a otros midpoints
        # for idx in mid_conns[:faltan]:
        #     to_drop.add(idx)
    
    # Eliminar y resetear índice
    if to_drop:
        edges = edges.drop(index=list(to_drop)).reset_index(drop=True)
    # Mantener CRS
    if hasattr(gdf_edges, 'crs'):
        edges.set_crs(gdf_edges.crs, inplace=True)
    return edges

# Ejemplo:
# gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
#     gdf_edges, gdf_midpoints, gdf_nodes
# )

gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
    gdf_edges,
    gdf_midpoints,
    gdf_nodes
)



gdf_edges = gdf_edges_filtrado


def conectar_midpoints_unicos(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada punto medio con solo una conexión, conecta este punto con los dos midpoints
    más cercanos en su frontera, si es posible.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints
    mid_keys = { key(p) for p in gdf_midpoints.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    # Nueva lista de aristas
    new_edges = []

    # 1) Identificar midpoints con una única conexión
    for mid_k, edge_idxs in incidence.items():
        if len(edge_idxs) == 1:
            # Identificar el sector de este midpoint
            midpoint = gdf_midpoints[gdf_midpoints.geometry.apply(key) == mid_k]
            if midpoint.empty:
                continue
            s1, s2 = midpoint.iloc[0]['sector_1'], midpoint.iloc[0]['sector_2']
            
            # Buscar otros midpoints con la misma frontera
            vecinos_potenciales = gdf_midpoints[(gdf_midpoints['sector_1'] == s1) & (gdf_midpoints['sector_2'] == s2)]
            
            # Eliminar el punto medio actual de la lista de vecinos
            vecinos_potenciales = vecinos_potenciales[vecinos_potenciales.geometry.apply(key) != mid_k]
            
            # Si hay más de uno, encontrar los dos más cercanos
            if len(vecinos_potenciales) > 1:
                coords_mid = midpoint.iloc[0].geometry.coords[0]
                vecinos_coords = [(v.geometry.coords[0], idx) for idx, v in vecinos_potenciales.iterrows()]
                
                # Calcular distancias
                # distancias = [(idx, np.sqrt((coords_mid[0] - v[0][0])**2 + (coords_mid[1] - v[0][1])**2)) for v, idx in vecinos_coords]
                distancias = [(idx,np.sqrt((coords_mid[0] - v[0])**2 +(coords_mid[1] - v[1])**2))for v, idx in vecinos_coords]

                # Ordenar por distancia
                distancias.sort(key=lambda x: x[1])
                
                # Seleccionar los dos más cercanos
                closest_idx = distancias[0][0]
                second_closest_idx = distancias[1][0]
                
                # Crear nuevas aristas entre el midpoint y los dos más cercanos
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[closest_idx].geometry])
                })
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[second_closest_idx].geometry])
                })
    
    # 2) Agregar las nuevas aristas al GeoDataFrame
    if new_edges:
        new_edges_gdf = gpd.GeoDataFrame(new_edges, geometry='geometry')
        edges = pd.concat([edges, new_edges_gdf], ignore_index=True)
    
    # Mantener CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        edges.set_crs(gdf_edges.crs, inplace=True)
    
    return edges


gdf_edges_nuevo = conectar_midpoints_unicos(gdf_edges, gdf_midpoints)
gdf_edges=gdf_edges_nuevo

def eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Elimina solo las aristas que conectan entre sí dos midpoints,
    cuando ambos tienen grado > 2.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas del grafo con geometrías LineString.
    gdf_midpoints : GeoDataFrame
        Midpoints con geometría Point.
    tol : float
        Tolerancia para comparación de coordenadas.

    Retorna
    -------
    GeoDataFrame
        gdf_edges sin las aristas que unen midpoints con grado > 2 entre sí.
    """
    from collections import defaultdict

    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # Crear set de claves de midpoints
    mid_keys = {key(p) for p in gdf_midpoints.geometry}

    # Calcular grado de cada midpoint
    degree_map = defaultdict(int)
    edge_extremos = {}

    for idx, line in gdf_edges.geometry.items():
        coords = list(line.coords)
        k1 = key(Point(*coords[0]))
        k2 = key(Point(*coords[-1]))
        edge_extremos[idx] = (k1, k2)

        if k1 in mid_keys:
            degree_map[k1] += 1
        if k2 in mid_keys:
            degree_map[k2] += 1

    # Identificar aristas entre midpoints de grado > 2
    to_drop = set()
    for idx, (k1, k2) in edge_extremos.items():
        if k1 in mid_keys and k2 in mid_keys:
            if degree_map[k1] > 2 and degree_map[k2] > 2:
                to_drop.add(idx)

    # Mostrar resumen
    print(f"Aristas eliminadas entre midpoints de grado > 2: {len(to_drop)}")

    # Eliminar aristas identificadas
    gdf_filtrado = gdf_edges.drop(index=list(to_drop)).reset_index(drop=True) if to_drop else gdf_edges.copy()

    # Conservar CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges = eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints)


def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from collections import defaultdict

def conectar_midpoints_aislados(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada frontera (sector_1, sector_2), identifica los midpoints
    que solo tienen una conexión (grado==1). Si en esa misma frontera
    hay exactamente dos midpoints de grado 1, añade una arista que los una.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas existentes con geometrías LINESTRING.
    gdf_midpoints : GeoDataFrame
        Midpoints con columnas ['sector_1','sector_2','geometry'].
    tol : float
        Tolerancia para comparar coordenadas (por defecto 1e-9).

    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    # 1) Función para obtener clave “redondeada” de un punto
    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # 2) Preparamos el conteo de grado para cada midpoint (clave->grado)
    #    y almacenamos la geometría de cada clave
    mid_keys = {}      # key -> Point geom
    for _, row in gdf_midpoints.iterrows():
        k = key(row.geometry)
        mid_keys[k] = row.geometry

    grado = defaultdict(int)
    # Recorremos cada arista y contamos sus endpoints si son midpoints
    for line in gdf_edges.geometry:
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        k0, k1 = key(type(line)(x0, y0)) if False else (None, None), (None, None)
        # mejor extraer directamente
        k0 = key(Point(x0, y0))
        k1 = key(Point(x1, y1))
        if k0 in mid_keys:
            grado[k0] += 1
        if k1 in mid_keys:
            grado[k1] += 1

    new_edges = []
    # 3) Agrupamos los midpoints por frontera
    for (s1, s2), group in gdf_midpoints.groupby(['sector_1', 'sector_2']):
        # Mapeamos clave->fila para este grupo
        key_to_row = { key(row.geometry): row for _, row in group.iterrows() }
        # Filtramos los que tengan grado == 1
        aislados = [k for k in key_to_row if grado.get(k, 0) == 1]
        # Si hay exactamente dos, los conectamos
        if len(aislados) == 2:
            geom1 = key_to_row[aislados[0]].geometry
            geom2 = key_to_row[aislados[1]].geometry
            new_edges.append({
                'sector_1': s1,
                'sector_2': s2,
                'geometry': LineString([geom1, geom2])
            })

    # 4) Si hay nuevas aristas, las concatenamos
    if new_edges:
        gdf_nuevas = gpd.GeoDataFrame(new_edges, geometry='geometry', crs=gdf_edges.crs)
        return pd.concat([gdf_edges, gdf_nuevas], ignore_index=True)
    else:
        return gdf_edges.copy()


gdf_edges = conectar_midpoints_aislados(gdf_edges, gdf_midpoints)


import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# 1) Reconstruir el contorno real del ACC (sin usar convex_hull)
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

if isinstance(union_poligonos, MultiPolygon):
    # Si la unión da MultiPolygon, tomamos cada polígono por separado
    lista_polys = list(union_poligonos.geoms)
else:
    # Si es un único Polygon
    lista_polys = [union_poligonos]

# 2) Extraer todas las coordenadas de los exteriores para ajustar límites
all_x = []
all_y = []
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    all_x.extend(x_poly)
    all_y.extend(y_poly)

min_lon = min(all_x) - 0.5
max_lon = max(all_x) + 0.5
min_lat = min(all_y) - 0.5
max_lat = max(all_y) + 0.5

# 3) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUDE [º]')
ax.set_ylabel('LATITUDE [º]')
ax.set_title('Final Sectorization')

# 4) Dibujar solo el contorno (exterior) de cada polígono de ACC, sin rellenar
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='_nolegend_')

# 5) Ahora superponemos cada LineString de gdf_edges en rojo
for linea in gdf_edges.geometry:
    xs, ys = linea.xy
    ax.plot(xs, ys, color='red', linewidth=1.2)

# Opcional: añadir una leyenda manual para gdf_edges
ax.plot([], [], color='red', linewidth=1.2, label='New Boundaries')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')

plt.tight_layout()
plt.show()

# 1) Filtrar puntos que están dentro de la delimitación del ACC
gdf_midpoints_in = gdf_midpoints[gdf_midpoints.geometry.within(union_poligonos)]
gdf_acc_touch_in = gdf_acc_touch
# gdf_acc_touch_in = gdf_acc_touch[gdf_acc_touch.geometry.within(union_poligonos)]
gdf_nodes_in = gdf_nodes[gdf_nodes.geometry.within(union_poligonos)]

# 2) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUDE [º]')
ax.set_ylabel('LATITUDE [º]')
ax.set_title('Points Used to Generate the Representation')

# 3) Dibujar contorno ACC
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='ACC Boundary')

# 4) Dibujar puntos
gdf_midpoints_in.plot(ax=ax, color='blue', markersize=20, label='Midpoints')
gdf_acc_touch_in.plot(ax=ax, color='green', markersize=30, marker='^', label='ACC Points')
gdf_nodes_in.plot(ax=ax, color='purple', markersize=40, marker='s', label='Nodes')

# 5) Leyenda y presentación
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
plt.tight_layout()
plt.show()







import matplotlib.pyplot as plt

# 1) Reconstruir el contorno real del ACC (sin usar convex_hull)
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

if isinstance(union_poligonos, MultiPolygon):
    lista_polys = list(union_poligonos.geoms)
else:
    lista_polys = [union_poligonos]

# 2) Extraer límites para ajustar la vista
all_x = []
all_y = []
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    all_x.extend(x_poly)
    all_y.extend(y_poly)

min_lon = min(all_x) - 0.5
max_lon = max(all_x) + 0.5
min_lat = min(all_y) - 0.5
max_lat = max(all_y) + 0.5

# 3) Crear la figura
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUD [º]')
ax.set_ylabel('LATITUD [º]')
ax.set_title('Sectorización Final con Flujos de Mayor Distancia')

# 4) Dibujar el contorno de los sectores
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='_nolegend_')

# 5) Dibujar las fronteras nuevas (gdf_edges)
for linea in gdf_edges.geometry:
    xs, ys = linea.xy
    ax.plot(xs, ys, color='red', linewidth=1.2)

# 6) Superponer los flujos de mayor distancia
for cell_name, flow_data in max_distance_flows.items():
    flow = flow_data['flow']
    if flow is not None:
        x_flow, y_flow = flow.xy
        ax.plot(x_flow, y_flow, color='blue', linewidth=2, linestyle='--', label='_nolegend_')  # Azul punteado

# 7) Añadir leyenda manual
ax.plot([], [], color='red', linewidth=1.2, label='Fronteras nuevas')
ax.plot([], [], color='blue', linewidth=2, linestyle='--', label='Flujos de mayor distancia')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize='small')
plt.tight_layout()
plt.show()













































# ESTO ES PARA CONECTAR LOS PUNTOS MEDIOS DE LAS CELDAS NO DE LOS TRAMOS



# 1) Asegúrate de que gdf_borders tenga un ID por fila
gdf_borders = gdf_borders.reset_index().rename(columns={'index':'border_id'})






def get_midpoints_per_segment(gdf_borders, tol=1e-9):
    """
    Para cada frontera (LineString) en gdf_borders, descompone sus segmentos
    consecutivos y genera un punto medio si el tramo es puramente horizontal
    o vertical. Conserva también el border_id y los sectores.
    """
    records = []
    for _, row in gdf_borders.iterrows():
        bid    = row['border_id']
        s1, s2 = row['sector_1'], row['sector_2']
        coords = list(row.geometry.coords)
        # recorre cada segmento (coords[i] → coords[i+1])
        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            # vertical
            if abs(x1 - x2) < tol:
                mx, my = x1,        (y1 + y2)/2.0
            # horizontal
            elif abs(y1 - y2) < tol:
                mx, my = (x1 + x2)/2.0, y1
            else:
                continue
            records.append({
                'border_id': bid,
                'sector_1' : s1,
                'sector_2' : s2,
                'geometry' : Point(mx, my)
            })

    return gpd.GeoDataFrame(records, crs=gdf_borders.crs)

# 2) Llamada reemplazando ambos pasos anteriores:
gdf_midpoints = get_midpoints_per_segment(gdf_borders)



from shapely.geometry import LineString
import geopandas as gpd

# 0) Prepara cells_info
cells_info = [
    (row['Cell_Name'], row['Polygon'], row['Sector'])
    for _, row in gdf_cells.iterrows()
]

# 1) get_shared_borders_and_nodes
gdf_borders, gdf_nodes = get_shared_borders_and_nodes(cells_info, gdf_cells.crs)

# 2) polígono ACC y su contorno
polygons = DF_info_conf['Contorno Sector'].tolist()
union_poly = unary_union(polygons)
if union_poly.geom_type == 'MultiPolygon':
    poligono_ACC = union_poly.convex_hull
else:
    poligono_ACC = Polygon(union_poly.exterior)
acc_boundary = poligono_ACC.boundary


# 3) puntos de contacto ACC
records = []
for _, row in gdf_borders.iterrows():
    s1, s2 = row.sector_1, row.sector_2
    for x, y in [row.geometry.coords[0], row.geometry.coords[-1]]:
        pt = Point(x, y)
        if pt.distance(acc_boundary) < 1e-9:
            records.append({'sector_1':s1,'sector_2':s2,'geometry':pt})
gdf_acc_touch = gpd.GeoDataFrame(records, crs=gdf_borders.crs) \
                     .drop_duplicates('geometry') \
                     .reset_index(drop=True)


# 4) cruces H-V
gdf_corners = extract_corner_points(gdf_borders)


from pyproj import Geod

def filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=10):
    """
    Elimina los puntos de gdf_midpoints que estén a menos de `max_nm` millas náuticas
    de cualquiera de los puntos en gdf_acc_touch.

    Si los GeoDataFrames no tienen CRS definido, asume WGS84 (grados).
    """
    # Convertir millas náuticas a metros
    threshold_m = max_nm * 1852

    # Determinar si es geográfico (lat/lon) o proyectado
    is_geographic = True
    if hasattr(gdf_midpoints, 'crs') and gdf_midpoints.crs is not None:
        try:
            is_geographic = gdf_midpoints.crs.is_geographic
        except AttributeError:
            is_geographic = True

    # Obtener coordenadas de puntos ACC
    acc_coords = [(pt.x, pt.y) for pt in gdf_acc_touch.geometry]

    if is_geographic:
        # Usar cálculo geodésico en WGS84
        geod = Geod(ellps="WGS84")
        keep_indices = []
        for idx, mid_pt in enumerate(gdf_midpoints.geometry):
            lon, lat = mid_pt.x, mid_pt.y
            # Calcular distancia a cada punto ACC
            distances = [geod.inv(lon, lat, acc_lon, acc_lat)[2] for acc_lon, acc_lat in acc_coords]
            # Conservar si todas las distancias > umbral
            if all(d > threshold_m for d in distances):
                keep_indices.append(idx)
        return gdf_midpoints.loc[keep_indices].reset_index(drop=True)
    else:
        # CRS proyectado (unidades en metros): usar buffer único
        acc_buffer = gdf_acc_touch.buffer(threshold_m).unary_union
        filtered = gdf_midpoints[~gdf_midpoints.geometry.within(acc_buffer)].copy()
        return filtered.reset_index(drop=True)

# Ejemplo de uso:
gdf_midpoints = filter_midpoints_by_acc_distance(gdf_midpoints, gdf_acc_touch, max_nm=5)




import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

def connect_all_points(gdf_midpoints, gdf_nodes, gdf_acc_touch):
    """
    Conecta:
      - gdf_midpoints: puntos medios con columnas ['sector_1','sector_2','geometry']
      - gdf_acc_touch: puntos ACC con columnas ['sector_1','sector_2','geometry']
      - gdf_nodes: puntos nodales con columnas ['sectors','geometry'],
                   donde 'sectors' es una lista de sectores (strings)
    Siguiendo estas reglas:
      • Cada punto ACC (tiene dos sectores [s1,s2]) se conecta a SU vecino MÁS CERCANO
        que comparta ambos sectores s1 y s2 (puede ser midpoint o node).
      • Cada punto medio (tiene dos sectores [s1,s2]) se conecta a SUS DOS vecinos MÁS CERCANOS
        que compartan el par {s1,s2}. Si solo hay uno, se conecta a ese.
      • Cada nodo (tiene lista de sectores [s1,s2,…,sK]) se conecta, para cada sector si en su lista,
        a SU vecino MÁS CERCANO que contenga ESE sector en su lista de sectores (puede ser midpoint,
        ACC o node distinto). De ese modo genera un número de aristas igual al número de sectores.
      • No se crean duplicados: si A se conecta a B, no se vuelve a crear B→A.

    Devuelve:
      GeoDataFrame con columnas ['sector_1','sector_2','geometry'], donde cada fila
      es una arista (LineString) que une dos puntos conforme a las reglas.
    """

    # 1) Construir lista unificada de “puntos” con:
    #     - uid: índice entero único
    #     - coords: (x,y) tupla de coordenadas
    #     - sectors: lista de sectores (para mid/acc, dos; para node, lista)
    #     - type: 'mid', 'acc' o 'node'
    puntos = []
    uid = 0

    # a) Midpoints
    for _, row in gdf_midpoints.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'mid'
        })
        uid += 1

    # b) ACC Touch
    for _, row in gdf_acc_touch.iterrows():
        s1, s2 = row['sector_1'], row['sector_2']
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': [s1, s2],
            'type': 'acc'
        })
        uid += 1

    # c) Nodes
    for _, row in gdf_nodes.iterrows():
        sec_list = row['sectors']  # asume lista de strings
        pts = row.geometry
        puntos.append({
            'uid': uid,
            'coords': (pts.x, pts.y),
            'sectors': list(sec_list),  # ya es lista
            'type': 'node'
        })
        uid += 1

    if not puntos:
        return gpd.GeoDataFrame(columns=['sector_1','sector_2','geometry'], geometry='geometry')

    # 2) Preparar arrays para cálculos de distancia
    n = len(puntos)
    coords = np.array([p['coords'] for p in puntos])  # shape (n,2)
    # Distancia euclídea² entre cada par (i,j)
    diffs = coords[:, None, :] - coords[None, :, :]    # shape (n,n,2)
    dist2 = np.sum(diffs**2, axis=2)                   # shape (n,n)
    np.fill_diagonal(dist2, np.inf)

    # 3) Función auxiliar: para un índice i, devuelve lista de índices de candidatos j donde:
    #      - La lista de sectores de j contiene TODOS los sectores de i (para mid y acc)
    #      - Para node: la lista de sectores de j contiene el sector sc que estamos procesando
    #    y j != i.
    def candidatos_para(i, modo, sector_obj=None):
        """
        i: índice del punto en 'puntos'
        modo: 'pair' si queremos pares exactos (mid/acc),
              'single' si es búsqueda por un solo sector (para node)
        sector_obj: en modo 'single', el sector en cuestión (string)
        Retorna lista de índices j válidos.
        """
        lista_j = []
        if modo == 'pair':
            set_i = set(puntos[i]['sectors'])  # debe tener exactamente 2
            for j in range(n):
                if j == i:
                    continue
                if set_i.issubset(set(puntos[j]['sectors'])):
                    lista_j.append(j)
        else:  # modo == 'single'
            # sector_obj es un string. Buscamos todos j != i con sector_obj in puntos[j]['sectors']
            for j in range(n):
                if j == i:
                    continue
                if sector_obj in puntos[j]['sectors']:
                    lista_j.append(j)
        return lista_j

    # 4) Construir aristas sin duplicados
    added = set()  # almacenará frozenset({i,j}) para evitar duplicados
    records = []

    for idx in range(n):
        punto = puntos[idx]
        tipo = punto['type']

        if tipo == 'acc':
            # buscar candidatos que contengan el mismo par de sectores
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # elegir el más cercano según dist2
            j_min = int(np.argmin(dist2[idx, cands]))
            vecino = cands[j_min]
            key = frozenset({idx, vecino})
            if key not in added:
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        elif tipo == 'mid':
            # buscar candidatos que contengan el par exacto
            cands = candidatos_para(idx, modo='pair')
            if not cands:
                continue
            # necesitamos hasta dos vecinos más cercanos
            k = min(2, len(cands))
            idxs_k = np.argsort(dist2[idx, cands])[:k]
            for pos in idxs_k:
                vecino = cands[pos]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': punto['sectors'][0],
                    'sector_2': punto['sectors'][1],
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

        else:  # tipo == 'node'
            # para cada sector en su lista, conectar al vecino más cercano que contenga ese sector
            for sector in punto['sectors']:
                cands = candidatos_para(idx, modo='single', sector_obj=sector)
                if not cands:
                    continue
                j_min = int(np.argmin(dist2[idx, cands]))
                vecino = cands[j_min]
                key = frozenset({idx, vecino})
                if key in added:
                    continue
                # La arista pertenece a la frontera entre el sector “sector”
                # y algún otro. Pero debemos decidir qué par guardamos en 'sector_1','sector_2'.
                # Tomamos el par formado por ese 'sector' y el otro sector compartido.
                # Encontrar intersección de listas de sectores:
                set_i = set(puntos[idx]['sectors'])
                set_j = set(puntos[vecino]['sectors'])
                comunes = set_i.intersection(set_j)
                # Debería contener al menos 'sector'. Ahora elegimos:
                #   - si hay exactamente 2 comunes, ese par es la frontera
                #   - si hay más de 2, tomamos arbitrariamente los dos primeros
                if len(comunes) >= 2:
                    s1, s2 = sorted(list(comunes))[:2]
                else:
                    # Si solo 'sector' está en común, buscamos en punto j un sector diferente:
                    otros_j = set_j - {sector}
                    if otros_j:
                        s1, s2 = sorted([sector, list(otros_j)[0]])
                    else:
                        # cae aquí si ambos puntos comparten exactamente ese sector;
                        # en ese caso, no hay "par" claro, we still put sector twice
                        s1, s2 = sector, sector
                p1 = coords[idx]
                p2 = coords[vecino]
                records.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([Point(*p1), Point(*p2)])
                })
                added.add(key)

    # 5) Devolver GeoDataFrame final
    gdf_edges = gpd.GeoDataFrame(records, geometry='geometry')
    # Asignar CRS heredado (si alguno de los inputs lo tiene)
    for df_in in (gdf_midpoints, gdf_acc_touch, gdf_nodes):
        if df_in is not None and hasattr(df_in, 'crs') and df_in.crs is not None:
            gdf_edges.set_crs(df_in.crs, inplace=True)
            break

    return gdf_edges


# Asumiendo que ya tienes:
#   gdf_midpoints con ['sector_1','sector_2','geometry']
#   gdf_acc_touch con ['sector_1','sector_2','geometry']
#   gdf_nodes con ['sectors','geometry']  (donde 'sectors' es lista de strings)

gdf_edges = connect_all_points(
    gdf_midpoints=gdf_midpoints,
    gdf_nodes=gdf_nodes,
    gdf_acc_touch=gdf_acc_touch
)

# El GeoDataFrame resultante 'gdf_edges' tendrá columnas:
#   - 'sector_1', 'sector_2'  (indicando la frontera asociada a cada arista)
#   - 'geometry' (LineString entre los dos puntos unidos)
print(gdf_edges)



import geopandas as gpd

def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados



import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict


import geopandas as gpd
from shapely.geometry import Point, LineString
from collections import defaultdict

def eliminar_aristas_prefer_node_mid(gdf_edges, gdf_midpoints, gdf_nodes, tol=1e-9):
    """
    Para cada punto medio con grado > 2, elimina primero las aristas que conectan
    con un nodo, y solo si faltan por eliminar, las que conectan con otro midpoint,
    hasta dejar grado = 2.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    gdf_nodes : GeoDataFrame
        Nodos (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges sin las aristas eliminadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints y nodos
    mid_keys  = { key(p) for p in gdf_midpoints.geometry }
    node_keys = { key(p) for p in gdf_nodes.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    to_drop = set()
    # Procesar cada midpoint con más de 2 conexiones
    for mid_k, edge_idxs in incidence.items():
        degree = len(edge_idxs)
        if degree <= 2:
            continue
        eliminar = degree - 2
        
        # Clasificar candidatos según destino node o midpoint
        node_conns = []
        mid_conns  = []
        for idx in edge_idxs:
            coords = list(edges.geometry.loc[idx].coords)
            e0 = (round(coords[0][0]/tol)*tol, round(coords[0][1]/tol)*tol)
            e1 = (round(coords[-1][0]/tol)*tol, round(coords[-1][1]/tol)*tol)
            other = e1 if e0 == mid_k else e0
            if other in node_keys:
                node_conns.append(idx)
            elif other in mid_keys:
                mid_conns.append(idx)
        
        # Primero eliminar conexiones a nodos
        for idx in node_conns[:eliminar]:
            to_drop.add(idx)
        faltan = eliminar - min(len(node_conns), eliminar)
        # # Si todavía faltan, eliminar conexiones a otros midpoints
        # for idx in mid_conns[:faltan]:
        #     to_drop.add(idx)
    
    # Eliminar y resetear índice
    if to_drop:
        edges = edges.drop(index=list(to_drop)).reset_index(drop=True)
    # Mantener CRS
    if hasattr(gdf_edges, 'crs'):
        edges.set_crs(gdf_edges.crs, inplace=True)
    return edges

# Ejemplo:
# gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
#     gdf_edges, gdf_midpoints, gdf_nodes
# )

gdf_edges_filtrado = eliminar_aristas_prefer_node_mid(
    gdf_edges,
    gdf_midpoints,
    gdf_nodes
)



gdf_edges = gdf_edges_filtrado


def conectar_midpoints_unicos(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada punto medio con solo una conexión, conecta este punto con los dos midpoints
    más cercanos en su frontera, si es posible.
    
    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Debe contener geometrías LINESTRING y un índice único.
    gdf_midpoints : GeoDataFrame
        Puntos medios (geom POINT).
    tol : float
        Tolerancia para comparar coordenadas.
    
    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    edges = gdf_edges.copy()
    
    # Función para obtener clave redondeada de un punto
    def key(pt):
        return (round(pt.x/tol)*tol, round(pt.y/tol)*tol)
    
    # Conjuntos de claves de midpoints
    mid_keys = { key(p) for p in gdf_midpoints.geometry }
    
    # Incidencia: midpoint -> lista de índices de edges conectados
    incidence = defaultdict(list)
    for idx, line in edges.geometry.items():
        coords = list(line.coords)
        for extremo in (coords[0], coords[-1]):
            k = (round(extremo[0]/tol)*tol, round(extremo[1]/tol)*tol)
            if k in mid_keys:
                incidence[k].append(idx)
    
    # Nueva lista de aristas
    new_edges = []

    # 1) Identificar midpoints con una única conexión
    for mid_k, edge_idxs in incidence.items():
        if len(edge_idxs) == 1:
            # Identificar el sector de este midpoint
            midpoint = gdf_midpoints[gdf_midpoints.geometry.apply(key) == mid_k]
            if midpoint.empty:
                continue
            s1, s2 = midpoint.iloc[0]['sector_1'], midpoint.iloc[0]['sector_2']
            
            # Buscar otros midpoints con la misma frontera
            vecinos_potenciales = gdf_midpoints[(gdf_midpoints['sector_1'] == s1) & (gdf_midpoints['sector_2'] == s2)]
            
            # Eliminar el punto medio actual de la lista de vecinos
            vecinos_potenciales = vecinos_potenciales[vecinos_potenciales.geometry.apply(key) != mid_k]
            
            # Si hay más de uno, encontrar los dos más cercanos
            if len(vecinos_potenciales) > 1:
                coords_mid = midpoint.iloc[0].geometry.coords[0]
                vecinos_coords = [(v.geometry.coords[0], idx) for idx, v in vecinos_potenciales.iterrows()]
                
                # Calcular distancias
                # distancias = [(idx, np.sqrt((coords_mid[0] - v[0][0])**2 + (coords_mid[1] - v[0][1])**2)) for v, idx in vecinos_coords]
                distancias = [(idx,np.sqrt((coords_mid[0] - v[0])**2 +(coords_mid[1] - v[1])**2))for v, idx in vecinos_coords]

                # Ordenar por distancia
                distancias.sort(key=lambda x: x[1])
                
                # Seleccionar los dos más cercanos
                closest_idx = distancias[0][0]
                second_closest_idx = distancias[1][0]
                
                # Crear nuevas aristas entre el midpoint y los dos más cercanos
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[closest_idx].geometry])
                })
                new_edges.append({
                    'sector_1': s1,
                    'sector_2': s2,
                    'geometry': LineString([midpoint.iloc[0].geometry, vecinos_potenciales.loc[second_closest_idx].geometry])
                })
    
    # 2) Agregar las nuevas aristas al GeoDataFrame
    if new_edges:
        new_edges_gdf = gpd.GeoDataFrame(new_edges, geometry='geometry')
        edges = pd.concat([edges, new_edges_gdf], ignore_index=True)
    
    # Mantener CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        edges.set_crs(gdf_edges.crs, inplace=True)
    
    return edges


gdf_edges_nuevo = conectar_midpoints_unicos(gdf_edges, gdf_midpoints)
gdf_edges=gdf_edges_nuevo


def eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Elimina solo las aristas que conectan entre sí dos midpoints,
    cuando ambos tienen grado > 2.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas del grafo con geometrías LineString.
    gdf_midpoints : GeoDataFrame
        Midpoints con geometría Point.
    tol : float
        Tolerancia para comparación de coordenadas.

    Retorna
    -------
    GeoDataFrame
        gdf_edges sin las aristas que unen midpoints con grado > 2 entre sí.
    """
    from collections import defaultdict

    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # Crear set de claves de midpoints
    mid_keys = {key(p) for p in gdf_midpoints.geometry}

    # Calcular grado de cada midpoint
    degree_map = defaultdict(int)
    edge_extremos = {}

    for idx, line in gdf_edges.geometry.items():
        coords = list(line.coords)
        k1 = key(Point(*coords[0]))
        k2 = key(Point(*coords[-1]))
        edge_extremos[idx] = (k1, k2)

        if k1 in mid_keys:
            degree_map[k1] += 1
        if k2 in mid_keys:
            degree_map[k2] += 1

    # Identificar aristas entre midpoints de grado > 2
    to_drop = set()
    for idx, (k1, k2) in edge_extremos.items():
        if k1 in mid_keys and k2 in mid_keys:
            if degree_map[k1] > 2 and degree_map[k2] > 2:
                to_drop.add(idx)

    # Mostrar resumen
    print(f"Aristas eliminadas entre midpoints de grado > 2: {len(to_drop)}")

    # Eliminar aristas identificadas
    gdf_filtrado = gdf_edges.drop(index=list(to_drop)).reset_index(drop=True) if to_drop else gdf_edges.copy()

    # Conservar CRS
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges = eliminar_aristas_entre_midpoints_con_grado_alto(gdf_edges, gdf_midpoints)



def eliminar_aristas_mas_largas_por_acc_duplicado(gdf_edges, gdf_acc_touch):
    """
    Para cada punto de gdf_acc_touch, busca en cuántas líneas de gdf_edges
    aparece. Si aparece en más de una, elimina la línea de mayor longitud
    entre las que comparten ese punto.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías LINESTRING.
    gdf_acc_touch : GeoDataFrame
        GeoDataFrame con columnas ['sector_1','sector_2','geometry'] y geometrías POINT
        (son los puntos ACC).

    Retorna
    -------
    GeoDataFrame
        Una copia de gdf_edges pero eliminando, para cada punto ACC que aparezca en
        más de una línea, la línea (LineString) de mayor longitud que contenga ese punto.
    """
    # 1) Construir un diccionario para contar "en qué edges aparece cada punto ACC".
    #    Usaremos las coordenadas exactas de cada Point de gdf_acc_touch para identificarlo.
    #    La llave será un tuple (x,y) y el valor, una lista de índices de gdf_edges donde aparece.
    occ = {}  # occ[(x_acc, y_acc)] = [índice_edge_1, índice_edge_2, ...]
    # Crear un set de coordenadas de ACC para comparación rápida:
    acc_coords = { (pt.x, pt.y) for pt in gdf_acc_touch.geometry }

    # 2) Recorrer todos los edges y ver si en sus extremos hay algún punto que esté en acc_coords.
    #    Asumimos que la única forma de “aparecer” un punto ACC en un LineString es que coincida
    #    exactamente con uno de sus endpoints (primera o última coordenada).
    for idx_edge, linea in gdf_edges.geometry.items():
        # Obtener los endpoints (coordenadas) del LINESTRING
        coords = list(linea.coords)
        extremo1 = tuple(coords[0])
        extremo2 = tuple(coords[-1])

        # Si extremo1 corresponde a un punto ACC, apuntamos el índice de esta línea
        if extremo1 in acc_coords:
            occ.setdefault(extremo1, []).append(idx_edge)
        # Si extremo2 corresponde a un punto ACC, apuntamos también
        if extremo2 in acc_coords:
            occ.setdefault(extremo2, []).append(idx_edge)

    # 3) Ahora, para cada punto ACC (cada key en occ), verificamos cuántas líneas lo contienen.
    #    Si está en más de una, calculamos la longitud de cada una de esas líneas y marcamos
    #    para eliminación la que resulte más larga.
    a_eliminar = set()
    for punto_acc, lista_indices in occ.items():
        if len(lista_indices) > 1:
            # Calcular longitudes de todas las líneas que comparten este punto ACC
            longitudes = [(idx, gdf_edges.geometry.loc[idx].length) for idx in lista_indices]
            # Ordenar por longitud
            longitudes.sort(key=lambda tpl: tpl[1], reverse=True)  # de mayor a menor
            # La primera tupla es (índice_de_la_línea_más_larga, su_longitud)
            idx_linea_mas_larga, _ = longitudes[0]
            a_eliminar.add(idx_linea_mas_larga)

    # 4) Hacer drop de esas filas de gdf_edges y devolver una copia “limpia”.
    if a_eliminar:
        # Drop por índices y resetear índice (opcional)
        gdf_filtrado = gdf_edges.drop(index=list(a_eliminar)).reset_index(drop=True)
    else:
        # Si no hay nada que eliminar, devolvemos una copia idéntica
        gdf_filtrado = gdf_edges.copy().reset_index(drop=True)

    # Conservamos el CRS original
    if hasattr(gdf_edges, 'crs') and gdf_edges.crs is not None:
        gdf_filtrado.set_crs(gdf_edges.crs, inplace=True)

    return gdf_filtrado


gdf_edges_sin_duplicados = eliminar_aristas_mas_largas_por_acc_duplicado(
    gdf_edges=gdf_edges,
    gdf_acc_touch=gdf_acc_touch
)

gdf_edges = gdf_edges_sin_duplicados

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from collections import defaultdict

def conectar_midpoints_aislados(gdf_edges, gdf_midpoints, tol=1e-9):
    """
    Para cada frontera (sector_1, sector_2), identifica los midpoints
    que solo tienen una conexión (grado==1). Si en esa misma frontera
    hay exactamente dos midpoints de grado 1, añade una arista que los una.

    Parámetros
    ----------
    gdf_edges : GeoDataFrame
        Aristas existentes con geometrías LINESTRING.
    gdf_midpoints : GeoDataFrame
        Midpoints con columnas ['sector_1','sector_2','geometry'].
    tol : float
        Tolerancia para comparar coordenadas (por defecto 1e-9).

    Retorna
    -------
    GeoDataFrame
        Copia de gdf_edges con las nuevas aristas agregadas.
    """
    # 1) Función para obtener clave “redondeada” de un punto
    def key(pt):
        return (round(pt.x / tol) * tol, round(pt.y / tol) * tol)

    # 2) Preparamos el conteo de grado para cada midpoint (clave->grado)
    #    y almacenamos la geometría de cada clave
    mid_keys = {}      # key -> Point geom
    for _, row in gdf_midpoints.iterrows():
        k = key(row.geometry)
        mid_keys[k] = row.geometry

    grado = defaultdict(int)
    # Recorremos cada arista y contamos sus endpoints si son midpoints
    for line in gdf_edges.geometry:
        x0, y0 = line.coords[0]
        x1, y1 = line.coords[-1]
        k0, k1 = key(type(line)(x0, y0)) if False else (None, None), (None, None)
        # mejor extraer directamente
        k0 = key(Point(x0, y0))
        k1 = key(Point(x1, y1))
        if k0 in mid_keys:
            grado[k0] += 1
        if k1 in mid_keys:
            grado[k1] += 1

    new_edges = []
    # 3) Agrupamos los midpoints por frontera
    for (s1, s2), group in gdf_midpoints.groupby(['sector_1', 'sector_2']):
        # Mapeamos clave->fila para este grupo
        key_to_row = { key(row.geometry): row for _, row in group.iterrows() }
        # Filtramos los que tengan grado == 1
        aislados = [k for k in key_to_row if grado.get(k, 0) == 1]
        # Si hay exactamente dos, los conectamos
        if len(aislados) == 2:
            geom1 = key_to_row[aislados[0]].geometry
            geom2 = key_to_row[aislados[1]].geometry
            new_edges.append({
                'sector_1': s1,
                'sector_2': s2,
                'geometry': LineString([geom1, geom2])
            })

    # 4) Si hay nuevas aristas, las concatenamos
    if new_edges:
        gdf_nuevas = gpd.GeoDataFrame(new_edges, geometry='geometry', crs=gdf_edges.crs)
        return pd.concat([gdf_edges, gdf_nuevas], ignore_index=True)
    else:
        return gdf_edges.copy()


gdf_edges = conectar_midpoints_aislados(gdf_edges, gdf_midpoints)


import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# 1) Reconstruir el contorno real del ACC (sin usar convex_hull)
poligonos_sectores = DF_info_conf['Contorno Sector'].tolist()
union_poligonos = unary_union(poligonos_sectores)

if isinstance(union_poligonos, MultiPolygon):
    # Si la unión da MultiPolygon, tomamos cada polígono por separado
    lista_polys = list(union_poligonos.geoms)
else:
    # Si es un único Polygon
    lista_polys = [union_poligonos]

# 2) Extraer todas las coordenadas de los exteriores para ajustar límites
all_x = []
all_y = []
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    all_x.extend(x_poly)
    all_y.extend(y_poly)

min_lon = min(all_x) - 0.5
max_lon = max(all_x) + 0.5
min_lat = min(all_y) - 0.5
max_lat = max(all_y) + 0.5

# 3) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUD [º]')
ax.set_ylabel('LATITUD [º]')
ax.set_title('Sectorización Final')

# 4) Dibujar solo el contorno (exterior) de cada polígono de ACC, sin rellenar
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='_nolegend_')

# 5) Ahora superponemos cada LineString de gdf_edges en rojo
for linea in gdf_edges.geometry:
    xs, ys = linea.xy
    ax.plot(xs, ys, color='red', linewidth=1.2)

# Opcional: añadir una leyenda manual para gdf_edges
ax.plot([], [], color='red', linewidth=1.2, label='Fronteras nuevas')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize='small')

plt.tight_layout()
plt.show()

# 1) Filtrar puntos que están dentro de la delimitación del ACC
gdf_midpoints_in = gdf_midpoints[gdf_midpoints.geometry.within(union_poligonos)]
gdf_acc_touch_in = gdf_acc_touch
# gdf_acc_touch_in = gdf_acc_touch[gdf_acc_touch.geometry.within(union_poligonos)]
gdf_nodes_in = gdf_nodes[gdf_nodes.geometry.within(union_poligonos)]

# 2) Crear figura y ejes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_aspect('equal')
ax.set_xlabel('LONGITUD [º]')
ax.set_ylabel('LATITUD [º]')
ax.set_title('Puntos Utilizados para Realizar la Representación')

# 3) Dibujar contorno ACC
for poly in lista_polys:
    x_poly, y_poly = poly.exterior.xy
    ax.plot(x_poly, y_poly, color='black', linewidth=1, label='Contorno ACC')

# 4) Dibujar puntos
gdf_midpoints_in.plot(ax=ax, color='blue', markersize=20, label='Puntos medios')
gdf_acc_touch_in.plot(ax=ax, color='green', markersize=30, marker='^', label='Puntos ACC')
gdf_nodes_in.plot(ax=ax, color='purple', markersize=40, marker='s', label='Nodos')

# 5) Leyenda y presentación
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')
plt.tight_layout()
plt.show()




