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
plt.xlabel('LONGITUDE[º]')
plt.ylabel('LATITUDE[º]')
plt.title('Representation of the Study Sectors')

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
# 5. DEFINICIÓN DE UNA NUEVA FUNCIÓN OBJETIVO CON REPARTO DE COMPLEJIDAD ENTRE SECTORES
# =============================================================================

# SIN TENER EN CUENTA EL NUMERO DE CELDAS
def improved_objective(assignment):
    """
    Función objetivo: minimiza la varianza muestral de la complejidad total entre los sectores,
    y añade una penalización por la diferencia entre la complejidad máxima y mínima.
    """
    # 1. Calculamos la complejidad total por sector
    comp_by_sector = {}
    for cell, sec in assignment.items():
        comp = df_cells.loc[df_cells['Cell_Name'] == cell, 'Complexity'].values[0]
        comp_by_sector.setdefault(sec, 0)
        comp_by_sector[sec] += comp

    # 2. Calculamos la varianza muestral de la complejidad entre sectores
    values = list(comp_by_sector.values())
    number_sectors = len(values)
    overall_avg = sum(values) / number_sectors

    if number_sectors > 1:
        variance = sum((v - overall_avg) ** 2 for v in values) / (number_sectors - 1)
    else:
        variance = 0  # Si solo hay un sector, no hay varianza

    # 3. Penalización por la diferencia entre la complejidad total máxima y mínima
    max_complexity = max(values)
    min_complexity = min(values)
    complexity_diff_penalty = max_complexity - min_complexity

    # 4. Función objetivo: se minimiza la suma de la varianza y la diferencia extrema
    objective_value = variance + complexity_diff_penalty

    return objective_value


# Función auxiliar para contar las celdas cambiadas por sector,
# comparando la asignación actual con la asignación original.
def changed_cells_count(assignment, original_assignment):
    counts = {}
    for cell, new_sec in assignment.items():
        orig_sec = original_assignment[cell]
        if new_sec != orig_sec:
            counts.setdefault(orig_sec, 0)
            counts[orig_sec] += 1
    return counts

# =============================================================================
# 6. ESTABLECER LA RESTRICCIÓN DE CAMBIOS MÁXIMOS POR SECTOR (ya no usamos max_change en el sentido neto)
# =============================================================================
initial_counts = df_cells.groupby('Sector').size().to_dict()
current_counts = initial_counts.copy()
max_change = 10

# La asignación original se guarda para poder comparar posteriormente
original_assignment = df_cells.set_index('Cell_Name')['Sector'].to_dict()

assignment = original_assignment.copy()

# =============================================================================
# 7. OPTIMIZACIÓN ITERATIVA CON REPARTO DE COMPLEJIDAD ENTRE SECTORES - 5 CORRIDAS
# =============================================================================
n_runs = 10
best_assignment = None
best_obj = float('inf')

for run in range(1, n_runs + 1):
    print(f"\n========== INICIO DE LA CORRIDA {run} ==========")
    # Reinicializamos la asignación y los contadores para cada corrida
    assignment_run = original_assignment.copy()
    current_counts_run = initial_counts.copy()
    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1
        current_obj = improved_objective(assignment_run)
        print(f"Iteración {iteration}: Objetivo = {current_obj:.2f}")

        # Obtener las celdas de borde
        border_cells = get_border_cells(assignment_run)
        random.shuffle(border_cells)  # para evitar ciclos deterministas

        # Calcular la complejidad acumulada por sector (usado para ordenar candidatos)
        comp_by_sector = {}
        for cell, sec in assignment_run.items():
            comp = df_cells.loc[df_cells['Cell_Name'] == cell, 'Complexity'].values[0]
            comp_by_sector.setdefault(sec, 0)
            comp_by_sector[sec] += comp

        # (Opcional) Ordenar todos los sectores por complejidad de mayor a menor (no se utiliza luego)
        sorted_sectors = sorted(comp_by_sector.items(), key=lambda item: item[1], reverse=True)

        for cell in border_cells:
            current_sec = assignment_run[cell]
            # Sectores vecinos: sectores de las celdas adyacentes
            candidate_sectors = {assignment_run[neighbor] for neighbor in G.neighbors(cell)}
            candidate_sectors.discard(current_sec)
            # Ordenar los sectores candidatos por su complejidad acumulada (del más bajo al más alto)
            candidate_sectors = sorted(candidate_sectors, key=lambda sec: comp_by_sector.get(sec, 0))

            for cand in candidate_sectors:
                # Verificar conectividad en ambos sectores tras mover la celda
                if not check_move_connectivity(assignment_run, current_sec, cand, cell):
                    continue
                # La celda debe tener al menos un vecino en el sector candidato
                if not any(assignment_run[neighbor] == cand for neighbor in G.neighbors(cell)):
                    continue
                # Verificar la restricción del cambio máximo (opcional, para la diferencia neta de celdas)
                new_count_origin = current_counts_run[current_sec] - 1
                new_count_candidate = current_counts_run[cand] + 1
                if (abs(new_count_origin - initial_counts[current_sec]) > max_change or
                    abs(new_count_candidate - initial_counts[cand]) > max_change):
                    continue

                # Simular la asignación cambiada
                new_assignment = assignment_run.copy()
                new_assignment[cell] = cand

                # Nueva restricción: cada sector (original) solo puede perder (o cambiar) x celdas
                changes = changed_cells_count(new_assignment, original_assignment)
                if any(count > max_change for count in changes.values()):
                    continue

                # Evaluar el impacto en la función objetivo
                new_obj = improved_objective(new_assignment)
                if new_obj < current_obj:
                    print(f"Moviendo celda {cell} de {current_sec} a {cand} mejora el objetivo de {current_obj:.2f} a {new_obj:.2f}")
                    assignment_run = new_assignment
                    current_counts_run[current_sec] = new_count_origin
                    current_counts_run[cand] = new_count_candidate
                    improved = True
                    current_obj = new_obj
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
ax2.legend(handles=patches_opt, loc='upper right', title='Optimized Sectors')
ax2.set_title("Optimized Sectorization")
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
max_distance_nautical = 15  # 20 millas náuticas

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






