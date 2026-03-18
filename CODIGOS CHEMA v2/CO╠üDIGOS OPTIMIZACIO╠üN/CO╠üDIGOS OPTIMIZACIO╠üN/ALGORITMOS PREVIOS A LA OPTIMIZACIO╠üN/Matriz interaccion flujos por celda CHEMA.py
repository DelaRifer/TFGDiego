#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- LIBRERÍAS QUE NECESITA EL CÓDIGO ----------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

import pandas as pd
from datetime import datetime, time
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads as wkt_loads
import ast
import time as tm
from sortedcontainers import SortedDict
import warnings
import joblib
from joblib import dump
from joblib import load
import pickle
import itertools
from shapely.geometry import LineString
from sklearn.preprocessing import LabelEncoder

# DIRECTORIOS
PATH_flujos = "C:\\TFG\\Codigos Chema\\Datos\\3. bloque optimizacion\\3. bloque optimizacion\\Resultados analisis flujo celda\\"
PATH_MATRIZ = "C:\\TFG\\Codigos Chema\\Datos\\3. bloque optimizacion\\3. bloque optimizacion\\Matrices de interaccion de flujos\\"
PATH_MATRIZ_CELDA = "C:\\TFG\\Codigos Chema\\Datos\\3. bloque optimizacion\\3. bloque optimizacion\\Matrices de interaccion de flujos\\matrices de interacción individuales por celda\\"

import time
start_time = time.time()

#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ IMPORTACIÓN DE BASE DE DATOS -------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


DF_celdas_por_flujo = pd.read_csv(PATH_flujos + 'dataset_celdas_por_flujo.csv', sep=";", encoding="latin1", 
                 dtype=None,         # intenta inferir tipos
                 parse_dates=True,   # intenta convertir fechas
                 low_memory=False)

# DATASET FLUJOS CLUSTERIZADOS QUE PASAN POR CADA CELDA
DF_flujos_por_celda = pd.read_csv(PATH_flujos + 'dataset_flujos_por_celda.csv', sep=";", encoding="latin1", 
                 dtype=None,         # intenta inferir tipos
                 parse_dates=True,   # intenta convertir fechas
                 low_memory=False)



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------- OBTENCIÓN DE LA TENDENCIA DEL FLUJO EN CADA CELDA ---------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Se asume que la tendencia del flujo en cada una de las celdas de paso es la misma que la tendencia del flujo en el sector
flow_trend_DF = DF_celdas_por_flujo.copy()



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------- FACTOR DE CRUCE DE FLUJOS POR CELDA ----------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Paso 1: Crear todas las combinaciones posibles de pares de flujos
flujos = DF_celdas_por_flujo['Clave_Flujo'].unique()
combinaciones_flujos = list(itertools.product(flujos, repeat=2))

# Crear el DataFrame para almacenar los resultados: el factor de cruce valdrá 1 si existe cruce entre flujos
puntos_cruce_flujos_DF = pd.DataFrame(combinaciones_flujos, columns=['Flujo_1', 'Flujo_2'])
puntos_cruce_flujos_DF['Flow_Comb'] = puntos_cruce_flujos_DF['Flujo_1'] + '-' + puntos_cruce_flujos_DF['Flujo_2']
puntos_cruce_flujos_DF['Sector_cruce'] = '-'  # Inicializar columna
puntos_cruce_flujos_DF['Cruce_SI(1)/NO(0)'] = 0  # Inicializar la columna de cruce con 0
puntos_cruce_flujos_DF['Coordenadas_Cruce'] = '-'  # Inicializar columna
puntos_cruce_flujos_DF['Celda_Cruce'] = '-'  # Inicializar columna

# Paso 2: Verificar si existe cruce entre los pares de flujos, calcular las coordenadas de dicho `punto y la celda en la que se produce el cruce
for i, row in puntos_cruce_flujos_DF.iterrows():

    # Obtener los datos de los flujos correspondientes
    flujo1 = DF_celdas_por_flujo[DF_celdas_por_flujo['Clave_Flujo'] == row['Flujo_1']]
    flujo2 = DF_celdas_por_flujo[DF_celdas_por_flujo['Clave_Flujo'] == row['Flujo_2']]

    # Crear líneas representativas de los flujos usando las coordenadas del punto inicial y final
    line_flujo1 = wkt_loads(flujo1['Line'].iloc[0])
    line_flujo2 = wkt_loads(flujo2['Line'].iloc[0])


    # print(line_flujo1, line_flujo2)

    # Sector de cada flujo
    sector_flujo1 = flujo1['Sector'].values[0]
    sector_flujo2 = flujo2['Sector'].values[0]

    # Polygon de celda
    DF_flujos_por_celda['Polygon'] = DF_flujos_por_celda['Polygon'].apply(
    lambda x: wkt_loads(x) if isinstance(x, str) else x)

    # print(type(line_flujo1), type(line_flujo2))
    # print(list(line_flujo1.coords))  
    # print(list(line_flujo2.coords))  

    
    
    # Solo se considera que existe cruce entre flujos si ambos flujos están en el mismo sector
    if sector_flujo1 == sector_flujo2:

        # Caso 1: Si las líneas que representan los flujos son iguales
        if line_flujo1.equals(line_flujo2):
            puntos_cruce_flujos_DF.at[i, 'Cruce_SI(1)/NO(0)'] = 1
            puntos_cruce_flujos_DF.at[i, 'Coordenadas_Cruce'] = line_flujo1
            celdas = DF_flujos_por_celda[DF_flujos_por_celda['Polygon'].apply(lambda poly: line_flujo1.intersects(poly))]
            puntos_cruce_flujos_DF.at[i, 'Celda_Cruce'] = celdas['Cell_Name'].tolist()  # Lista de celdas por las que pasa el flujo
            puntos_cruce_flujos_DF.at[i,'Sector_cruce'] = sector_flujo1

        # Caso 2: Si las líneas que representan los flujos se cruzan
        elif line_flujo1.intersects(line_flujo2):
            puntos_cruce_flujos_DF.at[i,'Sector_cruce'] = sector_flujo1
            interseccion = line_flujo1.intersection(line_flujo2)
            if isinstance(interseccion, Point):  # Si es un punto de cruce
                puntos_cruce_flujos_DF.at[i, 'Cruce_SI(1)/NO(0)'] = 1
                puntos_cruce_flujos_DF.at[i, 'Coordenadas_Cruce'] = f"({interseccion.x}, {interseccion.y})"
                # Buscar celdas que contengan el punto o cuyo contorno toque el punto
                celda = DF_flujos_por_celda[DF_flujos_por_celda['Polygon'].apply(lambda poly: poly.contains(interseccion) or poly.touches(interseccion))]
                if not celda.empty:
                    puntos_cruce_flujos_DF.at[i, 'Celda_Cruce'] = celda['Cell_Name'].tolist()


# Emplear una copia del dataframe anterior para futuras operaciones
factor_cruce_DF = puntos_cruce_flujos_DF.copy()


# COMPROBACIÓN GRÁFICA DE LOS PUNTOS DE CRUCE ENTRE FLUJOS

def representar_cruce(row, DF_cells, DF_Flujos):

    """
    Representa gráficamente el cruce de flujos según la información en una fila del DataFrame de resultados.

    Args:
        row (pd.Series): Fila del DataFrame `puntos_cruce_flujos_DF`.
        DF_cells (pd.DataFrame): DataFrame con las celdas y sus polígonos.
        DF_Flujos (pd.DataFrame): DataFrame con las coordenadas de los flujos.
    """

    # Inicializar el gráfico
    fig, ax = plt.subplots()

    min_lat = []
    max_lat = []
    min_lon = []
    max_lon = []

    # Dibujar el mallado y etiquetar las celdas
    for _, celda in DF_cells.iterrows():
        poly = celda['Polygon']
        x, y = poly.exterior.xy
        min_lat.append(min(y))
        max_lat.append(max(y))
        min_lon.append(min(x))
        max_lon.append(max(x))
        ax.plot(x, y, color='gray', alpha=0.5)  # Contorno de la celda

    min_lat = min(min_lat) - 0.5
    max_lat = max(max_lat) + 0.5
    min_lon = min(min_lon) - 0.5
    max_lon = max(max_lon) + 0.5

    # Establecer los límites de los ejes según las coordenadas de las celdas
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Obtener los datos de los flujos correspondientes
    flujo1 = DF_Flujos[DF_Flujos['Clave_Flujo'] == row['Flujo_1']]
    flujo2 = DF_Flujos[DF_Flujos['Clave_Flujo'] == row['Flujo_2']]

    # Crear líneas representativas de los flujos usando las coordenadas del punto inicial y final
    line_flujo1 = wkt_loads(flujo1['Line'].iloc[0])
    line_flujo2 = wkt_loads(flujo2['Line'].iloc[0])

    # Verificar las coordenadas de los flujos
    x1, y1 = line_flujo1.xy
    x2, y2 = line_flujo2.xy
    ax.plot(x1, y1, color='blue', linewidth=2, label=row['Flujo_1'])
    ax.plot(x2, y2, color='red', linewidth=2, label=row['Flujo_2'])

    # PRINT DE LA CELDA DONDE OCURRE EL CRUCE
    if row['Celda_Cruce'] != '-':
        print("Celda(s) donde ocurre el cruce:", row['Celda_Cruce'])
    else:
        print("No se ha identificado ninguna celda de cruce.")

    # Dibujar el punto de cruce, si existe
    if row['Cruce_SI(1)/NO(0)'] == 1 and row['Coordenadas_Cruce'] != '-' and not isinstance(row['Coordenadas_Cruce'], LineString):
        # Convertir coordenadas de cruce a tupla
        coords_cruce = tuple(map(float, row['Coordenadas_Cruce'][1:-1].split(', ')))
        ax.scatter(coords_cruce[0], coords_cruce[1], color='black', label='Punto de Cruce', zorder=1)

        # Resaltar la celda, si existe
        if row['Celda_Cruce'] != '-':
            celdas = row['Celda_Cruce']
            for celda_nombre in celdas:
                celda_poly = DF_cells[DF_cells['Cell_Name'] == celda_nombre].iloc[0]['Polygon']
                x, y = celda_poly.exterior.xy
                ax.fill(x, y, color='yellow', alpha=0.5, label=celda_nombre)
                ax.plot(x, y, color='yellow', linewidth=1, linestyle='-')

    # Dibujar recta de cruce si las rectas son coincidentes
    if row['Cruce_SI(1)/NO(0)'] == 1 and isinstance(row['Coordenadas_Cruce'], LineString):
        celdas = row['Celda_Cruce']
        for celda_nombre in celdas:
            celda_poly = DF_cells[DF_cells['Cell_Name'] == celda_nombre].iloc[0]['Polygon']
            x, y = celda_poly.exterior.xy
            ax.fill(x, y, color='yellow', alpha=0.5, label=celda_nombre)
            ax.plot(x, y, color='yellow', linewidth=1, linestyle='-')
        cruce = line_flujo1
        ax.plot(*cruce.xy, color='orange', linewidth=2, linestyle='-', label="Cruce")

    # Configuración final del gráfico
    ax.set_title('REPRESENTACIÓN DEL CRUCE DE FLUJOS SOBRE EL MALLADO', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlabel('LONGITUD [º]')
    ax.set_ylabel('LATITUD[º]')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize='small')
    plt.subplots_adjust(bottom=0.25)
    plt.show()

# Ejemplo de uso: seleccionar una fila del DataFrame puntos_cruce_flujos_DF
fila_ejemplo = puntos_cruce_flujos_DF.iloc[75]  # Cambiar el índice según la fila deseada
representar_cruce(fila_ejemplo, DF_flujos_por_celda, DF_celdas_por_flujo)



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- FACTOR DE INTERACCIÓN DE FLUJOS ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Creación del DF
Int_Factor_DF = pd.DataFrame(columns=['Flujo_1','Trend_Flujo_1','Flujo_2','Trend_Flujo_2','Flow_Comb','Trend_Comb','Int_Factor'])

# Paso 1: Unir las tendencias al dataset de cruces usando 'flow_trend_DF'
# Crear un diccionario para mapear 'Clave_Flujo' a 'Flow_Trend'
trend_dict = flow_trend_DF.set_index('Clave_Flujo')['Flow_Trend'].to_dict()

# Paso 2: Rellenar Int_Factor_DF con las combinaciones y calcular los factores
rows = [] # Crear una lista para almacenar cada fila como un diccionario
for i, row in factor_cruce_DF.iterrows():
    flujo_1 = row['Flujo_1']
    flujo_2 = row['Flujo_2']

    # Obtener las tendencias de los flujos
    trend_flujo_1 = trend_dict.get(flujo_1, None)
    trend_flujo_2 = trend_dict.get(flujo_2, None)

    # Crear los nombres combinados
    flow_comb = f"{flujo_1}-{flujo_2}"
    trend_comb = f"{trend_flujo_1}-{trend_flujo_2}"

    # Calcular el factor de interacción
    if flujo_1 == flujo_2:
        int_factor = 0.5
    elif trend_comb == 'CRUISE-CRUISE':
        int_factor = 1.0
    elif trend_comb in ['EVOLUTION-CRUISE', 'CRUISE-EVOLUTION']:
        int_factor = 2.0
    elif trend_comb == 'EVOLUTION-EVOLUTION':
        int_factor = 3.0
    else:
        int_factor = 0  # Por si surge una combinación no especificada

    # Crear un diccionario con los valores de la fila
    row_data = {
        'Flujo_1': flujo_1,
        'Trend_Flujo_1': trend_flujo_1,
        'Flujo_2': flujo_2,
        'Trend_Flujo_2': trend_flujo_2,
        'Flow_Comb': flow_comb,
        'Trend_Comb': trend_comb,
        'Int_Factor': int_factor
    }

    # Agregar el diccionario a la lista de filas
    rows.append(row_data)

# Paso 3: Convertir la lista en un DataFrame
Int_Factor_DF = pd.DataFrame(rows)



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------- MATRIZ DE INTERACCIÓN DE FLUJOS ------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# CREAR EL DATAFRAME AUXILIAR PARA LA MATRIZ DE INTERACCIÓN DE FLUJOS. SE AÑADE UNA COLUMNA CON EL VALOR DE LA INTERACCIÓN
DF_interaccion_flujos = pd.merge(Int_Factor_DF, factor_cruce_DF, on=['Flujo_1', 'Flujo_2','Flow_Comb'])
DF_interaccion_flujos['Factor_Interaccion'] = DF_interaccion_flujos['Cruce_SI(1)/NO(0)']*DF_interaccion_flujos['Int_Factor']
DF_interaccion_flujos = DF_interaccion_flujos[['Flujo_1','Trend_Flujo_1','Flujo_2','Trend_Flujo_2','Flow_Comb','Trend_Comb','Factor_Interaccion',
                                               'Celda_Cruce','Sector_cruce','Coordenadas_Cruce']]


# EXPANDIR EL DATAFRAME AUXILIAR DE FACTOR DE INTERACCIÓN DE FLUJOS POR CELDA. SE PASA A TENER UNA ENTRADA PARA CADA INTERACCIÓN DE FLUJOS Y CELDA
# Repetir filas según la longitud de las listas en la columna 'Celda_Cruce'
DF_interaccion_flujos_exploded = DF_interaccion_flujos.loc[DF_interaccion_flujos.index.repeat(DF_interaccion_flujos['Celda_Cruce'].str.len())].reset_index(drop=True)
# Expandir las columnas de listas para distribuir los valores
DF_interaccion_flujos_exploded['Celda_Cruce'] = DF_interaccion_flujos['Celda_Cruce'].explode(ignore_index=True)
# Resetear índices del nuevo DF de predicciones expandido
DF_interaccion_flujos_exploded.reset_index(drop=True, inplace=True)


# GENERAR UNA MATRIZ DE INTERACCIÓN DE FLUJOS PARA CADA CELDA
# Lista con las celdas del mallado
celdas_unicas = list(DF_flujos_por_celda['Cell_Name'].unique())

# Crear un diccionario para almacenar todas las matrices
DIC_matrices_interaccion_flujos_celda = {}

# Bucle para generar una matriz particular para cada celda
for celda in celdas_unicas:

    # Filtrar 'DF_interaccion_flujos_exploded' según la celda correspondiente
    DF_interaccion_flujos_celda = DF_interaccion_flujos_exploded[DF_interaccion_flujos_exploded['Celda_Cruce'] == celda]

    # Generar la lista de flujos que interaccionan en esa celda
    flujos_unicos_celda = sorted(DF_interaccion_flujos_celda['Flujo_1'].unique())

    # Inicializar la matriz vacía con índices y columnas de flujos únicos
    matriz_interaccion_flujos_celda = pd.DataFrame(0.0, index=flujos_unicos_celda, columns=flujos_unicos_celda)
    
    # Llenar la matriz de manera simétrica
    for _, row in DF_interaccion_flujos_celda.iterrows():
        flujo_1 = row['Flujo_1']
        flujo_2 = row['Flujo_2']
        factor_interaccion = row['Factor_Interaccion']

        # Asignar el valor en ambas posiciones para mantener la simetría
        matriz_interaccion_flujos_celda.at[flujo_1, flujo_2] = factor_interaccion
        matriz_interaccion_flujos_celda.at[flujo_2, flujo_1] = factor_interaccion

    # Crear un nombre dinámico para la matriz basado en el nombre de la celda
    nombre_matriz = f"Matriz_{celda}"

    # Guardar el dataframe en el diccionario con el nombre dinámico
    DIC_matrices_interaccion_flujos_celda[nombre_matriz] = matriz_interaccion_flujos_celda



#%%
# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- GUARDADO DE DATAFRAMES RELEVANTES ------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------------- #


# Guardado en formato .csv para visualización externa
flow_trend_DF.to_csv(PATH_MATRIZ + f'\\flow_trend_DF.csv', index=False, sep=';')
factor_cruce_DF.to_csv(PATH_MATRIZ + f'\\factor_cruce_DF.csv', index=False, sep=';')
Int_Factor_DF.to_csv(PATH_MATRIZ + f'\\Int_Factor_DF.csv', index=False, sep=';')
DF_interaccion_flujos.to_csv(PATH_MATRIZ + f'\\Dataset_interacción_flujos.csv', index=False, sep=';')
DF_interaccion_flujos_exploded.to_csv(PATH_MATRIZ + f'\\Dataset_interacción_flujos_exp.csv', index=False, sep=';')


# Guardado en formato .pkl para su empleo en otros códigos
flow_trend_DF.to_pickle(PATH_MATRIZ + f'\\flow_trend_DF.pkl')
factor_cruce_DF.to_pickle(PATH_MATRIZ + f'\\factor_cruce_DF.pkl')
Int_Factor_DF.to_pickle(PATH_MATRIZ + f'\\Int_Factor_DF.pkl')
DF_interaccion_flujos.to_pickle(PATH_MATRIZ + f'\\Dataset_interacción_flujos.pkl')
DF_interaccion_flujos_exploded.to_pickle(PATH_MATRIZ + f'\\Dataset_interacción_flujos_exp.pkl')
# Guardar el diccionario de matrices completo
with open(PATH_MATRIZ + f'\\Diccionario_Matrices_celda.pkl', "wb") as file:
    pickle.dump(DIC_matrices_interaccion_flujos_celda, file)


# Guardar cada matriz por separado
for celda, matriz in DIC_matrices_interaccion_flujos_celda.items():
    # Guardar la matriz como archivo .csv
    matriz.to_csv(PATH_MATRIZ_CELDA + f'\\{celda}.csv', index=False, sep=';')
    # Guardar la matriz como archivo .pkl
    matriz.to_pickle(PATH_MATRIZ_CELDA + f'\\{celda}.pkl')





end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")

# %%
