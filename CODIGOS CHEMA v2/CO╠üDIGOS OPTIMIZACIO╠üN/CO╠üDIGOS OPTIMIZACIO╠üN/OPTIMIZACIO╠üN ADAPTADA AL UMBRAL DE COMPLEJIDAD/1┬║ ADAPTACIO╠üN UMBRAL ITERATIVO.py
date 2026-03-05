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

# 0) Tu diccionario de configuraciones, por número de sectores:
configuraciones = {
    1: ('CNF1A', ['LECMR1I']),
    2: ('CNF2A', ['LECMDPI', 'LECMSAB']),
    3: ('CNF3A', ['LECMDGI', 'LECMPAI', 'LECMSAB']),
    4: ('CNF4A', ['LECMDGI', 'LECMBLI', 'LECMPAI', 'LECMSAI']),
    5: ('CNF5A', ['LECMASI', 'LECMBLI', 'LECMDGI', 'LECMPAI', 'LECMSAN']),
}



# 1) Función que: dado un list_sectors, ejecuta todo tu flujo
#    devuelve el assignment optimizado y el max_complexity.
def run_optimization_for_sectors(lista_sectores, configuracion_estudio, **paths):
    """
    1) Filtra DF_info_conf según list_sectors.
    2) Construye gdf_cells, gdf_sectors, el grafo G, etc.
    3) Lanza tus corridas de optimización y halla best_assignment.
    4) Calcula la complejidad total por sector tras best_assignment.
    Devuelve: best_assignment, max_complexity_per_sector.
    """

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

    #
    # -------------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------ IMPORTACIÓN DE BASE DE DATOS -------------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------------- #

    # CARGAR BASE DE DATOS DEL MALLADO DEL ACC DE ESTUDIO
    DF_MALLADO = pd.read_pickle(PATH_MALLADO_DATA + 'dataset_flujos_por_celda.pkl')


    # CARGAR BASE DE DATOS DE COMPLEJIDAD DE LOS SECTORES DE LA CONFIGURACIÓN PARA UNA FRANJA TEMPORAL DE ESTUDIO: 2022-06-01 14:00-15:00
    Complejidad_sectores_ = pd.read_pickle(PATH_COMPLEJIDAD_SECTOR + 'Complejidad_por_hora_2022-06-01_06-07.pkl')

    # Renombrar la columna
    Complejidad_sectores_.rename(columns={'Suma_Complejidad_total': 'Valor_Complejidad_Sector'}, inplace=True)

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


    # #PLOTEAR EL ACC
    # fig, sects = plt.subplots()
    # sects.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
    # sects.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
    # sects.set_aspect('equal')
    # plt.xlabel('LONGITUD[º]')
    # plt.ylabel('LATITUD[º]')
    # plt.title('REPRESENTACION DE LOS SECTORES DE ESTUDIO')

    # for index, row in DF_info_conf.iterrows():
    #     poligono = row['Contorno Sector']
    #     x, y = poligono.exterior.xy
    #     sects.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'{row["SECTOR_ID"]}')

    # # Añadir leyenda
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

    # # Guardar la figura
    # nombre_figura_ACC = PATH_COMPLEJIDAD_OPT + 'ACC Madrid Norte - conf 9A2.png'
    # plt.savefig(nombre_figura_ACC, format='png', dpi=300, bbox_inches='tight')

    # # Mostrar figura
    # plt.show()



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

    # fig, ACC = plt.subplots()
    # ACC.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
    # ACC.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
    # ACC.set_aspect('equal')
    # plt.xlabel('LONGITUD[º]')
    # plt.ylabel('LATITUD[º]')
    # plt.title('REPRESENTACION DEL ESPACIO AÉREO DE ESTUDIO - ACC MADRID NORTE')
    # ACC.fill(x, y, zorder=1, edgecolor='black',alpha=0.5, linewidth=1, label=f'LECMCTAN')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

    # # Guardar la figura
    # nombre_figura_espacio_ACC = PATH_COMPLEJIDAD_OPT + 'Espacio aéreo de estudio - ACC Madrid Norte.png'
    # plt.savefig(nombre_figura_espacio_ACC, format='png', dpi=300, bbox_inches='tight')

    # # Mostrar figura
    # plt.show()


    # -------------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------- REPRESENTACIÓN DEL MALLADO DEL ACC ---------------------------------------- #
    # -------------------------------------------------------------------------------------------------------------------- #

    # fig, ax_cells = plt.subplots()

    # # Dibujar el polígono del ACC
    # x, y = poligono_ACC.exterior.xy
    # ax_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

    # # Dibujar las celdas del mallado
    # for _, row in DF_MALLADO.iterrows():
    #     polygon = row['Polygon']  # Obtener el polígono
    #     x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno
    #     ax_cells.plot(x, y, color='gray', alpha=0.5)  # Dibujar el contorno de la celda

    # # Configurar la gráfica
    # ax_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
    # ax_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
    # ax_cells.set_aspect('equal')

    # ax_cells.set_title("AIRSPACE MESHING WITH 30NM x 30NM CELLS")
    # ax_cells.set_aspect('equal')
    # ax_cells.set_xlabel('LONGITUDE[º]')
    # ax_cells.set_ylabel('LATITUDE[º]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

    # # Guardar la figura
    # nombre_figura_mallado_ACC = PATH_COMPLEJIDAD_OPT + 'Mallado del espacio aéreo de estudio - ACC Madrid Norte.png'
    # plt.savefig(nombre_figura_mallado_ACC, format='png', dpi=300, bbox_inches='tight')

    # # Mostrar figura
    # plt.show()



    # -------------------------------------------------------------------------------------------------------------------- #
    # ---------------------------- REPRESENTACIÓN DE LA COMPLEJIDAD DE LOS SECTORES DEL ACC ------------------------------ #
    # -------------------------------------------------------------------------------------------------------------------- #

    # OBTENCIÓN DE UN DATAFRAME CON LA INFORMACIÓN GEOMÉTRICA DE LOS SECTORES Y SU COMPLEJIDAD ASOCIADA
    DF_info_conf_copia = DF_info_conf.copy()
    DF_info_conf_copia.rename(columns={'SECTOR_ID': 'Sector'}, inplace=True)
    DF_COMPLEJIDAD_SECTORES = pd.merge(DF_info_conf_copia[['Sector','Contorno Sector']], Complejidad_sectores, on="Sector", how="left")

    # # GRAFICAR UN MAPA DE COLOR A PARTIR DE LA COMPLEJIDAD DE LOS SECTORES
    # fig, ax_complejidad_sects = plt.subplots(figsize=(12, 8))

    # # Dibujar el polígono del ACC
    # x, y = poligono_ACC.exterior.xy
    # ax_complejidad_sects.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

    # # Lista para almacenar las posiciones de los centroides
    # centroid_positions = []

    # # Dibujar los sectores con colores basados en 'Valor_Complejidad_Sector'
    # for _, row in DF_COMPLEJIDAD_SECTORES.iterrows():
    #     polygon = row['Contorno Sector']  # Obtener el polígono
    #     x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno

    #     # Seleccionar el color basado en el valor de complejidad
    #     valor_complejidad = row['Valor_Complejidad_Sector']

    #     # Escala de colores (ajusta según los datos)
    #     color = plt.cm.viridis((valor_complejidad - DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min()) /
    #                         (DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].max() - DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min()))

    #     # Rellenar el polígono con el color correspondiente
    #     ax_complejidad_sects.fill(x, y, color=color, alpha=0.7)

    #     # Dibujar el contorno del sector
    #     ax_complejidad_sects.plot(x, y, color='black', alpha=0.5)

    #     # Calcular el centroide del polígono para colocar la etiqueta
    #     centroid = polygon.centroid
    #     centroid_x, centroid_y = centroid.x, centroid.y

    #     # Comprobar si el centroide actual está demasiado cerca de algún centroide ya registrado
    #     # Función para calcular la distancia entre dos puntos
    #     def distance(x1, y1, x2, y2):
    #         return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    #     displacement = 0
    #     new_centroid_x, new_centroid_y = centroid_x, centroid_y
    #     while any(distance(new_centroid_x, new_centroid_y, cx, cy) < 0.5 for cx, cy in centroid_positions):
    #         # Si el centroide está demasiado cerca, moverlo un poco (puedes ajustar el valor de 'displacement')
    #         displacement += 0.5  # Desplazar más si hay centroids cercanos
    #         new_centroid_x = centroid_x
    #         new_centroid_y = centroid_y + displacement

    #     # Añadir la nueva posición del centroide a la lista
    #     centroid_positions.append((new_centroid_x, new_centroid_y))

    #     # Colocar el valor de complejidad como etiqueta en el centroide, con desplazamiento
    #     ax_complejidad_sects.text(new_centroid_x, new_centroid_y, f'{valor_complejidad:.2f}', ha='center', va='center', fontsize=10, color='black', fontweight='bold')


    # # Crear la barra de colores (colorbar) para la escala de complejidad
    # sm_sect = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].min(),
    #                                                                 vmax=DF_COMPLEJIDAD_SECTORES['Valor_Complejidad_Sector'].max()))
    # sm_sect.set_array([])  # Se necesita para que funcione el colorbar
    # # Ajustar el tamaño de la barra de colores (colorbar)
    # cbar_sect = fig.colorbar(sm_sect, ax=ax_complejidad_sects, label='Valor Complejidad', shrink=0.8, aspect=10)

    # # Configurar la gráfica
    # ax_complejidad_sects.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
    # ax_complejidad_sects.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
    # ax_complejidad_sects.set_aspect('equal')

    # if tipo_datos == 'predicciones':
    #     ax_complejidad_sects.set_title("COMPLEJIDAD PREDICHA DE LOS SECTORES - MAPA DE COLOR")
    # elif tipo_datos == 'reales':
    #     ax_complejidad_sects.set_title("COMPLEJIDAD REAL DE LOS SECTORES - MAPA DE COLOR")

    # ax_complejidad_sects.set_aspect('equal')
    # ax_complejidad_sects.set_xlabel('LONGITUD[º]')
    # ax_complejidad_sects.set_ylabel('LATITUD[º]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')

    # # Guardar la figura
    # if tipo_datos == 'predicciones':
    #         nombre_figura_complejidad_sectores = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad predicha sectores - ACC Madrid Norte.png'
    # elif tipo_datos == 'reales':
    #     nombre_figura_complejidad_sectores = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad real sectores - ACC Madrid Norte.png'

    # plt.savefig(nombre_figura_complejidad_sectores, format='png', dpi=300, bbox_inches='tight')

    # # Mostrar figura
    # plt.show()



    # -------------------------------------------------------------------------------------------------------------------- #
    # ----------------------------- REPRESENTACIÓN DE LA COMPLEJIDAD DE LAS CELDAS DEL ACC ------------------------------- #
    # -------------------------------------------------------------------------------------------------------------------- #

    # OBTENCIÓN DE UN DATAFRAME CON LA INFORMACIÓN GEOMÉTRICA DE LAS CELDAS Y SU COMPLEJIDAD ASOCIADA
    DF_COMPLEJIDAD_CELDAS = pd.merge(DF_MALLADO[['Cell_Name','Polygon']], Complejidad_celdas, on="Cell_Name", how="left")

    # # GRAFICAR UN MAPA DE COLOR A PARTIR DE LA COMPLEJIDAD DE LAS CELDAS
    # fig, ax_complejidad_cells = plt.subplots(figsize=(12, 8))

    # # Dibujar el polígono del ACC
    # x, y = poligono_ACC.exterior.xy
    # ax_complejidad_cells.plot(x, y, color='black', linewidth=1, label=f'LECMCTAN')

    # # Dibujar las celdas del mallado con colores basados en 'Valor_Complejidad_Celda'
    # for _, row in DF_COMPLEJIDAD_CELDAS.iterrows():
    #     polygon = row['Polygon']  # Obtener el polígono
    #     x, y = polygon.exterior.xy  # Obtener las coordenadas del contorno

    #     # Seleccionar el color basado en el valor de complejidad
    #     valor_complejidad = row['Valor_Complejidad_Celda']

    #     # Escala de colores (ajusta según tus datos)
    #     color = plt.cm.viridis((valor_complejidad - DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min()) /
    #                         (DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].max() - DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min()))

    #     # Rellenar el polígono con el color correspondiente
    #     ax_complejidad_cells.fill(x, y, color=color, alpha=0.7)

    #     # Dibujar el contorno de la celda
    #     ax_complejidad_cells.plot(x, y, color='black', alpha=0.5)

    #     # Calcular el centroide del polígono para colocar la etiqueta
    #     centroid = polygon.centroid
    #     centroid_x, centroid_y = centroid.x, centroid.y

    #     # Colocar el valor de complejidad como etiqueta en el centroide
    #     ax_complejidad_cells.text(centroid_x, centroid_y, f'{valor_complejidad:.2f}', ha='center', va='center',fontsize=10, color='black', fontweight='bold')


    # # Crear la barra de colores (colorbar) para la escala de complejidad
    # sm_cell = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].min(),
    #                                                                 vmax=DF_COMPLEJIDAD_CELDAS['Valor_Complejidad_Celda'].max()))
    # sm_cell.set_array([])  # Se necesita para que funcione el colorbar
    # # Ajustar el tamaño de la barra de colores (colorbar)
    # cbar_cell = fig.colorbar(sm_cell, ax=ax_complejidad_cells, label='Valor Complejidad', shrink=0.8, aspect=10)

    # # Configurar la gráfica
    # ax_complejidad_cells.set_xlim(min_lon, max_lon)  # ajusta los límites en el eje x
    # ax_complejidad_cells.set_ylim(min_lat, max_lat)  # ajusta los límites en el eje y
    # ax_complejidad_cells.set_aspect('equal')

    # ax_complejidad_cells.set_title("COMPLEJIDAD DE LAS CELDAS - MAPA DE COLOR")
    # ax_complejidad_cells.set_aspect('equal')
    # ax_complejidad_cells.set_xlabel('LONGITUD[º]')
    # ax_complejidad_cells.set_ylabel('LATITUD[º]')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize='small')


    # # Guardar la figura
    # if tipo_datos == 'predicciones':
    #     nombre_figura_complejidad_celdas = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad predicha celdas - ACC Madrid Norte.png'
    # elif tipo_datos == 'reales':
    #     nombre_figura_complejidad_celdas = PATH_COMPLEJIDAD_OPT + 'Mapa de color complejidad real celdas - ACC Madrid Norte.png'

    # plt.savefig(nombre_figura_complejidad_celdas, format='png', dpi=300, bbox_inches='tight')

    # # Mostrar figura
    # plt.show()



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

    # def check_move_connectivity(assignment, old_sec, new_sec, cell):
    #     """
    #     Verifica que, tras mover 'cell' de old_sec a new_sec, tanto old_sec como new_sec queden conexos.
    #     """
    #     new_assignment = assignment.copy()
    #     new_assignment[cell] = new_sec
    #     return is_sector_connected(new_assignment, old_sec) and is_sector_connected(new_assignment, new_sec)


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
    n_runs = 10
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




    complexities = {
        sec: sum(df_cells.loc[df_cells['Optimized_Sector']==sec, 'Complexity'])
        for sec in set(best_assignment.values())
    }
    max_comp = max(complexities.values())
    return best_assignment, max_comp, complexities




# 2) Pedimos al usuario el umbral:
umbral = float(input("Introduce el umbral máximo de complejidad por sector: "))
# umbral = 40
resultado_final = None

# 3) Iteramos automáticamente desde 2 hasta 5 sectores
for n_sectores in range(2, 5):
    cfg_name, lista_sectores = configuraciones[n_sectores]
    print(f"\nProbando configuración de {n_sectores} sectores ({cfg_name}): {lista_sectores}")

    assignment, max_comp, all_comps = run_optimization_for_sectors(
        lista_sectores,            # lista de sectores para esta configuración
        cfg_name,                   # nombre de la configuración
        PATH_SECTOR_DATA=PATH_SECTOR_DATA,
        PATH_MALLADO_DATA=PATH_MALLADO_DATA,
        PATH_COMPLEJIDAD_SECTOR=PATH_COMPLEJIDAD_SECTOR,
        PATH_COMPLEJIDAD_CELDA=PATH_COMPLEJIDAD_CELDA,
        PATH_COMPLEJIDAD_OPT=PATH_COMPLEJIDAD_OPT
    )

    print(f" → Máxima complejidad resultante: {max_comp:.2f}")

    if max_comp <= umbral:
        print(f"✔️  Cumple umbral {umbral:.2f}. Detenemos el proceso con esta configuración.")
        resultado_final = {
            'configuracion': cfg_name,
            'lista_sectores': lista_sectores,
            'assignment': assignment,
            'complexities': all_comps
        }
        break
    else:
        print("✗ No cumple, probamos con más sectores…")

# 4) Informar resultado
if resultado_final is None:
    print("⚠️  Ninguna configuración (2–5 sectores) cumple el umbral especificado.")
else:
    print("\nResultado final:")
    print("Configuración elegida:", resultado_final['configuracion'])
    print("Complejidades por sector:", resultado_final['complexities'])



