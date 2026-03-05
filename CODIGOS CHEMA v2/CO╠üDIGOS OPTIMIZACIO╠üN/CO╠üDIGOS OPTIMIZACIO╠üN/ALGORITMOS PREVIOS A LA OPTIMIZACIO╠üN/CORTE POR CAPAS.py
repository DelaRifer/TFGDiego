# import os
# import pandas as pd

# # —————— Rutas de entrada ——————
# PATH_DATA_TRAIN       = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\1. bloque prediccion\\datos\\ACC Madrid Norte\\Data_Train\\'
# PATH_DATA_CRIDA       = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\1. bloque prediccion\\datos\\Data_CRIDA\\'
# PATH_CLUSTERIZADOS    = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\1. bloque prediccion\\datos\\ACC Madrid Norte\\CLUSTERIZADOS MADRID NORTE\\'

# # —————— Rutas de salida (LOWER) ——————
# PATH_DATA_TRAIN_CAPAS   = os.path.join(PATH_DATA_TRAIN,    'UPPER')
# PATH_DATA_CRIDA_CAPAS  = os.path.join(PATH_DATA_CRIDA,    'UPPER')
# PATH_CLUSTER_CAPAS      = os.path.join(PATH_CLUSTERIZADOS, 'UPPER')

# import os
# import pandas as pd


# # —————— Mapeo de entrada → salida ——————
# dirs = [
#     (PATH_DATA_TRAIN,    PATH_DATA_TRAIN_CAPAS),
#     (PATH_DATA_CRIDA,    PATH_DATA_CRIDA_CAPAS),
#     (PATH_CLUSTERIZADOS, PATH_CLUSTER_CAPAS),
# ]

# for src_dir, dst_dir in dirs:
#     for fname in os.listdir(src_dir):
#         if not fname.lower().endswith(('.csv', '.xls', '.xlsx')):
#             continue

#         src_path = os.path.join(src_dir, fname)
#         # 1) Leer (auto-detección de separador en CSV)
#         try:
#             if fname.lower().endswith('.csv'):
#                 df = pd.read_csv(src_path, sep=None, engine='python')
#             else:
#                 df = pd.read_excel(src_path)
#         except Exception as e:
#             print(f"⚠️ fallo al leer {src_path}: {e}")
#             continue

#         # 2) Filtrar filas con modoCIN < 245 si existe la columna
#         if 'modoCIN' in df.columns:
#             df = df[df['modoCIN'] >= 245]

#         # 3) Guardar con mismo nombre (CSV y PKL)
#         base, _ = os.path.splitext(fname)
#         try:
#             df.to_csv (os.path.join(dst_dir, base + '.csv'), index=False, sep=';')
#             df.to_pickle(os.path.join(dst_dir, base + '.pkl'))
#         except Exception as e:
#             print(f"⚠️ fallo al guardar {base}: {e}")
            
            


# import pandas as pd
# import os

# # Ruta al fichero Excel
# path = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

# # 1) Leer el Excel
# df = pd.read_excel(path)

# # 2) Filtrar: eliminar los sectores indeseados
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU'}
# if 'Sector' in df.columns:
#     df = df[~df['Sector'].isin(sectores_a_eliminar)]

# # 3) Sobrescribir el mismo Excel
# df.to_excel(path, index=False)

# print(f"Fichero filtrado y guardado en: {path}")


# csv_path = os.path.splitext(path)[0] + '.csv'
# df.to_csv(csv_path, index=False, sep=';')
# print(f"También guardado CSV en: {csv_path}")


# import pandas as pd
# import os

# # Ruta al fichero Excel
# path = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

# # 1) Leer el Excel
# df = pd.read_excel(path)

# # 2) Filtrar: eliminar los sectores indeseados
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU'}
# if 'Sector' in df.columns:
#     df = df[~df['Sector'].isin(sectores_a_eliminar)]

# # 3) Sobrescribir el mismo Excel
# df.to_excel(path, index=False)
# print(f"✔ Filtrado y guardado en: {path}")

# # 4) (Opcional) Mantener la versión CSV en el mismo directorio
# csv_path = os.path.splitext(path)[0] + '.csv'
# df.to_csv(csv_path, index=False, sep=';')
# print(f"✔ También guardado CSV en: {csv_path}")

# import pandas as pd
# import os

# # Ruta al fichero Excel
# path = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

# # 1) Leer el Excel
# df = pd.read_excel(path)

# # 2) Filtrar: eliminar los sectores indeseados
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU'}
# if 'Sector' in df.columns:
#     df = df[~df['Sector'].isin(sectores_a_eliminar)]

# # 3) Sobrescribir el mismo Excel
# df.to_excel(path, index=False)
# print(f"✔ Filtrado y guardado en: {path}")

# # 4) (Opcional) Mantener la versión CSV en el mismo directorio
# csv_path = os.path.splitext(path)[0] + '.csv'
# df.to_csv(csv_path, index=False, sep=';')
# print(f"✔ También guardado CSV en: {csv_path}")


# import pandas as pd
# import ast
# import os

# # 1) Ruta al fichero Excel
# path = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'

# # 2) Sectores a eliminar
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU'}

# # 3) Leer el Excel
# df = pd.read_excel(path, engine='openpyxl')

# # 4) Función auxiliar: filtra lista de flujos y coordenadas
# def filtrar_flujos_y_coords(flujos_raw, coords_raw):
#     # Parsear la cadena de lista a objeto Python, si viene como string
#     flujos = ast.literal_eval(flujos_raw) if isinstance(flujos_raw, str) else list(flujos_raw)
#     coords  = ast.literal_eval(coords_raw)  if isinstance(coords_raw,  str) else list(coords_raw)
    
#     flujos_filt = []
#     coords_filt = []
#     for flujo, coord in zip(flujos, coords):
#         # Extraer el código de sector (p.ej. '4_LECMSAO_CL' → 'LECMSAO')
#         partes = flujo.split('_')
#         sector = partes[1] if len(partes) > 1 else ''
#         if sector in sectores_a_eliminar:
#             # saltar este flujo (y su coordenada)
#             continue
#         flujos_filt.append(flujo)
#         coords_filt.append(coord)
    
#     return flujos_filt, coords_filt

# # 5) Aplicar el filtrado fila a fila
# n_f = []
# n_c = []
# for _, row in df.iterrows():
#     flujos_ok, coords_ok = filtrar_flujos_y_coords(row['Flujos_Clusterizados'],
#                                                    row['Coordinates'])
#     n_f.append(flujos_ok)
#     n_c.append(coords_ok)

# df['Flujos_Clusterizados'] = n_f
# df['Coordinates']           = n_c

# # 6) Sobrescribir el mismo Excel
# df.to_excel(path, index=False)

# # 7) (Opcional) También actualizar CSV paralelo
# csv_path = os.path.splitext(path)[0] + '.csv'
# df.to_csv(csv_path, index=False, sep=';')

# print(f"✔ He limpiado los flujos de los sectores {sectores_a_eliminar}")
# print(f"  → Excel actualizado: {path}")
# print(f"  → CSV actualizado:   {csv_path}")


# import pandas as pd
# import os

# # 1) Ruta al fichero Excel
# path = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\'

# # 2) Leer el Excel en un DataFrame
# df = pd.read_excel(path, engine='openpyxl')

# # 3) Sectores que queremos eliminar
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU'}

# # 4) Filtrar: quitar las filas donde 'Sector' esté en esa lista
# if 'Sector' in df.columns:
#     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]
# else:
#     raise KeyError("La columna 'Sector' no existe en el DataFrame")

# # 5) Sobrescribir el mismo Excel
# df.to_excel(path, index=False)

# # 6) (Opcional) También actualizar la versión CSV en el mismo directorio
# csv_path = os.path.splitext(path)[0] + '.csv'
# df.to_csv(csv_path, index=False, sep=';')

# print(f"✔ Filtrado completo. Excel actualizado: {path}")
# print(f"✔ CSV actualizado: {csv_path}")





# import pandas as pd
# import ast

# # —————— Configuración de rutas ——————
# PATH_ANALISIS = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Resultados analisis flujo celda\\'
# PATH_ENTRADA  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\'

# PATH_SALIDA_1  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma300\\'
# PATH_SALIDA_2  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me300\\'

# PATH_SALIDA_3  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma330\\'
# PATH_SALIDA_4  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me330\\'

# PATH_SALIDA_5  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma360\\'
# PATH_SALIDA_6  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me360\\'

# PATH_SALIDA_7  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma390\\'
# PATH_SALIDA_8  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me390\\'

# PATH_SALIDA_9  = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma420\\'
# PATH_SALIDA_10 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me420\\'

# PATH_SALIDA_11 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma450\\'
# PATH_SALIDA_12 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me450\\'

# PATH_SALIDA_13 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\Ma480\\'
# PATH_SALIDA_14 = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\me480\\'


# # —————— 5) DF_T_REAL_CELDA.pkl ——————
# df = pd.read_pickle(f"{PATH_ENTRADA}\\DF_T_REAL_CELDA.pkl")

# # # 5.1) Eliminar los sectores indeseados
# # if 'Sector' in df.columns:
# #     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]

# # 5.2) Eliminar filas con modoCIN < 345
# if 'modoCIN' in df.columns:
#     df1 = df.loc[df['modoCIN'] > 300]

# # 5.3) Guardar de nuevo en PKL y CSV
# df1.to_pickle(f"{PATH_SALIDA_1}\\DF_T_REAL_CELDA.pkl")
# df1.to_csv   (f"{PATH_SALIDA_1}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # 5.2) Eliminar filas con modoCIN < 345
# if 'modoCIN' in df.columns:
#     df2 = df.loc[df['modoCIN'] <= 300]

# # 5.3) Guardar de nuevo en PKL y CSV
# df2.to_pickle(f"{PATH_SALIDA_2}\\DF_T_REAL_CELDA.pkl")
# df2.to_csv   (f"{PATH_SALIDA_2}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # 5.2) Eliminar filas con modoCIN < 345
# if 'modoCIN' in df.columns:
#     df3 = df.loc[df['modoCIN'] > 330]

# # 5.3) Guardar de nuevo en PKL y CSV
# df3.to_pickle(f"{PATH_SALIDA_3}\\DF_T_REAL_CELDA.pkl")
# df3.to_csv   (f"{PATH_SALIDA_3}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # —————— Continuación de cortes y guardado ——————

# # Corte modoCIN <= 330
# if 'modoCIN' in df.columns:
#     df4 = df.loc[df['modoCIN'] <= 330]
# df4.to_pickle(f"{PATH_SALIDA_4}\\DF_T_REAL_CELDA.pkl")
# df4.to_csv   (f"{PATH_SALIDA_4}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN > 360
# if 'modoCIN' in df.columns:
#     df5 = df.loc[df['modoCIN'] > 360]
# df5.to_pickle(f"{PATH_SALIDA_5}\\DF_T_REAL_CELDA.pkl")
# df5.to_csv   (f"{PATH_SALIDA_5}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN <= 360
# if 'modoCIN' in df.columns:
#     df6 = df.loc[df['modoCIN'] <= 360]
# df6.to_pickle(f"{PATH_SALIDA_6}\\DF_T_REAL_CELDA.pkl")
# df6.to_csv   (f"{PATH_SALIDA_6}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN > 390
# if 'modoCIN' in df.columns:
#     df7 = df.loc[df['modoCIN'] > 390]
# df7.to_pickle(f"{PATH_SALIDA_7}\\DF_T_REAL_CELDA.pkl")
# df7.to_csv   (f"{PATH_SALIDA_7}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN <= 390
# if 'modoCIN' in df.columns:
#     df8 = df.loc[df['modoCIN'] <= 390]
# df8.to_pickle(f"{PATH_SALIDA_8}\\DF_T_REAL_CELDA.pkl")
# df8.to_csv   (f"{PATH_SALIDA_8}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN > 420
# if 'modoCIN' in df.columns:
#     df9 = df.loc[df['modoCIN'] > 420]
# df9.to_pickle(f"{PATH_SALIDA_9}\\DF_T_REAL_CELDA.pkl")
# df9.to_csv   (f"{PATH_SALIDA_9}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN <= 420
# if 'modoCIN' in df.columns:
#     df10 = df.loc[df['modoCIN'] <= 420]
# df10.to_pickle(f"{PATH_SALIDA_10}\\DF_T_REAL_CELDA.pkl")
# df10.to_csv   (f"{PATH_SALIDA_10}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN > 450
# if 'modoCIN' in df.columns:
#     df11 = df.loc[df['modoCIN'] > 450]
# df11.to_pickle(f"{PATH_SALIDA_11}\\DF_T_REAL_CELDA.pkl")
# df11.to_csv   (f"{PATH_SALIDA_11}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN <= 450
# if 'modoCIN' in df.columns:
#     df12 = df.loc[df['modoCIN'] <= 450]
# df12.to_pickle(f"{PATH_SALIDA_12}\\DF_T_REAL_CELDA.pkl")
# df12.to_csv   (f"{PATH_SALIDA_12}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN > 480
# if 'modoCIN' in df.columns:
#     df13 = df.loc[df['modoCIN'] > 480]
# df13.to_pickle(f"{PATH_SALIDA_13}\\DF_T_REAL_CELDA.pkl")
# df13.to_csv   (f"{PATH_SALIDA_13}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# # Corte modoCIN <= 480
# if 'modoCIN' in df.columns:
#     df14 = df.loc[df['modoCIN'] <= 480]
# df14.to_pickle(f"{PATH_SALIDA_14}\\DF_T_REAL_CELDA.pkl")
# df14.to_csv   (f"{PATH_SALIDA_14}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# print("✔ Todos los cortes de modoCIN (300, 330, 360, 390, 420, 450, 480) guardados en sus respectivas carpetas.")


# print("✔ DF_T_REAL_CELDA: PKL y CSV actualizados (sectores y modoCIN filtrados)")



import os
import pandas as pd

# Rutas base
BASE_PATH = 'C:\\Users\\Jose Maria A. Lopez\\Desktop\\Chema\\3. bloque optimizacion\\Datos de entrada eCOMMET\\'
SUBFOLDER = 'DF_T_REAL_CELDA'
PATH_ENTRADA = f"{BASE_PATH}\\entrada\\"  # Asegúrate de tener la ruta correcta para el archivo de entrada

# Cargar el dataframe
df = pd.read_pickle(f"{PATH_ENTRADA}\\DF_T_REAL_CELDA.pkl")

# Crear cortes de 10 en 10 desde modoCIN = 300 hasta 400
niveles = list(range(305, 405, 10))

for nivel in niveles:
    upper_df = df[df['modoCIN'] > nivel]
    lower_df = df[df['modoCIN'] <= nivel]

    # Formatear nombres de carpeta (ej: Ma300, me300, Ma310, me310, ...)
    path_upper = os.path.join(BASE_PATH, f"Ma20x20{nivel}")
    path_lower = os.path.join(BASE_PATH, f"me20x20{nivel}")

    # Crear carpetas si no existen
    os.makedirs(path_upper, exist_ok=True)
    os.makedirs(path_lower, exist_ok=True)

    # Guardar upper_df
    upper_df.to_pickle(os.path.join(path_upper, f'{SUBFOLDER}.pkl'))
    upper_df.to_csv(os.path.join(path_upper, f'{SUBFOLDER}.csv'), index=False, sep=';')

    # Guardar lower_df
    lower_df.to_pickle(os.path.join(path_lower, f'{SUBFOLDER}.pkl'))
    lower_df.to_csv(os.path.join(path_lower, f'{SUBFOLDER}.csv'), index=False, sep=';')

print("✔ Cortes de modoCIN cada 10 unidades entre 300 y 400 guardados correctamente.")



# # 5) DF_T_REAL_CELDA.pkl
# df = pd.read_pickle(f"{PATH_ENTRADA}\\DF_T_REAL_CELDA.pkl")

# # 5.1) (opcional) Eliminar sectores indeseados
# # if 'Sector' in df.columns:
# #     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]

# # # 5.2) Filtrar filas con modoCIN > 345
# # if 'modoCIN' in df.columns:
# #     df = df.loc[df['modoCIN'] > 345]

# # 5.2 bis) Filtrar vuelos entre FL 300 y FL 400
# # — Ajusta 'FlightLevel' al nombre de tu columna; puede ser 'FL', 'Nivel', etc.
# if 'modoCIN' in df.columns:
#     df = df.loc[
#         (df['modoCIN'] >= 300) &
#         (df['modoCIN'] <= 400)
#     ]
# # Si tu columna almacena cadenas como "FL300", conviértelas a int así:
# # if 'FlightLevel' in df.columns and df['FlightLevel'].dtype == object:
# #     df['FlightLevel'] = df['FlightLevel'].str.replace('FL', '').astype(int)
# #     df = df.loc[df['FlightLevel'].between(300, 400)]

# # 5.3) Guardar de nuevo en PKL y CSV
# df.to_pickle(f"{PATH_ENTRADA}\\DF_T_REAL_CELDA.pkl")
# df.to_csv   (f"{PATH_ENTRADA}\\DF_T_REAL_CELDA.csv", index=False, sep=';')

# print("✔ DF_T_REAL_CELDA: PKL y CSV actualizados (sectores, modoCIN y FL 300–400 filtrados)")




# —————— Sectores a eliminar ——————
# sectores_a_eliminar = {'LECMPAU', 'LECMBLU', 'LECMDGU','LECMASU'}

# # —————— 1) dataset_celdas_por_flujo.pkl ——————
# df = pd.read_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo.pkl")
# if 'Sector' in df.columns:
#     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]
# df.to_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo.pkl")
# df.to_csv   (f"{PATH_ANALISIS}\\dataset_celdas_por_flujo.csv", index=False, sep=';')
# print("✔ dataset_celdas_por_flujo: PKL y CSV actualizados")

# # —————— 2) dataset_celdas_por_flujo_completo.pkl ——————
# df = pd.read_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo.pkl")
# if 'Sector' in df.columns:
#     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]
# df.to_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo.pkl")
# df.to_csv   (f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo.csv", index=False, sep=';')
# print("✔ dataset_celdas_por_flujo_completo: PKL y CSV actualizados")

# # —————— 3) dataset_celdas_por_flujo_completo_exp.pkl ——————
# df = pd.read_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo_exp.pkl")
# if 'Sector' in df.columns:
#     df = df.loc[~df['Sector'].isin(sectores_a_eliminar)]
# df.to_pickle(f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo_exp.pkl")
# df.to_csv   (f"{PATH_ANALISIS}\\dataset_celdas_por_flujo_completo_exp.csv", index=False, sep=';')
# print("✔ dataset_celdas_por_flujo_completo_exp: PKL y CSV actualizados")

# # —————— 4) dataset_flujos_por_celda.pkl ——————
# def filtrar_flujos_y_coords(flujos_raw, coords_raw):
#     flujos = ast.literal_eval(flujos_raw) if isinstance(flujos_raw, str) else list(flujos_raw)
#     coords  = ast.literal_eval(coords_raw)  if isinstance(coords_raw,  str) else list(coords_raw)
#     flujos_f, coords_f = [], []
#     for flujo, coord in zip(flujos, coords):
#         partes = flujo.split('_')
#         sector = partes[1] if len(partes) > 1 else ''
#         if sector in sectores_a_eliminar:
#             continue
#         flujos_f.append(flujo)
#         coords_f.append(coord)
#     return flujos_f, coords_f

# df = pd.read_pickle(f"{PATH_ANALISIS}\\dataset_flujos_por_celda.pkl")
# n_f, n_c = [], []
# for _, row in df.iterrows():
#     f_ok, c_ok = filtrar_flujos_y_coords(row['Flujos_Clusterizados'], row['Coordinates'])
#     n_f.append(f_ok)
#     n_c.append(c_ok)
# df['Flujos_Clusterizados'] = n_f
# df['Coordinates']           = n_c
# df.to_pickle(f"{PATH_ANALISIS}\\dataset_flujos_por_celda.pkl")
# df.to_csv   (f"{PATH_ANALISIS}\\dataset_flujos_por_celda.csv", index=False, sep=';')
# print("✔ dataset_flujos_por_celda: PKL y CSV actualizados")
