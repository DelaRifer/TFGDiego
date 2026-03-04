import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d

# ---- PUNTOS SEMILLA (cámbialos por los tuyos) ----
puntos = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.2],
    [0.2, 0.6],
])

# Calculas el Voronoi con los mismos puntos 
vor = Voronoi(puntos)
vertices = vor.vertices

#-------------------------------------------------------------------------------------------------------------
# Triangulación de Delaunay (dual del Voronoi)
#-------------------------------------------------------------------------------------------------------------

tri = Delaunay(vor.vertices)
# ---- PLOT ----
fig, ax = plt.subplots(figsize=(7, 7))

# Triángulos de Delaunay
ax.triplot(vertices[:, 0], vertices[:, 1], tri.simplices)

# Puntos semilla
ax.plot(puntos[:, 0], puntos[:, 1], "o", color="red")  # círculos rojos

#Vertices del Voronoi
ax.plot(vertices[:, 0], vertices[:, 1], "o", label="Vértices Voronoi")

# (Opcional) etiqueta cada semilla con su índice
for i, (x, y) in enumerate(vertices):
    ax.text(x, y, f" {i}", va="center")

ax.set_aspect("equal", adjustable="box")
ax.set_title("Triangulación de Delaunay + puntos semilla")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

#-------------------------------------------------------------------------------------------------------------
# Mallado de Voronoi (aristas + vértices)
#-------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

# Dibuja aristas (equidistancia a 2 semillas) y vértices (equidistancia a 3 semillas)
voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=True)

# Dibuja semillas
ax.plot(puntos[:, 0], puntos[:, 1], "o")

ax.set_aspect("equal", adjustable="box")
ax.set_title("Mallado de Voronoi (aristas + vértices) y puntos semilla")
plt.show()