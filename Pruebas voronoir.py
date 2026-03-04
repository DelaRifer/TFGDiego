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
for i, (x, y) in enumerate(puntos):
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


# -------------------------------------------------------------------------------------------------
# Voronoi + Delaunay a los puntos semilla
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))

# 1) Dibujar Voronoi (aristas + vértices)
voronoi_plot_2d(
    vor,
    ax=ax,
    show_points=False,     # no dibuja las semillas (las pintamos nosotros)
    show_vertices=True     # muestra los vértices del Voronoi
)

# 2) Dibujar Delaunay (triángulos)
ax.triplot(
    vertices[:, 0], vertices[:, 1],
    tri.simplices,
    linewidth=1.0,
    alpha=0.8
)

# 3) Dibujar semillas (encima de todo)
ax.scatter(
    puntos[:, 0], puntos[:, 1],
    c="red",
    s=35,
    zorder=5
)

# Vértices de Voronoi (forzamos que se vean)
if len(vor.vertices) > 0:
    ax.scatter(vor.vertices[:, 0], vor.vertices[:, 1],
               marker="x", s=60, zorder=6)

    # Ajustar límites usando semillas + vértices
    all_xy = np.vstack([puntos, vor.vertices])
else:
    all_xy = puntos.copy()

xmin, ymin = all_xy.min(axis=0)
xmax, ymax = all_xy.max(axis=0)
dx, dy = xmax - xmin, ymax - ymin
pad = 0.15 * max(dx, dy) if max(dx, dy) > 0 else 1.0

ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)

# Etiquetas de índice
for i, (x, y) in enumerate(puntos):
    ax.text(x, y, f" {i}", va="center", ha="left", fontsize=9, color="red", zorder=6)

ax.set_aspect("equal", adjustable="box")
ax.set_title("Voronoi + Delaunay a los puntos semilla")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()