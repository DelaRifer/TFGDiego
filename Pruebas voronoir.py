import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay

# ---- PUNTOS SEMILLA (cámbialos por los tuyos) ----
puntos = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.2],
    [0.2, 0.6],
])

# (Opcional) Calculas el Voronoi con los mismos puntos (por si lo necesitas luego)
vor = Voronoi(puntos)

# Triangulación de Delaunay (dual del Voronoi)
tri = Delaunay(puntos)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(7, 7))

# Triángulos de Delaunay
ax.triplot(puntos[:, 0], puntos[:, 1], tri.simplices)

# Puntos semilla
ax.plot(puntos[:, 0], puntos[:, 1], "o")

# (Opcional) etiqueta cada semilla con su índice
for i, (x, y) in enumerate(puntos):
    ax.text(x, y, f" {i}", va="center")

ax.set_aspect("equal", adjustable="box")
ax.set_title("Triangulación de Delaunay + puntos semilla")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
