import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

import tkinter as tk
from tkinter import filedialog
# Use Tkinter to allow interactive selection of the SCF ALAT data file
root = tk.Tk()
root.withdraw()  # Hide the extra Tkinter window
filename = filedialog.askopenfilename(initialdir=".", title="Select a file")

if filename:
    print(f"Selected file: {filename}")
else:
    print("No file selected. Exiting.")
    exit()

# Read data from the file and split into x, y, and z components
data = np.genfromtxt(filename)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Reshape data into 2D arrays for plotting
N = int(np.sqrt(np.size(x)))  # Assume square grid of k-points
xx = np.reshape(x, (N, N))
yy = np.reshape(y, (N, N))
zz = np.reshape(z, (N, N))

# Mirror data along the y-axis and append
zz_fly = np.flipud(zz)
yy_fly = yy
xx_fly = -np.flipud(xx)

xx = np.concatenate([xx_fly, xx], axis=0)
yy = np.concatenate([yy_fly, yy], axis=0)
zz = np.concatenate([zz_fly, zz], axis=0)

# Mirror data along the x-axis and append
zz_fly = np.fliplr(zz)
yy_fly = -np.fliplr(yy)
xx_fly = xx

xx = np.concatenate([xx_fly, xx], axis=1)
yy = np.concatenate([yy_fly, yy], axis=1)
zz = np.concatenate([zz_fly, zz], axis=1)

# Plot the color mesh of the spectral function
plt.figure(figsize=(5, 5))
plt.pcolormesh(xx, yy, zz, cmap='gnuplot2', vmin=0.24, vmax=0.73)
# plt.colorbar()

# Add contour lines for the Fermi surface
contour = plt.contour(xx, yy, zz, levels=np.linspace(0, 0.8, 9), colors='white', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

# Define vertices and edges for the Brillouin zone (BZ) projection in the kx-ky plane
bz_vertices = np.array([
    [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5],  # Outer square
    [0.0, 0.0],  # Center (Γ point)
    [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5],  # Midpoints of edges
])

bz_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Outer square
    (0, 4), (1, 4), (2, 4), (3, 4),  # Diagonals to center (Γ)
    (5, 6), (6, 7), (7, 8), (8, 5)  # Cross edges
]

# Draw the Brillouin zone
for edge in bz_edges:
    x1, y1 = bz_vertices[edge[0]]
    x2, y2 = bz_vertices[edge[1]]
    plt.plot([x1, x2], [y1, y2], color='white', linewidth=0.5, linestyle='--')

# Annotate high-symmetry points in the BZ
symmetry_points = {
    "Γ": (0.0, 0.0),        # Center
    "M": (0.5, 0.5),        # Original
    "X": (0.5, 0.0),        # Original
    "M$_{x}$": (-0.5, 0.5),  # Mirrored along y-axis
    "M$_{y}$": (0.5, -0.5),  # Mirrored along x-axis
    "M$_{xy}$": (-0.5, -0.5),  # Mirrored along both axes
    "X$_{x}$": (-0.5, -0.0),  # Mirrored along both axes
}

# Plot and label symmetry points
for label, (kx, ky) in symmetry_points.items():
    plt.scatter(kx, ky, color='white', s=50)
    # plt.text(kx - 0.075, ky - 0.075, label, color='white', fontsize=10)

# Set plot limits and remove axis ticks and labels for a clean visualization
plt.xlim([-0.6, 0.6])
plt.ylim([-0.6, 0.6])
plt.xticks([])  # Remove x-axis ticks and numbers
plt.yticks([])  # Remove y-axis ticks and numbers
plt.tick_params(left=False, bottom=False)  # Remove tick markers

# Output the minimum and maximum values of the data for reference
print("Minimum spectral intensity:", np.min(zz[:]))
print("Maximum spectral intensity:", np.max(zz[:]))

# Construct the output file name and save the plot.
output_file_path = os.path.splitext(filename)[0] + '.png'
plt.savefig(output_file_path, dpi=500, bbox_inches='tight')

# Show the plot.
plt.show()