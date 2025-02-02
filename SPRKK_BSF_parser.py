import numpy as np
import matplotlib.pyplot as plt
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
    
# Load the data using numpy's genfromtxt.
data = np.genfromtxt(filename)

# Extract the x, y, and z values.
x_values = data[:, 0]
y_values = data[:, 1]
z_values = data[:, 2]

# Identify unique x values and count occurrences.
unique_x_values, x_value_counts = np.unique(x_values, return_counts=True)

# Define the standard number of y-values per x-value.
num_y_per_x = x_value_counts[0]

# Calculate the number of extra x rows for duplicated x-values.
extra_x_rows = np.sum((x_value_counts - num_y_per_x) // num_y_per_x)

# Determine the total number of x rows for the 2D grid.
total_x_rows = len(x_value_counts) + extra_x_rows

# Reshape the flat arrays into 2D arrays.
grid_x = np.reshape(x_values, (total_x_rows, num_y_per_x))
grid_y = np.reshape(y_values, (total_x_rows, num_y_per_x))
grid_z = np.reshape(z_values, (total_x_rows, num_y_per_x))

# Output the minimum and maximum spectral intensity values for reference.
print("Minimum spectral intensity:", np.min(grid_z))
print("Maximum spectral intensity:", np.max(grid_z))

# Create a new figure.
plt.figure(figsize=(5, 5))

# Generate a pseudocolor plot (heatmap) of the spectral data.
plt.pcolormesh(grid_x, grid_y, grid_z, cmap='gnuplot2', vmin=0, vmax=30)

# Set y-axis limits.
plt.ylim([-10, 5])
plt.ylabel('$E - E_F$ (eV)')

# Define x-axis tick positions and their corresponding labels.
xtick_positions = [0.0000, 0.5000, 1.0000, 1.5000, 2.3660, 3.0731]
xtick_labels = [r'$\Gamma$', 'X', 'M', 'R', r'$\Gamma$', 'M']

plt.xticks(xtick_positions, xtick_labels)

# Add thin white vertical lines at x-axis tick positions.
for pos in xtick_positions:
    plt.axvline(x=pos, color='black', linestyle='-', linewidth=0.5)  # Thin vertical lines

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Construct the output file name and save the plot.
output_file_path = os.path.splitext(filename)[0] + '.png'
plt.savefig(output_file_path, dpi=500, bbox_inches='tight')

# Show the plot.
plt.show()