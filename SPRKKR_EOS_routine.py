# -*- coding: utf-8 -*-
"""
Imports and processes data from SPRKKR outputs for EOS calculations
Fits the Birch-Murnaghan equation of state and exports the results Some

Author: David Redka
Date: 2025.01.01
"""

import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as au
import astropy.constants as ac

def birch_murnaghan(volume, e0, v0, b0, b0_prime):
    """
    Birch-Murnaghan equation of state

    Parameters
    ----------
    volume : array-like
        Volume values in Å³
    e0 : float
        Minimum energy at the equilibrium volume in eV
    v0 : float
        Equilibrium volume in Å³
    b0 : float
        Bulk modulus at the equilibrium volume in eV/Å³
    b0_prime : float
        Derivative of the bulk modulus with respect to pressure (dimensionless)

    Returns
    -------
    energy : array-like
        Total energy in eV as a function of volume
    """
    # Strain parameter
    eta = (v0 / volume) ** (2.0 / 3.0)
    # Two separate terms in the Birch-Murnaghan expansion
    term1 = (eta - 1.0) ** 3.0 * b0_prime
    term2 = (eta - 1.0) ** 2.0 * (6.0 - 4.0 * eta)
    # Full energy expression
    energy = e0 + (9.0 * v0 * b0 / 16.0) * (term1 + term2)
    return energy

# Use Tkinter to allow interactive selection of the SCF ALAT data file
root = tk.Tk()
root.withdraw()  # Hide the extra Tkinter window
filename = filedialog.askopenfilename(initialdir=".", title="Select a file")

if filename:
    print(f"Selected file: {filename}")
else:
    print("No file selected. Exiting.")
    exit()

# Load data, skipping the first 11 lines (data begins on line 12)
data = np.genfromtxt(filename, skip_header=11)

# Extract lattice parameters (ALAT) in atomic units
alat_au = data[:, 0]  # [a.u.]

# Convert ALAT from [a.u.] to Å and compute volume in Å³
alat_angstrom = alat_au * ac.a0.to(au.angstrom).value
volumes = alat_angstrom ** 3

# Extract total energy in Ry and convert to eV
energy_ry = data[:, 5]
energy_eV = energy_ry * (1.0 * au.Ry).to(au.eV).value

# Prepare initial guesses for the fit
# E0: approximate minimum energy
# V0: approximate volume at that energy
# B0: initial guess for bulk modulus
# B0_prime: derivative of bulk modulus
initial_guess = [
    np.min(energy_eV),
    volumes[np.argmin(energy_eV)],
    100.0,
    4.0
]

# Perform the curve fit using the Birch-Murnaghan EOS
params, covariance = curve_fit(
    birch_murnaghan,
    volumes,
    energy_eV,
    p0=initial_guess
)

# Unpack the fitted parameters
e0_fit, v0_fit, b0_fit, b0_prime_fit = params

# Compute the equilibrium lattice constant a0 in Å
a0_fit = v0_fit ** (1.0 / 3.0)

# Create a plot for the EOS fit
plt.figure()
plt.title("Birch-Murnaghan EOS fit")
plt.plot(v0_fit, e0_fit, "ro", label="Equilibrium point")  # Equilibrium point
plt.plot(volumes, energy_eV, "k*", label="Data points")    # Raw data

# Generate a range of volumes to plot the fitted EOS
v_array = np.linspace(0.8 * np.min(volumes), 1.5 * np.max(volumes), 200)
eos_fit = birch_murnaghan(v_array, e0_fit, v0_fit, b0_fit, b0_prime_fit)
plt.plot(v_array, eos_fit, label="Fitted EOS")

plt.xlabel("Volume (Å³)")
plt.ylabel("Total Energy (eV)")
plt.legend()
plt.tight_layout()
plt.show()

# Prepare and display final results
b0_gpa = (b0_fit * au.eV / au.angstrom**3).to(au.GPa).value
a0_au = (a0_fit * au.angstrom / ac.a0.to(au.angstrom)).si.value

fit_results = {
    "E0 (eV)": e0_fit,
    "V0 (Å³)": v0_fit,
    "a0 (Å)": a0_fit,
    "a0 (a.u.)": a0_au,
    "B0 (GPa)": b0_gpa,
    "B0_prime": b0_prime_fit
}

print("Fit results from Birch-Murnaghan EOS")
for key, val in fit_results.items():
    print(f"{key}: {val}")

# Save the fit results to a file
with open(filename + "_fit.dat", "w") as f:
    for key, val in fit_results.items():
        f.write(f"{key}: {val}\n")

# Export fitted EOS data
data_to_export_eos = np.column_stack((v_array, eos_fit))
np.savetxt(filename + "_eos.dat", data_to_export_eos)

# Export the original DFT data
data_to_export_points = np.column_stack((volumes, energy_eV))
np.savetxt(filename + "_points.dat", data_to_export_points)
