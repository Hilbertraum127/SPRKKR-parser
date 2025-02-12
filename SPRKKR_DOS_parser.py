import numpy as np

import tkinter as tk
from tkinter import filedialog

# Use Tkinter to allow interactive selection of the SCF ALAT data file
root = tk.Tk()
root.withdraw()  # Hide the extra Tkinter window
fname = filedialog.askopenfilename(initialdir=".", title="Select a file")

if fname:
    print(f"Selected file: {fname}")
else:
    print("No file selected. Exiting.")
    exit()

# Full read and safe data to data_parsed
# Read the file line by line
# This allows for parsing any similar SPR-KKR DOS file
with open(fname, "r") as file:
    data_file = file.readlines()  # Store all lines of the file in a list
# Deleting file reference to keep namespace clean
del file

print("Reading lines from file and prepare parsing...")

# Find the line where the DOS data starts, marked by 'DOS-FMT:  OLD-SPRKKR'
# This determines where the header ends and raw data begins
index = next(i for i, line in enumerate(data_file) if 'DOS-FMT:  OLD-SPRKKR' in line)

# Split the file into header and raw data
# Header contains metadata; raw data holds the numerical DOS information
data_header, data_raw = data_file[:index], data_file[index+1:]
# Deleting index as it is no longer needed
del index

# Extract the number of energy points (NE) from the header
# Look for the line starting with 'NE' and extract the number that follows
NE = int(next(line.split()[1] for line in data_header if line.startswith("NE")))
print(f"Number of energy points (NE): {NE}")

# Calculate the number of lines per block in the raw data
# This helps in identifying the structure of the data blocks
block_lines = len(data_raw) // NE

# Parse all blocks and store the data in a 2D NumPy array
# Each block corresponds to one energy point, with multiple associated values
data_parsed = np.array([
    [
        float(line[i:i+10].strip())  # Extract fixed-width values (10 characters per value)
        for line in data_raw[block_start:block_start + block_lines]  # Loop through lines in the block
        for i in range(0, len(line), 10)  # Extract values in chunks of 10 characters
        if line[i:i+10].strip()  # Ignore empty or whitespace-only chunks
    ]
    for block_start in range(0, len(data_raw), block_lines)  # Iterate through all blocks
])
# Deleting block_lines as it is no longer needed
del block_lines

# Extract all data from header for further parsing
# Define a helper function to extract values from the header based on keywords
def extract_header_value(keyword, data_type):
    line = next((line for line in data_header if keyword in line), None)
    if line:
        return data_type(line.split()[1])
    return None

# Extract specific values from the header
NQ_eff = extract_header_value('NQ_eff', int)
print("NQ_eff:", NQ_eff, "--> Number of different atoms")

NT_eff = extract_header_value('NT_eff', int)
print("NT_eff:", NT_eff, "--> Number of different elements")

EFERMI = extract_header_value('EFERMI', float)
print("EFERMI:", EFERMI, "--> Fermi energy")

IREL = extract_header_value('IREL', int)
print("IREL:", IREL, "--> ???")

# Locate lines containing 'IQ NLQ' and 'IT TXT_T CONC NAT IQAT'
iq_nlq_line = next((i for i, line in enumerate(data_header) if 'IQ' in line and 'NLQ' in line), None)
it_txt_line = next((i for i, line in enumerate(data_header) if 'IT' in line and 'TXT_T' in line), None)

# Extract lines between 'IQ NLQ' and 'IT TXT_T CONC NAT IQAT' into separate 1D arrays
if iq_nlq_line is not None and it_txt_line is not None and iq_nlq_line < it_txt_line:
    iq_data = []
    nlq_data = []
    for line in data_header[iq_nlq_line + 1:it_txt_line]:
        if line.strip():  # Ignore empty lines
            iq, nlq = map(int, line.split())  # Split and convert to integers
            iq_data.append(iq)
            nlq_data.append(nlq)
    IQ = np.array(iq_data, dtype=int)
    NLQ = np.array(nlq_data, dtype=int)
    # Map orbital quantum numbers to their respective labels (s, p, d, etc.)
    BANDS = {1: 's', 2: 'p', 3: 'd', 4: 'f', 5: 'g', 6: 'h', 7: 'i'}
    print("IQ:", IQ, "--> IDs of different atoms")
    if (IQ[-1] / NQ_eff) == 1:
        print("IQ_eff and IQ are consistent")
    else:
        print("Please check, IQ_eff and IQ are not consistent!")
    print("NLQ:", NLQ, "--> Number of bands for atoms IQ")
    print("Highest band is:", BANDS[NLQ[-1]])
else:
    IQ = NLQ = None
    print("Error! Unable to extract IQ and NLQ data...")
# Deleting temporary variables to keep namespace clean
del line, iq_nlq_line, iq, iq_data, nlq, nlq_data

# Extract lines between 'IT TXT_T CONC NAT IQAT' and end of header into structured arrays
if it_txt_line is not None:
    it_txt_data = [
        line.split()  # Split each line into components
        for line in data_header[it_txt_line + 1:]  # Lines from IT TXT_T onward
        if line.strip()  # Ignore empty lines
    ]

    # Parse components into separate arrays
    IT = np.array([int(entry[0]) for entry in it_txt_data], dtype=int)
    TXT_T = np.array([entry[1] for entry in it_txt_data], dtype=str)
    CONC = np.array([float(entry[2]) for entry in it_txt_data], dtype=float)
    NAT = np.array([int(entry[3]) for entry in it_txt_data], dtype=int)
    IQAT = np.array([int(entry[4]) for entry in it_txt_data], dtype=int)

    print("IT:", IT, "--> IDs of different elements (including same site)")
    print("TXT_T:", TXT_T, "--> Types of different elements (including same site)")
    print("CONC:", CONC, "--> Concentrations of different elements (including same site)")
    print("NAT:", NAT, "--> ???")
    print("IQAT:", IQAT, "--> Atom ID of different elements")
else:
    IT = TXT_T = CONC = NAT = IQAT = None
    print("Unable to extract IT, TXT_T, CONC, NAT, IQAT data.")
# Deleting temporary variables to keep namespace clean
del it_txt_data, it_txt_line

# Check if reading DOS data and header infos are consistent
if (np.size(data_parsed, 1) - 2 - IT[-1] * NLQ[-1] * 2) == 0:
    print('DOS data and header infos are consistent. Start parsing...')
else:
    print('Please check data, seems to be wrong. Parsing will be broken...')

# Create header as a NumPy array
print("Creating header: Reading elements of atoms... \n")
header_data = []

# Add initial columns "E" and "???"
header_data.append("E")
header_data.append("???")

for n in range(np.size(IQAT)):
    print("Element number: ", n + 1)
    # Find index of corresponding NLQ for the current IQAT
    index_NLQ = np.where(IQ == IQAT[n])[0][0]
    print("Atom number: ", IT[index_NLQ])
    print("Lmax:", NLQ[index_NLQ])

    for m in range(2):  # Loop for spin states (up and down)
        spin = "up" if m == 0 else "dn"

        for l in range(NLQ[index_NLQ]):  # Loop over angular momentum quantum numbers
            label = f"{TXT_T[n]} {BANDS[l+1]} {spin}"
            print(label)  # this string should go into the header array
            header_data.append(label)

    if n == np.size(IQAT) - 1:
        print("\nDone...")
    else:
        print("Next element... \n")

HEADER = np.array(header_data, dtype=str)

# Deleting temporary variables to keep namespace clean
del header_data, index_NLQ, l, m, n, spin, label

# Shift energy values relative to Fermi energy
data_parsed[:, 0] = data_parsed[:, 0] - EFERMI

# Helper function to get valid input from the user for unit conversion
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        print(f"Invalid input. Please choose one of the following: {', '.join(valid_options)}")

# Prompt user for preferred energy units (Ry or eV)
opt_unit = get_valid_input("\nProceed in units of Ry or eV? (Ry/eV): ", ["Ry", "eV"])
if opt_unit == "eV":
    conversion_factor = 13.605693122994  # Conversion factor from Ry to eV
    print("Converting energies and DOS to units of eV.")
else:
    conversion_factor = 1.0  # No conversion needed
    print("Keeping energies and DOS in units of Ry.")

# Apply unit conversion to energy values and DOS data
data_parsed[:, 0] = data_parsed[:, 0] * conversion_factor
data_parsed[:, 2:] = data_parsed[:, 2:] / conversion_factor
EFERMI = EFERMI * conversion_factor

# Deleting temporary variables to keep namespace clean
del conversion_factor

# Ask user if concentration correction should be applied
opt_conc = get_valid_input(
    "\nDo you want to apply concentration correction for element-resolved DOS? (y/n): ",
    ["y", "n"]
)

if opt_conc == "y":
    print("Applying concentration correction to element-resolved DOS...")
    
    # Apply concentration correction to DOS
    for n in range(np.size(IT)):
        conc_factor = CONC[n]  # Concentration of the element at this site
        print(f"Element {TXT_T[n]} with concentration {conc_factor}")

        # Find all columns in HEADER that correspond to the current element
        relevant_columns = [
            idx for idx, label in enumerate(HEADER)
            if label.startswith(TXT_T[n] + " ")  # Ensure exact match by checking prefix
        ]

        if not relevant_columns:
            print(f"Warning: No columns found for element {TXT_T[n]}")
            continue

        # Apply correction factor to the relevant columns
        data_parsed[:, relevant_columns] *= conc_factor
        print(f"Corrected columns for {TXT_T[n]}: {relevant_columns}")
    
    print("Concentration correction applied successfully.")
else:
    print("Concentration correction skipped.")

# Deleting temporary variables to keep namespace clean
del conc_factor, n, relevant_columns


# Step 1: Initialize the `data_tot` array with the energy column
data_tot = data_parsed[:, [0]]  # First column: energy

# Step 2: Create the new header `HEADER_tot`
HEADER_tot = ["E"]  # Start with "E" for energy

# Step 3: Process the elements and their bands (split by spin)
for n in range(len(TXT_T)):
    # Find the corresponding NLQ value and starting index for the current element
    index_NLQ = np.where(IQ == IQAT[n])[0][0]  # Match the atom ID (IQAT) with IQ
    lmax = NLQ[index_NLQ]  # Maximum number of bands (orbitals) for the current atom
    del index_NLQ
    
    # Initialize lists to store indices of spin-up and spin-down columns for this element
    spin_up_columns = []
    spin_dn_columns = []
    for l in range(lmax):
        # Locate the indices of spin-up and spin-down columns in the HEADER
        up_col = np.where(HEADER == f"{TXT_T[n]} {BANDS[l+1]} up")[0][0]
        dn_col = np.where(HEADER == f"{TXT_T[n]} {BANDS[l+1]} dn")[0][0]
        spin_up_columns.append(up_col)  # Append spin-up column index
        spin_dn_columns.append(dn_col)  # Append spin-down column index
    del l, up_col, dn_col, lmax

    # Compute the sum over all bands for each spin
    spin_up_sum = np.sum(data_parsed[:, spin_up_columns], axis=1)  # Sum spin-up columns
    spin_dn_sum = np.sum(data_parsed[:, spin_dn_columns], axis=1)  # Sum spin-down columns
    total_sum = spin_up_sum + spin_dn_sum  # Total DOS for this element

    # Add the computed spin-resolved and total columns to `data_tot`
    data_tot = np.column_stack((data_tot, spin_up_sum, spin_dn_sum, total_sum))
    del spin_up_sum, spin_dn_sum, total_sum, spin_dn_columns, spin_up_columns

    # Add header entries for the current element
    HEADER_tot.append(f"{TXT_T[n]} up")  # Spin-up entry for this element
    HEADER_tot.append(f"{TXT_T[n]} dn")  # Spin-down entry for this element
    HEADER_tot.append(f"{TXT_T[n]} tot")  # Total entry for this element
del n

# Step 4: Process atoms (summed DOS per atom)
# Create dictionaries to map atoms to their total, up, and dn contributions
atom_totals = {}
atom_up_totals = {}
atom_dn_totals = {}

for n in range(len(TXT_T)):
    # Extract the atom ID from IQAT[n]
    atom_id = IQAT[n]
    
    # Find the column indices for the current element in HEADER_tot
    up_col_index = HEADER_tot.index(f"{TXT_T[n]} up")
    dn_col_index = HEADER_tot.index(f"{TXT_T[n]} dn")
    tot_col_index = HEADER_tot.index(f"{TXT_T[n]} tot")
    
    # Initialize atom totals if not already present
    if atom_id not in atom_totals:
        atom_totals[atom_id] = np.zeros(data_tot.shape[0])
        atom_up_totals[atom_id] = np.zeros(data_tot.shape[0])
        atom_dn_totals[atom_id] = np.zeros(data_tot.shape[0])
    
    # Accumulate contributions to the atom totals
    atom_up_totals[atom_id] += data_tot[:, up_col_index]
    atom_dn_totals[atom_id] += data_tot[:, dn_col_index]
    atom_totals[atom_id] += data_tot[:, tot_col_index]
    del up_col_index, dn_col_index, tot_col_index

# Append atom contributions to `data_tot` and `HEADER_tot`
for atom_id in sorted(atom_totals.keys()):
    # Append total, up, and dn contributions for each atom
    data_tot = np.column_stack((data_tot, atom_up_totals[atom_id], atom_dn_totals[atom_id], atom_totals[atom_id]))
    HEADER_tot.append(f"ATOM_{atom_id} up")
    HEADER_tot.append(f"ATOM_{atom_id} dn")
    HEADER_tot.append(f"ATOM_{atom_id} tot")
del atom_totals, atom_up_totals, atom_dn_totals, atom_id

# Step 5: Verify totals by integrating up to the Fermi energy
print("\n--- Integration Results for Total DOS (up to Fermi Energy) ---")

# Process elements
for n in range(len(TXT_T)):
    # Find the column index of the `tot` column for the current element
    tot_col_index = HEADER_tot.index(f"{TXT_T[n]} tot")
    
    # Extract energy and total DOS for the current element
    energy = data_tot[:, 0]
    tot_dos = data_tot[:, tot_col_index]
    del tot_col_index
    
    # Integrate the DOS up to the Fermi energy (EFERMI)
    below_fermi_indices = energy <= 0.0
    integral = np.trapz(tot_dos[below_fermi_indices], energy[below_fermi_indices])
    del below_fermi_indices
    
    # Calculate pure atom properties
    pure_atom_property = integral / CONC[n]

    # Print the result for the current element
    print(f"Element {TXT_T[n]}: Total electrons (up to EFERMI = {EFERMI:.2f} eV) = {integral:.4f}, "
      f"pure atom = {pure_atom_property:.4f}")
    del integral, tot_dos, pure_atom_property
del n

# Process atoms
print("\n--- Integration Results for Atoms (up to Fermi Energy) ---")

for atom_id in range(1, max(IQAT) + 1):
    # Locate the `ATOM_X tot` column
    atom_tot_col_name = f"ATOM_{atom_id} tot"
    if atom_tot_col_name in HEADER_tot:
        atom_tot_col_index = HEADER_tot.index(atom_tot_col_name)
        del atom_tot_col_name
        # Extract energy and total DOS for the current atom
        atom_tot_dos = data_tot[:, atom_tot_col_index]
        del atom_tot_col_index
        
        # Integrate the DOS up to the Fermi energy (EFERMI)
        below_fermi_indices = energy <= 0.0
        atom_integral = np.trapz(atom_tot_dos[below_fermi_indices], energy[below_fermi_indices])
        del below_fermi_indices
        
        # Print the result for the current atom
        print(f"Atom {atom_id}: Total electrons (up to EFERMI = {EFERMI:.2f}) = {atom_integral:.4f} ")
        del atom_integral, atom_tot_dos
    else:
        print(f"Atom {atom_id}: No 'ATOM_{atom_id} tot' column found in HEADER_tot.")
del atom_id, energy

print("\nIntegration complete (simple Trapz, only for fast check). Verify the results above.")

# Export data into an dat file
# Determine suffix for concentration option
# If opt_conc is "y", append "_abs_conc" to filenames; otherwise, no additional suffix
conc_suffix = "_abs_conc" if opt_conc == "y" else ""

# Construct filenames dynamically
# For `data_parsed`, use the "bands" keyword
fname_bands = f"{fname}_bands_{opt_unit}{conc_suffix}.dat"

# For `data_tot`, use the "tot" keyword
fname_tot = f"{fname}_tot_{opt_unit}{conc_suffix}.dat"

# Export `data_parsed` to a file
# The file will contain comma-separated values (CSV format) and include a header row
np.savetxt(
    fname_bands,           # File name for the output
    data_parsed,           # Data to export (2D NumPy array)
    delimiter=",",         # Use a comma as the separator for CSV format
    header=",".join(HEADER),  # Convert the list of headers into a comma-separated string
    comments=""            # Ensure no "#" is prepended to the header (default behavior of savetxt)
)
print(f"Exported data_parsed to {fname_bands}")

# Export `data_tot` to a file
# This file will also be in CSV format and include a header row
np.savetxt(
    fname_tot,             # File name for the output
    data_tot,              # Data to export (2D NumPy array)
    delimiter=",",         # Use a comma as the separator for CSV format
    header=",".join(HEADER_tot),  # Convert the list of headers into a comma-separated string
    comments=""            # Ensure no "#" is prepended to the header
)
print(f"Exported data_tot to {fname_tot}")


del conc_suffix, fname_bands, fname_tot

