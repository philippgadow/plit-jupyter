# README for PLIT Variable Plotting Project

## Overview
This project is designed to plot variables from ntuples to train the Prompt Lepton Isolation Tagger (PLIT). The main components are a script to set up the conda environment (`setup_conda.sh`), a Python script (`plotting.py`), and a list of Python package requirements (`requirements.txt`).

### Files
- `setup_conda.sh`: Bash script for setting up the conda environment.
- `plotting.py`: Python script for plotting variables from ntuples.
- `requirements.txt`: List of Python packages required for the project.

## Setting Up the Conda Environment
1. Run `setup_conda.sh` to install Mamba locally and set up the conda environment.
   - Supports macOS (including M1 MacBooks) and Linux.
   - Not supported on other operating systems.

## Installing Python Packages
1. Ensure the conda environment is activated.
2. Install the required Python packages listed in `requirements.txt` using the following command:
   ```
   pip install -r requirements.txt
   ```

## Understanding `plotting.py`
This Python script performs the following functions:
- Imports necessary libraries like `h5py`, `pandas`, `numpy`, `matplotlib`, etc.
- Defines the path to the ntuple files.
- Specifies variables for muons and trackjets, along with their plotting configurations.
- Includes a function `plot()` to plot data for muons and muon tracks.
- In the `main()` function, data from ntuples is loaded, processed, and plotted for both muons and muon tracks.

### Key Features
- Uses `H5Reader` for reading H5 files.
- Dataframes are created for muons and muon tracks.
- Plots are generated for various variables like `pt_track`, `eta_track`, `phi_track`, etc., for both muons and tracks.
- The plots are saved as PNG files.

## Usage
To run the `plotting.py` script, navigate to the directory containing the script and execute:
```
python plotting.py
```

This will process the data from the specified ntuples and generate plots as configured in the script. Ensure that the required data files are accessible at the specified location in the script.

---

Remember to activate the conda environment before running the Python script, and ensure all dependencies listed in `requirements.txt` are installed.
