import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# ==========================================
# 1. UI SETUP
# ==========================================
st.set_page_config(page_title="Wing SHM Platform", layout="wide")
st.title("🛩️ Structural Health Monitoring (SHM) Dashboard")
st.markdown("Upload your ANSYS `.txt` node exports to detect structural damage using Modified Laplacian operators.")

# Sidebar for File Uploads
st.sidebar.header("Upload Data")
healthy_file = st.sidebar.file_uploader("Upload Healthy File (.txt)", type=['txt', 'csv'])
damaged_file = st.sidebar.file_uploader("Upload Damaged File (.txt)", type=['txt', 'csv'])

# Control Panel for user adjustments
st.sidebar.header("Analysis Settings")
resolution = st.sidebar.slider("Grid Resolution (Higher = slower but smoother)", 10, 5000, 300)
z_threshold = st.sidebar.slider("Z-Score Threshold (Filters noise)", 1.0, 5.0, 3.0)
blur_amount = st.sidebar.slider("Visual Blur (Sigma)", 0.0, 5.0, 1.5)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================

def load_and_clean(file):
    """Reads the ANSYS file, cleans headers, and extracts the top skin (Z=0)."""
    df = pd.read_csv(file, sep='\t')
    df.columns = df.columns.str.strip() # Remove spaces from ANSYS headers
    df_top = df[df['Z Location (m)'] == 0] # Filter for surface only
    return df_top

def create_dynamic_grid(df, res_x):
    """Automatically sizes the matrix grid based on the wing's actual dimensions."""
    x_min, x_max = df['X Location (m)'].min(), df['X Location (m)'].max()
    y_min, y_max = df['Y Location (m)'].min(), df['Y Location (m)'].max()
    
    # Calculate aspect ratio so the wing isn't stretched
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    res_y = int(res_x / aspect_ratio)
    
    # Create the blank canvas
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, res_x), y_min:y_max:complex(0, res_y)]
    return grid_x, grid_y

def calculate_curvature_energy(W_matrix):
    """The Math: Calculates (d^2w/dx^2)^2 + (d^2w/dy^2)^2 using finite differences."""
    # First derivative (Slope)
    slope_y, slope_x = np.gradient(W_matrix)
    
    # Second derivative (Curvature)
    _, d2w_dx2 = np.gradient(slope_x)
    d2w_dy2, _ = np.gradient(slope_y)
    
    # Strain Energy Proxy
    energy = (d2w_dx2**2) + (d2w_dy2**2)
    return energy

# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

if healthy_file and damaged_file:
    if st.button("Run Damage Analysis"):
        with st.spinner("Processing finite differences..."):
            
            # Step A: Load Data
            h_df = load_and_clean(healthy_file)
            d_df = load_and_clean(damaged_file)
            
            # Step B: Create Grid & Interpolate
            grid_x, grid_y = create_dynamic_grid(h_df, resolution)
            
            w_h = griddata((h_df['X Location (m)'], h_df['Y Location (m)']), 
                           h_df['Total Deformation (m)'], (grid_x, grid_y), method='cubic')
            w_d = griddata((d_df['X Location (m)'], d_df['Y Location (m)']), 
                           d_df['Total Deformation (m)'], (grid_x, grid_y), method='cubic')
            
            # Mask tracking: Remember where the wing 'exists' before we replace NaNs
            wing_mask = ~np.isnan(w_h)
            
            # Replace NaNs with 0 so math doesn't break
            w_h = np.nan_to_num(w_h)
            w_d = np.nan_to_num(w_d)
            
            # Step C: The Physics (Curvature)
            energy_h = calculate_curvature_energy(w_h)
            energy_d = calculate_curvature_energy(w_d)
            
            # Absolute Difference in Curvature Energy
            energy_diff = np.abs(energy_d - energy_h)
            
            # Step D: Filtering (Z-Score & Blur)
            # Calculate Z-score only on the actual wing surface, ignoring empty space
            mean_e = np.mean(energy_diff[wing_mask])
            std_e = np.std(energy_diff[wing_mask])
            z_score_matrix = (energy_diff - mean_e) / std_e
            
            # Apply mathematical threshold (Kill the noise)
            z_score_matrix[z_score_matrix < z_threshold] = 0
            
            # Apply visual filter (Gaussian glow)
            clean_heatmap = ndimage.gaussian_filter(z_score_matrix, sigma=blur_amount)
            
            # Force everything outside the wing boundary to be transparent/zero again
            clean_heatmap[~wing_mask] = np.nan
            
            # Step E: Visualization
            st.write("### Analysis Results")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Draw the heatmap
            contour = ax.contourf(grid_x, grid_y, clean_heatmap, levels=100, cmap='inferno')
            ax.set_title(f"Damage Localization Map (Z-Threshold: {z_threshold})")
            ax.set_xlabel("Span (X)")
            ax.set_ylabel("Chord (Y)")
            fig.colorbar(contour, label="Damage Severity (Filtered Z-Score)")
            
            # Send plot to Streamlit
            st.pyplot(fig)
            
            # Find exact coordinates of maximum damage
            max_val = np.nanmax(clean_heatmap)
            if max_val > 0:
                max_idx = np.unravel_index(np.nanargmax(clean_heatmap), clean_heatmap.shape)
                st.success(f"**Critical Damage Detected at:** X = {grid_x[max_idx]:.3f} m, Y = {grid_y[max_idx]:.3f} m")
            else:
                st.info("No damage detected above the current Z-Score threshold.")