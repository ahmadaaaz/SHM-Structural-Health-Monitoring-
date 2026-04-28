import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# ==========================================
# 1. UI SETUP
# ==========================================
st.set_page_config(page_title="Advanced Wing SHM", layout="wide")
st.title("🛩️ Advanced Damage Localization Platform")

st.sidebar.header("Data Input")
h_file = st.sidebar.file_uploader("Healthy State", type=['txt', 'csv'])
d_file = st.sidebar.file_uploader("Damaged State", type=['txt', 'csv'])

st.sidebar.header("Filtering & Stability")
resolution = st.sidebar.slider("Grid Resolution", 100, 500, 300)
z_thresh = st.sidebar.slider("Z-Score Threshold", 0.5, 10.0, 3.5)
# NEW: Erosion Slider to cut off noisy edges
edge_trim = st.sidebar.slider("Edge Trim (Pixels)", 0, 20, 5)

# ==========================================
# 2. UPDATED FUNCTIONS
# ==========================================

def load_and_project(file):
    """
    Handles 3D files. Projects all nodes to a 2D plane.
    If multiple nodes exist at same X,Y (Top/Bottom skin), it takes the Top (Max Z).
    """
    df = pd.read_csv(file, sep='\t')
    df.columns = df.columns.str.strip()
    
    # Sort by Z and keep the highest node for every X,Y coordinate 
    # (This ensures we only look at the top skin curvature)
    df = df.sort_values('Z Location (m)').drop_duplicates(subset=['X Location (m)', 'Y Location (m)'], keep='last')
    return df

def calculate_full_energy(W, nu=0.33):
    """Full Cornwell Equation with Bending, Torsion, and Poisson effects."""
    dy, dx = np.gradient(W)
    d2y, _ = np.gradient(dy)
    _, d2x = np.gradient(dx)
    _, d2xy = np.gradient(dy) # Cross derivative for torsion
    
    # Equation: (d2x^2 + d2y^2) + 2*nu*(d2x*d2y) + 2*(1-nu)*(d2xy^2)
    return (d2x**2 + d2y**2) + (2 * nu * d2x * d2y) + (2 * (1 - nu) * d2xy**2)

# ==========================================
# 3. ANALYSIS PIPELINE
# ==========================================

if h_file and d_file:
    # 1. Load and Project
    h_df = load_and_project(h_file)
    d_df = load_and_project(d_file)
    
    # 2. Create Dynamic Grid
    x_min, x_max = h_df['X Location (m)'].min(), h_df['X Location (m)'].max()
    y_min, y_max = h_df['Y Location (m)'].min(), h_df['Y Location (m)'].max()
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, resolution), y_min:y_max:complex(0, int(resolution/((x_max-x_min)/(y_max-y_min))))]
    
    # 3. Interpolate
    w_h = griddata((h_df['X Location (m)'], h_df['Y Location (m)']), h_df['Total Deformation (m)'], (grid_x, grid_y), method='cubic')
    w_d = griddata((d_df['X Location (m)'], d_df['Y Location (m)']), d_df['Total Deformation (m)'], (grid_x, grid_y), method='cubic')
    
    # Create Mask of the wing shape
    mask = ~np.isnan(w_h)
    
    # NEW: Erode the mask to remove edge artifacts
    if edge_trim > 0:
        mask = ndimage.binary_erosion(mask, iterations=edge_trim)
    
    # 4. Math Execution
    energy_h = calculate_full_energy(np.nan_to_num(w_h))
    energy_d = calculate_full_energy(np.nan_to_num(w_d))
    diff = np.abs(energy_d - energy_h)
    
    # 5. Z-Score (Only calculate on the masked/eroded area)
    valid_data = diff[mask]
    z_score = (diff - np.mean(valid_data)) / np.std(valid_data)
    z_score[z_score < z_thresh] = 0
    z_score[~mask] = np.nan # Hide anything outside the wing or in the trim zone
    
    # 6. Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(z_score.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='magma', aspect='auto')
    plt.colorbar(im, label="Damage Intensity (Z-Score)")
    ax.set_title("Damage Localization Heatmap (Eroded Boundaries)")
    st.pyplot(fig)
