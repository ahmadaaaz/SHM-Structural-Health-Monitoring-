import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# setting up
st.set_page_config(page_title="SHM", page_icon="https://github.com/ahmadaaaz/SHM-Structural-Health-Monitoring-/blob/1b40487d35657458dd4be5c577a0c1fc529e5b6f/ecf.png")
st.write('# kpuhtyte')

used_method = st.sidebar.selectbox("what method",["1","2","3"])
if used_method == "1":
    st.write('## ok')
elif used_method == "2":
    healthy_file = st.sidebar.file_uploader('Upload Healthy File', type="txt")
    damaged_file = st.sidebar.file_uploader('Upload Damaged File')
    st.write('## maybe')
    st.sidebar.header("2. Grid & Interpolation")
    # Kept resolution reasonable to prevent high-frequency noise artifacts
    resolution = st.sidebar.slider("Grid Resolution (X-axis)", 50, 250, 150)
    interp_method = "linear"
    st.sidebar.header("3. Signal Processing")

    # Pre-smoothing is critical before taking gradients
    pre_smooth = st.sidebar.slider("Pre-Gradient Smoothing (Sigma)", 1.0, 5.0, 2.0, step=0.5)

    # Epsilon prevents dividing by zero in rigid areas (like the wing tip)
    epsilon_pct = st.sidebar.slider("Denominator Epsilon (%)", 0.1, 10.0, 1.0, step=0.5, help="Stabilizes the Damage Index in areas of very low strain energy.")
    def load_and_project(file):
        df = pd.read_csv(file, sep='\t')
        df.columns = df.columns.str.strip()
        return df.sort_values('Z Location (m)').drop_duplicates(subset=['X Location (m)', 'Y Location (m)'], keep='last')
    def compute_strain_energy(w, dx, dy, nu=0.33):
        # First derivatives
        dw_dy, dw_dx = np.gradient(w, dy, dx)
        # Second derivatives
        d2w_dy2, d2w_dydx = np.gradient(dw_dy, dy, dx)
        d2w_dxdy, d2w_dx2 = np.gradient(dw_dx, dy, dx)
        # Strain energy formulation
        energy = (d2w_dx2**2 + d2w_dy2**2) + (2 * nu * d2w_dx2 * d2w_dy2) + (2 * (1 - nu) * d2w_dydx**2)
        return energy

    if healthy_file and damaged_file:
        h_df = load_and_project(healthy_file)
        d_df = load_and_project(damaged_file)
        x_min, x_max = h_df['X Location (m)'].min(), h_df['X Location (m)'].max()
        y_min, y_max = h_df['Y Location (m)'].min(), h_df['Y Location (m)'].max()
        aspect_ratio = (y_max-y_min)/(x_max-x_min)
        res_y = int(aspect_ratio * resolution)
    
        grid_x, grid_y = np.mgrid[x_min:x_max:complex(0, resolution), y_min:y_max:complex(0, res_y)]
        
        dx = (x_max - x_min) / resolution
        dy = (y_max - y_min) / res_y

        w_h = griddata((h_df['X Location (m)'], h_df['Y Location (m)']), h_df['Total Deformation (m)'], (grid_x, grid_y), method=interp_method)
        w_d = griddata((d_df['X Location (m)'], d_df['Y Location (m)']), d_df['Total Deformation (m)'], (grid_x, grid_y), method=interp_method)
        
        mask = ~np.isnan(w_h)
        mask_eroded = ndimage.binary_erosion(mask, iterations=3)

        w_h = np.nan_to_num(w_h)
        w_d = np.nan_to_num(w_d)

        w_h_smooth = ndimage.gaussian_filter(w_h, sigma=pre_smooth)
        w_d_smooth = ndimage.gaussian_filter(w_d, sigma=pre_smooth)

        energy_h = compute_strain_energy(w_h_smooth, dx, dy)
        energy_d = compute_strain_energy(w_d_smooth, dx, dy)

        energy_h_norm = energy_h / np.max(energy_h[mask_eroded])
        energy_d_norm = energy_d / np.max(energy_d[mask_eroded])
        
        epsilon = np.max(energy_h_norm[mask_eroded]) * (epsilon_pct / 100.0)
        #DI = (Ed-Eh)/(Eh+epsilon)
        damage_index = (energy_d_norm - energy_h_norm) / (energy_h_norm + epsilon)
        
        # Apply soft thresholding (ignore negative "healing" values, keep positive damage)
        damage_index[damage_index < 0] = 0
        damage_index[~mask_eroded] = np.nan # Hide outside edges

        #-----------------------    Code Walkthrough    -----------------------
        #st.write(h_df)

        # --- Step 7: Visualization ---
        fig, ax = plt.subplots(figsize=(10, 4))
        # Vmax sets the color scale to the 99th percentile, keeping one massive spike from ruining the colors
        vmax_val = np.nanpercentile(damage_index, 99) 
        im = ax.imshow(damage_index.T, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                       cmap='jet', aspect='auto', vmax=vmax_val)
        plt.colorbar(im, label="Damage Severity")
        ax.set_title("Damage Location Map")
        st.pyplot(fig)
elif used_method == "3":
    h1 = st.sidebar.file_uploader('Upload Healthy File 1', type="txt")
    h2 = st.sidebar.file_uploader('Upload Healthy File 2', type="txt")
    h3 = st.sidebar.file_uploader('Upload Healthy File 3', type="txt")
    d1 = st.sidebar.file_uploader('Upload Damaged File 1', type="txt")
    d2 = st.sidebar.file_uploader('Upload Damaged File 2', type="txt")
    d3 = st.sidebar.file_uploader('Upload Damaged File 3', type="txt")
    st.sidebar.header("2. Grid & Interpolation")
    # Kept resolution reasonable to prevent high-frequency noise artifacts
    resolution = st.sidebar.slider("Grid Resolution (X-axis)", 50, 250, 150)
    interp_method = "linear"
    st.sidebar.header("3. Signal Processing")
    # Pre-smoothing is critical before taking gradients
    pre_smooth = st.sidebar.slider("Pre-Gradient Smoothing (Sigma)", 1.0, 5.0, 2.0, step=0.5)
    # Epsilon prevents dividing by zero in rigid areas (like the wing tip)
    epsilon_pct = st.sidebar.slider("Denominator Epsilon (%)", 0.1, 10.0, 1.0, step=0.5, help="Stabilizes the Damage Index in areas of very low strain energy.")

    def get_damage_index(h_file, d_file, resolution, sigma, eps_p):
        """Computes Normalized Damage Index for a single mode pair."""
        # Load and clean
        df_h = pd.read_csv(h_file, sep='\t').rename(columns=lambda x: x.strip())
        df_d = pd.read_csv(d_file, sep='\t').rename(columns=lambda x: x.strip())
        
        # 2D Projection (Top Skin)
        proj_h = df_h.sort_values('Z Location (m)').drop_duplicates(subset=['X Location (m)', 'Y Location (m)'], keep='last')
        proj_d = df_d.sort_values('Z Location (m)').drop_duplicates(subset=['X Location (m)', 'Y Location (m)'], keep='last')

        # Grid Setup
        x, y = proj_h['X Location (m)'], proj_h['Y Location (m)']
        xi, yi = np.mgrid[x.min():x.max():complex(0, resolution), y.min():y.max():complex(0, int(resolution * (y.max()-y.min())/(x.max()-x.min())))]
        dx, dy = (x.max()-x.min())/resolution, (y.max()-y.min())/yi.shape[1]

        # Interpolation & Smoothing
        wi_h = ndimage.gaussian_filter(np.nan_to_num(griddata((x, y), proj_h['Total Deformation (m)'], (xi, yi), method='linear')), sigma)
        wi_d = ndimage.gaussian_filter(np.nan_to_num(griddata((x, y), proj_d['Total Deformation (m)'], (xi, yi), method='linear')), sigma)
        
        # Equation (15) from Reference: image_8475f4.png
        def energy(w):
            nu = 0.33
            g1y, g1x = np.gradient(w, dy, dx)
            g2y2, g2yx = np.gradient(g1y, dy, dx)
            g2xy, g2x2 = np.gradient(g1x, dy, dx)
            # Full bending + torsion + Poisson coupling
            return (g2x2**2 + g2y2**2) + (2 * nu * g2x2 * g2y2) + (2 * (1 - nu) * g2yx**2)

        e_h, e_d = energy(wi_h), energy(wi_d)
        
        # Local Normalization
        mask = ~np.isnan(griddata((x, y), proj_h['Total Deformation (m)'], (xi, yi), method='linear'))
        mask = ndimage.binary_erosion(mask, iterations=3)
        
        e_h_norm = e_h / np.max(e_h[mask])
        e_d_norm = e_d / np.max(e_d[mask])
        
        epsilon = np.max(e_h_norm[mask]) * (eps_p / 100.0)
        di = (e_d_norm - e_h_norm) / (e_h_norm + epsilon)
        di[di < 0] = 0
        di[~mask] = np.nan
        return di, xi, yi, proj_d
    if all([h1, h2, h3, d1, d2, d3]):
        # Compute for each mode
        di1, xi, yi, raw_d = get_damage_index(h1, d1, resolution, pre_smooth, epsilon_pct)
        di2, _, _, _ = get_damage_index(h2, d2, resolution, pre_smooth, epsilon_pct)
        di3, _, _, _ = get_damage_index(h3, d3, resolution, pre_smooth, epsilon_pct)
        s_p = st.sidebar.selectbox("Seri/Paralel", ["normal", "seri", "paralel","Product (Strict)", "RMS (Balanced)"])
        if s_p is "normal":
            di_total = ((di1/np.nanmax(di1)) + (di2/np.nanmax(di2)) + (di3/np.nanmax(di3)))
        elif s_p is "paralel":
            di_total = (1/(di1/np.nanmax(di1)) + 1/(di2/np.nanmax(di2)) + 1/(di3/np.nanmax(di3)))**-1
        elif s_p is "seri":
            di_total = ((di1/np.nanmax(di1)) + (di2/np.nanmax(di2)) + (di3/np.nanmax(di3)))/ ((di1/np.nanmax(di1)) * (di2/np.nanmax(di2)) * (di3/np.nanmax(di3)))
        elif s_p is "Product (Strict)":
            di_total = di_total / np.nanmax(di_total)
        elif s_p is "RMS (Balanced)":
            di_total = np.sqrt((di1_c**2 + di2_c**2 + di3_c**2) / 3)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        vmax_val = np.nanpercentile(di_total, 99) # Look at the 99th
        heat_color = st.selectbox("Choose color", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
        im = ax.imshow(di_total.T, origin='lower', extent=[xi.min(), xi.max(), yi.min(), yi.max()], cmap=heat_color, vmax=vmax_val)
        plt.colorbar(im, label="Fused Intensity")
        st.pyplot(fig)
    else:
        st.write("Please upload 6 mode files (healthy + Damaged) to perform Multi-Mode Fusion.")
    st.write('## not ok')
