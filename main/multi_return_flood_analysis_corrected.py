# -*- coding: utf-8 -*-
"""
Multi-Return Period Flood Risk Analysis - Simplified Version
Analysis of hydraulic clearances for different return periods and bulking factors

Changes from original:
1. Analyzes multiple return periods (2, 10, 25, 50, 100, 200, 500 years)
2. Uses uniform distribution of bulking factors (1.05 to 1.5)
3. Removed Strahler order and width/length calculations
4. Simplified tributary analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Import custom functions (keep the ones needed)
from river_network_functions import (
    extract_river_network_glob, 
    river_geometry_giandotti,
    foca_idf,
    snap2stream_mod
)
from plot_utils_functions import (
    plot_network,
    hydrosheds2flowobj, 
    load_bridge_data,
    calculate_hydraulic_risk,
    create_output_geodataframe,
    resample_grid,
    save_results
)

from integrated_analysis_workflow import perform_scientific_analysis_and_export

try:
    import topotoolbox as tt
except ImportError:
    print("Error: TopoToolbox for Python not installed")
    print("Install with: pip install topotoolbox")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d
from shapely.geometry import Point
import geopandas as gpd

# =================================================================
# CONFIGURATION PARAMETERS
# =================================================================

# Input data paths
BASE_PATH = Path(r"C:\Users\PVSLCN.609286\Desktop\FloodNodes v1.0\INPUT DATA")

input_files = {
    'dem': BASE_PATH / "DEM/Tevere_elv.tif",
    'flow_dir': BASE_PATH / "DEM/Tevere_dir.tif", 
    'accumulation': BASE_PATH / "DEM/Tevere_upa.tif"
}

# Multiple return periods analysis
return_periods = [2, 10, 25, 50, 100, 200, 500]

# Updated precipitation data paths for multiple return periods
precip_base_path = BASE_PATH / "extreme rainfall"
precip_files = {}
for tr in return_periods:
    precip_files[tr] = {
        'a_tr': precip_base_path / f"a/a_Tr_{tr}_wgs84.tif",
        'n_tr': precip_base_path / f"n/n_Tr_{tr}_wgs84.tif"
    }

# Hydrological model data (updated path)
hydro_file = BASE_PATH / "CN/S2.tif"  # Changed from S2 to S3

# Bridge data file - Choose one location
bridge_locations = {
    'san_vittorino': {
        'file': Path(r"C:\Users\PVSLCN.609286\Desktop\FloodNodes\SanVittorino rating curves.xlsx"),
        'outlet_coord': [12.75287, 41.942891],  # [lon, lat]
        'output_prefix': 'SanVittorino'
    },
    'tor_sapienza': {
        'file': Path(r"C:\Users\PVSLCN.609286\Desktop\FloodNodes\Tor sapienza rating curves.xlsx"),
        'outlet_coord': [12.58636, 41.91786],  # [lon, lat] 
        'output_prefix': 'TorSapienza'
    },
    'fregrizia': {
        'file': Path(r"C:\Users\PVSLCN.609286\Desktop\FloodNodes\Fregrizia rating curves.xlsx"),
        'outlet_coord': [12.73308, 41.93115],  # [lon, lat]
        'output_prefix': 'Fregrizia'
    }
}

# Select analysis location
LOCATION = 'fregrizia'  # Change this to switch locations
selected_location = bridge_locations[LOCATION]

# Analysis parameters (simplified)
params = {
    'amin_km2': 2.0,           # Minimum drainage area [km²]
    'reach_length_km': 1000,   # Target reach length [km] 
    'mingradient': 1e-4,       # Minimum gradient [-]
    'bulking_factor_range': [1.05, 1.5],  # Uniform distribution range
    'n_bulking_samples': 10    # Number of bulking factor samples
}

print(f"Starting multi-return period flood risk analysis for {selected_location['output_prefix']}")
print("="*80)

# =================================================================
# 1. LOAD AND PREPROCESS DEM DATA (unchanged)
# =================================================================

print("1. Loading and processing DEM data...")

# Load DEM
dem = tt.read_tif(str(input_files['dem']))
dem.z = dem.z.astype(np.float64)
print(f"   DEM loaded: {dem.shape} pixels, cellsize = {dem.cellsize}")

# Calculate flow directions directly from DEM
print("   Computing flow directions from DEM...")
fd = tt.FlowObject(dem)

# Calculate flow accumulation from flow directions
print("   Computing flow accumulation...")
a = fd.flow_accumulation()

print(f"   Flow accumulation range: {np.nanmin(a.z):.0f} - {np.nanmax(a.z):.0f}")

# =================================================================
# 2. DELINEATE DRAINAGE BASIN (unchanged)
# =================================================================

print("\n2. Delineating drainage basin...")

# Convert outlet coordinates to grid indices
outlet_coord = selected_location['outlet_coord']  # [lon, lat]

outlet_col = int((outlet_coord[0] - dem.bounds.left) / dem.cellsize)
outlet_row = int((dem.bounds.top - outlet_coord[1]) / dem.cellsize)

print(f"   Outlet coordinates: {outlet_coord}")
print(f"   Outlet grid indices: row={outlet_row}, col={outlet_col}")

# Create outlet grid and extract drainage basin
outlet_grid = dem.duplicate_with_new_data(np.zeros(dem.shape, dtype=bool))
outlet_grid.z[outlet_row, outlet_col] = True
outlet_indices = np.where(outlet_grid.z.ravel(order='F'))[0]

db = fd.drainagebasins(outlet_indices)
mask = db.duplicate_with_new_data((db.z == 1).astype(bool))

# Calculate basin bounds and crop data
basin_rows, basin_cols = np.where(mask.z)
min_row, max_row = np.min(basin_rows), np.max(basin_rows)
min_col, max_col = np.min(basin_cols), np.max(basin_cols)

left = dem.bounds.left + min_col * dem.cellsize
right = dem.bounds.left + (max_col + 1) * dem.cellsize
top = dem.bounds.top - min_row * dem.cellsize
bottom = dem.bounds.top - (max_row + 1) * dem.cellsize

print(f"   Basin bounds: left={left:.3f}, right={right:.3f}, top={top:.3f}, bottom={bottom:.3f}")

# Crop datasets to basin bounds
dem_crop = dem.crop(left, right, top, bottom, 'coordinate')
dem_crop.z = np.ascontiguousarray(dem_crop.z)

fd_crop = tt.FlowObject(dem_crop)
a_crop = a.crop(left, right, top, bottom, 'coordinate')
mask_crop = mask.crop(left, right, top, bottom, 'coordinate')

# Calculate basin area
lat_center = (dem_crop.bounds.top + dem_crop.bounds.bottom) / 2
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_center))
km_per_deg_lat = 111.32
cell_area_km2 = (dem_crop.cellsize * km_per_deg_lon) * (dem_crop.cellsize * km_per_deg_lat)
basin_area = np.sum(mask_crop.z) * cell_area_km2

print(f"   Basin area: {basin_area:.1f} km²")

# =================================================================
# 3. LOAD BRIDGE DATA (unchanged)
# =================================================================

print("\n3. Loading bridge data...")

bridge_file = selected_location['file']
bridges, lat_all, lon_all = load_bridge_data(str(bridge_file))
print(f"   Loaded {len(bridges)} bridges")

# =================================================================
# 4. EXTRACT RIVER NETWORK (simplified - removed Strahler calculations)
# =================================================================

print("\n4. Extracting river network...")

# Crop drainage basin
db_crop = db.crop(left, right, top, bottom, 'coordinate')
basin_mask = (db_crop.z == 1)

dem_crop.z[db_crop.z != 1] = np.nan
fd_crop = tt.FlowObject(dem_crop)
a_crop = fd_crop.flow_accumulation()
a_crop.z[db_crop.z != 1] = np.nan

amin_cells = 0.50 / cell_area_km2
w_stream = a_crop.duplicate_with_new_data(a_crop.z >= amin_cells)
s = tt.StreamObject(fd_crop, threshold=amin_cells, units='pixels')
s_connected = s.klargestconncomps(k=1)

print(f"   Stream network nodes: {len(s_connected.stream)}")

# =================================================================
# 4.8 - BRIDGE SNAPPING MIGLIORATO (snapping su reticolo principale)
# =================================================================

print("   4.8 Snapping bridges to main stream network (improved method)...")

# Parametri per il snapping migliorato
MAX_SNAP_DISTANCE_CELLS = 20  # Distanza massima di snapping (in celle)
MIN_DRAINAGE_AREA_KM2 = 10.0  # Area di drenaggio minima per reticolo principale

# Trova TUTTI i pixel dove w_stream.z == True
stream_pixel_rows, stream_pixel_cols = np.where(w_stream.z)

# Calcola l'area di drenaggio per ogni pixel di stream
stream_drainage_areas = a_crop.z[stream_pixel_rows, stream_pixel_cols] * cell_area_km2

print(f"   Found {len(stream_pixel_rows)} stream pixels")
print(f"   Drainage area range: {np.nanmin(stream_drainage_areas):.1f} - {np.nanmax(stream_drainage_areas):.1f} km²")

# Filtra solo i pixel del reticolo principale (area > soglia)
main_stream_mask = stream_drainage_areas >= MIN_DRAINAGE_AREA_KM2
main_stream_rows = stream_pixel_rows[main_stream_mask]
main_stream_cols = stream_pixel_cols[main_stream_mask]
main_stream_areas = stream_drainage_areas[main_stream_mask]

print(f"   Main stream pixels (area ≥ {MIN_DRAINAGE_AREA_KM2} km²): {len(main_stream_rows)}")

if len(main_stream_rows) == 0:
    print("   WARNING: No main stream pixels found, reducing threshold...")
    MIN_DRAINAGE_AREA_KM2 = 5.0  # Riduci la soglia
    main_stream_mask = stream_drainage_areas >= MIN_DRAINAGE_AREA_KM2
    main_stream_rows = stream_pixel_rows[main_stream_mask]
    main_stream_cols = stream_pixel_cols[main_stream_mask]
    main_stream_areas = stream_drainage_areas[main_stream_mask]
    print(f"   Main stream pixels (area ≥ {MIN_DRAINAGE_AREA_KM2} km²): {len(main_stream_rows)}")

breaknodes = []
valid_bridges = []
breaknode_lats = []
breaknode_lons = []
snapping_info = []  # Per debug

for i, bridge in enumerate(bridges):
    bridge_lon = bridge['lon']
    bridge_lat = bridge['lat']
    
    # Converti coordinate ponte in coordinate griglia
    bridge_col = int((bridge_lon - dem_crop.bounds.left) / dem_crop.cellsize)
    bridge_row = int((dem_crop.bounds.top - bridge_lat) / dem_crop.cellsize)
    
    # Verifica bounds
    if not (0 <= bridge_row < dem_crop.rows and 0 <= bridge_col < dem_crop.columns):
        print(f"   ✗ {bridge['name']}: outside bounds")
        continue
    
    # Calcola distanze dai pixel del reticolo principale
    distances = np.sqrt((main_stream_rows - bridge_row)**2 + (main_stream_cols - bridge_col)**2)
    
    # Trova pixel entro la distanza massima
    within_distance = distances <= MAX_SNAP_DISTANCE_CELLS
    
    if not np.any(within_distance):
        print(f"   ✗ {bridge['name']}: no main stream within {MAX_SNAP_DISTANCE_CELLS} cells")
        continue
    
    # Tra i pixel entro la distanza massima, scegli quello con maggiore area di drenaggio
    candidate_distances = distances[within_distance]
    candidate_areas = main_stream_areas[within_distance]
    candidate_rows = main_stream_rows[within_distance]
    candidate_cols = main_stream_cols[within_distance]
    
    # Strategia: priorità all'area di drenaggio, ma considera anche la distanza
    # Score = area_normalized - distance_penalty
    normalized_areas = candidate_areas / np.max(candidate_areas)
    normalized_distances = candidate_distances / np.max(candidate_distances)
    scores = normalized_areas - 0.3 * normalized_distances  # peso 0.3 alla distanza
    
    best_idx = np.argmax(scores)
    
    closest_stream_row = candidate_rows[best_idx]
    closest_stream_col = candidate_cols[best_idx]
    snap_distance = candidate_distances[best_idx]
    snap_area = candidate_areas[best_idx]
    
    # Converti in indice lineare (row-major order)
    closest_linear_idx = closest_stream_row * dem_crop.columns + closest_stream_col
    
    # Verifica che sia effettivamente su stream
    if w_stream.z[closest_stream_row, closest_stream_col]:
        # Calcola distanza in metri
        distance_m = snap_distance * cell_area_km2**0.5 * 1000
        
        breaknodes.append(closest_linear_idx)
        valid_bridges.append(bridge)
        
        # Calcola coordinate geografiche del punto snappato
        stream_lon = dem_crop.bounds.left + closest_stream_col * dem_crop.cellsize
        stream_lat = dem_crop.bounds.top - closest_stream_row * dem_crop.cellsize
        breaknode_lons.append(stream_lon)
        breaknode_lats.append(stream_lat)
        
        # Salva info per debug
        snapping_info.append({
            'name': bridge['name'],
            'distance_m': distance_m,
            'drainage_area_km2': snap_area,
            'snap_score': scores[best_idx]
        })
        
        print(f"   ✓ {bridge['name']}: snapped at {distance_m:.1f}m, drainage area {snap_area:.1f} km²")
    else:
        print(f"   ✗ {bridge['name']}: snapping verification failed")

breaknodes = np.array(breaknodes, dtype=np.int32)
print(f"   Successfully snapped {len(valid_bridges)} bridges to main stream network")

# Stampa riassunto snapping per debug
if snapping_info:
    print("\n   SNAPPING SUMMARY:")
    print("   " + "-" * 60)
    for info in snapping_info:
        print(f"   {info['name']:20s}: {info['distance_m']:6.1f}m, {info['drainage_area_km2']:6.1f} km², score={info['snap_score']:.3f}")

# =================================================================
# VERIFICA QUALITÀ SNAPPING
# =================================================================

print("\n   Verifying snapping quality...")

# Calcola l'area di drenaggio per ogni ponte snappato
ponte_drainage_areas = []
for i, bridge in enumerate(valid_bridges):
    node_linear_idx = breaknodes[i]
    node_row = node_linear_idx // dem_crop.columns  # Correzione: row-major order
    node_col = node_linear_idx % dem_crop.columns
    
    drainage_area = a_crop.z[node_row, node_col] * cell_area_km2
    ponte_drainage_areas.append(drainage_area)

ponte_drainage_areas = np.array(ponte_drainage_areas)

print(f"   Snapped bridge drainage areas: {np.min(ponte_drainage_areas):.1f} - {np.max(ponte_drainage_areas):.1f} km²")

# Identifica ponti potenzialmente problematici
problematic_bridges = []
for i, bridge in enumerate(valid_bridges):
    if ponte_drainage_areas[i] < MIN_DRAINAGE_AREA_KM2:
        problematic_bridges.append({
            'name': bridge['name'],
            'area': ponte_drainage_areas[i],
            'index': i
        })

if problematic_bridges:
    print(f"\n   WARNING: {len(problematic_bridges)} bridges may still be on tributaries:")
    for pb in problematic_bridges:
        print(f"   - {pb['name']}: {pb['area']:.1f} km² (< {MIN_DRAINAGE_AREA_KM2} km²)")
else:
    print("   ✓ All bridges successfully snapped to main stream network")

# =================================================================
# OPZIONE: RE-SNAPPING PER PONTI PROBLEMATICI
# =================================================================

if problematic_bridges and len(main_stream_rows) > 0:
    print("\n   Attempting re-snapping for problematic bridges...")
    
    for pb in problematic_bridges:
        bridge_idx = pb['index']
        bridge = valid_bridges[bridge_idx]
        
        print(f"   Re-snapping {bridge['name']}...")
        
        # Coordinate originali del ponte
        bridge_lon = bridge['lon']
        bridge_lat = bridge['lat']
        bridge_col = int((bridge_lon - dem_crop.bounds.left) / dem_crop.cellsize)
        bridge_row = int((dem_crop.bounds.top - bridge_lat) / dem_crop.cellsize)
        
        # Aumenta la distanza di ricerca
        extended_distance = MAX_SNAP_DISTANCE_CELLS * 2
        distances = np.sqrt((main_stream_rows - bridge_row)**2 + (main_stream_cols - bridge_col)**2)
        
        # Trova il pixel con area di drenaggio massima entro la distanza estesa
        within_extended_distance = distances <= extended_distance
        
        if np.any(within_extended_distance):
            candidate_areas = main_stream_areas[within_extended_distance]
            candidate_distances = distances[within_extended_distance]
            candidate_rows = main_stream_rows[within_extended_distance]
            candidate_cols = main_stream_cols[within_extended_distance]
            
            # Priorità assoluta all'area di drenaggio
            best_area_idx = np.argmax(candidate_areas)
            
            new_stream_row = candidate_rows[best_area_idx]
            new_stream_col = candidate_cols[best_area_idx]
            new_distance = candidate_distances[best_area_idx]
            new_area = candidate_areas[best_area_idx]
            
            # Aggiorna breaknode
            new_linear_idx = new_stream_row * dem_crop.columns + new_stream_col
            breaknodes[bridge_idx] = new_linear_idx
            
            # Aggiorna coordinate
            new_stream_lon = dem_crop.bounds.left + new_stream_col * dem_crop.cellsize
            new_stream_lat = dem_crop.bounds.top - new_stream_row * dem_crop.cellsize
            breaknode_lons[bridge_idx] = new_stream_lon
            breaknode_lats[bridge_idx] = new_stream_lat
            
            distance_m = new_distance * cell_area_km2**0.5 * 1000
            print(f"   ✓ Re-snapped {bridge['name']}: {distance_m:.1f}m, drainage area {new_area:.1f} km²")
        else:
            print(f"   ✗ Could not re-snap {bridge['name']}")

print(f"   Final snapping complete: {len(valid_bridges)} bridges")

# =================================================================
# SEZIONE 6 MODIFICATA: LOAD PRECIPITATION DATA FOR ALL RETURN PERIODS (FIX)
# =================================================================

print("\n6. Loading precipitation data for all return periods...")

# Dictionary to store all precipitation grids
precip_grids = {}

for tr in return_periods:
    print(f"   Loading return period {tr} years...")
    
    try:
        # Load IDF parameters
        a_tr_orig = tt.read_tif(str(precip_files[tr]['a_tr']))
        n_tr_orig = tt.read_tif(str(precip_files[tr]['n_tr']))
        
        print(f"     Original sizes - a_tr: {a_tr_orig.shape}, n_tr: {n_tr_orig.shape}")
        
        # Clean unrealistic values
        a_tr_orig.z[a_tr_orig.z > 500] = np.nan
        n_tr_orig.z[n_tr_orig.z > 2] = np.nan
        
        # Check if grids have same size
        if a_tr_orig.shape != n_tr_orig.shape:
            print(f"     WARNING: a_tr and n_tr have different sizes!")
            print(f"     a_tr: {a_tr_orig.shape}, n_tr: {n_tr_orig.shape}")
            
            # Find common bounds
            common_left = max(a_tr_orig.bounds.left, n_tr_orig.bounds.left)
            common_right = min(a_tr_orig.bounds.right, n_tr_orig.bounds.right)
            common_top = min(a_tr_orig.bounds.top, n_tr_orig.bounds.top)
            common_bottom = max(a_tr_orig.bounds.bottom, n_tr_orig.bounds.bottom)
            
            print(f"     Common bounds: left={common_left:.6f}, right={common_right:.6f}")
            print(f"                   top={common_top:.6f}, bottom={common_bottom:.6f}")
            
            # Crop to common bounds first
            a_tr = a_tr_orig.crop(common_left, common_right, common_top, common_bottom, 'coordinate')
            n_tr = n_tr_orig.crop(common_left, common_right, common_top, common_bottom, 'coordinate')
            
            print(f"     After common crop - a_tr: {a_tr.shape}, n_tr: {n_tr.shape}")
        else:
            a_tr = a_tr_orig
            n_tr = n_tr_orig
        
        # Resample to DEM resolution - with explicit size matching
        print(f"     DEM target size: {dem.shape}")
        print(f"     Resampling precipitation grids...")
        
        try:
            a_tr_resampled = resample_grid(a_tr, dem)
            n_tr_resampled = resample_grid(n_tr, dem)
            
            print(f"     Resampled sizes - a_tr: {a_tr_resampled.shape}, n_tr: {n_tr_resampled.shape}")
            
        except Exception as resample_error:
            print(f"     Resampling failed: {resample_error}")
            print(f"     Using direct interpolation method...")
            
            # Alternative resampling method using scipy
            from scipy.interpolate import RegularGridInterpolator
            
            # Create target coordinate arrays matching DEM
            target_lons = np.linspace(dem.bounds.left, dem.bounds.right, dem.columns)
            target_lats = np.linspace(dem.bounds.top, dem.bounds.bottom, dem.rows)
            target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
            
            # Original coordinate arrays
            orig_lons_a = np.linspace(a_tr.bounds.left, a_tr.bounds.right, a_tr.columns)
            orig_lats_a = np.linspace(a_tr.bounds.top, a_tr.bounds.bottom, a_tr.rows)
            
            orig_lons_n = np.linspace(n_tr.bounds.left, n_tr.bounds.right, n_tr.columns)
            orig_lats_n = np.linspace(n_tr.bounds.top, n_tr.bounds.bottom, n_tr.rows)
            
            # Interpolate a_tr
            interp_a = RegularGridInterpolator((orig_lats_a, orig_lons_a), a_tr.z, 
                                              method='linear', bounds_error=False, fill_value=np.nan)
            a_tr_interpolated = interp_a((target_lat_grid, target_lon_grid))
            
            # Interpolate n_tr
            interp_n = RegularGridInterpolator((orig_lats_n, orig_lons_n), n_tr.z, 
                                              method='linear', bounds_error=False, fill_value=np.nan)
            n_tr_interpolated = interp_n((target_lat_grid, target_lon_grid))
            
            # Create new grid objects with DEM properties
            a_tr_resampled = dem.duplicate_with_new_data(a_tr_interpolated)
            n_tr_resampled = dem.duplicate_with_new_data(n_tr_interpolated)
            
            print(f"     Interpolated sizes - a_tr: {a_tr_resampled.shape}, n_tr: {n_tr_resampled.shape}")
        
        # Crop to basin bounds
        print(f"     Cropping to basin bounds...")
        print(f"     Basin bounds: left={left:.6f}, right={right:.6f}, top={top:.6f}, bottom={bottom:.6f}")
        
        a_tr_grid = a_tr_resampled.crop(left, right, top, bottom, 'coordinate')
        n_tr_grid = n_tr_resampled.crop(left, right, top, bottom, 'coordinate')
        
        print(f"     Final cropped sizes - a_tr: {a_tr_grid.shape}, n_tr: {n_tr_grid.shape}")
        print(f"     DEM crop target size: {dem_crop.shape}")
        
        # Force exact dimensions to match dem_crop
        def force_exact_dimensions(grid, target_grid):
            """Force grid to have exact same dimensions as target_grid"""
            target_rows, target_cols = target_grid.shape
            current_rows, current_cols = grid.shape
            
            # Handle row dimension
            if current_rows < target_rows:
                # Add rows by repeating last row
                rows_to_add = target_rows - current_rows
                last_rows = np.tile(grid.z[-1:, :], (rows_to_add, 1))
                grid.z = np.vstack([grid.z, last_rows])
            elif current_rows > target_rows:
                # Remove extra rows
                grid.z = grid.z[:target_rows, :]
            
            # Handle column dimension
            if current_cols < target_cols:
                # Add columns by repeating last column
                cols_to_add = target_cols - current_cols
                last_cols = np.tile(grid.z[:, -1:], (1, cols_to_add))
                grid.z = np.hstack([grid.z, last_cols])
            elif current_cols > target_cols:
                # Remove extra columns
                grid.z = grid.z[:, :target_cols]
            
            # Create new grid with exact target dimensions
            return target_grid.duplicate_with_new_data(grid.z)
        
        # Force exact dimensions
        a_tr_grid = force_exact_dimensions(a_tr_grid, dem_crop)
        n_tr_grid = force_exact_dimensions(n_tr_grid, dem_crop)
        
        print(f"     Dimension-forced sizes - a_tr: {a_tr_grid.shape}, n_tr: {n_tr_grid.shape}")
        
        # Make arrays contiguous
        a_tr_grid.z = np.ascontiguousarray(a_tr_grid.z)
        n_tr_grid.z = np.ascontiguousarray(n_tr_grid.z)
        
        # Mask to drainage basin
        a_tr_grid.z[np.isnan(dem_crop.z)] = np.nan
        n_tr_grid.z[np.isnan(dem_crop.z)] = np.nan
        
        precip_grids[tr] = {
            'a_tr': a_tr_grid,
            'n_tr': n_tr_grid
        }
        
        print(f"     ✓ Successfully loaded Tr={tr}: a range {np.nanmin(a_tr_grid.z):.1f}-{np.nanmax(a_tr_grid.z):.1f}, "
              f"n range {np.nanmin(n_tr_grid.z):.2f}-{np.nanmax(n_tr_grid.z):.2f}")
        
    except Exception as e:
        print(f"     ✗ Error loading return period {tr}: {e}")
        import traceback
        traceback.print_exc()
        precip_grids[tr] = None

print(f"   Successfully loaded {len([tr for tr in return_periods if precip_grids[tr] is not None])} return periods")

# Verifica finale
print("\n   Final verification of precipitation grids:")
for tr in return_periods:
    if precip_grids[tr] is not None:
        a_shape = precip_grids[tr]['a_tr'].shape
        n_shape = precip_grids[tr]['n_tr'].shape
        dem_shape = dem_crop.shape
        print(f"   Tr={tr}: a_tr{a_shape}, n_tr{n_shape}, dem_crop{dem_shape} - {'✓' if a_shape == n_shape == dem_shape else '✗'}")
    else:
        print(f"   Tr={tr}: NOT LOADED")
# =================================================================
# 7. LOAD HYDROLOGICAL MODEL DATA (updated path)
# =================================================================

# =================================================================
# SEZIONE 7 MODIFICATA: LOAD HYDROLOGICAL MODEL DATA (FIX CN)
# =================================================================

print("\n7. Loading SCS curve number data...")

try:
    s_cn_orig = tt.read_tif(str(hydro_file))  # Now using S3.tif
    print(f"   Original CN size: {s_cn_orig.shape}")
    print(f"   CN bounds: left={s_cn_orig.bounds.left:.6f}, right={s_cn_orig.bounds.right:.6f}")
    print(f"               top={s_cn_orig.bounds.top:.6f}, bottom={s_cn_orig.bounds.bottom:.6f}")
    
    # Clean unrealistic values
    s_cn_orig.z[s_cn_orig.z > 10000] = np.nan
    print(f"   CN value range after cleaning: {np.nanmin(s_cn_orig.z):.1f} - {np.nanmax(s_cn_orig.z):.1f}")
    
    # Check compatibility with DEM
    print(f"   DEM size: {dem.shape}")
    print(f"   DEM bounds: left={dem.bounds.left:.6f}, right={dem.bounds.right:.6f}")
    print(f"               top={dem.bounds.top:.6f}, bottom={dem.bounds.bottom:.6f}")
    
    # Resample to DEM resolution with error handling
    try:
        print("   Attempting standard resampling...")
        s_cn_resample = resample_grid(s_cn_orig, dem)
        print(f"   Standard resampling successful: {s_cn_resample.shape}")
        
    except Exception as resample_error:
        print(f"   Standard resampling failed: {resample_error}")
        print(f"   Using alternative interpolation method...")
        
        # Alternative resampling method using scipy
        from scipy.interpolate import RegularGridInterpolator
        
        # Create target coordinate arrays matching DEM
        target_lons = np.linspace(dem.bounds.left, dem.bounds.right, dem.columns)
        target_lats = np.linspace(dem.bounds.top, dem.bounds.bottom, dem.rows)
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
        
        # Original coordinate arrays for CN
        orig_lons_cn = np.linspace(s_cn_orig.bounds.left, s_cn_orig.bounds.right, s_cn_orig.columns)
        orig_lats_cn = np.linspace(s_cn_orig.bounds.top, s_cn_orig.bounds.bottom, s_cn_orig.rows)
        
        # Handle potential bounds issues
        common_left = max(dem.bounds.left, s_cn_orig.bounds.left)
        common_right = min(dem.bounds.right, s_cn_orig.bounds.right)
        common_top = min(dem.bounds.top, s_cn_orig.bounds.top)
        common_bottom = max(dem.bounds.bottom, s_cn_orig.bounds.bottom)
        
        print(f"   Common bounds: left={common_left:.6f}, right={common_right:.6f}")
        print(f"                 top={common_top:.6f}, bottom={common_bottom:.6f}")
        
        # Create mask for valid interpolation area
        valid_mask = ((target_lon_grid >= common_left) & (target_lon_grid <= common_right) & 
                     (target_lat_grid >= common_bottom) & (target_lat_grid <= common_top))
        
        # Interpolate CN data
        try:
            interp_cn = RegularGridInterpolator((orig_lats_cn, orig_lons_cn), s_cn_orig.z, 
                                              method='linear', bounds_error=False, fill_value=np.nan)
            s_cn_interpolated = interp_cn((target_lat_grid, target_lon_grid))
            
            # Apply valid area mask
            s_cn_interpolated[~valid_mask] = np.nan
            
            # Create new grid object with DEM properties
            s_cn_resample = dem.duplicate_with_new_data(s_cn_interpolated)
            
            print(f"   Alternative interpolation successful: {s_cn_resample.shape}")
            
        except Exception as interp_error:
            print(f"   Interpolation also failed: {interp_error}")
            print("   Creating default CN grid...")
            
            # Create default constant CN grid (conservative approach)
            default_cn_value = 70.0  # Conservative CN value for mixed land use
            s_cn_interpolated = np.full(dem.shape, default_cn_value, dtype=np.float64)
            s_cn_resample = dem.duplicate_with_new_data(s_cn_interpolated)
            
            print(f"   Using default CN value: {default_cn_value}")
    
    # Crop to basin bounds
    print(f"   Cropping CN to basin bounds...")
    print(f"   Basin bounds: left={left:.6f}, right={right:.6f}, top={top:.6f}, bottom={bottom:.6f}")
    
    try:
        s_cn_grid = s_cn_resample.crop(left, right, top, bottom, 'coordinate')
        print(f"   Cropped CN size: {s_cn_grid.shape}")
        print(f"   Target dem_crop size: {dem_crop.shape}")
        
    except Exception as crop_error:
        print(f"   Cropping failed: {crop_error}")
        print("   Using direct extraction approach...")
        
        # Direct approach: extract subset based on indices
        # Calculate indices corresponding to basin bounds
        col_start = int((left - dem.bounds.left) / dem.cellsize)
        col_end = int((right - dem.bounds.left) / dem.cellsize)
        row_start = int((dem.bounds.top - top) / dem.cellsize) 
        row_end = int((dem.bounds.top - bottom) / dem.cellsize)
        
        # Ensure indices are within bounds
        col_start = max(0, min(col_start, s_cn_resample.columns-1))
        col_end = max(col_start+1, min(col_end, s_cn_resample.columns))
        row_start = max(0, min(row_start, s_cn_resample.rows-1))
        row_end = max(row_start+1, min(row_end, s_cn_resample.rows))
        
        print(f"   Extraction indices: rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]")
        
        s_cn_subset = s_cn_resample.z[row_start:row_end, col_start:col_end]
        s_cn_grid = dem_crop.duplicate_with_new_data(s_cn_subset)
    
    # Force exact dimensions to match dem_crop
    def force_exact_dimensions_cn(grid, target_grid):
        """Force CN grid to have exact same dimensions as target_grid"""
        target_rows, target_cols = target_grid.shape
        current_rows, current_cols = grid.shape
        
        print(f"   Forcing dimensions from {grid.shape} to {target_grid.shape}")
        
        # Handle row dimension
        if current_rows < target_rows:
            rows_to_add = target_rows - current_rows
            # Use mean of last row for padding
            mean_value = np.nanmean(grid.z[-1, :]) if not np.all(np.isnan(grid.z[-1, :])) else 50.0
            last_rows = np.full((rows_to_add, grid.z.shape[1]), mean_value)
            grid.z = np.vstack([grid.z, last_rows])
        elif current_rows > target_rows:
            grid.z = grid.z[:target_rows, :]
        
        # Handle column dimension
        current_rows, current_cols = grid.z.shape  # Update after row adjustment
        if current_cols < target_cols:
            cols_to_add = target_cols - current_cols
            # Use mean of last column for padding
            mean_value = np.nanmean(grid.z[:, -1]) if not np.all(np.isnan(grid.z[:, -1])) else 50.0
            last_cols = np.full((grid.z.shape[0], cols_to_add), mean_value)
            grid.z = np.hstack([grid.z, last_cols])
        elif current_cols > target_cols:
            grid.z = grid.z[:, :target_cols]
        
        # Create new grid with exact target dimensions
        return target_grid.duplicate_with_new_data(grid.z)
    
    # Force exact dimensions if needed
    if s_cn_grid.shape != dem_crop.shape:
        print(f"   CN grid size mismatch: {s_cn_grid.shape} vs {dem_crop.shape}")
        s_cn_grid = force_exact_dimensions_cn(s_cn_grid, dem_crop)
        print(f"   Dimension-forced CN size: {s_cn_grid.shape}")
    
    # Make array contiguous
    s_cn_grid.z = np.ascontiguousarray(s_cn_grid.z)
    
    # Mask to drainage basin
    s_cn_grid.z[np.isnan(dem_crop.z)] = np.nan
    
    # Final validation and statistics
    valid_cn_values = s_cn_grid.z[~np.isnan(s_cn_grid.z)]
    if len(valid_cn_values) > 0:
        cn_min, cn_max, cn_mean = np.min(valid_cn_values), np.max(valid_cn_values), np.mean(valid_cn_values)
        print(f"   ✓ CN processing successful")
        print(f"   Final CN statistics: min={cn_min:.1f}, mean={cn_mean:.1f}, max={cn_max:.1f}")
        print(f"   Valid CN pixels: {len(valid_cn_values)}/{s_cn_grid.z.size}")
    else:
        print(f"   WARNING: No valid CN values found, using default")
        s_cn_grid.z = np.full(dem_crop.shape, 50.0)  # Default CN value
        cn_min = cn_max = cn_mean = 50.0
        print(f"   Using default CN value: 50.0")
    
    # Data is already in S parameter format (no conversion needed)
    # Just validate and clean S values
    s_values = s_cn_grid.z.copy()
    s_values[s_values < 0] = 5.0  # Minimum reasonable S value
    s_values[s_values > 1000] = 254.0  # Maximum reasonable S value for very impermeable areas
    s_values[np.isinf(s_values)] = 50.0  # Handle any infinity values
    
    s_cn_grid.z = s_values
    
    print(f"   ✓ S parameter validation complete")
    print(f"   SCS S parameter range: {np.nanmin(s_cn_grid.z):.1f} - {np.nanmax(s_cn_grid.z):.1f} mm")

except Exception as e:
    print(f"   ✗ Error loading CN data: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"   Creating fallback CN grid...")
    # Create fallback grid with reasonable CN/S values
    fallback_s_values = np.full(dem_crop.shape, 50.0, dtype=np.float64)  # S = 50mm
    fallback_s_values[np.isnan(dem_crop.z)] = np.nan
    s_cn_grid = dem_crop.duplicate_with_new_data(fallback_s_values)
    
    print(f"   ✓ Fallback CN grid created with S = 50mm")

# Final verification
print(f"\n   Final CN grid verification:")
print(f"   CN grid shape: {s_cn_grid.shape}")
print(f"   DEM crop shape: {dem_crop.shape}")
print(f"   Shapes match: {'✓' if s_cn_grid.shape == dem_crop.shape else '✗'}")
print(f"   SCS S parameter final range: {np.nanmin(s_cn_grid.z):.1f} - {np.nanmax(s_cn_grid.z):.1f} mm")
# =================================================================
# 8. CALCULATE HYDROLOGICAL PARAMETERS (simplified)
# =================================================================

print("\n8. Calculating hydrological parameters...")

n_bridges = len(valid_bridges)
tau_b = np.zeros(n_bridges)  # Concentration time [h]
l_asta = np.zeros(n_bridges)  # Main channel length [km] 
ab = np.zeros(n_bridges)  # Basin area [km²]
hm = np.zeros(n_bridges)  # Mean elevation difference [m]

for i, bridge in enumerate(valid_bridges):
    node_linear_idx = breaknodes[i]
    node_row = node_linear_idx % dem_crop.rows
    node_col = node_linear_idx // dem_crop.rows
    
    try:
        # Calculate drainage basin for this breaknode
        basin = fd_crop.drainagebasins(breaknodes[[i]])
        basin_mask = (basin.z == 1)
        
        # Basin area [km²]
        basin_cells = np.sum(basin_mask)
        ab[i] = basin_cells * cell_area_km2
        
        # Simplified main channel length estimation [km]
        l_asta[i] = ab[i]**0.6  # Empirical relationship
        
        # Mean elevation difference [m]
        basin_elevations = dem_crop.z[basin_mask]
        basin_elevations = basin_elevations[~np.isnan(basin_elevations)]
        
        if len(basin_elevations) > 0:
            outlet_elevation = dem_crop.z[node_row, node_col]
            if not np.isnan(outlet_elevation):
                mean_basin_elevation = np.mean(basin_elevations)
                hm[i] = max(1.0, mean_basin_elevation - outlet_elevation)
            else:
                hm[i] = 100.0
        else:
            hm[i] = 100.0
        
        # Concentration time using Giandotti formula [h]
        tau_b[i] = (4 * np.sqrt(ab[i]) + 1.5 * l_asta[i]) / (0.8 * np.sqrt(hm[i]))
        tau_b[i] = max(0.5, min(tau_b[i], 48.0))  # Cap to reasonable values
        
    except Exception as e:
        print(f"   Error processing {bridge['name']}: {e}")
        tau_b[i] = 2.0
        l_asta[i] = 5.0
        ab[i] = 10.0
        hm[i] = 100.0

print(f"   Concentration times: {np.min(tau_b):.2f} - {np.max(tau_b):.2f} hours")
print(f"   Basin areas: {np.min(ab):.1f} - {np.max(ab):.1f} km²")

# =================================================================
# 9. MULTI-RETURN PERIOD ANALYSIS WITH UNIFORM BULKING FACTORS
# =================================================================

print("\n9. Multi-return period analysis with uniform bulking factors...")

# Generate uniform bulking factor samples
bulking_factors = np.linspace(params['bulking_factor_range'][0], 
                             params['bulking_factor_range'][1], 
                             params['n_bulking_samples'])

print(f"   Bulking factors: {bulking_factors}")
print(f"   Analyzing {len(return_periods)} return periods × {len(bulking_factors)} bulking factors")

# Data structure to store all results
all_results = []

# Bridge-breaknode mapping
ponte_breaknode_mapping = []
for i, bridge in enumerate(valid_bridges):
    ponte_breaknode_mapping.append({
        'ponte_idx': i,
        'breaknode_idx': i,  # Simplified 1:1 mapping
        'ponte_nome': bridge['name']
    })

print("\n   Processing each return period...")

for tr_idx, tr in enumerate(return_periods):
    if precip_grids[tr] is None:
        print(f"   Skipping return period {tr} (data not available)")
        continue
        
    print(f"   Return period {tr} years ({tr_idx+1}/{len(return_periods)})...")
    
    # Calculate design precipitation for this return period
    p_t = np.zeros(n_bridges)
    h_t = np.zeros(n_bridges)
    s_mean = np.zeros(n_bridges)
    
    for i, bridge in enumerate(valid_bridges):
        node_linear_idx = breaknodes[i]
        
        try:
            # Calculate drainage basin
            basin = fd_crop.drainagebasins(np.array([node_linear_idx]))
            basin_mask = (basin.z == 1)
            
            if not np.any(basin_mask):
                p_t[i] = 5.0
                h_t[i] = 10.0
                s_mean[i] = 50.0
                continue
            
            # Extract IDF parameters for basin
            a_values = precip_grids[tr]['a_tr'].z[basin_mask]
            n_values = precip_grids[tr]['n_tr'].z[basin_mask]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(a_values) | np.isnan(n_values))
            if np.any(valid_mask):
                a_values = a_values[valid_mask]
                n_values = n_values[valid_mask]
                
                # Calculate rainfall intensity: h = a * t^n
                h_tr_values = a_values * (tau_b[i] ** n_values)
                h_t[i] = np.mean(h_tr_values)
            else:
                h_t[i] = 10.0
            
            # Extract S parameter for basin
            s_values = s_cn_grid.z[basin_mask]
            s_values = s_values[~(np.isinf(s_values) | np.isnan(s_values))]
            
            if len(s_values) > 0:
                s_mean[i] = np.mean(s_values)
            else:
                s_mean[i] = 50.0
            
            # Calculate runoff using SCS method
            ia = 0.05 * s_mean[i]  # Initial abstraction
            
            if h_t[i] > ia:
                p_t[i] = ((h_t[i] - ia) ** 2) / (h_t[i] - ia + s_mean[i])
            else:
                p_t[i] = 0.02
            
            if np.isnan(p_t[i]) or p_t[i] <= 0:
                p_t[i] = 0.02
                
        except Exception as e:
            h_t[i] = 10.0
            s_mean[i] = 50.0
            p_t[i] = 5.0
    
    # Calculate base flow rates (no bulking)
    q_base = 0.278 * p_t * ab / tau_b
    
    # For each bulking factor, calculate hydraulic clearances
    for bf_idx, bulking_factor in enumerate(bulking_factors):
        print(f"     Bulking factor {bulking_factor:.3f} ({bf_idx+1}/{len(bulking_factors)})...")
        
        # Apply bulking factor
        q_bulked = q_base * bulking_factor
        
        # Calculate hydraulic clearances for each bridge
        for mapping in ponte_breaknode_mapping:
            ponte_idx = mapping['ponte_idx']
            breaknode_idx = mapping['breaknode_idx']
            bridge = valid_bridges[ponte_idx]
            
            Q_bulked_val = q_bulked[breaknode_idx]
            Q_base_val = q_base[breaknode_idx]
            
            # Extract bridge rating curve
            h_vals = bridge['rating_curve'][:, 0]
            Q_vals = bridge['rating_curve'][:, 1]
            
            try:
                # Interpolate rating curve
                f_interp = interp1d(Q_vals, h_vals, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                
                # Calculate hydraulic heights
                h_idraulico_bulked = f_interp(Q_bulked_val)
                h_idraulico_base = f_interp(Q_base_val)
                
                # Calculate hydraulic clearances
                low_deck = bridge.get('low_deck', bridge.get('LowDeck', 0))
                upper_deck = bridge.get('upper_deck', bridge.get('UpperDeck', 0))
                
                franco_bulked = low_deck - h_idraulico_bulked
                franco_base = low_deck - h_idraulico_base
                
                # Risk classification
                if h_idraulico_bulked < low_deck:
                    risk_bulk = 'Low'
                elif h_idraulico_bulked <= upper_deck:
                    risk_bulk = 'Medium'
                else:
                    risk_bulk = 'High'
                
                if h_idraulico_base < low_deck:
                    risk_base = 'Low'
                elif h_idraulico_base <= upper_deck:
                    risk_base = 'Medium'
                else:
                    risk_base = 'High'
                
            except Exception as e:
                h_idraulico_bulked = h_idraulico_base = 0
                franco_bulked = franco_base = low_deck
                risk_bulk = risk_base = 'Unknown'
            
            # Store result
            result = {
                'Return_Period': tr,
                'Bulking_Factor': bulking_factor,
                'Bridge_ID': ponte_idx + 1,
                'Bridge_Name': bridge['name'],
                'X': breaknode_lons[breaknode_idx],
                'Y': breaknode_lats[breaknode_idx],
                'Q_base': Q_base_val,
                'Q_bulked': Q_bulked_val,
                'LowDeck': low_deck,
                'UpperDeck': upper_deck,
                'H_hydraulic_base': h_idraulico_base,
                'H_hydraulic_bulked': h_idraulico_bulked,
                'Franco_base': franco_base,
                'Franco_bulked': franco_bulked,
                'Risk_base': risk_base,
                'Risk_bulked': risk_bulk,
                'P_t': p_t[breaknode_idx],
                'Tau_b': tau_b[breaknode_idx],
                'Area_km2': ab[breaknode_idx],
                'H_intensity': h_t[breaknode_idx],
                'S_mean': s_mean[breaknode_idx]
            }
            
            all_results.append(result)

print(f"\n   Total results generated: {len(all_results)}")

# =================================================================
# 10. SIMPLE PLOTTING ANALYSIS
# =================================================================

print("\n10. Creating simple analysis plots...")

# Convert all_results to DataFrame for easier plotting
import matplotlib.pyplot as plt
import pandas as pd

df_results = pd.DataFrame(all_results)
print(f"   Total results: {len(df_results)} scenarios")

# =================================================================
# PLOT 1: Single Bridge - All Bulking Factors
# =================================================================

print("   Creating Plot 1: Single bridge analysis...")

# Select first bridge for detailed analysis
selected_bridge = valid_bridges[0]['name']
bridge_data = df_results[df_results['Bridge_Name'] == selected_bridge].copy()

# Create the plots
plt.figure(figsize=(15, 6))

# Plot 1: Single bridge, multiple bulking factors
plt.subplot(1, 2, 1)

# Get unique bulking factors and sort them
unique_bfs = sorted(bridge_data['Bulking_Factor'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

print(f"   Plotting {selected_bridge} with {len(unique_bfs)} bulking factors")

for i, bf in enumerate(unique_bfs):
    bf_data = bridge_data[bridge_data['Bulking_Factor'] == bf].copy()
    bf_data = bf_data.sort_values('Return_Period')
    
    if len(bf_data) > 0:
        plt.plot(bf_data['Return_Period'], bf_data['Franco_bulked'], 
                'o-', color=colors[i % len(colors)], linewidth=2.5, markersize=6,
                label=f'BF = {bf:.2f}')

# Add reference lines
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 0 (CRITICO)')
plt.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 1.5m')

plt.xlabel('Tempo di Ritorno (anni)', fontsize=12)
plt.ylabel('Franco Idraulico (m)', fontsize=12)
plt.title(f'{selected_bridge} - Franco vs Tempo di Ritorno', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.xscale('log')

# Set x-axis ticks
return_periods = sorted(df_results['Return_Period'].unique())
plt.xticks(return_periods, [str(x) for x in return_periods])

# =================================================================
# PLOT 2: All Bridges - Average Line  
# =================================================================

print("   Creating Plot 2: All bridges average analysis...")

plt.subplot(1, 2, 2)

# Calculate overall average franco for each return period
avg_franco_by_tr = df_results.groupby('Return_Period')['Franco_bulked'].agg(['mean', 'std']).reset_index()

# Plot all individual bridge-bulking factor combinations (faded background)
bridge_names = df_results['Bridge_Name'].unique()
all_bfs = df_results['Bulking_Factor'].unique()

print(f"   Plotting {len(bridge_names)} bridges with {len(all_bfs)} bulking factors each")

# Plot individual lines (light gray, for context)
for bridge in bridge_names:
    for bf in all_bfs:
        bridge_bf_data = df_results[(df_results['Bridge_Name'] == bridge) & 
                                   (df_results['Bulking_Factor'] == bf)].copy()
        bridge_bf_data = bridge_bf_data.sort_values('Return_Period')
        
        if len(bridge_bf_data) > 0:
            plt.plot(bridge_bf_data['Return_Period'], bridge_bf_data['Franco_bulked'], 
                    '-', color='lightgray', alpha=0.3, linewidth=1)

# Plot average line (bold black)
plt.plot(avg_franco_by_tr['Return_Period'], avg_franco_by_tr['mean'], 
         'o-', color='black', linewidth=4, markersize=8, label='Media Generale')

# Plot confidence band (±1 standard deviation)
plt.fill_between(avg_franco_by_tr['Return_Period'], 
                avg_franco_by_tr['mean'] - avg_franco_by_tr['std'],
                avg_franco_by_tr['mean'] + avg_franco_by_tr['std'],
                color='black', alpha=0.2, label='±1 Deviazione Standard')

# Add reference lines
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 0 (CRITICO)')
plt.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 1.5m')

plt.xlabel('Tempo di Ritorno (anni)', fontsize=12)
plt.ylabel('Franco Idraulico (m)', fontsize=12)
plt.title('Tutti i Ponti - Andamento Medio Franco', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.xscale('log')

# Set x-axis ticks
plt.xticks(return_periods, [str(x) for x in return_periods])

plt.tight_layout()

# Save plot
output_plot_file = f"CN2_franco_analysis_{selected_location['output_prefix']}.png"
plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
print(f"   Plot saved: {output_plot_file}")

plt.show()

# =================================================================
# 11. ESPORTA RISULTATI PER TEMPO DI RITORNO (7 FILE SEPARATI)
# =================================================================

print("\n11. Exporting results by return period (7 separate files)...")

# Converti all_results in DataFrame
df_results = pd.DataFrame(all_results)
return_periods = sorted(df_results['Return_Period'].unique())

print(f"   Creating {len(return_periods)} files (one per return period)")

# Esporta un file per ogni tempo di ritorno
for tr in return_periods:
    print(f"   Processing return period {tr} years...")
    
    # Filtra i dati per questo tempo di ritorno
    tr_data = df_results[df_results['Return_Period'] == tr]
    
    # Crea risultati aggregati per ponte (mediando sui bulking factors)
    tr_results = []
    
    for bridge_name in tr_data['Bridge_Name'].unique():
        bridge_data = tr_data[tr_data['Bridge_Name'] == bridge_name]
        
        if len(bridge_data) > 0:
            # Prendi il primo record per le informazioni di base
            first_record = bridge_data.iloc[0]
            
            # Aggrega sui diversi bulking factors per questo tempo di ritorno
            result = {
                'Geometry': 'Point',
                'X': first_record['X'],
                'Y': first_record['Y'],
                'ID': first_record['Bridge_ID'],
                'Nome': bridge_name,
                'Return_Period': tr,
                'Q_median': bridge_data['Q_bulked'].median(),
                'Q_min': bridge_data['Q_bulked'].min(),
                'Q_max': bridge_data['Q_bulked'].max(),
                'LowDeck': first_record['LowDeck'],
                'UpperDeck': first_record['UpperDeck'],
                'Franco_median': bridge_data['Franco_bulked'].median(),
                'Franco_min': bridge_data['Franco_bulked'].min(),
                'Franco_max': bridge_data['Franco_bulked'].max(),
                'Critical_cases': len(bridge_data[bridge_data['Franco_bulked'] <= 0]),
                'Total_BF': len(bridge_data),  # Numero bulking factors
                'Critical_pct': (bridge_data['Franco_bulked'] <= 0).mean() * 100,
                'Area_km2': first_record['Area_km2'],
                'P_t_median': bridge_data['P_t'].median(),
                'Tau_b': first_record['Tau_b'],
                'BF_range': f"{bridge_data['Bulking_Factor'].min():.2f}-{bridge_data['Bulking_Factor'].max():.2f}"
            }
            
            tr_results.append(result)
    
    # Converti in GeoDataFrame e salva
    if len(tr_results) > 0:
        from shapely.geometry import Point
        import geopandas as gpd
        
        geometry = [Point(r['X'], r['Y']) for r in tr_results]
        gdf = gpd.GeoDataFrame(tr_results, geometry=geometry, crs='EPSG:4326')
        
        # Nome file per questo tempo di ritorno
        output_file = f"ponti_Tr{tr}y_{selected_location['output_prefix']}_CN2.shp"
        gdf.to_file(output_file)
        
        print(f"     ✓ Saved {len(gdf)} bridges to: {output_file}")
    else:
        print(f"     ✗ No data for return period {tr}")

print(f"\n   Export completed: {len(return_periods)} files created")
print("   Files created:")
for tr in return_periods:
    filename = f"ponti_Tr{tr}y_{selected_location['output_prefix']}_CN2.shp"
    print(f"     - {filename}")

# =================================================================
# SEZIONE MODIFICATA: GRAFICI SEMPLIFICATI (SOSTITUISCE QUELLA ESISTENTE)
# =================================================================

def create_simplified_plots(df_results, valid_bridges, selected_location):
    """
    Crea solo i 3 grafici richiesti: ponte singolo, quantili tutti i ponti, franco vs portata
    """
    print("\n   Creando grafici semplificati (3 grafici)...")
    
    # Parametri grafici
    return_periods = sorted(df_results['Return_Period'].unique())
    
    # Crea figure ottimizzate (2x2 grid, usando solo 3 posizioni)
    fig = plt.figure(figsize=(16, 12))
    
    # =================================================================
    # PLOT 1: Franco vs Tempo di Ritorno - Ponte Singolo
    # =================================================================
    plt.subplot(2, 2, 1)
    
    # Seleziona primo ponte per analisi dettagliata
    selected_bridge = valid_bridges[0]['name']
    bridge_data = df_results[df_results['Bridge_Name'] == selected_bridge].copy()
    
    # Colori per bulking factors
    unique_bfs = sorted(bridge_data['Bulking_Factor'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, bf in enumerate(unique_bfs):
        bf_data = bridge_data[bridge_data['Bulking_Factor'] == bf].copy()
        bf_data = bf_data.sort_values('Return_Period')
        
        if len(bf_data) > 0:
            plt.plot(bf_data['Return_Period'], bf_data['Franco_bulked'], 
                    'o-', color=colors[i % len(colors)], linewidth=2.5, markersize=6,
                    label=f'BF = {bf:.2f}')
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 0 (CRITICO)')
    plt.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 1.5m')
    
    plt.xlabel('Tempo di Ritorno (anni)', fontsize=12)
    plt.ylabel('Franco Idraulico (m)', fontsize=12)
    plt.title(f'{selected_bridge} - Franco vs Tempo di Ritorno', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    plt.xscale('log')
    plt.xticks(return_periods, [str(x) for x in return_periods])
    
    # =================================================================
    # PLOT 2: Franco vs Tempo di Ritorno - TUTTI I PONTI con QUANTILI
    # =================================================================
    plt.subplot(2, 2, 2)
    
    # Calcola quantili per ogni tempo di ritorno
    quantile_data = []
    for tr in sorted(return_periods):
        tr_data = df_results[df_results['Return_Period'] == tr]['Franco_bulked']
        if len(tr_data) > 0:
            quantile_data.append({
                'Return_Period': tr,
                'q10': tr_data.quantile(0.10),
                'q50': tr_data.quantile(0.50),  # Mediana
                'q90': tr_data.quantile(0.90),
                'min': tr_data.min(),
                'max': tr_data.max()
            })
    
    quantile_df = pd.DataFrame(quantile_data)
    
    # Plot banda quantili 10-90
    plt.fill_between(quantile_df['Return_Period'], 
                    quantile_df['q10'], quantile_df['q90'],
                    color='lightblue', alpha=0.6, label='Banda Q10-Q90')
    
    # Plot mediana (linea nera)
    plt.plot(quantile_df['Return_Period'], quantile_df['q50'], 
             'o-', color='black', linewidth=3, markersize=8, label='Mediana (Q50)')
    
    # Plot range min-max (linea sottile)
    plt.fill_between(quantile_df['Return_Period'], 
                    quantile_df['min'], quantile_df['max'],
                    color='gray', alpha=0.2, label='Range Min-Max')
    
    # Linee di riferimento
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 0 (CRITICO)')
    plt.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Franco = 1.5m')
    
    plt.xlabel('Tempo di Ritorno (anni)', fontsize=12)
    plt.ylabel('Franco Idraulico (m)', fontsize=12)
    plt.title('Tutti i Ponti - Quantili Franco vs Tempo di Ritorno', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xscale('log')
    plt.xticks(return_periods, [str(x) for x in return_periods])
    
    # =================================================================
    # PLOT 3: Franco vs Portata (log-log)
    # =================================================================
    plt.subplot(2, 2, 3)
    
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    
    # Normalizza colori per tempo di ritorno
    norm = colors.Normalize(vmin=min(return_periods), vmax=max(return_periods))
    cmap = cm.get_cmap('OrRd')  # Arancione-rosso (verde=bassi, rosso=alti)
    
    for tr in sorted(return_periods):
        tr_data = df_results[df_results['Return_Period'] == tr]
        if len(tr_data) > 0:
            plt.scatter(tr_data['Q_bulked'], tr_data['Franco_bulked'], 
                       c=cmap(norm(tr)), alpha=0.6, s=30, label=f'{tr} anni')
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.axhline(y=1.5, color='orange', linestyle='--', linewidth=1, alpha=0.6)
    
    plt.xlabel('Portata Q_bulked (m³/s)', fontsize=12)
    plt.ylabel('Franco Idraulico (m)', fontsize=12)
    plt.title('Franco vs Portata (con bulking)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Correlazione
    from scipy.stats import pearsonr
    valid_mask = (df_results['Q_bulked'] > 0) & (~np.isnan(df_results['Franco_bulked']))
    if np.any(valid_mask):
        corr_data = df_results[valid_mask]
        corr_pearson_q, p_pearson_q = pearsonr(np.log(corr_data['Q_bulked']), 
                                               corr_data['Franco_bulked'])
        plt.text(0.05, 0.95, f'r = {corr_pearson_q:.3f}', 
                 transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =================================================================
    # PLOT 4: Percentuale Sezioni Critiche per Tempo di Ritorno (scala di bacino)
    # =================================================================
    plt.subplot(2, 2, 4)
    
    # Calcola percentuale casi critici per ogni tempo di ritorno
    critical_stats = []
    for tr in sorted(return_periods):
        tr_data = df_results[df_results['Return_Period'] == tr]
        if len(tr_data) > 0:
            critical_count = len(tr_data[tr_data['Franco_bulked'] <= 0])
            total_count = len(tr_data)
            critical_pct = (critical_count / total_count) * 100
            
            critical_stats.append({
                'Return_Period': tr,
                'Critical_Pct': critical_pct,
                'Critical_Count': critical_count,
                'Total_Count': total_count
            })
    
    critical_df = pd.DataFrame(critical_stats)
    
    # Bar plot con grafico a linea sovrapposto
    fig_ax = plt.gca()
    
    # Bars
    bars = fig_ax.bar(critical_df['Return_Period'], critical_df['Critical_Pct'], 
                     color='darkred', alpha=0.7, edgecolor='black', linewidth=1.5,
                     label='% Sezioni Critiche')
    
    # Line plot sovrapposto
    fig_ax.plot(critical_df['Return_Period'], critical_df['Critical_Pct'], 
               'o-', color='red', linewidth=3, markersize=8, 
               markerfacecolor='white', markeredgecolor='red', markeredgewidth=2)
    
    # Aggiungi valori sulle barre
    for i, (bar, row) in enumerate(zip(bars, critical_df.itertuples())):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%\n({row.Critical_Count}/{row.Total_Count})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Linee di riferimento
    plt.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                label='50% Sezioni Critiche')
    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='100% Sezioni Critiche')
    
    plt.xlabel('Tempo di Ritorno (anni)', fontsize=12)
    plt.ylabel('Sezioni Critiche (%)', fontsize=12)
    plt.title('Percentuale Sezioni Critiche per Tempo di Ritorno\n(Scala di Bacino)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=10, loc='upper left')
    
    # Scala logaritmica per x
    plt.xscale('log')
    plt.xticks(return_periods, [str(x) for x in return_periods])
    
    # Limiti y per migliore visualizzazione
    plt.ylim(0, max(105, critical_df['Critical_Pct'].max() + 10))
    
    # Aggiungere informazioni addizionali
    total_bridges = len(df_results['Bridge_Name'].unique())
    total_scenarios = len(df_results)
    
    info_text = f"BACINO: TOR SAPIENZA\n"
    info_text += f"Ponti analizzati: {total_bridges}\n"
    info_text += f"Scenari totali: {total_scenarios}\n"
    info_text += f"Bulking factors: {params['n_bulking_samples']}"
    
    plt.text(0.98, 0.02, info_text, transform=fig_ax.transAxes, 
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva il grafico
    output_plot = f"simplified_analysis_CN2_{selected_location['output_prefix']}.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"   Grafico semplificato salvato: {output_plot}")
    
    
    return quantile_df

# Questa sezione va inserita dopo la generazione di all_results, sostituendo le sezioni 10-13

print("\n" + "="*80)
print("ANALISI SEMPLIFICATA E EXPORT GIS")
print("="*80)

# Converti results a DataFrame
df_results = pd.DataFrame(all_results)
print(f"Total results: {len(df_results)} scenarios")

# Crea grafici semplificati
quantile_df = create_simplified_plots(df_results, valid_bridges, selected_location)

for _, row in quantile_df.iterrows():
    tr = int(row['Return_Period'])
    print(f"Tr = {tr:3d} anni: Q10 = {row['q10']:5.2f}m, Q50 = {row['q50']:5.2f}m, Q90 = {row['q90']:5.2f}m")

print(f"\nCASI CRITICI PER TEMPO DI RITORNO:")
print("-" * 50)
for tr in sorted(return_periods):
    tr_data = df_results[df_results['Return_Period'] == tr]
    if len(tr_data) > 0:
        critical_count = len(tr_data[tr_data['Franco_bulked'] <= 0])
        total_count = len(tr_data)
        critical_pct = (critical_count / total_count) * 100
        print(f"Tr = {tr:3d} anni: {critical_pct:4.1f}% ({critical_count:2d}/{total_count:2d} casi)")

print(f"\nSTATISTICHE PER PONTE (Franco Idraulico):")
print("-" * 50)
bridge_names = df_results['Bridge_Name'].unique()
for bridge in sorted(bridge_names):
    bridge_data_full = df_results[df_results['Bridge_Name'] == bridge]
    critical_cases = len(bridge_data_full[bridge_data_full['Franco_bulked'] <= 0])
    median_franco = bridge_data_full['Franco_bulked'].median()
    q10_franco = bridge_data_full['Franco_bulked'].quantile(0.10)
    q90_franco = bridge_data_full['Franco_bulked'].quantile(0.90)
    total_cases = len(bridge_data_full)
    low_deck = bridge_data_full['LowDeck'].iloc[0]
    
    print(f"{bridge:15s}: Altezza = {low_deck:4.1f}m, "
          f"Franco [Q10,Q50,Q90] = [{q10_franco:5.2f},{median_franco:5.2f},{q90_franco:5.2f}]m, "
          f"Critici = {critical_cases:2d}/{total_cases:2d} ({critical_cases/total_cases*100:4.1f}%)")




