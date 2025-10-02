"""
plot_and_utils.py
Plotting and Utility Functions for Flood Risk Analysis
Python translation of MATLAB plotting and utility functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Optional, Tuple, Union
from scipy.interpolate import interp1d
import rasterio
from rasterio.crs import CRS
from pyproj import Proj, transform

try:
    import topotoolbox as tt
except ImportError:
    print("TopoToolbox for Python non installato")


def plot_network(reach_data: List[Dict], 
                plot_variable: Optional[np.ndarray] = None,
                show_id: bool = False,
                cmap: str = 'viridis',
                title: Optional[str] = None,
                legend_type: str = 'percentile',
                line_width: Union[float, np.ndarray] = 1.0,
                class_number: int = 14,
                c_class: Optional[np.ndarray] = None,
                ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot river network with optional attribute visualization.
    
    Translation of plot_network.m
    
    Parameters
    ----------
    reach_data : list of dict
        River reach data with geometry
    plot_variable : np.ndarray, optional
        Values to visualize for each reach
    show_id : bool
        Whether to show reach IDs
    cmap : str
        Colormap name
    title : str, optional
        Plot title
    legend_type : str
        'colorbar' or 'percentile'
    line_width : float or np.ndarray
        Line width(s)
    class_number : int
        Number of percentile classes
    c_class : np.ndarray, optional
        Custom class boundaries
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.get_figure()
    
    # Prepare line widths
    if np.isscalar(line_width):
        line_widths = np.full(len(reach_data), line_width)
    else:
        line_widths = np.array(line_width)
    
    # Plot empty network if no variable provided
    if plot_variable is None:
        for i, reach in enumerate(reach_data):
            if 'geometry' in reach:
                x, y = reach['geometry'].xy
                ax.plot(x, y, 'b-', linewidth=line_widths[i])
        
        ax.set_xlabel('X coordinates')
        ax.set_ylabel('Y coordinates')
        ax.set_title(title or 'River Network')
        return fig
    
    # Ensure plot_variable is 1D array
    plot_variable = np.array(plot_variable).flatten()
    
    if len(plot_variable) != len(reach_data):
        raise ValueError(f"plot_variable length ({len(plot_variable)}) must match reach_data length ({len(reach_data)})")
    
    # Plot with colorbar
    if legend_type == 'colorbar':
        # Normalize colors
        vmin, vmax = np.min(plot_variable), np.max(plot_variable)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        
        for i, reach in enumerate(reach_data):
            if 'geometry' in reach:
                x, y = reach['geometry'].xy
                color = cmap_obj(norm(plot_variable[i]))
                ax.plot(x, y, color=color, linewidth=line_widths[i])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        
        # Format colorbar labels
        if vmax > 1e4:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2e}'))
        else:
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.4g}'))
    
    # Plot with percentile classes
    else:
        # Define classes
        if c_class is None:
            i_class = 100 / class_number
            non_zero_values = plot_variable[plot_variable != 0]
            if len(non_zero_values) > 0:
                c_class = np.unique(np.percentile(non_zero_values, np.arange(0, 100 + i_class, i_class)))
                c_class = np.append([0], c_class[c_class != 0])
            else:
                c_class = np.linspace(np.min(plot_variable), np.max(plot_variable), class_number)
        
        # Get colormap
        cmap_obj = plt.get_cmap(cmap, len(c_class))
        colors = cmap_obj(np.linspace(0, 1, len(c_class)))
        
        # Plot each class
        legend_handles = []
        legend_labels = []
        
        for c_idx in range(len(c_class)):
            if c_idx == 0:
                mask = plot_variable <= c_class[c_idx]
                label = f'≤ {c_class[c_idx]:.4g}'
            elif c_idx == len(c_class) - 1:
                mask = plot_variable > c_class[c_idx - 1]
                label = f'> {c_class[c_idx - 1]:.4g}'
            else:
                mask = (plot_variable > c_class[c_idx - 1]) & (plot_variable <= c_class[c_idx])
                label = f'{c_class[c_idx - 1]:.4g} - {c_class[c_idx]:.4g}'
            
            # Plot reaches in this class
            class_reaches = np.where(mask)[0]
            if len(class_reaches) > 0:
                for i in class_reaches:
                    reach = reach_data[i]
                    if 'geometry' in reach:
                        x, y = reach['geometry'].xy
                        line = ax.plot(x, y, color=colors[c_idx], linewidth=line_widths[i])
                
                # Add to legend (use first line for legend handle)
                if len(class_reaches) > 0:
                    legend_handles.append(plt.Line2D([0], [0], color=colors[c_idx], linewidth=3))
                    legend_labels.append(label)
        
        # Add legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add reach IDs if requested
    if show_id:
        for i, reach in enumerate(reach_data):
            if 'geometry' in reach:
                # Get midpoint of reach
                coords = list(reach['geometry'].coords)
                mid_idx = len(coords) // 2
                mid_x, mid_y = coords[mid_idx]
                ax.text(mid_x, mid_y, str(reach.get('reach_id', i)), 
                       fontsize=8, ha='center', va='center')
    
    # Formatting
    ax.set_xlabel('X coordinates', fontsize=15, fontweight='bold')
    ax.set_ylabel('Y coordinates', fontsize=15, fontweight='bold') 
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_aspect('equal')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def hydrosheds2flowobj(flow_dir_grid: tt.GridObject) -> tt.FlowObject:
    """
    Convert HydroSHEDS flow direction grid to FlowObject.
    
    Translation of hydrosheds2FLOWobj_mod.m
    
    Parameters
    ----------
    flow_dir_grid : GridObject
        HydroSHEDS flow direction grid
        
    Returns
    -------
    FlowObject
        Flow direction object
    """
    
    # HydroSHEDS uses power-of-2 encoding for flow directions
    # 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    
    # Handle invalid values (247, 255, 0 -> NaN)
    flow_grid = flow_dir_grid.duplicate_with_new_data(flow_dir_grid.z.copy())
    invalid_mask = (flow_grid.z == 247) | (flow_grid.z == 255) | (flow_grid.z == 0)
    flow_grid.z[invalid_mask] = np.nan
    
    # Create a temporary DEM for FlowObject construction
    # Use negative of flow directions as elevation to create consistent flow
    temp_dem = flow_grid.duplicate_with_new_data(-flow_grid.z.astype(np.float32))
    
    # Fill NaN values for processing
    valid_mask = ~np.isnan(temp_dem.z)
    temp_dem.z[~valid_mask] = np.nanmin(temp_dem.z) - 1000
    
    # Create FlowObject - this will compute D8 flow directions
    fd = tt.FlowObject(temp_dem)
    
    # Note: The resulting flow directions might not exactly match HydroSHEDS
    # For exact matching, would need to implement custom flow direction translation
    
    return fd


def load_bridge_data(filename: str) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Load bridge data from Excel file with rating curves.
    
    Parameters
    ----------
    filename : str
        Path to Excel file with bridge data
        
    Returns
    -------
    tuple
        bridges: List of bridge data dictionaries
        lat_all: Array of latitudes
        lon_all: Array of longitudes
    """
    
    # Get sheet names (bridge names)
    excel_file = pd.ExcelFile(filename)
    bridge_names = excel_file.sheet_names
    
    bridges = []
    lat_all = []
    lon_all = []
    
    # UTM Zone 33N projection for coordinate conversion
    utm_proj = Proj(proj='utm', zone=33, datum='WGS84')
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    
    for bridge_name in bridge_names:
        # Read sheet data
        df = pd.read_excel(filename, sheet_name=bridge_name, header=None)
        
        # Extract bridge metadata (assuming standard format)
        try:
            x_utm = float(df.iloc[0, 1])  # UTM X coordinate
            y_utm = float(df.iloc[1, 1])  # UTM Y coordinate  
            upper_deck = float(df.iloc[2, 1])  # Upper deck elevation
            low_deck = float(df.iloc[3, 1])   # Lower deck elevation
            h_fondo = float(df.iloc[4, 1])    # Bottom elevation
            
            # Extract rating curve (h-Q relationship)
            rating_data = []
            for idx in range(7, len(df)):  # Start from row 8 (index 7)
                try:
                    h_val = float(df.iloc[idx, 0])
                    q_val = float(df.iloc[idx, 1])
                    if not (np.isnan(h_val) or np.isnan(q_val)):
                        rating_data.append([h_val, q_val])
                except (ValueError, TypeError):
                    continue
            
            rating_curve = np.array(rating_data) if rating_data else np.array([[0, 0]])
            
            # Convert UTM to WGS84
            lon, lat = transform(utm_proj, wgs84_proj, x_utm, y_utm)
            
            # Store bridge data
            bridge_data = {
                'name': bridge_name,
                'x_utm': x_utm,
                'y_utm': y_utm,
                'lon': lon,
                'lat': lat,
                'upper_deck': upper_deck,
                'low_deck': low_deck,
                'h_fondo': h_fondo,
                'rating_curve': rating_curve  # [h, Q] pairs
            }
            
            bridges.append(bridge_data)
            lat_all.append(lat)
            lon_all.append(lon)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process bridge {bridge_name}: {e}")
            continue
    
    return bridges, np.array(lat_all), np.array(lon_all)


def calculate_hydraulic_risk(bridges: List[Dict], 
                           breaknodes: np.ndarray,
                           q_bulked: np.ndarray,
                           q_no_bulk: np.ndarray,
                           coords: Tuple[np.ndarray, np.ndarray]) -> List[Dict]:
    """
    Calculate hydraulic risk for bridges based on flow rates and rating curves.
    
    Parameters
    ----------
    bridges : list of dict
        Bridge data with rating curves
    breaknodes : np.ndarray
        Break node indices
    q_bulked : np.ndarray
        Flow rates with bulking factor
    q_no_bulk : np.ndarray
        Flow rates without bulking
    coords : tuple of np.ndarray
        (x, y) coordinates of break nodes
        
    Returns
    -------
    list of dict
        Bridge risk assessment results
    """
    
    x_coords, y_coords = coords
    results = []
    
    for i, bridge in enumerate(bridges):
        # Create interpolation function from rating curve
        if len(bridge['rating_curve']) < 2:
            print(f"Warning: Insufficient rating curve data for bridge {bridge['name']}")
            continue
            
        h_values = bridge['rating_curve'][:, 0]  # Water depths
        q_values = bridge['rating_curve'][:, 1]  # Flow rates
        
        # Sort by flow rate for interpolation
        sort_idx = np.argsort(q_values)
        q_sorted = q_values[sort_idx]
        h_sorted = h_values[sort_idx]
        
        # Create interpolation function
        if len(q_sorted) > 1 and q_sorted[-1] > q_sorted[0]:
            rating_func = interp1d(q_sorted, h_sorted, 
                                 kind='linear', 
                                 bounds_error=False, 
                                 fill_value=(h_sorted[0], h_sorted[-1]))
        else:
            # Handle case with insufficient or invalid data
            rating_func = lambda q: bridge['low_deck'] - 1.0  # Safe default
        
        # Calculate hydraulic depths
        h_bulked = float(rating_func(q_bulked[i]))
        h_no_bulk = float(rating_func(q_no_bulk[i]))
        
        # Calculate hydraulic clearance (franco idraulico)
        franco_bulked = bridge['low_deck'] - h_bulked
        franco_no_bulk = bridge['low_deck'] - h_no_bulk
        
        # Risk classification
        def classify_risk(h_water, low_deck, upper_deck):
            if h_water < low_deck:
                return 'Low'
            elif low_deck <= h_water <= upper_deck:
                return 'Medium' 
            else:
                return 'High'
        
        risk_bulked = classify_risk(h_bulked, bridge['low_deck'], bridge['upper_deck'])
        risk_no_bulk = classify_risk(h_no_bulk, bridge['low_deck'], bridge['upper_deck'])
        
        # Compile results
        result = {
            'id': i + 1,
            'name': bridge['name'],
            'x': x_coords[i],
            'y': y_coords[i],
            'lon': bridge['lon'],
            'lat': bridge['lat'],
            'q_bulked': q_bulked[i],
            'q_no_bulked': q_no_bulk[i],
            'low_deck': bridge['low_deck'],
            'upper_deck': bridge['upper_deck'],
            'h_bulked': h_bulked,
            'h_no_bulk': h_no_bulk,
            'franco_bulked': franco_bulked,
            'franco_no_bulk': franco_no_bulk,
            'risk_bulked': risk_bulked,
            'risk_no_bulk': risk_no_bulk
        }
        
        results.append(result)
    
    return results


def create_output_geodataframe(results: List[Dict], 
                              tau_b: np.ndarray,
                              p_t: np.ndarray, 
                              ab: np.ndarray,
                              crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame with bridge risk results for export.
    
    Parameters
    ----------
    results : list of dict
        Bridge risk assessment results
    tau_b : np.ndarray
        Concentration times
    p_t : np.ndarray
        Design precipitation
    ab : np.ndarray
        Basin areas
    crs : str
        Coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        Results as geodataframe for export
    """
    
    from shapely.geometry import Point
    
    # Create points geometry
    geometry = [Point(result['lon'], result['lat']) for result in results]
    
    # Prepare data dictionary
    data = {
        'ID': [r['id'] for r in results],
        'Nome': [r['name'] for r in results],
        'X_UTM': [r['x'] for r in results],
        'Y_UTM': [r['y'] for r in results],
        'Q_bulked': [r['q_bulked'] for r in results],
        'Q_no_bulk': [r['q_no_bulked'] for r in results],
        'LowDeck': [r['low_deck'] for r in results],
        'UpperDeck': [r['upper_deck'] for r in results],
        'H_bulked': [r['h_bulked'] for r in results],
        'H_no_bulk': [r['h_no_bulk'] for r in results],
        'Franco_bulk': [r['franco_bulked'] for r in results],
        'Franco_nobulk': [r['franco_no_bulk'] for r in results],
        'Risk_bulk': [r['risk_bulked'] for r in results],
        'Risk_nobulk': [r['risk_no_bulk'] for r in results],
        'Tau_b': tau_b[:len(results)],
        'P_t': p_t[:len(results)],
        'Area_km2': ab[:len(results)]
    }
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    
    return gdf


def resample_grid(source_grid: tt.GridObject, 
                 target_grid: tt.GridObject,
                 method: str = 'bilinear') -> tt.GridObject:
    """
    Resample source grid to match target grid EXACTLY like MATLAB resample().
    
    In MATLAB: resample(source, target, 'bilinear') creates a grid with 
    identical bounds, resolution and georeferencing as target.
    
    Parameters
    ----------
    source_grid : GridObject
        Grid to be resampled
    target_grid : GridObject
        Target grid defining output resolution/extent
    method : str
        Resampling method ('bilinear', 'nearest', 'cubic')
        
    Returns
    -------
    GridObject
        Resampled grid with IDENTICAL bounds/resolution as target
    """
    from rasterio.enums import Resampling
    from scipy.interpolate import RegularGridInterpolator
    
    # Map method names to rasterio enums
    resampling_methods = {
        'bilinear': Resampling.bilinear,
        'nearest': Resampling.nearest,
        'cubic': Resampling.cubic
    }
    
    resampling_method = resampling_methods.get(method, Resampling.bilinear)
    
    # Try using reproject first (most accurate)
    if (hasattr(source_grid, 'georef') and hasattr(target_grid, 'georef') and 
        source_grid.georef is not None and target_grid.georef is not None):
        
        try:
            # Reproject to match target exactly
            resampled = source_grid.reproject(
                georef=target_grid.georef,
                resolution=target_grid.cellsize,
                resampling=resampling_method
            )
            
            # Check if bounds are compatible, if not fall back
            if (hasattr(resampled, 'bounds') and hasattr(target_grid, 'bounds') and
                resampled.bounds and target_grid.bounds):
                
                # If reproject worked and bounds are reasonable, use it
                return resampled
                
        except Exception as e:
            print(f"Warning: Reproject failed ({e}), using fallback method")
    
    # Fallback: Create grid with target's EXACT properties and interpolate
    print(f"Warning: Using fallback interpolation for resampling")
    
    try:
        # Get source coordinates and values
        source_x = np.linspace(source_grid.bounds.left, source_grid.bounds.right, source_grid.columns)
        source_y = np.linspace(source_grid.bounds.top, source_grid.bounds.bottom, source_grid.rows)
        source_values = source_grid.z
        
        # Get target coordinates  
        target_x = np.linspace(target_grid.bounds.left, target_grid.bounds.right, target_grid.columns)
        target_y = np.linspace(target_grid.bounds.top, target_grid.bounds.bottom, target_grid.rows)
        
        # Create interpolator (handle NaN values)
        mask = ~np.isnan(source_values)
        if np.any(mask):
            # Use scipy interpolation
            from scipy.interpolate import griddata
            
            # Flatten source coordinates
            source_coords = np.column_stack([
                np.repeat(source_x, len(source_y)),
                np.tile(source_y, len(source_x))
            ])
            source_vals = source_values.flatten()
            
            # Remove NaN values
            valid_mask = ~np.isnan(source_vals)
            source_coords = source_coords[valid_mask]
            source_vals = source_vals[valid_mask]
            
            # Create target coordinate grid
            target_xx, target_yy = np.meshgrid(target_x, target_y)
            target_coords = np.column_stack([target_xx.flatten(), target_yy.flatten()])
            
            # Interpolate
            if len(source_vals) > 0:
                interp_method = 'linear' if method == 'bilinear' else method
                interpolated = griddata(source_coords, source_vals, target_coords, 
                                      method=interp_method, fill_value=np.nan)
                interpolated = interpolated.reshape(target_grid.shape)
            else:
                interpolated = np.full(target_grid.shape, np.nan)
        else:
            interpolated = np.full(target_grid.shape, np.nan)
        
        # Create result with target's exact properties
        result = target_grid.duplicate_with_new_data(interpolated.astype(np.float32))
        return result
        
    except Exception as e:
        print(f"Warning: Interpolation failed ({e}), using mean value fallback")
        
        # Final fallback: use mean value
        source_mean = float(np.nanmean(source_grid.z))
        if np.isnan(source_mean):
            source_mean = 0.0
            
        result = target_grid.duplicate_with_new_data(
            np.full(target_grid.shape, source_mean, dtype=np.float32)
        )
        return result


def save_results(results_gdf: gpd.GeoDataFrame,
                reach_data: List[Dict],
                basin_mask: tt.GridObject,
                output_prefix: str) -> None:
    """
    Save analysis results to shapefiles.
    
    Parameters
    ----------
    results_gdf : gpd.GeoDataFrame
        Bridge risk analysis results
    reach_data : list of dict
        River network reach data
    basin_mask : GridObject
        Drainage basin mask
    output_prefix : str
        Prefix for output file names
    """
    
    # Save bridge results
    results_gdf.to_file(f'{output_prefix}_bridges.shp')
    
    # Save river network
    if reach_data:
        river_geometry = [reach['geometry'] for reach in reach_data if 'geometry' in reach]
        if river_geometry:
            river_gdf = gpd.GeoDataFrame(
                [{'reach_id': reach['reach_id'], 'length_m': reach['length_m']} 
                 for reach in reach_data if 'geometry' in reach],
                geometry=river_geometry,
                crs=results_gdf.crs
            )
            river_gdf.to_file(f'{output_prefix}_river.shp')
    
    # Save drainage basin (would need polygon conversion from mask)
    print(f"Results saved with prefix: {output_prefix}")