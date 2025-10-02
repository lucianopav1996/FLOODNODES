import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors
import os

def visualize_bridge_basins(bridges, breaknodes, fd, dem_crop, a_crop, s, 
                            tau_b, p_t, ab, q_valle, output_dir='bridge_basins'):
    """
    Crea visualizzazioni dettagliate per ogni ponte mostrando:
    - Il bacino di drenaggio
    - La sezione di chiusura
    - I parametri idrologici calcolati
    - La rete idrografica
    """
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Per ogni ponte/breaknode
    for i, (bridge, node_idx) in enumerate(zip(bridges[:10], breaknodes[:10])):  # Primi 10 per test
        
        print(f"\nCreating visualization for bridge {i}: {bridge['name']}")
        
        # Crea figura con subplot
        fig = plt.figure(figsize=(20, 12))
        
        # === SUBPLOT 1: Bacino di drenaggio ===
        ax1 = plt.subplot(2, 3, 1)
        
        try:
            # Calcola il bacino di drenaggio per questo nodo
            basin = fd.drainagebasins(np.array([node_idx]))
            basin_mask = (basin.z == 1)
            
            # Visualizza il DEM con il bacino evidenziato
            dem_display = dem_crop.z.copy()
            dem_display[~basin_mask] = np.nan
            
            im1 = ax1.imshow(dem_display, cmap='terrain', aspect='equal')
            plt.colorbar(im1, ax=ax1, label='Elevation (m)')
            
            # Aggiungi punto di chiusura
            row = node_idx % dem_crop.rows
            col = node_idx // dem_crop.rows
            ax1.plot(col, row, 'r*', markersize=15, label=f'Bridge {i}')
            
            ax1.set_title(f'Drainage Basin - {bridge["name"]}')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            ax1.legend()
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {e}', transform=ax1.transAxes, ha='center')
            ax1.set_title(f'Drainage Basin - ERROR')
        
        # === SUBPLOT 2: Area di accumulo ===
        ax2 = plt.subplot(2, 3, 2)
        
        try:
            # Visualizza l'area di accumulo con il bacino
            a_display = a_crop.z.copy()
            a_display[~basin_mask] = np.nan
            
            im2 = ax2.imshow(np.log10(a_display + 1), cmap='Blues', aspect='equal')
            plt.colorbar(im2, ax=ax2, label='Log10(Flow Accumulation)')
            
            # Punto di chiusura
            ax2.plot(col, row, 'r*', markersize=15)
            
            ax2.set_title('Flow Accumulation')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            
        except:
            ax2.text(0.5, 0.5, 'Error displaying accumulation', transform=ax2.transAxes, ha='center')
        
        # === SUBPLOT 3: Rete idrografica nel bacino ===
        ax3 = plt.subplot(2, 3, 3)
        
        try:
            # Estrai la rete idrografica nel bacino
            stream_mask = a_crop.z > (2.0 / ((dem_crop.cellsize/1000)**2))  # Soglia 2 km²
            stream_in_basin = stream_mask & basin_mask
            
            # Visualizza
            display = np.zeros_like(dem_crop.z)
            display[basin_mask] = 1  # Bacino in grigio
            display[stream_in_basin] = 2  # Stream in blu
            
            cmap = mcolors.ListedColormap(['white', 'lightgray', 'blue'])
            im3 = ax3.imshow(display, cmap=cmap, aspect='equal')
            
            # Punto di chiusura
            ax3.plot(col, row, 'r*', markersize=15, label='Outlet')
            
            ax3.set_title('Stream Network in Basin')
            ax3.set_xlabel('Column')
            ax3.set_ylabel('Row')
            ax3.legend()
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error: {e}', transform=ax3.transAxes, ha='center')
        
        # === SUBPLOT 4: Parametri idrologici (tabella) ===
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        # Crea tabella con i parametri
        params_text = f"""
        BRIDGE: {bridge['name']}
        
        Location:
        - UTM X: {bridge.get('x_utm', 'N/A'):.1f}
        - UTM Y: {bridge.get('y_utm', 'N/A'):.1f}
        - Lon: {bridge['lon']:.6f}
        - Lat: {bridge['lat']:.6f}
        
        Basin Parameters:
        - Area: {ab[i]:.2f} km²
        - Concentration time: {tau_b[i]:.2f} hours
        - Design precipitation: {p_t[i]:.1f} mm
        - Design flow rate: {q_valle[i]:.1f} m³/s
        
        Bridge Data:
        - Lower deck: {bridge['low_deck']:.2f} m
        - Upper deck: {bridge['upper_deck']:.2f} m
        - Bottom elevation: {bridge['h_fondo']:.2f} m
        
        Node Index: {node_idx}
        Grid Position: row={row}, col={col}
        """
        
        ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Hydrological Parameters')
        
        # === SUBPLOT 5: Profilo altimetrico lungo l'asta principale ===
        ax5 = plt.subplot(2, 3, 5)
        
        try:
            # Trova il percorso più lungo fino alla sezione di chiusura
            # Questo richiede di tracciare il flusso upstream
            profile_points = []
            current_idx = node_idx
            max_iterations = 1000
            iter_count = 0
            
            while iter_count < max_iterations:
                row_curr = current_idx % dem_crop.rows
                col_curr = current_idx // dem_crop.rows
                
                if 0 <= row_curr < dem_crop.rows and 0 <= col_curr < dem_crop.columns:
                    elev = dem_crop.z[row_curr, col_curr]
                    if not np.isnan(elev):
                        profile_points.append(elev)
                
                # Trova il prossimo punto upstream (questo è semplificato)
                # In realtà dovresti usare fd per trovare il vero percorso
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    new_row = row_curr + dr
                    new_col = col_curr + dc
                    if 0 <= new_row < dem_crop.rows and 0 <= new_col < dem_crop.columns:
                        new_idx = new_row + new_col * dem_crop.rows
                        if basin_mask.flat[new_idx] and a_crop.z.flat[new_idx] < a_crop.z.flat[current_idx]:
                            neighbors.append((new_idx, a_crop.z.flat[new_idx]))
                
                if neighbors:
                    # Scegli il neighbor con maggiore accumulo (più upstream)
                    neighbors.sort(key=lambda x: x[1], reverse=True)
                    current_idx = neighbors[0][0]
                else:
                    break
                
                iter_count += 1
            
            if profile_points:
                distances = np.arange(len(profile_points)) * dem_crop.cellsize
                ax5.plot(distances, profile_points, 'b-', linewidth=2)
                ax5.set_xlabel('Distance from outlet (m)')
                ax5.set_ylabel('Elevation (m)')
                ax5.set_title('Elevation Profile')
                ax5.grid(True, alpha=0.3)
                ax5.invert_xaxis()  # Outlet a destra
            else:
                ax5.text(0.5, 0.5, 'No profile data', transform=ax5.transAxes, ha='center')
            
        except Exception as e:
            ax5.text(0.5, 0.5, f'Error: {e}', transform=ax5.transAxes, ha='center')
        
        # === SUBPLOT 6: Rating curve del ponte ===
        ax6 = plt.subplot(2, 3, 6)
        
        try:
            if 'rating_curve' in bridge and len(bridge['rating_curve']) > 0:
                h_values = bridge['rating_curve'][:, 0]
                q_values = bridge['rating_curve'][:, 1]
                
                ax6.plot(q_values, h_values, 'b-', linewidth=2, label='Rating curve')
                
                # Aggiungi il punto di design
                ax6.axvline(q_valle[i], color='red', linestyle='--', 
                           label=f'Design Q = {q_valle[i]:.1f} m³/s')
                
                # Aggiungi livelli del ponte
                ax6.axhline(bridge['low_deck'], color='orange', linestyle='--', 
                           label=f'Low deck = {bridge["low_deck"]:.1f} m')
                ax6.axhline(bridge['upper_deck'], color='darkred', linestyle='--',
                           label=f'Upper deck = {bridge["upper_deck"]:.1f} m')
                
                ax6.set_xlabel('Flow rate (m³/s)')
                ax6.set_ylabel('Water level (m)')
                ax6.set_title('Bridge Rating Curve')
                ax6.legend(loc='best', fontsize=8)
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No rating curve data', transform=ax6.transAxes, ha='center')
                
        except Exception as e:
            ax6.text(0.5, 0.5, f'Error: {e}', transform=ax6.transAxes, ha='center')
        
        # Titolo generale
        fig.suptitle(f'Hydrological Analysis - Bridge {i}: {bridge["name"]}', 
                    fontsize=14, fontweight='bold')
        
        # Salva figura
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'bridge_{i:02d}_{bridge["name"].replace(" ", "_")}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: {output_file}")
    
    print(f"\nAll visualizations saved in {output_dir}/")
    
    # Crea anche una figura riassuntiva
    create_summary_plot(bridges[:10], ab[:10], tau_b[:10], p_t[:10], q_valle[:10], output_dir)


def create_summary_plot(bridges, ab, tau_b, p_t, q_valle, output_dir):
    """Crea un plot riassuntivo con tutti i parametri idrologici"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bridge_names = [b['name'][:15] for b in bridges]  # Abbrevia i nomi
    x_pos = np.arange(len(bridges))
    
    # Plot 1: Aree dei bacini
    ax1 = axes[0, 0]
    ax1.bar(x_pos, ab, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Bridge')
    ax1.set_ylabel('Basin Area (km²)')
    ax1.set_title('Drainage Basin Areas')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bridge_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Tempi di concentrazione
    ax2 = axes[0, 1]
    ax2.bar(x_pos, tau_b, color='lightgreen', edgecolor='darkgreen')
    ax2.set_xlabel('Bridge')
    ax2.set_ylabel('Concentration Time (hours)')
    ax2.set_title('Concentration Times')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bridge_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Precipitazioni di progetto
    ax3 = axes[1, 0]
    ax3.bar(x_pos, p_t, color='lightcoral', edgecolor='darkred')
    ax3.set_xlabel('Bridge')
    ax3.set_ylabel('Design Precipitation (mm)')
    ax3.set_title('Design Precipitation (Tr=200 years)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bridge_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Portate di progetto
    ax4 = axes[1, 1]
    ax4.bar(x_pos, q_valle, color='gold', edgecolor='darkorange')
    ax4.set_xlabel('Bridge')
    ax4.set_ylabel('Design Flow Rate (m³/s)')
    ax4.set_title('Design Flow Rates')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bridge_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Hydrological Parameters Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'summary_hydrological_params.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {output_file}")


