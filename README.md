# FLOODNODES
GIS tool for identifying flood hotspots in river networks. Combines terrain analysis, hydrological modeling, and hydraulic assessment for multiple return periods (2-500 years). Evaluates sediment transport impact on bridge safety. Developed for basins ≤150 km². Italian PNRR project.
## Description

FLOODNODES is an **integrated workflow** combining geomorphological terrain analysis, hydrological modeling, and hydraulic simulation algorithms to identify critical flood nodes (flood hotspots) along river networks at large scale.

The solution implements data, methods, and computational codes originating from **15 years of research** by the IDRAN team, providing:

### Key Features

- **Multi-return period flood analysis** (2, 10, 25, 50, 100, 200, 500 years)
- **DEM-based rapid assessment** for large-scale applications
- **Sediment transport modeling** with bulking factors (1.05 to 1.5)
- **Bridge hydraulic clearance analysis** for infrastructure risk assessment
- **GIS-ready outputs** (Shapefiles, thematic maps, statistical reports)


### What FLOODNODES Does

1. Extracts river network from DEM data
2. Calculates hydrological parameters (concentration time, runoff)
3. Computes design discharges for multiple return periods
4. Assesses hydraulic clearances at bridge locations
5. Evaluates sediment transport effects (bulking/blockage)
6. Generates risk maps and critical section identification

---

**IMPORTANT:** This code has been **developed and validated for watersheds up to 150 km²**.

### Why This Matters

The implemented methodologies have specific scale assumptions:

- **Giandotti formula** for concentration time is calibrated for small-medium basins
- **SCS Curve Number method** performs best at basin scale < 150 km²
- **Simplified hydraulic modeling** may not capture complex routing in large basins
- **Computational efficiency** decreases significantly beyond this threshold

### For Larger Basins (> 150 km²)

If you need to analyze larger watersheds:
- Recalibrate hydrological parameters
- Consider distributed hydrological models (not lumped)
- Validate results against observed data
- Use more sophisticated hydraulic routing
- Consult with hydrological experts

**The authors cannot guarantee result accuracy beyond the 150 km² limit.**

---

## Requirements

### Software Dependencies

- **Python 3.8 or higher**
- **GDAL/OGR** (for geospatial processing)
- **TopoToolbox for Python** (terrain analysis)

### Python Libraries

Install all dependencies:
```bash
