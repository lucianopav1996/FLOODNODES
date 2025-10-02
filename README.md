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


## IDF Parameters Data Source

The **IDF (Intensity-Duration-Frequency) parameters** used in this analysis are derived from:

- **Database:** FOCA (Italian national extreme rainfall database)
- **Methodology:** Regional frequency analysis following **Hosking and Wallis (1997)** procedure
- **Parameters:** 
  - `a` (scale parameter) - mm/h
  - `n` (exponential parameter) - dimensionless

---

## Data Sources and Methodology Notes

### IDF Parameters

The **IDF (Intensity-Duration-Frequency) parameters** used in this analysis are derived from:

- **Database:** FOCA (Italian national extreme rainfall database)
- **Methodology:** Regional frequency analysis following **Hosking and Wallis (1997)** procedure
- **Parameters:** 
  - `a` (scale parameter) - mm/h
  - `n` (exponential parameter) - dimensionless

**Data Availability:** The complete IDF parameter datasets (a and n) for all of Italy are **too large to be hosted on GitHub** due to file size limitations. Data must be obtained from ISPRA, regional environmental agencies, or the FOCA database repository.

**Reference:** Hosking, J. R. M., & Wallis, J. R. (1997). *Regional frequency analysis: an approach based on L-moments*. Cambridge University Press.

### Infiltration Model (SCS Curve Number)

The infiltration model provides the **S parameter (potential maximum soil moisture retention)** directly in **millimeters [mm]**, extracted from the FOCA database.

- **S2.tif:** Represents **average antecedent moisture conditions (AMC II)** - normal conditions
- **S3.tif:** Represents **wet antecedent moisture conditions (AMC III)** - saturated/high moisture conditions
- **Methodology:** Based on **Hawkins et al. (2010)** asymptotic fitting procedure
- **Usage:** Direct input to SCS runoff equation without unit conversion

**Note:** The current implementation uses **S2 (average conditions)** by default. For conservative flood risk assessment in already saturated conditions, S3 should be used instead.

**Reference:** Hawkins, R. H., Ward, T. J., Woodward, D. E., & Van Mullem, J. A. (2010). *Continuing evolution of rainfall-runoff and the curve number precedent*. In 2nd Joint Federal Interagency Conference, Las Vegas, NV.

---

## Case Study Application

This methodology has been applied and validated on **tributaries of the Aniene River** within the **Tiber River Basin** (Central Italy).

### Study Area Characteristics

- **Main River:** Aniene River (tributary of Tiber/Tevere)
- **Basin:** Tiber River Basin (Bacino del Tevere)
- **Location:** Central Italy, Lazio Region
- **Typical Basin Sizes:** 10-150 km²
- **Analysis Sites:** San Vittorino, Tor Sapienza, Fregrizia

### Digital Elevation Model

**MERIT Hydro DEM** is used as the primary elevation dataset:

- **Source:** Yamazaki et al. (2019) - Multi-Error-Removed Improved-Terrain Hydro DEM
- **Resolution:** ~90m (3 arc-second)
- **Coverage:** Global
- **Advantages:** 
  - Hydrologically conditioned (removed vegetation bias, stripe noise, speckle noise)
  - Improved accuracy for flow direction and drainage network extraction
  - Suitable for ungauged basins analysis
  - Freely available

**Reference:** Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., & Pavelsky, T. M. (2019). MERIT Hydro: A high-resolution global hydrography map based on latest topography dataset. *Water Resources Research*, 55(6), 5053-5073. https://doi.org/10.1029/2019WR024873

**Alternative DEMs:** The code also supports higher resolution DEMs (e.g., LiDAR 1m, TINItaly 10m, drone-derived DEMs) when available for specific study areas.

---
---
