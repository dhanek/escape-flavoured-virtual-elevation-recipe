# Virtual Elevation Analyzer

<img src=".assets/VE_icon.png" width="128" height="128" alt="Virtual Elevation Analyzer Icon">

## TODO
- [ ] Add unit tests for core VirtualElevation functionality
- [ ] Add documentation for obtaining DEM elevation data sources by country/region
  - [ ] list of DEM file sources for regions beyond France
- [ ] test moving rho and ETA in the slider view
- [ ] investigate change from fitparse to either fitdecode or to garmin_fit_sdk
- [ ] add support for fit files without power data by forcing all power to 0 (right now it doesn't work)
- [ ] investigate the problem when courses cross the boundaries of elevation data files

## Introduction
This tool implements the Virtual Elevation method ("Chung Method") developed by Robert Chung, which allows for the estimation of a cyclist's aerodynamic parameters (CdA) and rolling resistance (Crr) using power, speed and elevation data from cycling activities.

The Virtual Elevation method works by calculating what the elevation profile would look like based on the measured power output, speed, and assumed aerodynamic and rolling resistance parameters. By optimizing these parameters to make the virtual elevation match the actual measured elevation as closely as possible, we can hopefully accurately estimate a rider's CdA and Crr values.

## Features
- **FIT File Import**: Load cycling data from Garmin and other devices
- **Multiple Analysis Methods**:
  - Standard analysis with manual lap selection
  - GPS-based lap splitting for repeated laps
  - Out-and-back analysis for bi-directional courses
  - GPS gate analysis for specific course segments
- **Parameter Optimization**: Adjust CdA and Crr values with real-time visualization
- **Wind Correction**: Account for wind speed and direction in calculations
- **Map Visualization**: View GPS tracks with lap markers and analysis segments
- **Results Export**: Save analysis results to CSV with plots

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- PySide6
- pandas
- numpy
- matplotlib
- fitparse
- folium
- platformdirs

## Usage

### Starting the Application
```bash
python main.py
```

### Basic Workflow
1. **Select FIT File**: Click "Browse" to choose a cycling activity file
2. **Choose Analysis Type**: Select from available analysis methods
3. **Configure Parameters**:
   - System mass (rider + bike weight)
   - Air density (rho)
   - Fixed or variable CdA/Crr values
   - Wind conditions (optional)
4. **Select Laps**: Choose which laps to analyze
5. **Adjust Analysis**: Use sliders to fine-tune parameters and trim data
6. **Save Results**: Export analysis results and plots

### Analysis Methods

#### Standard Analysis
Basic Virtual Elevation analysis for selected laps with manual trim controls.

#### GPS Lap Splitting
Automatically detects repeated laps based on GPS position markers. Ideal for velodrome or circuit testing.

#### Out-and-Back Analysis
Specialized analysis for courses where you ride out and return on the same route. Compares outbound vs inbound segments.

#### GPS Gate Analysis
Define start/end gates for analyzing specific course segments. Supports multiple gate sets for complex courses.

## Technical Details

### Virtual Elevation Calculation
The tool calculates virtual elevation using the power balance equation:
```
Power = Aero_drag + Rolling_resistance + Gravity + Acceleration
```

### Correcting GPS Elevation
For France:
 - Download 1m rgealti here: https://geoservices.ign.fr/rgealti#telechargement1m
 - Extract the 7zip
 - cd RGEALTI/1_DONNEES_LIVRAISON_2021-01-00157/RGEALTI_MNT_1M_ASC_LAMB93_IGN69_D075_20210118
 - gdalbuildvrt -a_srs EPSG:2154 -allow_projection_difference mosaic_l93.vrt *.asc

Use this mosaic_l93.vrt file from the "Correct Elevation" dialog.

For USA:
 - Go to https://apps.nationalmap.gov/downloader/
 - Select "Elevation Products (3D Elevation Program Products and Services)"
 - Search your area (Right corner)
 - Download the tiff files that cover your area (prioritize 1 meter DEM)
 - In the VE application, select one .tiff file (cf. gdalbuildvrt above if your ride is across several tiff files)

### Parameters
- **CdA**: Coefficient of drag × frontal area (m²)
- **Crr**: Coefficient of rolling resistance
- **System Mass**: Total mass of rider + equipment (kg)
- **Rho**: Air density (kg/m³)
- **Eta**: Drivetrain efficiency (default 0.98)

### File Storage
- Configuration files: OS-specific config directory
- Results: Documents/VirtualElevationRecipes/

## Requirements
- Windows 10/11, macOS 10.14+, or Linux
- 4GB RAM minimum
- Display resolution 1280×720 or higher

## License
GNU GENERAL PUBLIC LICENSE

## Acknowledgements
- Robert Chung for developing the "Virtual Elevation method"
- https://escapecollective.com/the-chung-method-explained-how-to-aero-test-in-the-real-world-2/
- John Karrasch (https://www.instagram.com/flexfitbyjohn/) for applying the Virtual Elevation method to real-world gravel surface testing
- https://escapecollective.com/performance-process-mtb-v-gravel-tyre-testing-how-to-diy-test-your-setup/