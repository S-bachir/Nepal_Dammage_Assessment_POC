# Nepal Earthquake Damage Assessment Using GeoAI

This repository contains a comprehensive workflow for post-disaster damage assessment using satellite imagery and GeoAI techniques, specifically developed for the November 3, 2023 earthquake in Nepal.

## Event Information
- **Date**: November 3, 2023
- **Magnitude**: 6.4 ML (5.7 Mw)
- **Epicenter**: Ramidanda, Jajarkot District (28.84°N, 82.19°E)
- **Affected Districts**: Jajarkot, Rukum West, Salyan

## Project Overview

This project implements state-of-the-art satellite imagery analysis and GeoAI techniques for earthquake damage assessment, including:
- Multi-temporal satellite imagery analysis (Sentinel-2, Landsat 8/9, Sentinel-1)
- Spectral change detection using various indices (NDVI, NBR, NDBI)
- Machine learning classification for damage assessment
- Building-level damage analysis
- Landslide detection
- Comprehensive reporting and visualization

## Requirements

### Prerequisites
1. Python 3.9+
2. Google Earth Engine account and authentication
3. API keys for satellite data providers (optional)
4. Sufficient storage for satellite imagery

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nepal-earthquake-assessment.git
cd nepal-earthquake-assessment

# Create conda environment
conda create -n earthquake-assessment python=3.9
conda activate earthquake-assessment

# Install dependencies
pip install -r requirements.txt


# Workflow

The complete workflow is documented in the main notebook: `notebooks/damage_assessment_workflow.ipynb`


## Pipeline Steps

### 1. Data Acquisition (`scripts/data_acquisition.py`)

- Fetches satellite imagery from multiple sources:
  - Sentinel-2 Level-2A (ESA Copernicus)
  - Sentinel-1 SAR GRD (ESA Copernicus)
  - Landsat 8/9 Collection 2 (USGS)
- Downloads ancillary data:
  - OpenStreetMap building footprints
  - SRTM Digital Elevation Model (NASA)
  - Population density (CIESIN GPW v4.11)

### 2. Preprocessing (`scripts/preprocessing.py`)

- Image co-registration
- Radiometric correction
- Cloud masking
- Creation of analysis-ready data

### 3. Damage Analysis (`scripts/damage_analysis.py`)

- Spectral change detection
- Machine learning classification
- Building-level damage assessment
- Landslide detection

### 4. Visualization (`scripts/visualization.py`)

- Before/after satellite image comparisons
- Damage classification maps
- Interactive visualizations (e.g., via Folium or Plotly)
- 3D terrain analysis

### 5. Reporting (`scripts/reporting.py`)

- PDF reports (e.g., via ReportLab or LaTeX)
- Excel summaries (e.g., via Pandas)
- GIS outputs (e.g., Shapefiles, GeoJSON)
- Web dashboards (e.g., via Dash or Streamlit)

## Data Sources

- Sentinel-2 Level-2A (ESA Copernicus)
- Sentinel-1 SAR GRD (ESA Copernicus)
- Landsat 8/9 Collection 2 (USGS)
- OpenStreetMap building footprints
- SRTM Digital Elevation Model (NASA)
- Population density (CIESIN GPW v4.11)

## Results

The workflow produces:

- Damage classification maps
- Building-level damage statistics
- Landslide susceptibility maps
- Comprehensive reports for disaster response

## License

This project is licensed under the **Mozilla Public License 2.0**. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ESA Copernicus Programme for Sentinel data
- USGS for Landsat imagery
- OpenStreetMap contributors
- NASA for SRTM elevation data

## Contributing

Contributions are welcome! Please submit pull requests or open issues for improvements and bug fixes.