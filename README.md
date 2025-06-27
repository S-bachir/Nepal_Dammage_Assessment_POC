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