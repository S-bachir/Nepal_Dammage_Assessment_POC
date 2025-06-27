"""
01_data_acquisition.py
Data Acquisition Script for Nepal Earthquake Damage Assessment
Collects satellite imagery from multiple sources including GEE, Planet, and Maxar
"""
# import platform, sys, types

# # ==== begin Windows‐shim for earthengine-api ====
# import sys, types, platform

# if platform.system() == "Windows":
#     # fake‐modules to satisfy ee’s imports
#     shims = {
#         "_curses":    {},
#         "fcntl":      {"ioctl": lambda *a, **k: 0},
#         "termios":    {"TIOCGWINSZ": None},
#         "blessings":  {"Terminal": type("Terminal", (), {})},
#     }
#     for name, attrs in shims.items():
#         if name not in sys.modules:
#             mod = types.ModuleType(name)
#             for k, v in attrs.items():
#                 setattr(mod, k, v)
#             sys.modules[name] = mod
# # ==== end shim ====

import ee
import os
import json
import requests
import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_acquisition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarthquakeDataAcquisition:
    """
    Handles data acquisition from multiple satellite sources for earthquake damage assessment
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize with configuration file"""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_dir'])
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize Google Earth Engine
        try:
            ee.Initialize()
            logger.info("Google Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default configuration
            default_config = {
                "data_dir": "./data",
                "earthquake": {
                    "date": "2023-11-03",
                    "epicenter": [82.19, 28.84],
                    "magnitude": 6.4,
                    "affected_districts": ["Jajarkot", "Rukum West", "Salyan"]
                },
                "date_ranges": {
                    "pre_start": "2023-09-01",
                    "pre_end": "2023-11-02",
                    "post_start": "2023-11-04",
                    "post_end": "2023-12-15"
                },
                "aoi_buffer_km": 50,
                "cloud_threshold": 20,
                "apis": {
                    "planet_api_key": "",
                    "maxar_api_key": "",
                    "copernicus_user": "",
                    "copernicus_pass": ""
                }
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default configuration at {config_path}")
            return default_config
    
    def create_aoi(self, buffer_km: Optional[float] = None) -> ee.Geometry:
        """Create Area of Interest around epicenter"""
        epicenter = self.config['earthquake']['epicenter']
        buffer = buffer_km or self.config['aoi_buffer_km']
        
        # Create point and buffer
        point = ee.Geometry.Point(epicenter)
        aoi = point.buffer(buffer * 1000).bounds()
        
        logger.info(f"Created AOI with {buffer}km buffer around epicenter")
        return aoi
    
    def get_district_boundaries(self) -> gpd.GeoDataFrame:
        """Load district boundaries for affected areas"""
        # In practice, load from actual shapefile
        # This is a simplified version
        districts = []
        
        # Approximate boundaries for affected districts
        district_coords = {
            "Jajarkot": [[82.0, 28.7], [82.4, 28.7], [82.4, 29.0], [82.0, 29.0], [82.0, 28.7]],
            "Rukum West": [[82.3, 28.5], [82.7, 28.5], [82.7, 28.8], [82.3, 28.8], [82.3, 28.5]],
            "Salyan": [[81.8, 28.3], [82.2, 28.3], [82.2, 28.6], [81.8, 28.6], [81.8, 28.3]]
        }
        
        for district, coords in district_coords.items():
            if district in self.config['earthquake']['affected_districts']:
                poly = Polygon(coords)
                districts.append({
                    'district': district,
                    'geometry': poly
                })
        
        gdf = gpd.GeoDataFrame(districts)
        gdf.crs = 'EPSG:4326'
        
        # Save to file
        output_path = self.data_dir / 'aoi' / 'affected_districts.geojson'
        output_path.parent.mkdir(exist_ok=True)
        gdf.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Saved district boundaries to {output_path}")
        return gdf
    
    def get_sentinel2_data(self, date_start: str, date_end: str, aoi: ee.Geometry) -> ee.ImageCollection:
        """Retrieve Sentinel-2 imagery from Google Earth Engine"""
        logger.info(f"Fetching Sentinel-2 data from {date_start} to {date_end}")
        
        # Sentinel-2 Surface Reflectance
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_start, date_end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.config['cloud_threshold']))
        
        # Add cloud masking function
        def mask_clouds(image):
            qa = image.select('QA60')
            cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(
                qa.bitwiseAnd(1 << 11).eq(0)
            )
            return image.updateMask(cloud_mask).divide(10000) \
                .copyProperties(image, ['system:time_start'])
        
        collection = collection.map(mask_clouds)
        
        # Get collection size
        size = collection.size()
        logger.info(f"Found {size.getInfo()} Sentinel-2 images")
        
        return collection
    
    def get_landsat_data(self, date_start: str, date_end: str, aoi: ee.Geometry) -> ee.ImageCollection:
        """Retrieve Landsat 8/9 imagery from Google Earth Engine"""
        logger.info(f"Fetching Landsat data from {date_start} to {date_end}")
        
        # Landsat 8
        landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(aoi) \
            .filterDate(date_start, date_end) \
            .filter(ee.Filter.lt('CLOUD_COVER', self.config['cloud_threshold']))
        
        # Landsat 9
        landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(aoi) \
            .filterDate(date_start, date_end) \
            .filter(ee.Filter.lt('CLOUD_COVER', self.config['cloud_threshold']))
        
        # Merge collections
        collection = landsat8.merge(landsat9)
        
        # Apply scaling factors
        def apply_scale_factors(image):
            optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(optical, None, True) \
                .addBands(thermal, None, True) \
                .copyProperties(image, ['system:time_start'])
        
        collection = collection.map(apply_scale_factors)
        
        logger.info(f"Found {collection.size().getInfo()} Landsat images")
        return collection
    
    def get_sentinel1_data(self, date_start: str, date_end: str, aoi: ee.Geometry) -> ee.ImageCollection:
        """Retrieve Sentinel-1 SAR data for all-weather monitoring"""
        logger.info(f"Fetching Sentinel-1 SAR data from {date_start} to {date_end}")
        
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(aoi) \
            .filterDate(date_start, date_end) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])
        
        # Apply speckle filter
        def apply_speckle_filter(image):
            # Simplified Lee filter
            return image.focal_mean(radius=3, units='pixels') \
                .copyProperties(image, ['system:time_start'])
        
        collection = collection.map(apply_speckle_filter)
        
        logger.info(f"Found {collection.size().getInfo()} Sentinel-1 images")
        return collection
    
    def export_gee_imagery(self, image: ee.Image, description: str, folder: str = 'earthquake_assessment'):
        """Export Earth Engine imagery to Google Drive"""
        aoi = self.create_aoi()
        
        export_params = {
            'image': image,
            'description': description,
            'folder': folder,
            'region': aoi,
            'scale': 10,  # 10m for Sentinel-2
            'maxPixels': 1e13,
            'fileFormat': 'GeoTIFF'
        }
        
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        
        logger.info(f"Export task '{description}' started. Check Google Drive folder '{folder}'")
        return task
    
    def get_planet_data(self, date_start: str, date_end: str, aoi_geojson: Dict) -> List[Dict]:
        """
        Retrieve Planet imagery metadata
        Requires Planet API key in config
        """
        api_key = self.config['apis'].get('planet_api_key')
        if not api_key:
            logger.warning("Planet API key not configured. Skipping Planet data.")
            return []
        
        logger.info(f"Fetching Planet data from {date_start} to {date_end}")
        
        # Planet API endpoint
        search_url = "https://api.planet.com/data/v1/quick-search"
        
        # Create search filter
        date_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {
                "gte": f"{date_start}T00:00:00.000Z",
                "lte": f"{date_end}T23:59:59.999Z"
            }
        }
        
        geo_filter = {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": aoi_geojson
        }
        
        cloud_filter = {
            "type": "RangeFilter",
            "field_name": "cloud_cover",
            "config": {
                "lte": self.config['cloud_threshold'] / 100.0
            }
        }
        
        # Combine filters
        combined_filter = {
            "type": "AndFilter",
            "config": [date_filter, geo_filter, cloud_filter]
        }
        
        # Search request
        search_request = {
            "item_types": ["PSScene"],
            "filter": combined_filter
        }
        
        # Make request
        headers = {'Authorization': f'api-key {api_key}'}
        response = requests.post(search_url, json=search_request, headers=headers)
        
        if response.status_code == 200:
            features = response.json().get('features', [])
            logger.info(f"Found {len(features)} Planet scenes")
            return features
        else:
            logger.error(f"Planet API error: {response.status_code}")
            return []
    
    def download_copernicus_emergency_maps(self):
        """
        Download Copernicus Emergency Management Service maps if available
        """
        ems_url = "https://emergency.copernicus.eu/mapping/list-of-components/EMSR"
        
        # Check for Nepal earthquake activation
        # This would need the actual EMSR number for the event
        logger.info("Checking Copernicus EMS for emergency activation maps...")
        
        # Save placeholder for manual download
        ems_dir = self.data_dir / 'ems_maps'
        ems_dir.mkdir(exist_ok=True)
        
        info_file = ems_dir / 'download_info.txt'
        with open(info_file, 'w') as f:
            f.write(f"Check {ems_url} for Nepal earthquake activation\n")
            f.write(f"Event date: {self.config['earthquake']['date']}\n")
            f.write(f"Look for damage assessment maps and grading products\n")
        
        logger.info(f"EMS download information saved to {info_file}")
    
    def collect_ancillary_data(self):
        """Collect additional data sources for validation"""
        ancillary_dir = self.data_dir / 'ancillary'
        ancillary_dir.mkdir(exist_ok=True)
        
        # 1. Download OpenStreetMap building footprints
        logger.info("Downloading OSM building footprints...")
        
        # Using Overpass API (simplified example)
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        # Query for buildings in affected area
        bbox = "28.5,81.8,29.0,82.7"  # min_lat,min_lon,max_lat,max_lon
        query = f"""
        [out:json];
        (
          way["building"]({bbox});
          relation["building"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.get(overpass_url, params={'data': query})
            if response.status_code == 200:
                osm_data = response.json()
                with open(ancillary_dir / 'osm_buildings.json', 'w') as f:
                    json.dump(osm_data, f)
                logger.info("OSM data downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download OSM data: {e}")
        
        # 2. Download SRTM elevation data
        logger.info("Preparing SRTM elevation data...")
        srtm = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(srtm)
        aspect = ee.Terrain.aspect(srtm)
        
        # Export elevation products
        self.export_gee_imagery(
            srtm.addBands([slope, aspect]).clip(self.create_aoi()),
            'nepal_elevation_slope_aspect',
            'earthquake_assessment'
        )
        
        # 3. Population density
        logger.info("Downloading population density data...")
        population = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density") \
            .filterDate('2020-01-01', '2020-12-31') \
            .mean()
        
        self.export_gee_imagery(
            population.clip(self.create_aoi()),
            'nepal_population_density_2020',
            'earthquake_assessment'
        )
    
    def create_metadata_catalog(self):
        """Create a metadata catalog of all acquired data"""
        metadata = {
            'acquisition_date': datetime.datetime.now().isoformat(),
            'earthquake_info': self.config['earthquake'],
            'date_ranges': self.config['date_ranges'],
            'data_sources': {
                'sentinel2': {'platform': 'ESA Copernicus', 'resolution': '10m'},
                'landsat': {'platform': 'USGS', 'resolution': '30m'},
                'sentinel1': {'platform': 'ESA Copernicus', 'type': 'SAR'},
                'planet': {'platform': 'Planet Labs', 'resolution': '3-5m'},
                'ancillary': {
                    'osm': 'OpenStreetMap building footprints',
                    'srtm': 'USGS SRTM elevation data',
                    'population': 'CIESIN GPW v4.11'
                }
            }
        }
        
        catalog_path = self.data_dir / 'data_catalog.json'
        with open(catalog_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Data catalog created at {catalog_path}")
    
    def run_acquisition_pipeline(self):
        """Run the complete data acquisition pipeline"""
        logger.info("Starting data acquisition pipeline for Nepal earthquake assessment")
        
        # Create AOI
        aoi = self.create_aoi()
        
        # Get district boundaries
        districts = self.get_district_boundaries()
        
        # Define date ranges
        pre_start = self.config['date_ranges']['pre_start']
        pre_end = self.config['date_ranges']['pre_end']
        post_start = self.config['date_ranges']['post_start']
        post_end = self.config['date_ranges']['post_end']
        
        # Collect satellite imagery
        logger.info("Collecting pre-earthquake imagery...")
        pre_s2 = self.get_sentinel2_data(pre_start, pre_end, aoi)
        pre_landsat = self.get_landsat_data(pre_start, pre_end, aoi)
        pre_s1 = self.get_sentinel1_data(pre_start, pre_end, aoi)
        
        logger.info("Collecting post-earthquake imagery...")
        post_s2 = self.get_sentinel2_data(post_start, post_end, aoi)
        post_landsat = self.get_landsat_data(post_start, post_end, aoi)
        post_s1 = self.get_sentinel1_data(post_start, post_end, aoi)
        
        # Create composites and export
        logger.info("Creating and exporting image composites...")
        
        # Sentinel-2 composites
        pre_s2_composite = pre_s2.median()
        post_s2_composite = post_s2.median()
        
        self.export_gee_imagery(pre_s2_composite, 'sentinel2_pre_earthquake')
        self.export_gee_imagery(post_s2_composite, 'sentinel2_post_earthquake')
        
        # Landsat composites
        if pre_landsat.size().getInfo() > 0:
            pre_landsat_composite = pre_landsat.median()
            self.export_gee_imagery(pre_landsat_composite, 'landsat_pre_earthquake')
        
        if post_landsat.size().getInfo() > 0:
            post_landsat_composite = post_landsat.median()
            self.export_gee_imagery(post_landsat_composite, 'landsat_post_earthquake')
        
        # Sentinel-1 composites
        pre_s1_composite = pre_s1.mean()
        post_s1_composite = post_s1.mean()
        
        self.export_gee_imagery(pre_s1_composite, 'sentinel1_pre_earthquake')
        self.export_gee_imagery(post_s1_composite, 'sentinel1_post_earthquake')
        
        # Get Planet data (if API key available)
        aoi_geojson = json.loads(districts.to_json())['features'][0]['geometry']
        planet_pre = self.get_planet_data(pre_start, pre_end, aoi_geojson)
        planet_post = self.get_planet_data(post_start, post_end, aoi_geojson)
        
        # Download emergency maps
        self.download_copernicus_emergency_maps()
        
        # Collect ancillary data
        self.collect_ancillary_data()
        
        # Create metadata catalog
        self.create_metadata_catalog()
        
        logger.info("Data acquisition pipeline completed successfully!")
        logger.info("Check Google Drive for exported imagery")
        logger.info(f"Local data saved to: {self.data_dir}")


def main():
    """Main execution function"""
    # Initialize data acquisition
    acquisition = EarthquakeDataAcquisition('config.json')
    
    # Run the pipeline
    acquisition.run_acquisition_pipeline()


if __name__ == "__main__":
    main()