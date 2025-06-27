"""
04_visualization.py
Visualization Script for Nepal Earthquake Damage Assessment
Creates interactive maps, damage visualizations, and comparison views
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show, plotting_extent
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML
import cv2
from pathlib import Path
import json
import base64
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DamageVisualizer:
    """
    Create visualizations for earthquake damage assessment
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize visualizer"""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_dir'])
        self.results_dir = self.data_dir / 'results'
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # Color schemes
        self.damage_colors = {
            'No damage': '#2ecc71',
            'Low': '#f1c40f',
            'Moderate': '#e67e22',
            'High': '#e74c3c',
            'Severe': '#8b0000'
        }
        
        self.landslide_cmap = ListedColormap(['white', '#8B4513'])
        
        # Damage colormap
        self.damage_cmap = ListedColormap([
            '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b0000'
        ])
        self.damage_bounds = [0, 1, 2, 3, 4, 5]
        self.damage_norm = BoundaryNorm(self.damage_bounds, self.damage_cmap.N)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_before_after_comparison(self, pre_image_path: str, post_image_path: str,
                                     output_path: Optional[str] = None) -> plt.Figure:
        """Create side-by-side before/after comparison"""
        logger.info("Creating before/after comparison visualization")
        
        # Read images
        with rasterio.open(pre_image_path) as src:
            pre_img = src.read([3, 2, 1])  # RGB bands
            pre_img = np.moveaxis(pre_img, 0, -1)
            extent = plotting_extent(src)
            transform = src.transform
        
        with rasterio.open(post_image_path) as src:
            post_img = src.read([3, 2, 1])
            post_img = np.moveaxis(post_img, 0, -1)
        
        # Normalize for display
        pre_img = self._normalize_image(pre_img)
        post_img = self._normalize_image(post_img)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot images
        ax1.imshow(pre_img, extent=extent)
        ax1.set_title('Pre-Earthquake', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        ax2.imshow(post_img, extent=extent)
        ax2.set_title('Post-Earthquake', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        # Add earthquake info
        eq_date = self.config['earthquake']['date']
        eq_mag = self.config['earthquake']['magnitude']
        fig.suptitle(f'Nepal Earthquake ({eq_date}, M{eq_mag}) - Before/After Comparison',
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison to {output_path}")
        
        return fig
    
    def visualize_damage_map(self, damage_map_path: str, 
                           output_path: Optional[str] = None) -> plt.Figure:
        """Visualize damage classification map"""
        logger.info("Creating damage map visualization")
        
        # Read damage map
        with rasterio.open(damage_map_path) as src:
            damage_map = src.read(1)
            extent = plotting_extent(src)
            transform = src.transform
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot damage map
        im = ax.imshow(damage_map, extent=extent, cmap=self.damage_cmap,
                      norm=self.damage_norm, interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Damage Severity', fontsize=12)
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_ticklabels(['No damage', 'Low', 'Moderate', 'High', 'Severe'])
        
        # Add title and labels
        ax.set_title('Earthquake Damage Assessment Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add epicenter
        epicenter = self.config['earthquake']['epicenter']
        ax.plot(epicenter[0], epicenter[1], 'r*', markersize=20, 
               markeredgecolor='white', markeredgewidth=2, label='Epicenter')
        
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved damage map to {output_path}")
        
        return fig
    
    def create_interactive_map(self, damage_data: Union[str, gpd.GeoDataFrame],
                             output_path: Optional[str] = None) -> folium.Map:
        """Create interactive Folium map with damage data"""
        logger.info("Creating interactive damage map")
        
        # Load damage data
        if isinstance(damage_data, str):
            if damage_data.endswith('.geojson'):
                damage_gdf = gpd.read_file(damage_data)
            else:
                # Assume it's a raster, convert to polygons
                damage_gdf = self._raster_to_polygons(damage_data)
        else:
            damage_gdf = damage_data
        
        # Get epicenter
        epicenter = self.config['earthquake']['epicenter']
        
        # Create base map
        m = folium.Map(
            location=[epicenter[1], epicenter[0]],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add satellite imagery option
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add damage layer
        if 'damage_class' in damage_gdf.columns:
            # Building damage
            for idx, row in damage_gdf.iterrows():
                color = self.damage_colors.get(row['damage_class'], '#cccccc')
                
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                ).add_child(
                    folium.Tooltip(f"Building: {row.get('building_id', idx)}<br>"
                                 f"Damage: {row['damage_class']}<br>"
                                 f"Score: {row.get('damage_score', 'N/A'):.2f}")
                ).add_to(m)
        else:
            # Area damage
            folium.Choropleth(
                geo_data=damage_gdf.to_json(),
                name='Damage Assessment',
                data=damage_gdf,
                columns=['id', 'damage_level'],
                key_on='feature.properties.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Damage Level'
            ).add_to(m)
        
        # Add epicenter marker
        folium.Marker(
            [epicenter[1], epicenter[0]],
            popup=f"Epicenter<br>Magnitude: {self.config['earthquake']['magnitude']}",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)
        
        # Add affected districts
        for district in self.config['earthquake']['affected_districts']:
            folium.Marker(
                [epicenter[1] + np.random.uniform(-0.3, 0.3), 
                 epicenter[0] + np.random.uniform(-0.3, 0.3)],
                popup=f"{district} District",
                icon=folium.Icon(color='orange', icon='info-sign')
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Add measurement tool
        plugins.MeasureControl().add_to(m)
        
        # Save map
        if output_path:
            m.save(output_path)
            logger.info(f"Saved interactive map to {output_path}")
        
        return m
    
    def plot_damage_statistics(self, stats_path: str, 
                             output_path: Optional[str] = None) -> plt.Figure:
        """Create statistical plots for damage assessment"""
        logger.info("Creating damage statistics visualizations")
        
        # Load statistics
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Create subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Damage distribution pie chart
        ax1 = fig.add_subplot(gs[0, :2])
        if 'damage_distribution' in stats:
            damage_dist = stats['damage_distribution']
            labels = list(damage_dist.keys())
            sizes = [d['percentage'] for d in damage_dist.values()]
            colors = [self.damage_colors.get(l, '#cccccc') for l in labels]
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax1.set_title('Overall Damage Distribution', fontsize=14, fontweight='bold')
        
        # 2. Area affected bar chart
        ax2 = fig.add_subplot(gs[0, 2])
        if 'damage_distribution' in stats:
            areas = [d['area_km2'] for d in damage_dist.values()]
            y_pos = np.arange(len(labels))
            
            bars = ax2.barh(y_pos, areas, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel('Area (km²)')
            ax2.set_title('Affected Area by Damage Level', fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (bar, area) in enumerate(zip(bars, areas)):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{area:.1f}', va='center')
        
        # 3. Building damage summary
        ax3 = fig.add_subplot(gs[1, :])
        if 'building_damage' in stats:
            building_damage = stats['building_damage']['damage_summary']
            
            categories = list(building_damage.keys())
            values = list(building_damage.values())
            colors_building = [self.damage_colors.get(c, '#cccccc') for c in categories]
            
            bars = ax3.bar(categories, values, color=colors_building)
            ax3.set_xlabel('Damage Category')
            ax3.set_ylabel('Number of Buildings')
            ax3.set_title(f'Building Damage Assessment (Total: {stats["building_damage"]["total_buildings"]})',
                         fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        # 4. Summary statistics text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        summary_text = f"""
        EARTHQUAKE DAMAGE ASSESSMENT SUMMARY
        
        Event Date: {self.config['earthquake']['date']}
        Magnitude: {self.config['earthquake']['magnitude']}
        Epicenter: {self.config['earthquake']['epicenter']}
        
        Total Area Assessed: {stats.get('total_area_km2', 'N/A'):.2f} km²
        Assessment Date: {stats.get('assessment_date', 'N/A')}
        """
        
        if 'landslide_area_km2' in stats:
            summary_text += f"\nLandslide Affected Area: {stats['landslide_area_km2']:.2f} km²"
        
        ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Nepal Earthquake Damage Assessment Statistics', 
                    fontsize=16, fontweight='bold')
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics plot to {output_path}")
        
        return fig
    
    def create_3d_damage_visualization(self, damage_map_path: str, dem_path: str,
                                     output_path: Optional[str] = None) -> go.Figure:
        """Create 3D visualization of damage overlaid on terrain"""
        logger.info("Creating 3D damage visualization")
        
        # Read data
        with rasterio.open(damage_map_path) as src:
            damage = src.read(1)
            transform = src.transform
        
        with rasterio.open(dem_path) as src:
            elevation = src.read(1)
        
        # Downsample for performance
        step = 10
        damage_ds = damage[::step, ::step]
        elevation_ds = elevation[::step, ::step]
        
        # Create coordinate grids
        rows, cols = damage_ds.shape
        x = np.arange(cols) * step * transform[0] + transform[2]
        y = np.arange(rows) * step * transform[4] + transform[5]
        X, Y = np.meshgrid(x, y)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=elevation_ds,
            surfacecolor=damage_ds,
            colorscale=[[0, '#2ecc71'], [0.25, '#f1c40f'], 
                       [0.5, '#e67e22'], [0.75, '#e74c3c'], [1, '#8b0000']],
            colorbar=dict(
                title="Damage Level",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=['No damage', 'Low', 'Moderate', 'High', 'Severe']
            )
        )])
        
        # Update layout
        fig.update_layout(
            title='3D Terrain with Damage Assessment',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved 3D visualization to {output_path}")
        
        return fig
    
    def create_temporal_change_animation(self, image_series: List[str],
                                       output_path: Optional[str] = None):
        """Create animation showing temporal changes"""
        logger.info("Creating temporal change animation")
        
        frames = []
        
        for img_path in image_series:
            with rasterio.open(img_path) as src:
                img = src.read([3, 2, 1])
                img = np.moveaxis(img, 0, -1)
                img = self._normalize_image(img)
                
                # Convert to uint8
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Add timestamp
                timestamp = Path(img_path).stem
                cv2.putText(img_uint8, timestamp, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                frames.append(img_uint8)
        
        if output_path and frames:
            # Create video
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 2.0, (width, height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            logger.info(f"Saved animation to {output_path}")
    
    def create_damage_report_dashboard(self, results_dir: str,
                                     output_path: Optional[str] = None) -> str:
        """Create comprehensive HTML dashboard with all visualizations"""
        logger.info("Creating damage report dashboard")
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Nepal Earthquake Damage Assessment Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                .info-box {{
                    background-color: #ecf0f1;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    box-shadow: 0 0 5px rgba(0,0,0,0.3);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: #3498db;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 36px;
                    font-weight: bold;
                }}
                .stat-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                iframe {{
                    width: 100%;
                    height: 600px;
                    border: none;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Nepal Earthquake Damage Assessment Report</h1>
                
                <div class="info-box">
                    <h3>Event Information</h3>
                    <p><strong>Date:</strong> {event_date}</p>
                    <p><strong>Magnitude:</strong> {magnitude}</p>
                    <p><strong>Epicenter:</strong> {epicenter}</p>
                    <p><strong>Affected Districts:</strong> {districts}</p>
                    <p><strong>Assessment Date:</strong> {assessment_date}</p>
                </div>
                
                <h2>Key Statistics</h2>
                <div class="stats-grid">
                    {stat_cards}
                </div>
                
                <h2>Before/After Comparison</h2>
                <div class="image-container">
                    <img src="{before_after_img}" alt="Before/After Comparison">
                </div>
                
                <h2>Damage Assessment Map</h2>
                <div class="image-container">
                    <img src="{damage_map_img}" alt="Damage Assessment Map">
                </div>
                
                <h2>Interactive Map</h2>
                <iframe src="{interactive_map}"></iframe>
                
                <h2>Damage Statistics</h2>
                <div class="image-container">
                    <img src="{stats_plot}" alt="Damage Statistics">
                </div>
                
                <h2>3D Terrain Visualization</h2>
                <iframe src="{terrain_3d}"></iframe>
                
                <div class="info-box">
                    <h3>Methodology</h3>
                    <p>This assessment was conducted using satellite imagery analysis, including:</p>
                    <ul>
                        <li>Spectral change detection (NDVI, NBR, NDBI)</li>
                        <li>Texture analysis for structural damage</li>
                        <li>Machine learning classification</li>
                        <li>SAR coherence analysis (where available)</li>
                    </ul>
                </div>
                
                <div class="info-box">
                    <h3>Data Sources</h3>
                    <ul>
                        <li>Sentinel-2 optical imagery (ESA Copernicus)</li>
                        <li>Sentinel-1 SAR data (ESA Copernicus)</li>
                        <li>Landsat 8/9 imagery (USGS)</li>
                        <li>OpenStreetMap building footprints</li>
                        <li>SRTM elevation data</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Load statistics
        stats_path = Path(results_dir) / 'damage_assessment_report.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}
        
        # Create stat cards
        stat_cards = ""
        if 'damage_distribution' in stats:
            total_damaged = sum(d['area_km2'] for k, d in stats['damage_distribution'].items() 
                              if k != 'No damage')
            stat_cards += f"""
            <div class="stat-card" style="background-color: #e74c3c;">
                <div class="stat-value">{total_damaged:.1f} km²</div>
                <div class="stat-label">Total Damaged Area</div>
            </div>
            """
        
        if 'building_damage' in stats:
            total_buildings = stats['building_damage']['total_buildings']
            stat_cards += f"""
            <div class="stat-card" style="background-color: #f39c12;">
                <div class="stat-value">{total_buildings}</div>
                <div class="stat-label">Buildings Assessed</div>
            </div>
            """
        
        if 'landslide_area_km2' in stats:
            stat_cards += f"""
            <div class="stat-card" style="background-color: #8B4513;">
                <div class="stat-value">{stats['landslide_area_km2']:.1f} km²</div>
                <div class="stat-label">Landslide Area</div>
            </div>
            """
        
        # Convert images to base64 for embedding
        def img_to_base64(img_path):
            if Path(img_path).exists():
                with open(img_path, 'rb') as f:
                    return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
            return ""
        
        # Fill template
        html_content = html_template.format(
            event_date=self.config['earthquake']['date'],
            magnitude=self.config['earthquake']['magnitude'],
            epicenter=f"{self.config['earthquake']['epicenter'][1]:.4f}°N, {self.config['earthquake']['epicenter'][0]:.4f}°E",
            districts=', '.join(self.config['earthquake']['affected_districts']),
            assessment_date=stats.get('assessment_date', 'N/A'),
            stat_cards=stat_cards,
            before_after_img=img_to_base64(str(self.viz_dir / 'before_after_comparison.png')),
            damage_map_img=img_to_base64(str(self.viz_dir / 'damage_map.png')),
            interactive_map='interactive_damage_map.html',
            stats_plot=img_to_base64(str(self.viz_dir / 'damage_statistics.png')),
            terrain_3d='3d_terrain_damage.html'
        )
        
        # Save dashboard
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            logger.info(f"Saved dashboard to {output_path}")
        
        return html_content
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image for display"""
        img_norm = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            band = img[:, :, i].astype(np.float32)
            p2, p98 = np.percentile(band[~np.isnan(band)], (2, 98))
            img_norm[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        return img_norm
    
    def _raster_to_polygons(self, raster_path: str) -> gpd.GeoDataFrame:
        """Convert damage raster to polygons for visualization"""
        with rasterio.open(raster_path) as src:
            image = src.read(1)
            transform = src.transform
            
        # Simplify by grouping similar values
        simplified = np.round(image).astype(int)
        
        # Extract shapes
        from rasterio.features import shapes
        
        polygons = []
        for geom, value in shapes(simplified, transform=transform):
            if value > 0:  # Ignore no-damage areas
                polygons.append({
                    'geometry': geom,
                    'damage_level': int(value),
                    'id': len(polygons)
                })
        
        gdf = gpd.GeoDataFrame(polygons)
        gdf.crs = src.crs
        
        return gdf
    
    def create_all_visualizations(self, results_dir: str):
        """Create all visualizations from results directory"""
        logger.info("Creating all visualizations")
        
        results_path = Path(results_dir)
        
        # 1. Before/After comparison
        pre_img = results_path.parent / 'processed' / 'ard' / 'pre_ard.tif'
        post_img = results_path.parent / 'processed' / 'ard' / 'post_ard.tif'
        
        if pre_img.exists() and post_img.exists():
            self.create_before_after_comparison(
                str(pre_img), str(post_img),
                str(self.viz_dir / 'before_after_comparison.png')
            )
        
        # 2. Damage map
        damage_map = results_path / 'damage_classification.tif'
        if damage_map.exists():
            self.visualize_damage_map(
                str(damage_map),
                str(self.viz_dir / 'damage_map.png')
            )
        
        # 3. Interactive map
        building_damage = results_path / 'building_damage_assessment.geojson'
        if building_damage.exists():
            self.create_interactive_map(
                str(building_damage),
                str(self.viz_dir / 'interactive_damage_map.html')
            )
        elif damage_map.exists():
            self.create_interactive_map(
                str(damage_map),
                str(self.viz_dir / 'interactive_damage_map.html')
            )
        
        # 4. Statistics plot
        stats_file = results_path / 'damage_assessment_report.json'
        if stats_file.exists():
            self.plot_damage_statistics(
                str(stats_file),
                str(self.viz_dir / 'damage_statistics.png')
            )
        
        # 5. 3D visualization (if DEM available)
        dem = results_path.parent / 'ancillary' / 'elevation.tif'
        if damage_map.exists() and dem.exists():
            self.create_3d_damage_visualization(
                str(damage_map), str(dem),
                str(self.viz_dir / '3d_terrain_damage.html')
            )
        
        # 6. Create dashboard
        self.create_damage_report_dashboard(
            str(results_path),
            str(self.viz_dir / 'damage_assessment_dashboard.html')
        )
        
        logger.info(f"All visualizations created in {self.viz_dir}")


def main():
    """Main execution function"""
    # Initialize visualizer
    visualizer = DamageVisualizer('config.json')
    
    # Create all visualizations
    results_dir = visualizer.data_dir / 'results'
    
    if results_dir.exists():
        visualizer.create_all_visualizations(str(results_dir))
        logger.info("Visualization complete!")
        logger.info(f"Open {visualizer.viz_dir / 'damage_assessment_dashboard.html'} to view the report")
    else:
        logger.error("Results directory not found. Run damage analysis first.")


if __name__ == "__main__":
    main()