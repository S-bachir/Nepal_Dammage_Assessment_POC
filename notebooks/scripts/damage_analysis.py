"""
03_damage_analysis.py
Damage Analysis Script for Nepal Earthquake Assessment
Implements various damage detection algorithms including GeoAI approaches
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage import feature, segmentation, morphology
from skimage.filters import sobel, gaussian
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import xgboost as xgb
from scipy import ndimage, stats
from shapely.geometry import Point, Polygon
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('damage_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DamageAnalyzer:
    """
    Comprehensive damage analysis using traditional and GeoAI methods
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize damage analyzer"""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_dir'])
        self.results_dir = self.data_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Damage thresholds
        self.damage_thresholds = {
            'ndvi_change': [-0.4, -0.25, -0.1],  # severe, high, moderate
            'nbr_change': [0.66, 0.44, 0.27, 0.1],  # thresholds for dNBR
            'coherence_loss': [0.7, 0.5, 0.3],  # for SAR coherence
            'texture_change': [0.4, 0.3, 0.2]  # for texture analysis
        }
        
        # GeoAI model paths (if pre-trained models available)
        self.model_paths = {
            'building_damage': 'models/building_damage_cnn.h5',
            'landslide': 'models/landslide_detector.pkl',
            'infrastructure': 'models/infrastructure_damage.h5'
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def calculate_spectral_indices(self, image_path: str) -> Dict[str, np.ndarray]:
        """Calculate various spectral indices for damage assessment"""
        logger.info(f"Calculating spectral indices for {image_path}")
        
        with rasterio.open(image_path) as src:
            # Read bands (assuming Sentinel-2 band order)
            blue = src.read(1).astype(float)
            green = src.read(2).astype(float)
            red = src.read(3).astype(float)
            nir = src.read(4).astype(float)
            swir1 = src.read(5).astype(float) if src.count > 4 else nir
            swir2 = src.read(6).astype(float) if src.count > 5 else swir1
            
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'width': src.width,
                'height': src.height
            }
        
        indices = {}
        
        # NDVI - Vegetation health
        indices['ndvi'] = (nir - red) / (nir + red + 1e-8)
        
        # NBR - Burn ratio (useful for landslides too)
        indices['nbr'] = (nir - swir2) / (nir + swir2 + 1e-8)
        
        # NDBI - Built-up index
        indices['ndbi'] = (swir1 - nir) / (swir1 + nir + 1e-8)
        
        # NDWI - Water index
        indices['ndwi'] = (green - nir) / (green + nir + 1e-8)
        
        # BSI - Bare Soil Index
        indices['bsi'] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)
        
        # UI - Urban Index
        indices['ui'] = (swir2 - nir) / (swir2 + nir + 1e-8)
        
        return indices, metadata
    
    def change_detection(self, pre_indices: Dict[str, np.ndarray],
                        post_indices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate change in spectral indices"""
        logger.info("Performing change detection analysis")
        
        changes = {}
        
        for index_name in pre_indices.keys():
            if index_name in post_indices:
                # Calculate difference
                change = post_indices[index_name] - pre_indices[index_name]
                changes[f'd{index_name}'] = change
                
                # Calculate relative change
                rel_change = change / (np.abs(pre_indices[index_name]) + 1e-8)
                changes[f'rel_{index_name}'] = rel_change
        
        return changes
    
    def texture_analysis(self, image_path: str, window_size: int = 13) -> Dict[str, np.ndarray]:
        """Extract texture features using GLCM"""
        logger.info(f"Performing texture analysis on {image_path}")
        
        with rasterio.open(image_path) as src:
            # Use red band for texture
            band = src.read(3)
            metadata = {'transform': src.transform, 'crs': src.crs}
        
        # Normalize to 0-255 for GLCM
        band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate GLCM properties
        textures = {}
        
        # Compute GLCM
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Initialize feature arrays
        textures['contrast'] = np.zeros_like(band, dtype=float)
        textures['dissimilarity'] = np.zeros_like(band, dtype=float)
        textures['homogeneity'] = np.zeros_like(band, dtype=float)
        textures['energy'] = np.zeros_like(band, dtype=float)
        textures['correlation'] = np.zeros_like(band, dtype=float)
        textures['entropy'] = np.zeros_like(band, dtype=float)
        
        # Calculate texture for sliding windows
        pad = window_size // 2
        padded = np.pad(band_norm, pad, mode='reflect')
        
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                
                # Calculate GLCM features
                glcm = self._calculate_glcm(window)
                
                textures['contrast'][i-pad, j-pad] = self._glcm_contrast(glcm)
                textures['homogeneity'][i-pad, j-pad] = self._glcm_homogeneity(glcm)
                textures['energy'][i-pad, j-pad] = self._glcm_energy(glcm)
                textures['entropy'][i-pad, j-pad] = self._glcm_entropy(glcm)
        
        return textures, metadata
    
    def _calculate_glcm(self, window: np.ndarray) -> np.ndarray:
        """Calculate Gray Level Co-occurrence Matrix"""
        levels = 256
        glcm = np.zeros((levels, levels), dtype=np.float64)
        
        # Simple GLCM for horizontal direction
        for i in range(window.shape[0]):
            for j in range(window.shape[1] - 1):
                glcm[window[i, j], window[i, j + 1]] += 1
        
        # Normalize
        glcm = glcm / (np.sum(glcm) + 1e-8)
        
        return glcm
    
    def _glcm_contrast(self, glcm: np.ndarray) -> float:
        """Calculate GLCM contrast"""
        i, j = np.ogrid[0:glcm.shape[0], 0:glcm.shape[1]]
        return np.sum(glcm * (i - j) ** 2)
    
    def _glcm_homogeneity(self, glcm: np.ndarray) -> float:
        """Calculate GLCM homogeneity"""
        i, j = np.ogrid[0:glcm.shape[0], 0:glcm.shape[1]]
        return np.sum(glcm / (1.0 + np.abs(i - j)))
    
    def _glcm_energy(self, glcm: np.ndarray) -> float:
        """Calculate GLCM energy"""
        return np.sum(glcm ** 2)
    
    def _glcm_entropy(self, glcm: np.ndarray) -> float:
        """Calculate GLCM entropy"""
        # Avoid log(0)
        glcm_pos = glcm[glcm > 0]
        return -np.sum(glcm_pos * np.log2(glcm_pos))
    
    def sar_coherence_analysis(self, pre_sar_path: str, post_sar_path: str) -> np.ndarray:
        """
        Analyze SAR coherence loss for damage detection
        Coherence loss indicates structural changes
        """
        logger.info("Performing SAR coherence analysis")
        
        # Read SAR data
        with rasterio.open(pre_sar_path) as src:
            pre_vv = src.read(1).astype(complex)
            pre_vh = src.read(2).astype(complex) if src.count > 1 else pre_vv
            metadata = {'transform': src.transform, 'crs': src.crs}
        
        with rasterio.open(post_sar_path) as src:
            post_vv = src.read(1).astype(complex)
            post_vh = src.read(2).astype(complex) if src.count > 1 else post_vv
        
        # Calculate coherence
        coherence_vv = self._calculate_coherence(pre_vv, post_vv)
        coherence_vh = self._calculate_coherence(pre_vh, post_vh)
        
        # Average coherence
        coherence = (coherence_vv + coherence_vh) / 2
        
        # Coherence loss (1 - coherence) indicates damage
        coherence_loss = 1 - coherence
        
        return coherence_loss, metadata
    
    def _calculate_coherence(self, img1: np.ndarray, img2: np.ndarray,
                           window_size: int = 5) -> np.ndarray:
        """Calculate coherence between two complex SAR images"""
        # Ensure complex type
        if img1.dtype != complex:
            img1 = img1.astype(complex)
        if img2.dtype != complex:
            img2 = img2.astype(complex)
        
        # Pad images
        pad = window_size // 2
        img1_pad = np.pad(img1, pad, mode='reflect')
        img2_pad = np.pad(img2, pad, mode='reflect')
        
        coherence = np.zeros(img1.shape, dtype=float)
        
        for i in range(pad, img1_pad.shape[0] - pad):
            for j in range(pad, img1_pad.shape[1] - pad):
                # Extract windows
                w1 = img1_pad[i-pad:i+pad+1, j-pad:j+pad+1]
                w2 = img2_pad[i-pad:i+pad+1, j-pad:j+pad+1]
                
                # Calculate coherence
                numerator = np.abs(np.sum(w1 * np.conj(w2)))
                denominator = np.sqrt(np.sum(np.abs(w1)**2) * np.sum(np.abs(w2)**2))
                
                coherence[i-pad, j-pad] = numerator / (denominator + 1e-8)
        
        return coherence
    
    def ml_damage_classification(self, features: np.ndarray, 
                                labels: Optional[np.ndarray] = None,
                                model_type: str = 'random_forest') -> Union[RandomForestClassifier, xgb.XGBClassifier]:
        """
        Machine learning based damage classification
        """
        logger.info(f"Training {model_type} classifier for damage assessment")
        
        if labels is None:
            # Unsupervised clustering for damage levels
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            labels = kmeans.fit_predict(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'gradient_boost':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            logger.info("\nTop 10 Feature Importances:")
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices):
                logger.info(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")
        
        return model, scaler
    
    def deep_learning_damage_detection(self, image_shape: Tuple[int, int, int],
                                     num_classes: int = 5) -> tf.keras.Model:
        """
        Build CNN model for damage detection using GeoAI approach
        """
        logger.info("Building deep learning model for damage detection")
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def building_damage_assessment(self, pre_image_path: str, post_image_path: str,
                                 building_footprints_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Assess damage to individual buildings
        """
        logger.info("Performing building-level damage assessment")
        
        # Calculate indices for both images
        pre_indices, metadata = self.calculate_spectral_indices(pre_image_path)
        post_indices, _ = self.calculate_spectral_indices(post_image_path)
        
        # Calculate changes
        changes = self.change_detection(pre_indices, post_indices)
        
        # Get texture changes
        pre_texture, _ = self.texture_analysis(pre_image_path)
        post_texture, _ = self.texture_analysis(post_image_path)
        
        texture_changes = {}
        for key in pre_texture.keys():
            texture_changes[f'd_{key}'] = post_texture[key] - pre_texture[key]
        
        # Load or create building footprints
        if building_footprints_path and Path(building_footprints_path).exists():
            buildings = gpd.read_file(building_footprints_path)
        else:
            # Create sample buildings (in practice, use actual footprints)
            buildings = self._create_sample_buildings(metadata)
        
        # Analyze each building
        damage_scores = []
        
        for idx, building in buildings.iterrows():
            # Get building mask
            mask = rasterize(
                [(building.geometry, 1)],
                out_shape=(metadata['height'], metadata['width']),
                transform=metadata['transform'],
                fill=0,
                dtype=np.uint8
            ).astype(bool)
            
            # Extract statistics for building
            stats_dict = {}
            
            # Spectral change statistics
            for key, change_map in changes.items():
                if mask.any():
                    values = change_map[mask]
                    stats_dict[f'{key}_mean'] = np.nanmean(values)
                    stats_dict[f'{key}_std'] = np.nanstd(values)
                    stats_dict[f'{key}_max'] = np.nanmax(values)
            
            # Texture change statistics
            for key, texture_map in texture_changes.items():
                if mask.any():
                    values = texture_map[mask]
                    stats_dict[f'{key}_mean'] = np.nanmean(values)
            
            # Calculate damage score
            damage_score = self._calculate_building_damage_score(stats_dict)
            damage_scores.append(damage_score)
        
        buildings['damage_score'] = damage_scores
        buildings['damage_class'] = pd.cut(
            buildings['damage_score'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['No damage', 'Minor', 'Moderate', 'Major', 'Destroyed']
        )
        
        # Save results
        output_path = self.results_dir / 'building_damage_assessment.geojson'
        buildings.to_file(output_path, driver='GeoJSON')
        
        logger.info(f"Building damage assessment saved to {output_path}")
        return buildings
    
    def _calculate_building_damage_score(self, stats: Dict[str, float]) -> float:
        """Calculate damage score for a building based on statistics"""
        score = 0.0
        weights = {
            'dndvi_mean': 0.3,
            'dnbr_mean': 0.2,
            'dndbi_mean': 0.2,
            'd_contrast_mean': 0.15,
            'd_homogeneity_mean': 0.15
        }
        
        for key, weight in weights.items():
            if key in stats:
                # Normalize to 0-1 based on expected ranges
                if 'ndvi' in key:
                    # NDVI decrease indicates damage
                    normalized = np.clip(-stats[key] / 0.5, 0, 1)
                elif 'nbr' in key:
                    # NBR increase indicates damage
                    normalized = np.clip(stats[key] / 0.5, 0, 1)
                elif 'ndbi' in key:
                    # NDBI change indicates structural change
                    normalized = np.clip(abs(stats[key]) / 0.3, 0, 1)
                else:
                    # Texture changes
                    normalized = np.clip(abs(stats[key]) / 0.5, 0, 1)
                
                score += weight * normalized
        
        return np.clip(score, 0, 1)
    
    def landslide_detection(self, pre_image_path: str, post_image_path: str,
                           slope_path: str) -> np.ndarray:
        """
        Detect landslides using spectral and topographic analysis
        """
        logger.info("Detecting landslides")
        
        # Calculate indices
        pre_indices, metadata = self.calculate_spectral_indices(pre_image_path)
        post_indices, _ = self.calculate_spectral_indices(post_image_path)
        
        # Load slope
        with rasterio.open(slope_path) as src:
            slope = src.read(1)
        
        # Calculate changes
        ndvi_change = post_indices['ndvi'] - pre_indices['ndvi']
        nbr_change = pre_indices['nbr'] - post_indices['nbr']  # dNBR
        
        # Landslide indicators:
        # 1. Significant NDVI decrease
        # 2. High dNBR values
        # 3. Steep slopes
        # 4. Bare soil increase
        
        landslide_probability = np.zeros_like(ndvi_change)
        
        # NDVI decrease weight
        ndvi_weight = np.clip(-ndvi_change / 0.3, 0, 1) * 0.3
        
        # NBR change weight
        nbr_weight = np.clip(nbr_change / 0.4, 0, 1) * 0.3
        
        # Slope weight (higher probability on steep slopes)
        slope_weight = np.clip(slope / 45, 0, 1) * 0.2
        
        # Bare soil increase
        bsi_change = post_indices['bsi'] - pre_indices['bsi']
        bsi_weight = np.clip(bsi_change / 0.3, 0, 1) * 0.2
        
        # Combine weights
        landslide_probability = (ndvi_weight + nbr_weight + 
                               slope_weight + bsi_weight)
        
        # Apply morphological operations to clean up
        landslide_binary = landslide_probability > 0.6
        landslide_binary = morphology.remove_small_objects(landslide_binary, 100)
        landslide_binary = morphology.binary_closing(landslide_binary, morphology.disk(3))
        
        # Save results
        output_path = self.results_dir / 'landslide_detection.tif'
        self._save_raster(landslide_probability, metadata, output_path)
        
        logger.info(f"Landslide detection saved to {output_path}")
        return landslide_probability
    
    def comprehensive_damage_assessment(self, pre_image_path: str, post_image_path: str,
                                      pre_sar_path: Optional[str] = None,
                                      post_sar_path: Optional[str] = None,
                                      building_footprints_path: Optional[str] = None,
                                      slope_path: Optional[str] = None) -> Dict:
        """
        Perform comprehensive damage assessment using all available data
        """
        logger.info("Starting comprehensive damage assessment")
        
        results = {}
        
        # 1. Spectral analysis
        pre_indices, metadata = self.calculate_spectral_indices(pre_image_path)
        post_indices, _ = self.calculate_spectral_indices(post_image_path)
        changes = self.change_detection(pre_indices, post_indices)
        
        # 2. Texture analysis
        pre_texture, _ = self.texture_analysis(pre_image_path)
        post_texture, _ = self.texture_analysis(post_image_path)
        
        # 3. SAR coherence (if available)
        if pre_sar_path and post_sar_path:
            coherence_loss, _ = self.sar_coherence_analysis(pre_sar_path, post_sar_path)
            results['coherence_loss'] = coherence_loss
        
        # 4. Create feature stack for ML
        feature_stack = []
        feature_names = []
        
        for name, array in changes.items():
            feature_stack.append(array.flatten())
            feature_names.append(name)
        
        for name in ['contrast', 'homogeneity', 'entropy']:
            texture_change = post_texture[name] - pre_texture[name]
            feature_stack.append(texture_change.flatten())
            feature_names.append(f'd_{name}')
        
        features = np.column_stack(feature_stack)
        
        # Remove NaN values
        valid_mask = ~np.any(np.isnan(features), axis=1)
        features_clean = features[valid_mask]
        
        # 5. ML-based damage classification
        if features_clean.shape[0] > 1000:  # Enough samples
            model, scaler = self.ml_damage_classification(features_clean)
            
            # Predict damage for all pixels
            features_scaled = scaler.transform(features)
            damage_prediction = np.full(features.shape[0], -1)
            damage_prediction[valid_mask] = model.predict(features_scaled[valid_mask])
            
            # Reshape to image
            damage_map = damage_prediction.reshape(metadata['height'], metadata['width'])
            results['damage_map'] = damage_map
            
            # Save damage map
            output_path = self.results_dir / 'damage_classification.tif'
            self._save_raster(damage_map.astype(np.float32), metadata, output_path)
        
        # 6. Building damage assessment
        if building_footprints_path:
            buildings = self.building_damage_assessment(
                pre_image_path, post_image_path, building_footprints_path
            )
            results['building_damage'] = buildings
        
        # 7. Landslide detection
        if slope_path:
            landslides = self.landslide_detection(
                pre_image_path, post_image_path, slope_path
            )
            results['landslides'] = landslides
        
        # 8. Generate statistics
        stats = self._generate_damage_statistics(results, metadata)
        results['statistics'] = stats
        
        # Save comprehensive report
        report_path = self.results_dir / 'damage_assessment_report.json'
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        logger.info(f"Comprehensive assessment complete. Report saved to {report_path}")
        return results
    
    def _generate_damage_statistics(self, results: Dict, metadata: Dict) -> Dict:
        """Generate statistics from damage assessment results"""
        stats = {
            'assessment_date': pd.Timestamp.now().isoformat(),
            'total_area_km2': (metadata['width'] * metadata['height'] * 100) / 1e6,  # assuming 10m pixels
        }
        
        if 'damage_map' in results:
            damage_map = results['damage_map']
            unique, counts = np.unique(damage_map[damage_map >= 0], return_counts=True)
            
            damage_labels = ['No damage', 'Low', 'Moderate', 'High', 'Severe']
            stats['damage_distribution'] = {}
            
            for i, count in zip(unique, counts):
                if i < len(damage_labels):
                    label = damage_labels[int(i)]
                    area_km2 = (count * 100) / 1e6
                    stats['damage_distribution'][label] = {
                        'pixels': int(count),
                        'area_km2': round(area_km2, 2),
                        'percentage': round(count / damage_map.size * 100, 2)
                    }
        
        if 'building_damage' in results:
            buildings = results['building_damage']
            stats['building_damage'] = {
                'total_buildings': len(buildings),
                'damage_summary': buildings['damage_class'].value_counts().to_dict()
            }
        
        if 'landslides' in results:
            landslide_pixels = np.sum(results['landslides'] > 0.6)
            stats['landslide_area_km2'] = round((landslide_pixels * 100) / 1e6, 2)
        
        return stats
    
    def _save_raster(self, array: np.ndarray, metadata: Dict, output_path: str):
        """Save array as GeoTIFF"""
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # Update metadata
        meta = metadata.copy()
        meta.update({
            'driver': 'GTiff',
            'dtype': array.dtype,
            'count': 1 if len(array.shape) == 2 else array.shape[0],
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            if len(array.shape) == 2:
                dst.write(array, 1)
            else:
                dst.write(array)
    
    def _create_sample_buildings(self, metadata: Dict) -> gpd.GeoDataFrame:
        """Create sample building footprints for demonstration"""
        # In practice, load actual building footprints
        buildings = []
        
        # Create a grid of sample buildings
        transform = metadata['transform']
        
        for i in range(10, metadata['height']-10, 50):
            for j in range(10, metadata['width']-10, 50):
                # Convert pixel to coordinates
                x, y = transform * (j, i)
                
                # Create building polygon (20x20m)
                building = Polygon([
                    (x, y), (x+20, y), (x+20, y-20), (x, y-20)
                ])
                
                buildings.append({
                    'geometry': building,
                    'building_id': f'BLD_{i}_{j}'
                })
        
        gdf = gpd.GeoDataFrame(buildings)
        gdf.crs = metadata['crs']
        
        return gdf


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = DamageAnalyzer('config.json')
    
    # Define input paths
    data_dir = Path(analyzer.config['data_dir'])
    processed_dir = data_dir / 'processed' / 'ard'
    
    # Input files
    pre_image = processed_dir / 'pre_ard.tif'
    post_image = processed_dir / 'post_ard.tif'
    pre_sar = processed_dir / 'sentinel1_pre_earthquake.tif'
    post_sar = processed_dir / 'sentinel1_post_earthquake.tif'
    buildings = data_dir / 'ancillary' / 'building_footprints.geojson'
    slope = data_dir / 'ancillary' / 'slope.tif'
    
    # Check if required files exist
    if not pre_image.exists() or not post_image.exists():
        logger.error("Pre/post images not found. Run preprocessing first.")
        return
    
    # Run comprehensive assessment
    results = analyzer.comprehensive_damage_assessment(
        str(pre_image),
        str(post_image),
        str(pre_sar) if pre_sar.exists() else None,
        str(post_sar) if post_sar.exists() else None,
        str(buildings) if buildings.exists() else None,
        str(slope) if slope.exists() else None
    )
    
    logger.info("Damage analysis complete!")
    logger.info(f"Results saved to: {analyzer.results_dir}")


if __name__ == "__main__":
    main()