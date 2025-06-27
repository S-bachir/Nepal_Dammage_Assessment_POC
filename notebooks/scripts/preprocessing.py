"""
02_preprocessing.py
Preprocessing Script for Nepal Earthquake Damage Assessment
Handles image preprocessing, co-registration, and normalization
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
import cv2
from skimage import exposure, morphology, filters
from skimage.exposure import match_histograms
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional
import concurrent.futures
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Handles preprocessing of satellite imagery for damage assessment
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize preprocessor with configuration"""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_dir'])
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(exist_ok=True)
        
        # Standard band mappings for different sensors
        self.band_mappings = {
            'sentinel2': {
                'blue': 'B2', 'green': 'B3', 'red': 'B4', 'nir': 'B8',
                'swir1': 'B11', 'swir2': 'B12', 'rededge1': 'B5',
                'rededge2': 'B6', 'rededge3': 'B7', 'rededge4': 'B8A'
            },
            'landsat8': {
                'blue': 'B2', 'green': 'B3', 'red': 'B4', 'nir': 'B5',
                'swir1': 'B6', 'swir2': 'B7', 'pan': 'B8'
            },
            'sentinel1': {
                'vv': 'VV', 'vh': 'VH'
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def read_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Read satellite image and metadata"""
        logger.info(f"Reading image: {image_path}")
        
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()
            
            # Get metadata
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'bounds': src.bounds
            }
            
            # Read band descriptions if available
            if src.descriptions:
                metadata['band_names'] = list(src.descriptions)
            
        return image, metadata
    
    def coregister_images(self, reference_path: str, target_path: str, 
                         output_path: str, method: str = 'feature') -> str:
        """
        Co-register target image to reference image
        Methods: 'feature' (feature-based) or 'phase' (phase correlation)
        """
        logger.info(f"Co-registering {target_path} to {reference_path}")
        
        # Read images
        ref_img, ref_meta = self.read_image(reference_path)
        tgt_img, tgt_meta = self.read_image(target_path)
        
        if method == 'feature':
            # Feature-based registration using SIFT
            shift = self._feature_based_registration(ref_img, tgt_img)
        else:
            # Phase correlation
            shift = self._phase_correlation_registration(ref_img, tgt_img)
        
        # Apply transformation
        registered_img = self._apply_shift(tgt_img, shift)
        
        # Save registered image
        self._save_image(registered_img, ref_meta, output_path)
        
        logger.info(f"Registration complete. Shift: {shift}")
        return output_path
    
    def _feature_based_registration(self, ref_img: np.ndarray, 
                                   tgt_img: np.ndarray) -> Tuple[float, float]:
        """Feature-based image registration using SIFT"""
        # Convert to 8-bit grayscale for feature detection
        ref_gray = self._to_8bit_gray(ref_img)
        tgt_gray = self._to_8bit_gray(tgt_img)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(ref_gray, None)
        kp2, des2 = sift.detectAndCompute(tgt_gray, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            logger.warning("Not enough matches found for registration")
            return (0, 0)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Find homography
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Extract translation from homography
        shift_x = M[0, 2]
        shift_y = M[1, 2]
        
        return (shift_x, shift_y)
    
    def _phase_correlation_registration(self, ref_img: np.ndarray, 
                                       tgt_img: np.ndarray) -> Tuple[float, float]:
        """Phase correlation for sub-pixel registration"""
        ref_gray = self._to_gray(ref_img)
        tgt_gray = self._to_gray(tgt_img)
        
        # Apply window to reduce edge effects
        window = np.outer(np.hanning(ref_gray.shape[0]), 
                         np.hanning(ref_gray.shape[1]))
        ref_windowed = ref_gray * window
        tgt_windowed = tgt_gray * window
        
        # Compute phase correlation
        shift, error, diffphase = cv2.phaseCorrelate(
            ref_windowed.astype(np.float32),
            tgt_windowed.astype(np.float32)
        )
        
        return shift
    
    def _to_8bit_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert multi-band image to 8-bit grayscale"""
        if len(image.shape) == 3:
            # Use NIR band if available, otherwise first band
            gray = image[min(3, image.shape[0]-1)]
        else:
            gray = image
        
        # Normalize to 0-255
        gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return gray_norm.astype(np.uint8)
    
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale (float)"""
        if len(image.shape) == 3:
            return image[min(3, image.shape[0]-1)]
        return image
    
    def _apply_shift(self, image: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
        """Apply sub-pixel shift to image"""
        shifted = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            shifted[i] = ndimage.shift(image[i], (shift[1], shift[0]), 
                                     order=3, mode='constant')
        
        return shifted
    
    def normalize_image(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize image values
        Methods: 'minmax', 'zscore', 'histogram'
        """
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            band = image[i].astype(np.float32)
            
            if method == 'minmax':
                # Min-max normalization
                band_min = np.percentile(band, 2)
                band_max = np.percentile(band, 98)
                normalized[i] = np.clip((band - band_min) / (band_max - band_min), 0, 1)
                
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(band)
                std = np.std(band)
                normalized[i] = (band - mean) / (std + 1e-8)
                
            elif method == 'histogram':
                # Histogram equalization
                band_uint = (band * 255).astype(np.uint8)
                equalized = cv2.equalizeHist(band_uint)
                normalized[i] = equalized.astype(np.float32) / 255.0
        
        return normalized
    
    def pan_sharpen(self, multispectral_path: str, panchromatic_path: str,
                    output_path: str, method: str = 'brovey') -> str:
        """
        Pan-sharpen multispectral image using panchromatic band
        Methods: 'brovey', 'ihs', 'pca'
        """
        logger.info(f"Pan-sharpening {multispectral_path}")
        
        # Read images
        ms_img, ms_meta = self.read_image(multispectral_path)
        pan_img, pan_meta = self.read_image(panchromatic_path)
        
        # Resample multispectral to panchromatic resolution
        ms_resampled = self._resample_to_pan(ms_img, ms_meta, pan_img, pan_meta)
        
        if method == 'brovey':
            sharpened = self._brovey_transform(ms_resampled, pan_img[0])
        elif method == 'ihs':
            sharpened = self._ihs_transform(ms_resampled, pan_img[0])
        else:  # pca
            sharpened = self._pca_transform(ms_resampled, pan_img[0])
        
        # Save result
        self._save_image(sharpened, pan_meta, output_path)
        
        logger.info(f"Pan-sharpening complete: {output_path}")
        return output_path
    
    def _brovey_transform(self, ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
        """Brovey transform for pan-sharpening"""
        # Calculate intensity as mean of RGB bands
        intensity = np.mean(ms[:3], axis=0)
        
        # Apply Brovey transform
        sharpened = np.zeros_like(ms)
        for i in range(ms.shape[0]):
            sharpened[i] = ms[i] * pan / (intensity + 1e-8)
        
        return sharpened
    
    def cloud_mask(self, image_path: str, output_path: str, 
                   cloud_threshold: float = 0.3) -> Tuple[str, float]:
        """
        Create and apply cloud mask
        Returns path to masked image and cloud percentage
        """
        logger.info(f"Creating cloud mask for {image_path}")
        
        image, metadata = self.read_image(image_path)
        
        # Simple cloud detection using brightness and NDSI
        # For Sentinel-2
        if 'sentinel2' in str(image_path).lower():
            cloud_mask = self._sentinel2_cloud_mask(image)
        else:
            # Generic brightness-based cloud detection
            cloud_mask = self._brightness_cloud_mask(image)
        
        # Calculate cloud percentage
        cloud_percent = (np.sum(cloud_mask) / cloud_mask.size) * 100
        
        # Apply mask
        masked_image = image.copy()
        for i in range(image.shape[0]):
            masked_image[i][cloud_mask] = np.nan
        
        # Save masked image
        self._save_image(masked_image, metadata, output_path)
        
        logger.info(f"Cloud percentage: {cloud_percent:.2f}%")
        return output_path, cloud_percent
    
    def _sentinel2_cloud_mask(self, image: np.ndarray) -> np.ndarray:
        """Cloud mask for Sentinel-2 imagery"""
        # Assuming bands: B2(blue), B3(green), B4(red), B8(NIR), B11(SWIR1)
        blue = image[0]
        green = image[1]
        red = image[2]
        nir = image[3] if image.shape[0] > 3 else red
        swir = image[4] if image.shape[0] > 4 else nir
        
        # Brightness test
        brightness = (blue + green + red) / 3
        bright_mask = brightness > 0.3
        
        # NDSI (Normalized Difference Snow Index)
        ndsi = (green - swir) / (green + swir + 1e-8)
        snow_mask = ndsi > 0.4
        
        # Combine masks
        cloud_mask = bright_mask & ~snow_mask
        
        # Morphological operations to clean up
        cloud_mask = morphology.opening(cloud_mask, morphology.disk(3))
        cloud_mask = morphology.closing(cloud_mask, morphology.disk(3))
        
        return cloud_mask
    
    def create_image_stack(self, image_paths: List[str], output_path: str,
                          band_indices: Optional[List[int]] = None) -> str:
        """Create multi-temporal or multi-sensor image stack"""
        logger.info(f"Creating image stack from {len(image_paths)} images")
        
        # Read first image for reference
        ref_img, ref_meta = self.read_image(image_paths[0])
        
        # Select bands if specified
        if band_indices:
            ref_img = ref_img[band_indices]
        
        # Initialize stack
        stack = [ref_img]
        
        # Add other images
        for img_path in image_paths[1:]:
            img, _ = self.read_image(img_path)
            if band_indices:
                img = img[band_indices]
            
            # Ensure same shape
            if img.shape[1:] != ref_img.shape[1:]:
                img = cv2.resize(img.transpose(1, 2, 0), 
                               (ref_img.shape[2], ref_img.shape[1])).transpose(2, 0, 1)
            
            stack.append(img)
        
        # Concatenate along band dimension
        stacked = np.concatenate(stack, axis=0)
        
        # Update metadata
        ref_meta['count'] = stacked.shape[0]
        
        # Save stack
        self._save_image(stacked, ref_meta, output_path)
        
        logger.info(f"Stack created with {stacked.shape[0]} bands")
        return output_path
    
    def apply_radiometric_correction(self, image_path: str, output_path: str,
                                   correction_type: str = 'dos') -> str:
        """
        Apply radiometric correction
        Types: 'dos' (Dark Object Subtraction), 'histogram_matching'
        """
        logger.info(f"Applying {correction_type} correction to {image_path}")
        
        image, metadata = self.read_image(image_path)
        
        if correction_type == 'dos':
            corrected = self._dark_object_subtraction(image)
        elif correction_type == 'histogram_matching':
            # Use reference image for matching
            ref_path = self.processed_dir / 'reference_image.tif'
            if ref_path.exists():
                ref_img, _ = self.read_image(str(ref_path))
                corrected = self._histogram_matching(image, ref_img)
            else:
                logger.warning("No reference image found for histogram matching")
                corrected = image
        else:
            corrected = image
        
        # Save corrected image
        self._save_image(corrected, metadata, output_path)
        
        return output_path
    
    def _dark_object_subtraction(self, image: np.ndarray) -> np.ndarray:
        """Dark Object Subtraction for atmospheric correction"""
        corrected = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            band = image[i]
            # Find dark object value (1st percentile)
            dark_value = np.percentile(band[band > 0], 1)
            # Subtract dark object value
            corrected[i] = np.maximum(band - dark_value, 0)
        
        return corrected
    
    def _histogram_matching(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of image to reference"""
        matched = np.zeros_like(image)
        
        for i in range(min(image.shape[0], reference.shape[0])):
            matched[i] = match_histograms(image[i], reference[i])
        
        # Copy remaining bands if any
        if image.shape[0] > reference.shape[0]:
            matched[reference.shape[0]:] = image[reference.shape[0]:]
        
        return matched
    
    def create_analysis_ready_data(self, pre_image_path: str, post_image_path: str,
                                  aoi_path: str) -> Dict[str, str]:
        """
        Create analysis-ready data (ARD) from raw imagery
        """
        logger.info("Creating analysis-ready data...")
        
        # Output paths
        ard_dir = self.processed_dir / 'ard'
        ard_dir.mkdir(exist_ok=True)
        
        outputs = {}
        
        # 1. Co-register post image to pre image
        coreg_post = ard_dir / 'post_coregistered.tif'
        self.coregister_images(pre_image_path, post_image_path, str(coreg_post))
        
        # 2. Apply radiometric correction
        pre_corrected = ard_dir / 'pre_corrected.tif'
        post_corrected = ard_dir / 'post_corrected.tif'
        
        self.apply_radiometric_correction(pre_image_path, str(pre_corrected))
        self.apply_radiometric_correction(str(coreg_post), str(post_corrected))
        
        # 3. Create cloud masks
        pre_masked = ard_dir / 'pre_masked.tif'
        post_masked = ard_dir / 'post_masked.tif'
        
        _, pre_cloud_pct = self.cloud_mask(str(pre_corrected), str(pre_masked))
        _, post_cloud_pct = self.cloud_mask(str(post_corrected), str(post_masked))
        
        # 4. Clip to AOI
        if aoi_path and Path(aoi_path).exists():
            pre_clipped = ard_dir / 'pre_ard.tif'
            post_clipped = ard_dir / 'post_ard.tif'
            
            self._clip_to_aoi(str(pre_masked), aoi_path, str(pre_clipped))
            self._clip_to_aoi(str(post_masked), aoi_path, str(post_clipped))
            
            outputs['pre'] = str(pre_clipped)
            outputs['post'] = str(post_clipped)
        else:
            outputs['pre'] = str(pre_masked)
            outputs['post'] = str(post_masked)
        
        # 5. Create metadata
        outputs['metadata'] = {
            'pre_cloud_percentage': pre_cloud_pct,
            'post_cloud_percentage': post_cloud_pct,
            'coregistration_applied': True,
            'radiometric_correction': 'dos',
            'processing_date': pd.Timestamp.now().isoformat()
        }
        
        # Save metadata
        with open(ard_dir / 'ard_metadata.json', 'w') as f:
            json.dump(outputs['metadata'], f, indent=4)
        
        logger.info("Analysis-ready data created successfully")
        return outputs
    
    def _clip_to_aoi(self, image_path: str, aoi_path: str, output_path: str):
        """Clip image to area of interest"""
        # Read AOI
        aoi = gpd.read_file(aoi_path)
        
        with rasterio.open(image_path) as src:
            # Get AOI in image CRS
            aoi_reproj = aoi.to_crs(src.crs)
            
            # Clip
            out_image, out_transform = mask(src, aoi_reproj.geometry, crop=True)
            
            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Save
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
    
    def _save_image(self, image: np.ndarray, metadata: Dict, output_path: str):
        """Save image with metadata"""
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata for saving
        save_meta = metadata.copy()
        save_meta.update({
            'driver': 'GTiff',
            'dtype': image.dtype,
            'count': image.shape[0],
            'compress': 'lzw'
        })
        
        # Save
        with rasterio.open(output_path, 'w', **save_meta) as dst:
            dst.write(image)
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     operations: List[str] = ['normalize', 'cloud_mask']):
        """Process multiple images with specified operations"""
        logger.info(f"Batch processing images in {input_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all GeoTIFF files
        image_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for img_file in image_files:
                future = executor.submit(
                    self._process_single_image,
                    img_file, output_path, operations
                )
                futures.append((img_file, future))
            
            for img_file, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Processed: {img_file.name}")
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
        
        # Save processing report
        report = pd.DataFrame(results)
        report.to_csv(output_path / 'processing_report.csv', index=False)
        
        logger.info(f"Batch processing complete. Processed {len(results)} images")
    
    def _process_single_image(self, img_path: Path, output_dir: Path, 
                            operations: List[str]) -> Dict:
        """Process single image with specified operations"""
        result = {'input': str(img_path), 'status': 'success'}
        
        try:
            # Read image
            image, metadata = self.read_image(str(img_path))
            current_path = str(img_path)
            
            # Apply operations in sequence
            for op in operations:
                output_name = f"{img_path.stem}_{op}{img_path.suffix}"
                output_path = output_dir / output_name
                
                if op == 'normalize':
                    image = self.normalize_image(image)
                    self._save_image(image, metadata, str(output_path))
                    
                elif op == 'cloud_mask':
                    current_path, cloud_pct = self.cloud_mask(
                        current_path, str(output_path)
                    )
                    result['cloud_percentage'] = cloud_pct
                    
                elif op == 'radiometric':
                    current_path = self.apply_radiometric_correction(
                        current_path, str(output_path)
                    )
                
                result[op] = str(output_path)
                current_path = str(output_path)
                
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result


def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = ImagePreprocessor('config.json')
    
    # Example: Create analysis-ready data
    data_dir = Path(preprocessor.config['data_dir'])
    
    # Paths to downloaded images (from data acquisition)
    pre_image = data_dir / 'downloads' / 'sentinel2_pre_earthquake.tif'
    post_image = data_dir / 'downloads' / 'sentinel2_post_earthquake.tif'
    aoi = data_dir / 'aoi' / 'affected_districts.geojson'
    
    if pre_image.exists() and post_image.exists():
        ard = preprocessor.create_analysis_ready_data(
            str(pre_image), 
            str(post_image),
            str(aoi) if aoi.exists() else None
        )
        logger.info(f"ARD created: {ard}")
    else:
        logger.warning("Input images not found. Run data acquisition first.")
        
        # Process any available images
        if (data_dir / 'downloads').exists():
            preprocessor.process_batch(
                str(data_dir / 'downloads'),
                str(preprocessor.processed_dir),
                operations=['normalize', 'cloud_mask', 'radiometric']
            )


if __name__ == "__main__":
    main()