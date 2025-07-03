"""
Google Drive Download Helper for Earth Engine Exports
Helps download completed exports from Google Drive to local directory
"""

import os
import json
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import pickle
import logging

logger = logging.getLogger(__name__)

class GoogleDriveDownloader:
    """Download Earth Engine exports from Google Drive"""
    
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    def __init__(self, local_dir: str = 'data/downloads'):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.service = self._authenticate()
    
    def _authenticate(self):
        """Authenticate and return Google Drive service"""
        creds = None
        token_file = 'token.pickle'
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # You'll need to download credentials.json from Google Cloud Console
                if os.path.exists('credentials.json'):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        'credentials.json', self.SCOPES)
                    creds = flow.run_local_server(port=0)
                else:
                    logger.error("credentials.json not found!")
                    logger.info("Please follow these steps:")
                    logger.info("1. Go to https://console.cloud.google.com/")
                    logger.info("2. Create/select a project")
                    logger.info("3. Enable Google Drive API")
                    logger.info("4. Create credentials (OAuth 2.0 Client ID)")
                    logger.info("5. Download as credentials.json")
                    return None
            
            # Save the credentials for the next run
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('drive', 'v3', credentials=creds)
    
    def list_files_in_folder(self, folder_name: str = 'earthquake_assessment'):
        """List all files in specified Google Drive folder"""
        if not self.service:
            return []
        
        try:
            # First, find the folder
            folder_query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            folder_results = self.service.files().list(
                q=folder_query,
                fields="files(id, name)"
            ).execute()
            
            folders = folder_results.get('files', [])
            if not folders:
                logger.error(f"Folder '{folder_name}' not found in Google Drive")
                return []
            
            folder_id = folders[0]['id']
            logger.info(f"Found folder: {folder_name} (ID: {folder_id})")
            
            # List files in the folder
            files_query = f"'{folder_id}' in parents"
            results = self.service.files().list(
                q=files_query,
                fields="files(id, name, size, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            
            logger.info(f"\nFiles in '{folder_name}':")
            logger.info("-" * 60)
            for file in files:
                size_mb = int(file.get('size', 0)) / (1024 * 1024)
                logger.info(f"{file['name']:<40} {size_mb:>10.1f} MB")
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str):
        """Download a file from Google Drive"""
        if not self.service:
            return False
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_path = self.local_dir / file_name
            
            logger.info(f"Downloading: {file_name}")
            
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logger.info(f"  Progress: {int(status.progress() * 100)}%")
            
            # Write to file
            fh.seek(0)
            with open(file_path, 'wb') as f:
                f.write(fh.read())
            
            logger.info(f"✅ Downloaded to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {file_name}: {e}")
            return False
    
    def download_earthquake_data(self, folder_name: str = 'earthquake_assessment'):
        """Download all earthquake assessment data"""
        logger.info("="*60)
        logger.info("GOOGLE DRIVE DOWNLOAD HELPER")
        logger.info("="*60)
        
        # List files
        files = self.list_files_in_folder(folder_name)
        
        if not files:
            logger.warning("No files found to download")
            return
        
        # Filter for GeoTIFF files
        tif_files = [f for f in files if f['name'].endswith('.tif')]
        
        logger.info(f"\nFound {len(tif_files)} GeoTIFF files to download")
        
        # Download each file
        downloaded = 0
        for file in tif_files:
            file_path = self.local_dir / file['name']
            
            # Skip if already exists
            if file_path.exists():
                logger.info(f"⏭️ Skipping {file['name']} (already exists)")
                continue
            
            if self.download_file(file['id'], file['name']):
                downloaded += 1
        
        logger.info(f"\n✅ Download complete! {downloaded} new files downloaded")
        logger.info(f"📁 Files saved to: {self.local_dir.absolute()}")


# Alternative: Manual download instructions
def create_manual_download_instructions(export_results: Dict, output_dir: str = 'data/downloads'):
    """Create manual download instructions for users without API setup"""
    
    instructions_path = Path(output_dir) / 'download_instructions.txt'
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(instructions_path, 'w') as f:
        f.write("MANUAL DOWNLOAD INSTRUCTIONS\n")
        f.write("="*60 + "\n\n")
        f.write("1. Open Google Drive in your browser\n")
        f.write("2. Navigate to the 'earthquake_assessment' folder\n")
        f.write("3. Download the following files:\n\n")
        
        if 'completed' in export_results:
            for task in export_results['completed']:
                f.write(f"   ✓ {task['description']}.tif")
                if task.get('estimated_size_mb'):
                    f.write(f" (~{task['estimated_size_mb']:.1f} MB)")
                f.write("\n")
        
        f.write(f"\n4. Save all files to: {Path(output_dir).absolute()}\n")
        f.write("\n5. Once downloaded, you can proceed with preprocessing\n")
    
    logger.info(f"\n📄 Manual download instructions saved to: {instructions_path}")
    
    # Also create a simple download checklist
    checklist_path = Path(output_dir) / 'download_checklist.json'
    expected_files = [
        'sentinel2_pre_earthquake.tif',
        'sentinel2_post_earthquake.tif',
        'landsat_pre_earthquake.tif',
        'landsat_post_earthquake.tif',
        'sentinel1_pre_earthquake.tif',
        'sentinel1_post_earthquake.tif',
        'nepal_elevation_slope_aspect.tif',
        'nepal_population_density_2020.tif'
    ]
    
    checklist = {
        'expected_files': expected_files,
        'download_status': {f: os.path.exists(Path(output_dir) / f) for f in expected_files},
        'completed_exports': [task['description'] + '.tif' for task in export_results.get('completed', [])]
    }
    
    with open(checklist_path, 'w') as f:
        json.dump(checklist, f, indent=2)
    
    logger.info(f"📋 Download checklist saved to: {checklist_path}")


def verify_downloads(download_dir: str = 'data/downloads') -> Dict:
    """Verify downloaded files and check their properties"""
    import rasterio
    
    download_path = Path(download_dir)
    logger.info("\n" + "="*60)
    logger.info("VERIFYING DOWNLOADED FILES")
    logger.info("="*60)
    
    verification_results = {
        'found_files': [],
        'missing_files': [],
        'file_details': {}
    }
    
    expected_files = [
        'sentinel2_pre_earthquake.tif',
        'sentinel2_post_earthquake.tif',
        'landsat_pre_earthquake.tif',
        'landsat_post_earthquake.tif',
        'sentinel1_pre_earthquake.tif',
        'sentinel1_post_earthquake.tif',
        'nepal_elevation_slope_aspect.tif',
        'nepal_population_density_2020.tif'
    ]
    
    for filename in expected_files:
        file_path = download_path / filename
        
        if file_path.exists():
            verification_results['found_files'].append(filename)
            
            # Get file details
            try:
                with rasterio.open(file_path) as src:
                    details = {
                        'size_mb': file_path.stat().st_size / (1024 * 1024),
                        'width': src.width,
                        'height': src.height,
                        'bands': src.count,
                        'crs': str(src.crs),
                        'resolution': src.res,
                        'bounds': src.bounds,
                        'dtype': str(src.dtypes[0])
                    }
                    verification_results['file_details'][filename] = details
                    
                    logger.info(f"\n✅ {filename}")
                    logger.info(f"   Size: {details['size_mb']:.1f} MB")
                    logger.info(f"   Dimensions: {details['width']} x {details['height']} pixels")
                    logger.info(f"   Bands: {details['bands']}")
                    logger.info(f"   Resolution: {details['resolution'][0]:.1f}m x {details['resolution'][1]:.1f}m")
                    
            except Exception as e:
                logger.error(f"   Error reading {filename}: {e}")
        else:
            verification_results['missing_files'].append(filename)
            logger.warning(f"\n❌ Missing: {filename}")
    
    # Summary
    logger.info("\n" + "-"*60)
    logger.info(f"Found: {len(verification_results['found_files'])} / {len(expected_files)} files")
    
    if verification_results['missing_files']:
        logger.warning("\n⚠️ Missing files:")
        for f in verification_results['missing_files']:
            logger.warning(f"   - {f}")
        logger.info("\nPlease download missing files from Google Drive")
    else:
        logger.info("\n✅ All expected files are present!")
        logger.info("You can now proceed with preprocessing")
    
    # Save verification results
    results_path = download_path / 'verification_results.json'
    with open(results_path, 'w') as f:
        json.dump(verification_results, f, indent=2)
    
    return verification_results


# Main execution helper
def setup_downloads(export_results: Dict = None):
    """Setup downloads after Earth Engine exports"""
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SETUP ASSISTANT")
    logger.info("="*60)
    
    # Check if we have Google Drive API setup
    if os.path.exists('credentials.json'):
        logger.info("✅ Google Drive API credentials found")
        logger.info("Attempting automated download...\n")
        
        downloader = GoogleDriveDownloader()
        if downloader.service:
            downloader.download_earthquake_data()
        else:
            logger.warning("Failed to authenticate with Google Drive")
            logger.info("Falling back to manual download instructions")
            if export_results:
                create_manual_download_instructions(export_results)
    else:
        logger.info("Google Drive API not configured")
        logger.info("Creating manual download instructions...\n")
        
        if export_results:
            create_manual_download_instructions(export_results)
        
        logger.info("\nTo enable automated downloads:")
        logger.info("1. Go to https://console.cloud.google.com/")
        logger.info("2. Create a new project or select existing")
        logger.info("3. Enable Google Drive API")
        logger.info("4. Create OAuth 2.0 credentials")
        logger.info("5. Download as 'credentials.json' to this directory")
    
    # Verify what we have
    logger.info("\nVerifying local files...")
    verification = verify_downloads()
    
    return verification


if __name__ == "__main__":
    # This can be run standalone to download files
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        # Just verify existing downloads
        verify_downloads()
    else:
        # Try to download
        setup_downloads()