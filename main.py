## pseudo code
## load configs from the harmonized version for the given aoi
## call UrbanVectorDownloader to download all specified vector datasets for that AOI
## call UrbanRasterDownloader to download all specified raster datasets for that AOI
## load the respective paths of the download datasets and perform validation based on the datatype (vector vs raster)
## save the visualizations for the metrics 

import os
import warnings
warnings.filterwarnings("ignore")
import yaml
from google.colab import drive
from src.downloader import UrbanDownloader

drive.mount('/content/drive')
os.chdir("/content/drive/MyDrive/Gates Foundation/Building Dataset Validation/")

# vector download pipeline from aoi_tracker
BASE = "/content/drive/MyDrive/Gates Foundation/Building Dataset Validation"
CONFIG = f"{BASE}/configs/data_configs.yaml"
UrbanDownloader(CONFIG).download_vector()
UrbanDownloader(CONFIG).download_raster()

# TODO: add vector validation pipeline 
# TODO: add raster validation pipeline
# TODO: add visualization pipeline f