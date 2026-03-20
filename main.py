## pseudo code
## load configs from the harmonized version for the given aoi
## call UrbanVectorDownloader to download all specified vector datasets for that AOI
## call UrbanRasterDownloader to download all specified raster datasets for that AOI
## load the respective paths of the download datasets and perform validation based on the datatype (vector vs raster)
## save the visualizations for the metrics 

from google.colab import drive
drive.mount('/content/drive')

import os
from src.vectordownloader import UrbanVectorDownloader

os.chdir("/content/drive/MyDrive/Gates Foundation/Building Dataset Validation/")

# vector download pipeline from aoi_tracker
BASE = "/content/drive/MyDrive/Gates Foundation/Building Dataset Validation"
CONFIG = f"{BASE}/configs/data_configs.yaml"
UrbanVectorDownloader(CONFIG).run_connection()

# TODO: add raster download pipeline from aoi_tracker

# TODO: add vector validation pipeline 
# TODO: add raster validation pipeline
# TODO: add visualization pipeline f