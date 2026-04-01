<div align="center">

# Assessment of Satellite-derived Building Datasets 

</div>

## Project overview
This repository contains a pipeline for downloading global satellite-derived building footprints datasets filtered down to selected AOIs. It contains pipeline for vector datasets (`Overture`, `Global Building Atlas`, and `3D-GloBFP`) and raster datasets (`Google Open Building Temporal 2.5D`, `TEMPO`, `WSF-Tracker`, and `GHSL Built-up and Height`) for AOIs with high-quality reference datasets. 

The pipeline is config driven and rely on an `aoi_tracker` to identify the folders containing the `aoi` geojson and the `reference` data. 

## Code organization


## Usage Example
### Setup requirements

### Data Preparation and Download 
#### Vector datasets download pipeline
```python
from src.vectordownloader import UrbanVectorDownloader

BASE = "/content/drive/MyDrive/Gates Foundation/Building Dataset Validation"
CONFIG = f"{BASE}/configs/data_configs.yaml"
UrbanVectorDownloader(CONFIG).run_connection()
```

#### Raster  Download Pipeline
```python
TODO
```

### Data Validation Pipeline 
#### Vector datasets validation 
```python
TODO
```
#### Raster datasets validation

### Result visualization 


## Data availability 
Our datasets and validation are provided for these AOIs: 

![AOIs with high quality references](outputs/sample_AOIs.png)

Dataset would be provided publicly here: 

### File Organization
The datasets are organized as follows:


## License 
This code repository and corresponding datasets are distributed under the MIT License. See `LICENSE` for more information 

## Citation

```
@misc{gfdrr2026,
  title={Urban Validation},
  author={},
  year={2026},
  organization={GFDRR},
  type={Dataset},
  howpublished={\url{https://github.com/GFDRR/urban_validation}}
}
```

## Contributors
Rufai Omowunmi Balogun,
Derrick Mirindi,
Caroline Gevaert,
Pierre Chrzanowski,

## Acknowledgment
GFDRR, World Bank
Gates Foundation
HOTOSM Datasets
World Bank Datasets 
Partner Countries for on ground validation datasets
Feedback from other Team in the World Bank Digital Earth Partnership 