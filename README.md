<div align="center">

# Assessment of Satellite-derived Building Datasets 

</div>

## Project overview
This repository contains a pipeline for downloading global satellite-derived building footprints datasets filtered down to selected AOIs. It contains pipeline for vector datasets (`Overture`, `Global Building Atlas`, and `3D-GloBFP`) and raster datasets (`Google Open Building Temporal 2.5D`, `TEMPO`, `WSF-Tracker`, and `GHSL Built-up and Height`) for AOIs with high-quality reference datasets. 

The pipeline is config driven and rely on an `aoi_tracker` to identify the folders containing the `aoi` geojson and the `reference` data. 

<div align="center">

## Data availability 
Our datasets and validation are provided for these AOIs: 

</div>

![AOIs with high quality references](figures/sample_AOIs.png)

Dataset would be provided publicly here: 

## Code organization

| Module | Role |
|--------|------|
| `src/downloader.py` | `UrbanDownloader` — downloads vector (Overture, GBA, GloBFP) and raster (Google OBT, TEMPO, GHSL) datasets for all cities in the AOI inventory |
| `src/validator.py` | `Validator` — runs the full vector and raster validation pipeline per city |
| `src/metrics.py` | IoU-based building matching (vector) and pixel-level accuracy metrics (raster) |
| `src/output.py` | City-level summaries and all standard figures |
| `src/utils.py` | AOI loading, tiling, building loading, and raster I/O helpers |
| `src/config.py` | Typed dataclass config for the download pipeline |

Configuration is split across two files:
- `configs/data_configs.yaml` — controls which datasets to download and from where
- `configs/validation_configs.yaml` — controls validation thresholds, candidate datasets, and output format


## Usage Example
### Setup requirements

```bash
conda env create -f environment.yaml
conda activate urban_validation
pip install duckdb psutil earthengine-api
earthengine authenticate   # required for Google OBT and GHSL downloads
```

### Data Preparation and Download 
#### Vector datasets download pipeline
```python
from src.downloader import UrbanDownloader

UrbanDownloader("configs/data_configs.yaml").download_vector()
```

#### Raster download pipeline
```python
from src.downloader import UrbanDownloader

UrbanDownloader("configs/data_configs.yaml").download_raster()
```

Raster files are saved as:
- Google OBT: `data/01_raw/<city>/raster/<city_slug>_obt_<year>.tif`
- Microsoft TEMPO: `data/01_raw/<city>/raster/<city_slug>_tempo_<quarter>.tif`
- GHSL: `data/01_raw/<city>/raster/<city_slug>_ghsl_<product>_<year>.tif`

### Data Validation Pipeline 
#### Vector datasets validation
```python
from src.validator import Validator

v = Validator("configs/validation_configs.yaml")
v.validate_vector()
```

Outputs per city are written to `outputs/metrics/<city>/` and `outputs/figures/<city>/`. The candidate datasets (Overture, GBA, GloBFP) and preprocessing thresholds are controlled via `configs/validation_configs.yaml`.

See `notebooks/vector_validator.ipynb` for the Colab-ready notebook.

#### Raster datasets validation
```python
from src.validator import Validator

v = Validator("configs/validation_configs.yaml")
v.validate_raster()
```

Each raster dataset entry in `configs/validation_configs.yaml` specifies a `name`, `year`, and binarization method. The pipeline resolves the exact file for each city from the `year` field — for example, setting `year: 2020` for `ghsl_built_s` loads `<city_slug>_ghsl_built_s_2020.tif`. Multiple years of the same product can be validated by adding separate entries.

See `notebooks/raster_validator.ipynb` for the Colab-ready notebook.

### Result visualization 
[TODO]

### File Organization
The datasets are organized as follows:
[TODO]: when publicly available. 
[TODO]: check Zenodo/ Source Coop for sharing benchmark datasets (geoparquet) 


## License 
This code repository and corresponding datasets are distributed under the MIT License. See `LICENSE` for more information 

## Citation

```
@misc{gfdrr2026,
  title={An assessment of satellite derived global urban datasets for operational and analytical use cases},
  author={},
  year={2026},
  organization={GFDRR, The World Bank Group},
  type={Dataset},
  howpublished={\url{https://github.com/GFDRR/urban_validation}}
}
```

## Contributors
Rufai Omowunmi Balogun,
Caroline Gevaert,
Derrick Mirindi,
Pierre Chrzanowski,

## Acknowledgment
GFDRR, World Bank,

Gates Foundation,

HOTOSM Datasets, 

World Bank Datasets, 

Partner Countries for on ground validation datasets,
 
Feedback from other Team in the World Bank Digital Earth Partnership 