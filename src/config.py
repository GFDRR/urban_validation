from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class AOIConfig:
    path: str
    id_col: str
    crs_out: str = "EPSG:4326"
    layer: Optional[str] = None
    buffer_meters: float = 0.0
    base_dir: str = ""      
    filter_suitable: bool = True
    aoi_subdir: str = "aoi"
    high_quality_only: bool = False

@dataclass
class OutputConfig:
    root_dir: str = ""
    overwrite: bool = False
    use_base_dir_for_output: bool = True

@dataclass
class OvertureConfig:
    enabled: bool = False 
    provider: str = "aws"
    release: str = "latest"
    theme: str = "buildings"
    types: List[str] = None
    s3_region: str = "us-west-2"
    s3_url: str = "s3://overturemaps-us-west-2/release/"
    def __post_init__(self):
        object.__setattr__(self, "types", self.types or [])

@dataclass
class GBAConfig:
    enabled: bool = False    
    s3_url: str = ""
    s3_region: str = "us-west-2"
    country_iso: str = ""
    out_name: str = "gba_lod1.parquet"

@dataclass
class GloBFPConfig:
    enabled: bool = False
    local_dir: str = "data/globfp_cache"   # where world_grid + tiles are cached
    zenodo_record: int = 15487006           # Zenodo record ID for world_grid.zip

@dataclass
class GoogleOBTConfig:
    enabled: bool = True
    ee_collection_id: str = "GOOGLE/Research/open-buildings-temporal/v1"
    years: List[int] = field(default_factory=lambda: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    bands: List[str] = field(default_factory=lambda: ["building_presence", "building_height", "building_fractional_count"])
    scale: int = 4

@dataclass
class MicrosoftTempoConfig:
    enabled: bool = True
    tile_index_url: str = "https://opendata.aiforgood.ai/building-density/tile_index.gpkg"
    tile_index_cache: str = "data/cache/tempo_tile_index.gpkg"
    tile_cache_dir: str = "data/cache/tempo_tiles"
    make_mosaic: bool = True

@dataclass
class GHSLProductConfig:
    ee_id: str = ""
    band: str = ""
    scale: int = 100
    years: List[int] = field(default_factory=list)

@dataclass
class GHSLConfig:
    enabled: bool = True
    products: dict = field(default_factory=lambda: {
        "BUILT_H": GHSLProductConfig(
            ee_id="JRC/GHSL/P2023A/GHS_BUILT_H",
            band="built_height",
            scale=100,
            years=[2018]
        ),
        "BUILT_S": GHSLProductConfig(
            ee_id="JRC/GHSL/P2023A/GHS_BUILT_S",
            band="built_surface",
            scale=100,
            years=[1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
        ),
        "BUILT_V": GHSLProductConfig(
            ee_id="JRC/GHSL/P2023A/GHS_BUILT_V",
            band="built_volume_total",
            scale=100,
            years=[1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
        )
    })

@dataclass
class DatasetsConfig:
    google_open_buildings_temporal: GoogleOBTConfig = field(default_factory=GoogleOBTConfig)
    microsoft_tempo: MicrosoftTempoConfig = field(default_factory=MicrosoftTempoConfig)
    ghsl: GHSLConfig = field(default_factory=GHSLConfig)

@dataclass
class UrbanConfig:
    aoi: AOIConfig
    output: OutputConfig
    overture: OvertureConfig
    gba: GBAConfig
    globfp: GloBFPConfig = field(default_factory=GloBFPConfig)
    ee_project: str = "urbanvalidation"
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)


def load_config(path: str) -> UrbanConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw_datasets = raw.get("datasets", {})
    raw_ghsl     = raw_datasets.get("ghsl", {})

    ghsl_products = {
        name: GHSLProductConfig(**prod)
        for name, prod in raw_ghsl.get("products", {}).items()
    }

    return UrbanConfig(
        aoi=AOIConfig(**raw["aoi"]),
        output=OutputConfig(**raw.get("output", {})),
        overture=OvertureConfig(**raw.get("overture", {"enabled": False})),
        gba=GBAConfig(**raw.get("gba", {"enabled": False})),
        globfp=GloBFPConfig(**raw.get("globfp", {"enabled": False})),
        ee_project=raw.get("ee_project", "urbanvalidation"),
        datasets=DatasetsConfig(
            google_open_buildings_temporal=GoogleOBTConfig(
                **raw_datasets.get("google_open_buildings_temporal", {})
            ),
            microsoft_tempo=MicrosoftTempoConfig(
                **raw_datasets.get("microsoft_tempo", {})
            ),
            ghsl=GHSLConfig(
                enabled=raw_ghsl.get("enabled", True),
                products=ghsl_products or GHSLConfig().products,
            ),
        ),
    )
