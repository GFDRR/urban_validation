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

@dataclass
class OutputConfig:
    root_dir: str = ""
    overwrite: bool = False

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
class UrbanVectorConfig:
    aoi: AOIConfig
    output: OutputConfig
    overture: OvertureConfig
    gba: GBAConfig
    globfp: GloBFPConfig = field(default_factory=GloBFPConfig)

    
def load_config(path: str) -> UrbanVectorConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return UrbanVectorConfig(
        aoi=AOIConfig(**raw["aoi"]),
        output=OutputConfig(**raw.get("output", {})), 
        overture=OvertureConfig(**raw.get("overture", {"enabled": False})),
        gba=GBAConfig(**raw.get("gba", {"enabled": False})),
        globfp=GloBFPConfig(**raw.get("globfp", {"enabled": False})),
    )
