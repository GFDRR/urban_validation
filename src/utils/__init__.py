"""
Utility subpackage for the Global Satellite Derived Urban Dataset Validation pipeline.

Exposes geometric, geospatial, AOI inventory, and I/O helpers via direct submodule imports:
- src.utils.aoi_inventory  - AOI loading and inventory management
- src.utils.buildings      - Building footprint loaders
- src.utils.ee_utils       - Earth Engine authentication and helpers
- src.utils.geometry       - Geometry and CRS utilities
- src.utils.glopfp        - GloBFP grid management
- src.utils.matches        - Match consolidation utilities
- src.utils.memory         - Memory logging
- src.utils.raster_io      - Raster I/O and masking
- src.utils.runtime        - Runtime environment detection
- src.utils.tiling         - AOI tiling and subset utilities
- src.utils.wsf_utils      - WSF Tracker raster utilities
"""
