"""
Validate package for the Global Satellite Derived Urban Dataset Validation pipeline.

Contains runners for vector and raster validation along with supporting utilities:
- src.validate.base          - BaseValidationRunner base class
- src.validate.vector_runner - VectorValidationRunner orchestrator
- src.validate.raster_runner - RasterValidationRunner orchestrator
- src.validate.match_writer  - MatchChunkWriter for efficient per-tile match buffering
"""