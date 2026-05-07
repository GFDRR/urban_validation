"""
Download package for the Global Satellite Derived Urban Dataset Validation pipeline.

Contains runner classes for vector and raster building footprint dataset sources:
- src.download.base       - BaseRunner and shared runner base classes
- src.download.vector    - OvertureRunner, GBARunner, GloBFPRunner
- src.download.raster    - OBTRunner, TEMPORunner, GHSLRunner, WSFTrackerRunner
- src.download.db_connection - DuckDB utilities for Overture Maps
"""