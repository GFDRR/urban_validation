"""
Download package for the Global Satellite Derived Urban Dataset Validation pipeline.

Exposes runner classes for vector (Overture, GBA, GloBFP) and raster
(OBT, TEMPO, GHSL, WSF Tracker) building footprint datasets.
"""
from src.download.base import BaseRunner, BaseVectorRunner, BaseRasterRunner
from src.download.vector import OvertureRunner, GBARunner, GloBFPRunner
from src.download.raster import (
    OBTRunner,
    TEMPORunner,
    GHSLRunner,
    WSFTrackerRunner,
)

__all__ = [
    "BaseRunner",
    "BaseVectorRunner",
    "BaseRasterRunner",
    "OvertureRunner",
    "GBARunner",
    "GloBFPRunner",
    "OBTRunner",
    "TEMPORunner",
    "GHSLRunner",
    "WSFTrackerRunner",
]