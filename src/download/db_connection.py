"""
DuckDB connection setup and Overture release resolution.

Centralises database connection logic so vector runners do not have to
configure spatial/httpfs/s3 extensions themselves.
"""
from __future__ import annotations

import logging

import duckdb
import requests

log = logging.getLogger("UrbanDownloader.db")

OVERTURE_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"


def connect_duckdb(config) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB connection with spatial, httpfs and s3 extensions loaded
    and configured for Overture / GBA S3 buckets.
    """
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs;  LOAD httpfs;")
    con.execute("INSTALL s3;      LOAD s3;")

    region = (
        config.overture.s3_region
        if config.overture.enabled
        else config.gba.s3_region
    )
    con.execute(f"SET s3_region='{region}';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
    con.execute("SET s3_use_ssl=true;")
    return con


def resolve_overture_base(config) -> str:
    """Resolve the Overture S3 base path for the configured release."""
    release = str(config.overture.release)
    if release == "latest":
        log.info("Resolving latest Overture release from STAC …")
        j = requests.get(OVERTURE_STAC_ROOT, timeout=30).json()
        release = j["latest"]
    return f"{config.overture.s3_url}{release}".rstrip("/")