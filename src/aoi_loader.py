import geopandas as gpd

def read_aoi(self):
    """Read the AOI (Area of Interest) from the specified path."""
    aoi = gpd.read_file(self.aoi_path)
    if aoi.crs is None:
        self.logger.warning("AOI CRS missing; assuming EPSG:4326")
        aoi = aoi.set_crs("EPSG:4326")
    aoi = aoi.to_crs(self.config["aoi"]["crs_out"])


    buff = float(self.config.aoi.buffer_meters or 0)
    if buff > 0 and aoi.crs is not None and aoi.crs.is_projected:
        self.logger.info("Buffering AOI geometries by %sm", buff)
        aoi["geometry"] = aoi.geometry.buffer(buff)

    # convert to crs_out 
    crs_out = self.cfg.aoi.crs_out
    if str(aoi.crs) != str(crs_out):
        self.logger.info("Reprojecting AOI | %s -> %s", aoi.crs, crs_out)
        aoi_gdf = aoi.to_crs(crs_out)

    self.logger.info("Successfully loaded AOI  | rows=%d | crs=%s", len(aoi_gdf), aoi_gdf.crs)

    return aoi_gdf
