"""
Point cloud masking to keep only bare ground points

Revised from notebook version to reproduce & run jobs in parallel.
Given dem_mask.py-generated mask raster, creates polygons of bare regions,
and uses these in a PDAL filter for each input LAZ tile.

Best to save PDAL pipeline .json files and run them separately in parallel
"""
import pathlib
import json
import time
import os
import subprocess

import rasterio
import shapely.geometry
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt

out_crs = "EPSG:32610"
root_dir = pathlib.Path("/mnt/Backups/sethv/USGS_LPC_WA_MtBaker_2015")

laz_files = [str(p) for p in root_dir.glob("*.laz")]

with rasterio.open("baker_2015_utm_m_30m_ref.tif") as mask_src:
    polygons = list(
        rasterio.features.dataset_features(
            mask_src, band=False, as_mask=True, geographic=False
        )
    )
    print(len(polygons), polygons[0])

gdf_32610 = gpd.GeoDataFrame(
    geometry=[shapely.geometry.shape(p["geometry"]) for p in polygons], crs="EPSG:32610"
)

gdf = gdf_32610.to_crs("EPSG:3740")
gdf["c"] = gdf.index

gdf.plot(column="c", cmap="tab10")
plt.savefig("test_fig_dem_mask_30m_epsg3740.png")

filtered_dir = "usgs_all616_laz_filtered_dem_mask_outlier_filtered"
filtered_dir = os.path.abspath(filtered_dir)

plots_dir = os.path.join(filtered_dir, "plots")

os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

list_of_tile_gdfs = []

for i, laz_input_fn in enumerate(laz_files):
    start = time.time()
    print(f"{i} Processing tile {laz_input_fn}")
    laz_file_without_extension = os.path.basename(laz_input_fn)[:-4]
    filtered_laz_fn = f"{filtered_dir}/{laz_file_without_extension}_filtered.laz"

    # trim the polygons to only be within THIS tile

    # first extract PDAL bounds
    command = ["pdal", "info", "--metadata", laz_input_fn]

    # print(f"\n***Filtering tile {laz_file_without_extension}...")
    # print(os.path.getsize(laz_input_fn))  # TODO convert to human-readable

    # print(f"Extracting bbox with {command}")
    result = subprocess.run(
        ["pdal", "info", "--metadata", laz_input_fn],
        stderr=subprocess.PIPE,  # stderr and stdout get
        stdout=subprocess.PIPE,
    )  # captured as bytestrings

    # decode stdout from bytestring and convert to a dictionary
    json_result = json.loads(result.stdout.decode())

    # NOTE check against initial DEM/orthos extent should happen when generating tile list

    # TODO bbox will fail for uneven tiles at the edges, if there are any that overlap with
    minx, miny, maxx, maxy = (
        json_result["metadata"][f] for f in ["minx", "miny", "maxx", "maxy"]
    )

    tile_box = shapely.geometry.box(minx, miny, maxx, maxy)
    polygons_for_tile = gdf.clip(tile_box)
    list_of_tile_gdfs.append(polygons_for_tile)
    if len(polygons_for_tile) == 0:
        print("Skipping this tile")
        continue  # could skip but creates confusion if fewer filtered .laz files than inputs
        # print("No polygons, skipping")
        # continue
    print(f"Found {len(polygons_for_tile)} different mask polygons within this tile")
    polygon_list = list(polygons_for_tile.to_wkt().geometry)

    # pipeline = pdal.Pipeline()
    # pipeline |= pdal.Reader(laz_input_fn)
    # pipeline |= pdal.Filter.returns(groups="first,only")
    pipeline = [
        laz_input_fn,
        # Crop in native CRS to remove as many points as possible
        {"type": "filters.crop", "polygon": polygon_list},
        {"type": "filters.returns", "groups": "first,only"},
        # Filter outliers to label as noise points (slow?)
        {"type": "filters.outlier"},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "filters.reprojection", "out_srs": out_crs},
        {"type": "writers.las", "filename": filtered_laz_fn, "compression": "true"},
    ]

    pipeline_dir = os.path.join(filtered_dir, "pipelines")
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline_fn = f"pdal_filter_crop_{laz_file_without_extension}.json"
    pipeline_path = os.path.join(pipeline_dir, pipeline_fn)

    with open(pipeline_path, "w") as of:
        json.dump(pipeline, of, indent=2)

    continue  # skip the actual PDAL execution, do that separately in parallel

    print("Running pdal filter pipeline")
    plot_mask = False
    if plot_mask:
        polygons_for_tile.plot()
        plt.savefig(os.path.join(plots_dir, f"mask_{laz_file_without_extension}.png"))
    # plt.show()
    # if not os.path.exists(filtered_laz_fn):
    # pdal.Pipeline(pipeline).execute()
    # !pdal pipeline filter_pipeline.json
    print("output file:")
    print(os.path.getsize(filtered_laz_fn))  # TODO convert to human-readable
    print(
        f"Processing time for tile {laz_file_without_extension}: {time.time() - start} seconds"
    )
