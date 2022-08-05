"""
Prepare stack of pixel-aligned rasters

Inputs include (but should be flexible):
- USGS Lidar Point Cloud LAZ tiles
- DEM(s) e.g. from Satellite Stereo

For each lidar tile, we want a separate folder containing
- N pixel-aligned rasters in the same projection
    - initial DEM
    - lidar DTM/DSM
    - orthoimages
    - ???
- Logs/related outputs to verify dataset quality/find processing errors
- Unlikely to keep intermediate clouds etc. outside debugging runs (file sizes)
    - could be an option or just handle excluding LAZ in some separate script

What variables we need to pass in:
- Parameters configuration (should not be here in the script, rather a YAML)
    - resolution
    - could include a lot of options for the PDAL calls
- Probably some bounds or a shapefile to skip bad tiles
- 
"""

# Baseline approach is to formulate the config as a dict/JSON
# then invoke the bash script correctly
# and use shutil and pathlib to move the files around as needed

# Read config JSON
from fileinput import filename
import json
import os

import subprocess


# Set args in bash script

# Run
# need to encode the ortho inputs

res = 1.0 # pixel size, meters

# call the script for now

#Warning 1: Computed -srcwin 14592 35555 1757 1757 falls completely outside raster extent. Going on however.

# Grab list of tiles from usgs_tiles.ipynb
# TODO use fixed dataset list of tile IDs that we want to include in train/val/test datasets

# laz_filenames = [
#     # "/mnt/1.0_TB_VOLUME/sethv/shashank_data/usgs_lpc_cache/USGS_LPC_WA_MtBaker_2015_10UEU8592_LAS_2017.laz"
#     "/mnt/1.0_TB_VOLUME/sethv/shashank_data/usgs_lpc_cache/USGS_LPC_WA_MtBaker_2015_10UEU8597_LAS_2017.laz"
# ]

laz_ids = [
    # '10UEU8295',

    # '10UEU8296',
    # '10UEU8297',
    # '10UEU8298',
    # '10UEU8395',
    # '10UEU8396',
    # '10UEU8397',
    # '10UEU8398',
    # '10UEU8495',
    # '10UEU8496',
    # '10UEU8497',
    # '10UEU8498'
    '10UEU8499'
]

# Rely on LAZ already being downloaded (will be true on Pleiades)
filename_template = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/usgs_lpc_cache/USGS_LPC_WA_MtBaker_2015_<tile_id>_LAS_2017.laz"
laz_filenames = [filename_template.replace("<tile_id>", laz_id) for laz_id in laz_ids]


#  For each lidar tile, call the stacking script
for laz_filename in laz_filenames:
    laz_id = os.path.splitext(os.path.basename(laz_filename))[0]

    print(f"stack.py tile id = {laz_id}")

    # first extract PDAL bounds
    result = subprocess.run(['pdal', 'info', laz_filename],
                            stderr = subprocess.PIPE,  # stderr and stdout get
                            stdout = subprocess.PIPE)  # captured as bytestrings

    # decode stdout from bytestring and convert to a dictionary
    json_result = json.loads(result.stdout.decode())

    # NOTE check against initial DEM/orthos extent should happen when generating tile list

    # TODO bbox will fail for uneven tiles at the edges, if there are any that overlap with
    bbox = json_result['stats']['bbox']['native']['bbox']
    print(bbox)

    ulx, uly, lrx, lry = bbox["minx"], bbox["maxy"], bbox["maxx"], bbox["miny"]

    # Make the bounds slightly smaller than tile
    # TODO check if OK assume offset of only a few pixels on each side, now losing 10% of data
    ulx += 25
    uly -= 25
    lrx -= 25
    lry += 25
    ulx = int(ulx)
    uly = int(uly)
    lrx = int(lrx)
    lry = int(lry)

    # Set up bounds for script
    # projwin = "585025 5397974 585974 5397025"
    # pdal_bounds = "([585025, 585974], [5397025, 5397974])"
    projwin = f"{ulx} {uly} {lrx} {lry}"
    pdal_bounds = f"([{ulx}, {lrx}], [{lry}, {uly}])"

    print(f"projwin = {projwin}")
    print(f"pdal_bounds = {pdal_bounds}")

    outdir = f"/mnt/1.0_TB_VOLUME/sethv/shashank_data/tile_stacks/pc_laz_prep_full_outputs_{laz_id}"

    # invoke the script for now
    # TODO will want named arguments to pass more args if this stays in bash mode
    result = subprocess.run(
        [
            "./pc_laz_prep_full.sh",
            laz_id,
            laz_filename,
            projwin,
            pdal_bounds,
            outdir
        ],
        # shell=False
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    # TODO assuming directory names won't change in script
    top_dir = "/mnt/1.0_TB_VOLUME/sethv/shashank_data/tile_stacks"
    dest_dir = os.path.join(top_dir, f"pc_laz_prep_full_outputs_{laz_id}")

    os.makedirs(dest_dir, exist_ok=True)

    # save stderr, stdout
    with open(os.path.join(dest_dir, "stdout.txt"), 'w') as f:
        f.write(result.stdout.decode())

    with open(os.path.join(dest_dir, "stderr.txt"), 'w') as f:
        f.write(result.stderr.decode())
