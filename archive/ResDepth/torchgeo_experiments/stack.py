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
"""

# Baseline approach is to formulate the config as a dict/JSON
# then invoke the bash script correctly
# can use shutil and pathlib to move the files around as needed

# Read config JSON
import json
import os

import requests
import subprocess
import sys

assert len(sys.argv) == 2, "Usage: stack.py <name_of_dataset>"
# Define dataset title, e.g. "tile_stacks_prelim_west_half_baker_mapproject_w_asp_refdem_11OCT2022"
run_name = sys.argv[1]
top_dir = f"/mnt/1.0_TB_VOLUME/sethv/shashank_data/{run_name}"

run_script = False  # just save script invocation for each tile, run separately with `parallel`
skip_existing = True  # want to continue stacks with more IDs in the same folder, though can't check for script version mismatch
res = 1.0  # pixel size, meters

# Full list of Baker core tiles
laz_ids = [
    "10UEU8795",
    "10UEU8895",
    "10UEU8995",
    "10UEU9095",
    "10UEU8796",
    "10UEU8896",
    "10UEU8996",
    "10UEU9096",
    "10UEU8797",
    "10UEU8897",
    "10UEU8997",
    "10UEU9097",
    "10UEU8798",
    "10UEU8898",
    "10UEU8998",
    "10UEU9098",
    "10UEU8799",
    "10UEU8899",
    "10UEU8999",
    "10UEU9099",
    "10UEV8700",
    "10UEV8800",
    "10UEV8900",
    "10UEV9000",
    "10UEV8701",
    "10UEV8801",
    "10UEV8901",
    "10UEV9001",
    "10UEV8702",
    "10UEV8802",
    "10UEV8902",
    "10UEV9002",
    "10UEV8703",
    "10UEV8803",
    "10UEV8903",
    "10UEV9003",
    "10UEV8704",
    "10UEV8804",
    "10UEV8904",
    "10UEV9004",
    "10UEV8705",
    "10UEV8805",
    "10UEV8905",
    "10UEV9005",
    "10UEV8706",
    "10UEV8806",
    "10UEV8906",
    "10UEV9006",
    "10UEV8707",
    "10UEV8807",
    "10UEV8907",
    "10UEV9007",
    "10UEV8708",
    "10UEV8808",
    "10UEV8908",
    "10UEV9008",
    "10UEV8709",
    "10UEV8809",
    "10UEV8909",
    "10UEV9009",
    "10UEV8710",
    "10UEV8810",
    "10UEV8910",
    "10UEV9010",
    "10UEU8395",
    "10UEU8495",
    "10UEU8595",
    "10UEU8695",
    "10UEU8396",
    "10UEU8496",
    "10UEU8596",
    "10UEU8696",
    "10UEU8397",
    "10UEU8497",
    "10UEU8597",
    "10UEU8697",
    "10UEU8398",
    "10UEU8498",
    "10UEU8598",
    "10UEU8698",
    "10UEU8399",
    "10UEU8499",
    "10UEU8599",
    "10UEU8699",
    "10UEV8300",
    "10UEV8400",
    "10UEV8500",
    "10UEV8600",
    "10UEV8301",
    "10UEV8401",
    "10UEV8501",
    "10UEV8601",
    "10UEV8302",
    "10UEV8402",
    "10UEV8502",
    "10UEV8602",
    "10UEV8303",
    "10UEV8403",
    "10UEV8503",
    "10UEV8603",
    "10UEV8304",
    "10UEV8404",
    "10UEV8504",
    "10UEV8604",
    "10UEV8305",
    "10UEV8405",
    "10UEV8505",
    "10UEV8605",
    "10UEV8306",
    "10UEV8406",
    "10UEV8506",
    "10UEV8606",
    "10UEV8307",
    "10UEV8407",
    "10UEV8507",
    "10UEV8607",
    "10UEV8308",
    "10UEV8408",
    "10UEV8508",
    "10UEV8608",
    "10UEV8309",
    "10UEV8409",
    "10UEV8509",
    "10UEV8609",
    "10UEV8310",
    "10UEV8410",
    "10UEV8510",
    "10UEV8610",
]

# Rely on LAZ already being downloaded (should be true on Pleiades)
filename_template = "/mnt/Backups/sethv/USGS_LPC_WA_MtBaker_2015/USGS_LPC_WA_MtBaker_2015_<tile_id>_LAS_2017.laz"
laz_filenames = [filename_template.replace("<tile_id>", laz_id) for laz_id in laz_ids]

script_calls = []

print(f"Processing {len(laz_filenames)} tiles...")
#  For each lidar tile, call the stacking script
for laz_filename in laz_filenames:
    laz_id = os.path.splitext(os.path.basename(laz_filename))[0]

    print(f"stack.py tile id = {laz_id}")

    outdir = os.path.join(top_dir, f"pc_laz_prep_full_outputs_{laz_id}")
    final_dem_path = os.path.join(
        outdir,
        "lower_easton3/files_to_zip/",
        "pc_align_tr-trans_source_1.0m-DEM_holes_filled.tif",
    )
    if skip_existing and os.path.exists(outdir) and os.path.exists(final_dem_path):
        # Already made the stack for this laz tile
        # Note: a more thorough check would be appropriate but for now just the DEM existing would be enough
        print("Output already exists for this tile, skipping it.")
        continue

    # Otherwise continue with the workflow
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(laz_filename):
        print("LAZ file not found, need to download")
        laz_url = f"https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/{laz_id}.laz"
        # download this to folder

        print(f"Download LAZ from {laz_url}")
        resp = requests.get(laz_url)
        with open(laz_filename, "wb") as f:
            f.write(resp.content)
        print("Download complete")
        assert os.path.exists(laz_filename)

    # first extract PDAL bounds
    print("Extracting PDAL Info (slow)...")
    result = subprocess.run(
        ["pdal", "info", "--metadata", laz_filename],
        stderr=subprocess.PIPE,  # stderr and stdout get
        stdout=subprocess.PIPE,
    )  # captured as bytestrings

    # decode stdout from bytestring and convert to a dictionary
    json_result = json.loads(result.stdout.decode())
    minx, miny, maxx, maxy = (
        json_result["metadata"][f] for f in ["minx", "miny", "maxx", "maxy"]
    )

    # TODO bbox could fail for uneven tiles at the edges, if there are any that overlap with
    ulx, uly, lrx, lry = minx, maxy, maxx, miny

    # Use full tile extent, may have nodata at edges, handle those elsewhere
    ulx = int(ulx)
    uly = int(uly)
    lrx = int(lrx)
    lry = int(lry)

    # Set up bounds for script
    projwin = f"{ulx} {uly} {lrx} {lry}"
    pdal_bounds = f"([{ulx}, {lrx}], [{lry}, {uly}])"

    print(f"projwin = {projwin}")
    print(f"pdal_bounds = {pdal_bounds}")
    script_call = f'./pc_laz_prep_full.sh "{laz_id}" "{laz_filename}" "{projwin}" "{pdal_bounds}" "{outdir}" > "{outdir}/stdout.txt" 2> "{outdir}/stderr.txt"\n'
    print("\n")
    print(script_call)
    print("\n")
    script_calls.append(script_call)

    if run_script:
        # TODO may want named arguments to pass more args if this stays in bash mode
        result = subprocess.run(
            [
                "./pc_laz_prep_full.sh",
                laz_id,
                laz_filename,
                projwin,
                pdal_bounds,
                outdir,
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        # save stderr, stdout
        with open(os.path.join(outdir, "stdout.txt"), "w") as f:
            f.write(result.stdout.decode())

        with open(os.path.join(outdir, "stderr.txt"), "w") as f:
            f.write(result.stderr.decode())

# Save commands used/to be used to generate this dataset
with open(f"{run_name}_script_calls.txt", "w") as f:
    f.writelines(script_calls)
