#!/usr/bin/env python
"""Uses skimage.restoration.inpaint_biharmonic() to "fill in" the nodata areas
   in the provided dem.  If there are large nodata areas, and a suitable other
   terrain model (the filldem) is provided, this can be used to fill in the
   bulk of particular nodata areas, and the inpaint_biharmonic() function is
   used to join them.
"""

# Copyright 2022, Ross A. Beyer (rbeyer@rossbeyer.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import sys
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.interpolate import RBFInterpolator
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk
from skimage.restoration import inpaint_biharmonic


def arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "-f", "--filldem",
        type=Path,
        help="DEM to use if a nodata blob's equivalent semi-minor axis is "
             "less than --limit.  If --limit is given and --filldem isn't, "
             "then nodata blobs larger than --limit just won't get filled."
    )
    parser.add_argument(
        "-l", "--limit",
        type=float,
        help="If given, nodata blobs are characterized by the length of the "
             "minor axis of an ellipse that has the same normalized second "
             "central moments as the blob (see skimage.measure.regionprops). "
             "Blobs with a minor axis larger than --limit will not be filled. "
             "If --filldem is also given, then that dem will be used to "
             "'fill in' any nodata blobs that meet the --limit criteria.  "
             "Pixels farther than --limit from the edge of nodata blob will "
             "be identical to the --filldem  pixels, in the zone between "
             "them, fitting is performed. Default: %(default)s"
    )
    parser.add_argument(
        "indem",
        type=Path,
        help="Path to the input DEM."
    )
    parser.add_argument(
        "outdem",
        type=Path,
        help="Path to the output DEM."
    )

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    # Read in file
    with rasterio.open(args.indem) as dataset:
        indem = dataset.read(1, masked=True)
        profile = dataset.profile

    if args.filldem is not None:
        with rasterio.open(args.filldem) as filldata:
            fd = filldata.read(1, masked=True)
    else:
        fd = None

    # Fill
    if args.limit is None:
        # Just inpaint all of the nodata areas.
        outdem = inpaint_biharmonic(indem.data, indem.mask)
    else:
        try:
            outdem = inpaint_fill(indem, filllimit=args.limit, filldem=fd)
        except ValueError as err:
            sys.exit(str(err))

    # write output file
    with rasterio.Env():
        with rasterio.open(args.outdem, 'w', **profile) as dst:
            dst.write(outdem.astype(rasterio.float32), 1)

    return


def inpaint_fill(dem, filllimit: float, filldem=None):

    if filldem is not None and dem.shape != filldem.shape:
        raise ValueError("The dem and the filldem do not have the same shape.")

    # Find nodata blobs and get their properties
    nodata_blobs = label(dem.mask)
    nodata_regions = regionprops(nodata_blobs)

    if len(nodata_regions) == 0:
        raise ValueError("There are no nodata regions in dem.")

    # Make copy to write changes to
    tempdem = dem.copy()

    for props in nodata_regions:
        logging.info(
            f"{props.label}/{len(nodata_regions)}, area: {props.area}"
        )
        if props.axis_minor_length > filllimit:
            # print(props.axis_minor_length)
            logging.info(f"{props.label} filling with filldem")
            if filldem is not None:
                bbox_slice = (
                    slice(props.bbox[0], props.bbox[2]),
                    slice(props.bbox[1], props.bbox[3])
                )
                # cropfdem = filldem[bbox_slice]
                # blobmask = np.full_like(cropfdem, False)
                # for c in props.coords:
                #     blobmask[c[0] - props.bbox[0], c[1] - props.bbox[1]] = True

                # This step can be slow for large areas.
                eroded_mask = binary_erosion(
                    # blobmask, disk(filllimit, dtype=bool)
                    props.image, disk(filllimit, dtype=bool)
                )
                tempdem[bbox_slice][eroded_mask] = filldem[bbox_slice][eroded_mask]
            else:
                # The blob is larger than we want to just spline fill,
                # so remove it from the mask.
                logging.info(f"Skipping {props.label}")
                for coord in props.coords:
                    tempdem.mask[coord[0], coord[1]] = False
                continue
        # else the nodata blob is small enough to inpaint, and should be left
        # in the mask as-is.

    outdem = inpaint_biharmonic(tempdem.data, tempdem.mask)

    return outdem


def spline_fill(dem, pad=10, filllimit=None, filldem=None):
    # This function was an experiment in spline-fitting
    # It works very well for small masked patches, but as the patches
    # get bigger, the performance gets worse, and eventually crashes
    # somewhere between props.axis_minor_length 138.5 and 744.9.
    # Turns out that inpaint_biharmonic() applied to the individual patches
    # doesn't work too great, probably because of the limited patch size
    # However, applying it to a whole image has very similar results
    # to the spline-fitting, with the benefit of being much more
    # performant.

    if filldem is not None and dem.shape != filldem.shape:
        raise ValueError("The dem and the filldem do not have the same shape.")

    # Make copy to write changes to
    outdem = dem.copy()

    # Find nodata blobs and get their properties
    nodata_blobs = label(dem.mask)
    nodata_regions = regionprops(nodata_blobs)

    if len(nodata_regions) == 0:
        raise ValueError("There are no nodata regions in dem.")

    for props in nodata_regions:
        logging.info(
            f"{props.label}/{len(nodata_regions)}, area: {props.area}"
        )
        minrow = max(0, props.bbox[0] - pad)
        maxrow = min(dem.shape[0], props.bbox[2] + pad)
        mincol = max(0, props.bbox[1] - pad)
        maxcol = min(dem.shape[1], props.bbox[3] + pad)
        translated_coords = props.coords - [minrow, mincol]

        cropdem = dem[minrow:maxrow, mincol:maxcol]

        if filllimit is not None and props.axis_minor_length > filllimit:
            print(props.axis_minor_length)
            if filldem is not None:
                cropfdem = filldem[minrow:maxrow, mincol:maxcol]
                blobmask = np.full_like(cropfdem, False)
                for c in translated_coords:
                    blobmask[c[0], c[1]] = True

                eroded = binary_erosion(blobmask, disk(filllimit, dtype=bool))
                cropdem[eroded] = cropfdem[eroded]
            else:
                # The blob is larger than we want to just spline fill,
                # so skip it.
                logging.info(f"Skipping {props.label}")
                continue

        # Spline-fitting is good on small areas, but as the bounding box
        # reaches many hundreds of pixels across, this makes the spline-fit
        # very compute-intensive.
        good_pixels = np.ma.where(~cropdem.mask)

        # plt.ioff()
        # plt.imshow(cropdem)
        # plt.show()

        z = list()
        for row, col in zip(*good_pixels):
            z.append(cropdem[row, col])

        # spline fit hole
        coordgrid = np.mgrid[0:cropdem.shape[0], 0:cropdem.shape[1]]
        coordflat = coordgrid.reshape(2, -1).T
        # print(good_pixels)
        # print(np.stack(good_pixels, axis=-1))
        iflat = RBFInterpolator(np.stack(good_pixels, axis=-1), z)(coordflat)
        igrid = iflat.reshape(*cropdem.shape)

        # plt.ioff()
        # plt.imshow(igrid)
        # plt.show()

        # Replace the missing data with the spline data, which unmasks that
        # position.

        # Seems like this could be done more smoothly somehow?
        for coord, tcoord in zip(props.coords, translated_coords):
            outdem[coord[0], coord[1]] = igrid[tcoord[0], tcoord[1]]

    return outdem


if __name__ == "__main__":
    sys.exit(main())
