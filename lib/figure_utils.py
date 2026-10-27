"""Data and metrics helpers for the DeepDEM paper-figure notebook (notebooks/3-Paper_Figures.ipynb).

The statistics functions (`compute_residual_dict`, `compute_error_stats`, `compute_ssim_stats`) are the ones used in the
working notebooks, unchanged. What is new here:

* `ensure_worldcover` / `load_worldcover`: ESA WorldCover is read from a cached GeoTIFF on the lidar grid when the file
  exists, and downloaded (once) with easysnowdata and saved there when it does not.
* `read_comparison_files` / `read_comparison_files_ssim`: tile readers for the sample-tile figures. They replace the
  notebook functions of the same name; the notebook binds the site-specific paths with functools.partial so the plotting
  cells call them exactly as before.
* `compute_DSM_stats` / `compute_vals` read only the validation (or training) window of every raster instead of
  loading and resampling full scenes, and hold the nodata mask as uint8 instead of int64. Results are unchanged
  (checked against the original full-scene implementation on synthetic rasters: identical outputs).
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.windows as window
import rioxarray as rxr
import shapely
from scipy.stats import median_abs_deviation
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform
from sklearn.metrics import mean_absolute_error

# ESA WorldCover 2021 class table (as returned by easysnowdata, kept here so a cached map can be used offline)
ESA_WORLDCOVER_CLASS_INFO = {
    10: {'name': 'Tree cover', 'color': '#006400'},
    20: {'name': 'Shrubland', 'color': '#FFBB22'},
    30: {'name': 'Grassland', 'color': '#FFFF4C'},
    40: {'name': 'Cropland', 'color': '#F096FF'},
    50: {'name': 'Built-up', 'color': '#FA0000'},
    60: {'name': 'Bare / sparse vegetation', 'color': '#B4B4B4'},
    70: {'name': 'Snow and ice', 'color': '#F0F0F0'},
    80: {'name': 'Permanent water bodies', 'color': '#0064C8'},
    90: {'name': 'Herbaceous wetland', 'color': '#0096A0'},
    95: {'name': 'Mangroves', 'color': '#00CF75'},
    100: {'name': 'Moss and lichen', 'color': '#FAE6A0'},
}



# ------------------------------------------------------------------------------------------------------------------
# land cover
# ------------------------------------------------------------------------------------------------------------------
def _bounds_4326(raster):
    """Lon/lat bounds of a rioxarray raster."""
    transformer = pyproj.Transformer.from_crs(pyproj.CRS(raster.rio.crs), pyproj.CRS('EPSG:4326'), always_xy=True)
    return shapely_transform(transformer.transform, box(*raster.rio.bounds())).bounds


def ensure_worldcover(lidar_path, cache_path):
    """Return `cache_path`; if the file does not exist, download ESA WorldCover for the lidar extent,
    resample it to the lidar grid and save it there (uint8 GeoTIFF)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f'WorldCover: using existing {cache_path}')
        return cache_path
    import easysnowdata
    print(f'WorldCover: {cache_path} not found, downloading ...')
    lidar = rxr.open_rasterio(lidar_path).squeeze()
    wc = easysnowdata.remote_sensing.get_esa_worldcover(_bounds_4326(lidar))
    wc = wc.rio.reproject_match(lidar).astype('uint8')
    wc.rio.write_nodata(0, inplace=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wc.rio.to_raster(cache_path, dtype='uint8', compress='DEFLATE', tiled=True)
    print(f'WorldCover: saved {cache_path}')
    return cache_path


def load_worldcover(cache_path, ref_raster):
    """WorldCover classes on the grid of `ref_raster` (only the window covered by `ref_raster` is read).
    The returned DataArray carries the class table in attrs['class_info'] (as easysnowdata does)."""
    import copy
    wc = rxr.open_rasterio(cache_path).squeeze().rio.clip_box(*ref_raster.rio.bounds())
    same_grid = wc.shape == ref_raster.shape and wc.rio.transform() == ref_raster.rio.transform()
    if not same_grid:
        wc = wc.rio.reproject_match(ref_raster)
    wc.attrs['class_info'] = copy.deepcopy(ESA_WORLDCOVER_CLASS_INFO)
    return wc


def get_worldcover_map(ref_raster, cache_path):
    """(class array on the grid of `ref_raster`, class table)"""
    wc = load_worldcover(cache_path, ref_raster)
    return wc.data, wc.attrs['class_info']


# ------------------------------------------------------------------------------------------------------------------
# window reads
# ------------------------------------------------------------------------------------------------------------------
def read_window(da, ref, buffer_px=8):
    """Read `da` only around the extent of `ref` (plus a small buffer, so resampling at the window edge sees the same
    neighbours as a full-scene resample would) and resample it onto the grid of `ref`."""
    pad = buffer_px * abs(da.rio.resolution()[0])
    xmin, ymin, xmax, ymax = ref.rio.bounds()
    return da.rio.clip_box(xmin - pad, ymin - pad, xmax + pad, ymax + pad).rio.reproject_match(ref)


# ------------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------------
def compute_DSM_stats(file_list, labels, input_data_path, train_split, worldcover_path,
                      lidar_path='final_lidar_data.tif', nodata_path='final_nodata_mask.tif',
                      compute_train=False, ssim_maps=True):
    """MAE / bias / NMAD (and mean SSIM when `ssim_maps`) per land-cover class for every raster in `file_list`.

    Only the validation region (right 1 - `train_split` of the lidar x-extent, intersected with the extent of every
    raster) is read; the training region as well when `compute_train`. With `ssim_maps` the SSIM map of each raster is
    expected next to it (`<name>_ssim_map.tif`).
    Returns (df_val, df_train, class_info, data_dict_val, data_dict_train); the train entries are None unless requested.
    """
    input_data_path = Path(input_data_path)
    lidar_dsm = rxr.open_rasterio(input_data_path / lidar_path).squeeze()
    nodata_ds = rxr.open_rasterio(input_data_path / nodata_path).squeeze()

    xmin, ymin, xmax, ymax = lidar_dsm.rio.bounds()
    xsplit = xmin + train_split * (xmax - xmin)
    regions = {'val': box(xsplit, ymin, xmax, ymax)}
    if compute_train:
        regions['train'] = box(xmin, ymin, xsplit, ymax)

    # common region = intersection with the extent of every file (files are opened lazily, nothing is loaded yet)
    data_rxr, ssim_rxr = [], []
    for file in file_list:
        data = rxr.open_rasterio(file).squeeze()
        data_bounds = box(*data.rio.bounds())
        regions = {k: shapely.intersection(v, data_bounds) for k, v in regions.items()}
        data_rxr.append(data)
        if ssim_maps:
            ssim_rxr.append(rxr.open_rasterio(str(file).replace('.tif', '_ssim_map.tif')).squeeze())

    results = {}
    for category, region in regions.items():
        lidar_c = lidar_dsm.rio.clip_box(*region.bounds)
        nodata_c = nodata_ds.rio.clip_box(*region.bounds).data.astype('uint8')
        data_c = [read_window(d, lidar_c).data for d in data_rxr]
        ssim_c = [s.rio.clip_box(*region.bounds).data for s in ssim_rxr] if ssim_maps else None
        land_classification, class_info = get_worldcover_map(lidar_c, worldcover_path)
        results[category] = compute_residual_dict(data_c, lidar_c.data, nodata_c, labels, class_info, land_classification,
                                                  ssim_rxr=ssim_c, category=category)
        del data_c, ssim_c, land_classification, nodata_c, lidar_c
        gc.collect()

    df_val, data_dict_val = results['val']
    if compute_train:
        df_train, data_dict_train = results['train']
        return df_val, df_train, class_info, data_dict_val, data_dict_train
    return df_val, None, class_info, data_dict_val, None


def compute_vals(pred_path, nodata_path, worldcover_path, cls='all', bbox=None):
    """Valid per-pixel values of a raster on the grid of the nodata mask, optionally for one WorldCover class `cls`
    and restricted to `bbox` = (xmin, ymin, xmax, ymax). Values outside [-1, 1] are dropped (used for SSIM maps)."""
    nodata_win = rxr.open_rasterio(nodata_path).squeeze()
    if bbox is not None:
        nodata_win = nodata_win.rio.clip_box(*bbox)
    pred_ds = read_window(rxr.open_rasterio(pred_path).squeeze(), nodata_win).values
    nodata_ds = nodata_win.values.astype('uint8')

    nodata_ds[(pred_ds > 1) | (pred_ds < -1)] = 0
    if cls == 'all':
        pred_val = pred_ds[(nodata_ds == 1)]
    else:
        lulc = load_worldcover(worldcover_path, nodata_win).values
        pred_val = pred_ds[(nodata_ds == 1) & (lulc == cls)]

    print(len(pred_val))
    return pred_val


# ------------------------------------------------------------------------------------------------------------------
# tile readers
# ------------------------------------------------------------------------------------------------------------------
def read_comparison_files(xmin_list, ymin_list, chipsize, idx, *, input_data_path, output_data_path, models,
                          lidar_data_path='final_lidar_data.tif', lidar_hs_data_path='final_lidar_data_hs_multi.tif',
                          nodata_mask_path='final_nodata_mask.tif'):
    """Tile `idx` of the stereo DSM, four model outputs, lidar, nodata mask and ortho image 1.

    `models` are the file names of the four inference rasters (slot order of the original `image1_path` .. `image4_path`).
    Returns (asp, asp_hs, m1, m1_hs, m2, m2_hs, m3, m3_hs, m4, m4_hs, lidar, lidar_hs, nodata, ortho1).
    """
    xmin, ymin, xmax, ymax = get_bounds(xmin_list, ymin_list, chipsize, idx)
    geometry = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    input_data_path, output_data_path = Path(input_data_path), Path(output_data_path)

    def read(path):
        with rasterio.open(path) as ds:
            return ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))

    asp_dsm_img_hs = read(input_data_path / 'final_asp_dsm_hs_multi.tif')
    asp_dsm_img = read(input_data_path / 'final_asp_dsm.tif')
    ortho_channel1_img = read(input_data_path / 'final_ortho_channel1.tif')
    nodata_img = read(input_data_path / nodata_mask_path)
    lidar_dsm_img = read(input_data_path / lidar_data_path)
    lidar_dsm_img_hs = read(input_data_path / lidar_hs_data_path)

    inferences = [read_inf_files(m, output_data_path, None, geometry=geometry) for m in models]

    # Nodata mask
    asp_dsm_img = np.ma.masked_where(nodata_img == 0, asp_dsm_img)
    asp_dsm_img_hs = np.ma.masked_where(nodata_img == 0, asp_dsm_img_hs)

    return (asp_dsm_img, asp_dsm_img_hs, *inferences[0], *inferences[1], *inferences[2], *inferences[3],
            lidar_dsm_img, lidar_dsm_img_hs, nodata_img, ortho_channel1_img)


def read_comparison_files_ssim(xmin_list, ymin_list, chipsize, idx, *, input_data_path, output_data_path, models,
                               lidar_data_path='final_lidar_data.tif', nodata_mask_path='final_nodata_mask.tif'):
    """As `read_comparison_files`, but also reads the hillshade-SSIM map of the stereo DSM and of every model, and ortho 2.

    Returns (asp, asp_hs, asp_ssim, m1, m1_hs, m1_ssim, m2, ..., m4_ssim, lidar, lidar_hs, nodata, ortho1, ortho2).
    """
    xmin, ymin, xmax, ymax = get_bounds(xmin_list, ymin_list, chipsize, idx)
    geometry = Polygon.from_bounds(xmin, ymin, xmax, ymax)
    input_data_path, output_data_path = Path(input_data_path), Path(output_data_path)

    def read(path):
        with rasterio.open(path) as ds:
            return ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))

    asp_dsm_img_hs = read(input_data_path / 'final_asp_dsm_hs_multi.tif')
    asp_dsm_img = read(input_data_path / 'final_asp_dsm.tif')
    asp_dsm_img_ssim = read(input_data_path / 'final_asp_dsm_ssim_map.tif')
    ortho_channel1_img = read(input_data_path / 'final_ortho_channel1.tif')
    ortho_channel2_img = read(input_data_path / 'final_ortho_channel2.tif')
    nodata_img = read(input_data_path / nodata_mask_path)
    lidar_dsm_img = read(input_data_path / lidar_data_path)
    lidar_dsm_img_hs = read(input_data_path / 'final_lidar_data_hs_multi.tif')

    inferences = [read_inf_files_ssim(m, output_data_path, nodata_img, geometry=geometry) for m in models]

    # Nodata mask
    asp_dsm_img = np.ma.masked_where(nodata_img == 0, asp_dsm_img)
    asp_dsm_img_hs = np.ma.masked_where(nodata_img == 0, asp_dsm_img_hs)
    asp_dsm_img_ssim = np.ma.masked_where(nodata_img == 0, asp_dsm_img_ssim)

    return (asp_dsm_img, asp_dsm_img_hs, asp_dsm_img_ssim, *inferences[0], *inferences[1], *inferences[2], *inferences[3],
            lidar_dsm_img, lidar_dsm_img_hs, nodata_img, ortho_channel1_img, ortho_channel2_img)


# ------------------------------------------------------------------------------------------------------------------
# lifted unchanged from notebooks/3-Figure_paper_ssim.ipynb
# ------------------------------------------------------------------------------------------------------------------
def compute_residual_dict(data_rxr, lidar_dsm, nodata, labels, class_info, land_classification, ssim_rxr=None, category='val'):
    residual_dict = {}
    data_dict = {}

    for cls in class_info.keys():
        
        if cls in [100, 60]:
            continue
        
        if cls == 30:
            cls_mask_val = (land_classification == 30) | (land_classification == 60)| (land_classification == 100) 
        else:
            cls_mask_val = land_classification == cls
            
        
        if cls_mask_val.sum() < 600000:
            continue
        
        residual_dict[cls] = {'nmad': [], 'mae': [], 'median_resid': [], 'ssim': []}
        data_dict[cls] = {}
        print(f"Class {cls} - {class_info[cls]['name']}:")
        
        # Compute RMSE, MAE, and Median residuals
        for i, label in enumerate(labels):
            # print(label)
            # print("Validation Region:")
            mae, nmad, median_resid, dsm_filter, lidar_filter = compute_error_stats(data_rxr[i][cls_mask_val], lidar_dsm[cls_mask_val], nodata[cls_mask_val])
            ssim = compute_ssim_stats(ssim_rxr[i][cls_mask_val], nodata[cls_mask_val]) if ssim_rxr is not None else None
            # print(f"MAE: {mae}| RMSE: {rmse}| Bias: {median_resid}")
            residual_dict[cls]['nmad'].append(nmad)
            residual_dict[cls]['mae'].append(mae)
            residual_dict[cls]['median_resid'].append(median_resid)
            data_dict[cls][label] = [dsm_filter, lidar_filter]
            if ssim_rxr is not None:
                residual_dict[cls]['ssim'].append(ssim)
        
        # print("==========================================\n")
    
    data_dict['all'] = {}
    mae_overall, nmad_overall, median_resid_overall = [], [], []
    ssim_overall = []
    for i, label in enumerate(labels):
        mae, nmad, median_resid, dsm_filter, lidar_filter = compute_error_stats(data_rxr[i], lidar_dsm, nodata)
        ssim = compute_ssim_stats(ssim_rxr[i], nodata) if ssim_rxr is not None else None
        mae_overall.append(mae)
        nmad_overall.append(nmad)
        median_resid_overall.append(median_resid)
        data_dict['all'][label] = [dsm_filter, lidar_filter]
        if ssim_rxr is not None:
            ssim_overall.append(ssim)
        
    columns = {}
    for cls in residual_dict.keys():
        print(f"Class {cls} - {class_info[cls]['name']}:")
        columns[f'mae_{category}_{cls}'] = residual_dict[cls]['mae']
        columns[f'bias_{category}_{cls}'] = residual_dict[cls]['median_resid']
        columns[f'nmad_{category}_{cls}'] = residual_dict[cls]['nmad']
        if ssim_rxr is not None:
            columns[f'ssim_{category}_{cls}'] = residual_dict[cls]['ssim']
    
    columns[f'mae_{category}'] = mae_overall
    columns[f'bias_{category}'] = median_resid_overall
    columns[f'nmad_{category}'] = nmad_overall
    if ssim_rxr is not None:
        columns[f'ssim_{category}'] = ssim_overall
    
    columns['labels'] = labels
    
    df = pd.DataFrame(data=columns)
    df.set_index('labels', inplace=True)
    
    return df, data_dict

def compute_ssim_stats(ssim_img, nodata_img):
    ssim_img = np.ma.masked_where(nodata_img==0, ssim_img)
    mean_ssim = np.mean(ssim_img)
    return mean_ssim

def compute_error_stats(dsm, lidar_dsm, nodata_img):
    nodata_img[dsm<0] = 0
    nodata_img[lidar_dsm<0] = 0
    dsm_filtered = dsm[nodata_img==1]
    lidar_dsm_filtered = lidar_dsm[nodata_img==1]
    mae = mean_absolute_error(lidar_dsm_filtered, dsm_filtered)
    # print(mean_squared_error(lidar_dsm[nodata_img==1], dsm[nodata_img==1]), lidar_dsm[nodata_img==1].min(), dsm[nodata_img==1].min())
    nmad = median_abs_deviation((dsm - lidar_dsm)[nodata_img==1], scale='normal', nan_policy='omit')
    median_resid = np.median((dsm_filtered - lidar_dsm_filtered))
    return mae, nmad, median_resid, dsm_filtered, lidar_dsm_filtered


# ------------------------------------------------------------------------------------------------------------------
# tile helpers, lifted unchanged from notebooks/3-Figure_paper_ssim.ipynb and 3-Figures3 copy.ipynb
# (read_inf_files_ssim is the 3-return read_inf_files of the transfer figure)
# ------------------------------------------------------------------------------------------------------------------
def get_bounds(xmin_list, ymax_list, chipsizes, idx):
    xmin, ymax = xmin_list[idx], ymax_list[idx]
    chipsize = chipsizes[idx]
    xmax = xmin + chipsize
    ymin = ymax - chipsize
    return xmin, ymin, xmax, ymax

def read_inf_files(inference_path, output_data_path, nodata_img=None, geometry=None):
    inference_hs_path = inference_path.replace('.tif', '_hs_multi.tif')
    print(f'Looking for {inference_hs_path}')
    if (output_data_path / inference_hs_path).exists() is False:
        inference_hs_path = inference_path.replace('.tif', '_hs.tif')
        print(f'Using {inference_hs_path} instead of {inference_path.replace(".tif", "_hs_multi.tif")}')
    
    if geometry is not None:        
        with rasterio.open(output_data_path / inference_path) as ds:
            model_inference_img = ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))
        with rasterio.open(output_data_path / inference_hs_path) as ds:
            model_inference_img_hs = ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))
    else:
        with rasterio.open(output_data_path / inference_path) as ds:
            model_inference_img = ds.read(1)
        with rasterio.open(output_data_path / inference_hs_path) as ds:
            model_inference_img_hs = ds.read(1)
            
    model_inference_img = np.ma.masked_where(nodata_img==0, model_inference_img)
    model_inference_img_hs = np.ma.masked_where(nodata_img==0, model_inference_img_hs)
            
    return model_inference_img, model_inference_img_hs

def compute_residuals(dsm, lidar_dsm, nodata_img):
    dsm = np.ma.masked_where(nodata_img==0, dsm)
    residuals = dsm - lidar_dsm
    return residuals

def read_inf_files_ssim(inference_path, output_data_path, nodata_img=None, geometry=None):
    inference_hs_path = inference_path.replace('.tif', '_hs_multi.tif')
    inference_ssim_path = inference_path.replace('.tif', '_ssim_map.tif')
    if geometry is not None:        
        with rasterio.open(output_data_path / inference_path) as ds:
            model_inference_img = ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))
        with rasterio.open(output_data_path / inference_hs_path) as ds:
            model_inference_img_hs = ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))
        with rasterio.open(output_data_path / inference_ssim_path) as ds:
            model_inference_img_ssim = ds.read(1, window=window.from_bounds(*geometry.bounds, transform=ds.transform))
    else:
        with rasterio.open(output_data_path / inference_path) as ds:
            model_inference_img = ds.read(1)
        with rasterio.open(output_data_path / inference_hs_path) as ds:
            model_inference_img_hs = ds.read(1)
        with rasterio.open(output_data_path / inference_ssim_path) as ds:
            model_inference_img_ssim = ds.read(1)
            
    # model_inference_img = np.ma.masked_where(nodata_img==0, model_inference_img)
    # model_inference_img_hs = np.ma.masked_where(nodata_img==0, model_inference_img_hs)
    model_inference_img_ssim = np.ma.masked_where(nodata_img==0, model_inference_img_ssim)
            
    return model_inference_img, model_inference_img_hs, model_inference_img_ssim
