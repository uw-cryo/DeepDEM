import numpy as np

def filter_z(ins, outs):
    # pdal args is defined in the PDAL pipeline that calls this script
    z_val = pdalargs['z_val']
    mean = np.nanmean(ins['Z'])
    std = np.nanstd(ins['Z'])
    z_scores = (ins['Z'] - mean) / std
    filtered_data = np.where(np.abs(z_scores) < z_val, ins['Z'], -9999)
    outs['Z'] = filtered_data
    return True