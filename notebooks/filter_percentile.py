import numpy as np
import scipy

# This method accepts a percentile threshold as part of pdal args, converts it to a Z-score
# and then filters out points that fall outside this range. We use a nodata value (-9999) for 
# points that fall outside of the specified threshold
def filter_percentile(ins, outs):
    # pdal args is defined in the PDAL pipeline that calls this script
    percentile_threshold = pdalargs['percentile_threshold'] # type: ignore
    z_val = scipy.stats.norm.ppf(percentile_threshold)
    mean = np.nanmean(ins['Z'])
    std = np.nanstd(ins['Z'])
    z_scores = (ins['Z'] - mean) / std
    filtered_classification = np.where(z_scores > z_val, 18, ins['Classification'])
    outs['Classification'] = filtered_classification
    return True