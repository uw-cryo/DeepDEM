#! /bin/bash
#Testing ASP and PDAL point cloud processing and gridding
#David Shean
#9/20/21
#Modified by Seth Vanderwilt
#2022

#Using example for Easton
#Running on rfe bro node on Pleiades - Bus Error for longer PDAL command on pfe26


echo $(date)
PS4='+ $(date --iso-8601=seconds)\011 '
set -x

# Inputs
# ./pc_laz_prep_full.sh output_directory_suffix laz_url bounds
echo $0 $1 $2 $3 $4 $5
laz_full_path="$2" # reuse from local LAZ cache instead of downloading
laz_fn=$(basename $laz_full_path) # when copying
echo "$laz_fn"
projwin="$3"
pdal_bounds="$4"
outdir="$5"



#Assumes that input pair has been processed fully by ASP using MGM/SGM with mapproject
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, Set the output directory suffix with a cmdline argument"
    exit
fi


#topdir=. #/nobackupp17/deshean/baker_resdepth_20210917 #TODO SV
topdir=/mnt/1.0_TB_VOLUME/sethv/shashank_data #/WV01_20150911_1020010042D39D00_1020010043455300
pair=WV01_20150911_1020010042D39D00_1020010043455300
# outdir="/mnt/1.0_TB_VOLUME/sethv/shashank_data/pc_laz_prep_full_outputs_$1"
mkdir -p $outdir
# outdir=$(ls -trd $topdir/$pair/dem* | tail -1)
site=lower_easton3

# stdev_tif_fn="$outdir/$site/${laz_fn%.*}_32610_first_filt_v1.3_1.0m_intensity_all.tif"
# if [ -e $stdev_tif_fn ] ; then
#     echo "Already reached"
#     exit
# fi


cd $outdir
# Copy script for reference
echo "Saving copy of $(realpath $0) in output folder"
cp "$(realpath $0)" "pc_laz_prep_full_as_run$(date -Iminutes).sh"

$pwd
mkdir -p $site

gdal_opt='-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER'

#Define output projection
#PDAL doesn't like this, though it should be proper Compund CRS definition
#proj="EPSG:32610+4979"
#The "init=" definition is depreciated
#proj="init=EPSG:32610 +geoidgrids=us_noaa_g2018u0.tif"
#This is PROJ string with geoidgrids
#proj='+proj=utm +zone=10 +datum=WGS84 +units=m +vunits=m +geoidgrids=us_noaa_g2018u0.tif +no_defs'
#Simple, PDAL seems to acknowledge vertical datum definition and uses unknown geoidgrid for transformation
proj=EPSG:32610

#Output grid res
res=1.0

# #Output extent
# #Can extract directly from laz bounds
# #Slightly larger than laz tile
# #This is from QGIS
# projwin="584974 5397967 586025 5399032"
# #This is what gdal_translate expects (ulx uly lrx lry)
# projwin="584974 5399032 586025 5397967"
# #This is what PDAL expects
# pdal_bounds="([584974, 586025], [5397967, 5399032])"

# #Slightly smaller than laz tile
# projwin="585023 5398014 585982 5398985"
# projwin="585023 5398985 585982 5398014"
# pdal_bounds="([585023, 585982], [5398014, 5398985])"
# #TODO change back to above
# # Below is SV change for tile 8597
# projwin="585023 5397985 585982 5397014"
# pdal_bounds="([585023, 585982], [5397014, 5397985])"
# # Below is SV change for tile 8599
# projwin="585023 5399985 585982 5399014"
# pdal_bounds="([585023, 585982], [5399014, 5399985])"
# # Below is SV change for tile 8285
# url_3dep_laz='https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/USGS_LPC_WA_MtBaker_2015_10UEU8285_LAS_2017.laz'
# # have to extract from pdal info JSON
# # projwin=json_dict["stats"]["bbox"]["native"]["bbox"]["maxx"]... # etc.
# # probably have to round up or add 5-10 pixels of room
# # UEV8910
# 590000
# projwin="582000 5385257 582726 5386000"
# pdal_bounds="([582000, 582726], [5385257,5386000])"


# url_3dep_laz="$2"
# 

# url_3dep_laz='https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/USGS_LPC_WA_MtBaker_2015_10UEU8599_LAS_2017.laz'

# TODO remove, just moved to reorder
# #ASP PC 
# #For parallel_stereo this PC.tif is just a vrt
# pc_path=$(ls $topdir/$pair/*PC.tif) #$(ls *PC.tif)
# #Create single full PC file from the large vrt
# #gdal_translate $gdal_opt $pc ${pc%.*}_full.tif
# echo "$pc_path"
# pc=$(basename $pc_path)

# #Crop ASP PC.tif (assuming map-projected inputs used) - fast, easy way to crop PC
# #To do: revisit to make sure output origin is multiple of res, may need -r cubic
# if [ ! -e $site/${pc%.*}_crop.tif ] ; then 
#     gdal_translate $gdal_opt -projwin $projwin $pc_path $site/${pc%.*}_crop.tif
# fi

# #Crop original orthoimages (mapped on low-res ref DEM)
# #ortho_list=$(ls *ortho_*m.tif)

cd $site

# #Create laz from cropped ASP PC.tif
# if [ ! -e ${pc%.*}_crop.laz ] ; then 
#     point2las -c ${pc%.*}_crop.tif
# fi

#Search 3DEP LidarExplorer, identify tile, get URL 
#url_3dep_dtm='https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/OPR/WA_MtBaker_2015/USGS_NED_OPR_WA_MtBaker_2015_bh_10UEU8598_IMG_2017.zip'
# url_3dep_laz='https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/USGS_LPC_WA_MtBaker_2015_10UEU8597_LAS_2017.laz'
# url_3dep_laz='https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/USGS_LPC_WA_MtBaker_2015_10UEU8599_LAS_2017.laz'
# url_3dep_laz='https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MtBaker_2015_LAS_2017/laz/USGS_LPC_WA_MtBaker_2015_10UEU8598_LAS_2017.laz'
#Remember, wget won't work from rfe, no connection to outside
# wget -v -nc $url_3dep_laz
# laz_fn=$(basename $url_3dep_laz)
# if [ ! -e laz_fn ] ; then 
#     wget -v -nc $url_3dep_laz
# fi


# Copy from
cp -n "$laz_full_path" "$laz_fn"

#Should extract bounds from the tile and use that throughout
#pdal info returns json with bbox dictionary, or boundary
#pdal info $laz_fn | grep bounds

#Transform 3DEP laz to EPSG:32610 with elevations as height above ellipsoid
#This requires that the user has already run `projsync --all` to retrieve geoid offset grids
#PDAL should be smart enough to read PROJ info from laz header and pick appropriate grid/transform
laz_fn_32610=${laz_fn%.*}_32610_first_filt.laz

if [ ! -e $laz_fn_32610 ] ; then
    proj=EPSG:32610
	#This does the reprojeciton only. However, we want to do filtering as well, for consistency between grids from PDAL point2dem 
    #pdal translate -i $laz_fn -o $laz_fn_32610 -f filters.reprojection --filters.reprojection.out_srs="$proj" 
    #First+only returns, with outlier filter to remove bogus points (SLOW) 
	pdal translate -i $laz_fn -o $laz_fn_32610 -f filters.reprojection --filters.reprojection.out_srs="$proj" -f filters.returns --filters.returns.groups="first,only" -f filters.outlier -f filters.range --filters.range.limits='Classification![7:7]' 
fi
laz_fn=$laz_fn_32610

#Convert to LAS 1.3 - needed to view in OS X CloudCompare and process with ASP point2dem (uses older liblas)
if [ ! -e ${laz_fn%.*}_v1.3.laz ] ; then 
    pdal translate -i $laz_fn -o ${laz_fn%.*}_v1.3.laz --writers.las.format="1.3"
fi
laz_fn=${laz_fn%.*}_v1.3.laz

#Use PDAL to grid the 3DEP laz
# TODO try the other options for PDAL in EarthLab_AQ_lidar_download_processing_function.ipynb
# e.g. first, only
out_fn=${laz_fn%.*}_${res}m_pdal_all.tif
#Note: the outlier filter is slow, but removes problematic fist returns
if [ ! -e $out_fn ] ; then 
    #For some reason, breaking these up with \ doesnt work
    pdal translate -i $laz_fn -o $out_fn -f filters.crop --filters.crop.bounds="$pdal_bounds" --writers.gdal.resolution=$res --writers.gdal.output_type="all" --writers.gdal.data_type="float32" --writers.gdal.gdalopts="COMPRESS=LZW,TILED=YES,BIGTIFF=IF_SAFER" 
fi

#Extract each of the grids from the PDAL 6-band output
pdal_stats="min max mean idw count stdev" 
pdal_bands="1 2 3 4 5 6"
parallel --delay 1 --link -v "gdal_translate $gdal_opt -b {1} $out_fn ${out_fn%.*}_{2}.tif" ::: $pdal_bands ::: $pdal_stats

#Use PDAL to grid the 3DEP Intensity
out_fn=${laz_fn%.*}_${res}m_intensity_all.tif
#Note: the outlier filter is slow, but removes problematic fist returns
if [ ! -e $out_fn ] ; then 
    #Using UInt16 here, as that is native for Intensity values, will round stdev, which is fine
    #At end of day, we want mean or idw stat
    pdal translate -i $laz_fn -o $out_fn -f filters.crop --filters.crop.bounds="$pdal_bounds" --writers.gdal.dimension="Intensity" --writers.gdal.resolution=$res --writers.gdal.output_type="all" --writers.gdal.data_type="UInt16" --writers.gdal.gdalopts="COMPRESS=LZW,TILED=YES,BIGTIFF=IF_SAFER" 
fi

#Extract each of the grids from the PDAL 6-band output
pdal_stats="min max mean idw count stdev" 
pdal_bands="1 2 3 4 5 6"
parallel --delay 1 --link -v "gdal_translate $gdal_opt -b {1} $out_fn ${out_fn%.*}_{2}.tif" ::: $pdal_bands ::: $pdal_stats

# echo "Stopping early because ASP reading is not fixed yet"
# echo $(date)


##### New updated version where ASP PC is already aligned

#ASP PC 
#For parallel_stereo this PC.tif is just a vrt
pc_path="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/usgs_all616_laz_filtered_dem_mask_nlcd_rock_exclude_glaciers/pc_align_all/pc_align_all-trans_source.tif"
# pc_path=$(ls $topdir/$pair/*PC.tif) #$(ls *PC.tif)
#Create single full PC file from the large vrt
#gdal_translate $gdal_opt $pc ${pc%.*}_full.tif
echo "$pc_path"
pc="WV01_20150911_1020010042D39D00_1020010043455300_aligned.tif" # want a clearer name
# pc=$(basename $pc_path)

#Crop ASP PC.tif (assuming map-projected inputs used) - fast, easy way to crop PC
#To do: revisit to make sure output origin is multiple of res, may need -r cubic
# gdal_translate $gdal_opt -projwin $projwin $pc_path ${pc%.*}_crop.tif
if [ ! -e $site/${pc%.*}_crop.tif ] ; then 
#     # gdal_translate $gdal_opt -projwin $projwin $pc_path $site/${pc%.*}_crop.tif
    gdal_translate $gdal_opt -projwin $projwin $pc_path ${pc%.*}_crop.tif
fi

#Crop original orthoimages (mapped on low-res ref DEM)
#ortho_list=$(ls *ortho_*m.tif)

# cd $site

#Create laz from cropped ASP PC.tif
if [ ! -e ${pc%.*}_crop.laz ] ; then 
    point2las -c ${pc%.*}_crop.tif
fi

#Use PDAL to grid the ASP PC
#Can skip this and only grid the aligned PC laz
# out_fn=${pc%.*}_crop_${res}m_pdal_all.tif
# if [ ! -e $out_fn ] ; then
#     #Shouldn't need reprojection here if ASP PC is in EPSG:32610
# 	pdal translate -i ${pc%.*}_crop.laz -o $out_fn -f filters.reprojection --filters.reprojection.out_srs="$proj" -f filters.crop --filters.crop.bounds="$pdal_bounds" --writers.gdal.resolution=$res --writers.gdal.output_type="all" --writers.gdal.data_type="float32" --writers.gdal.gdalopts="COMPRESS=LZW,TILED=YES,BIGTIFF=IF_SAFER" 
# fi

# pdal_stats="min max mean idw count stdev" 
# pdal_bands="1 2 3 4 5 6"
# parallel --delay 1 --link -v "gdal_translate $gdal_opt -b {1} $out_fn ${out_fn%.*}_{2}.tif" ::: $pdal_bands ::: $pdal_stats

#Test to see if using full PC with projwin was different: the actual problem is holes in the orthoimage inputs??? than the output cropped using gdal_translate
#Note: while this says Cropping to... with small output dimensions, the Stats, Bounding Box and QuadTree take forever, all appear to use uncropped version
#point2dem --t_srs "$proj" --t_projwin $projwin --tr $res $pc 

#The cropped outputs look identical (good!), but have edge artifact in hs
#point2dem --t_srs "$proj" --tr $res ${pc%.*}_crop.laz 

#Use ASP point2dem to grid the 3DEP laz
#Note: ASP ships older GDAL, which can't find gcs.csv (removed in GDAL3) 
#Using EPSG:32610 here results in Error: GdalIO: Unable to open EPSG support file gcs.csv.
#Manually define proj string
proj='+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs'

#Auxiliary statistics for 3DEP laz
#weighted_average, min, max, mean, median, stddev, count, nmad, n-pct (80-pct)
asp_stats="weighted_average max" #min max mean median stddev count nmad"
# TODO skipping the pdal stats
parallel --delay 1 -v "point2dem --filter {} --t_srs \"${proj}\" --t_projwin $projwin --tr $res $laz_fn -o ${laz_fn%.*}_${res}m" ::: $asp_stats

# #Use ASP point2dem to grid cropped version of ASP PC
#Can skip this and only grid the aligned PC 
parallel --delay 1 -v "point2dem --filter {} --t_srs \"${proj}\" --t_projwin $projwin --tr $res ${pc%.*}_crop.tif -o ${pc%.*}_crop_${res}m" ::: $asp_stats
#Run once to create TriError image
point2dem --errorimage --no-dem --t_srs "${proj}" --t_projwin $projwin --tr $res ${pc%.*}_crop.tif -o ${pc%.*}_crop_${res}m

#Aligned DEM
fn_asp_dem=${pc_align_out}-trans_source_${res}m-DEM.tif
#3DEP DEM from point2dem
fn_3dep_dem=${laz_fn%.*}_${res}m-DEM.tif

exit 
# TODO THIS PART DIDN"T WORK

# TODO modified 
pc_align_out="${pc%.*}"

# #pc_align
# #Important to use the v1.3 and 32610 version here, as the original ASP can't handle v1.4 and NAD83/NAVD88, geoid offset of ~15 m
# pc_align_out=pc_align_tr/pc_align_tr
# pc_align $laz_fn ${pc%.*}_crop.tif --max-displacement 10 --compute-translation-only --save-transformed-source-points -o $pc_align_out 
#For some reason, script exits after pc_align finishes.  Maybe error code issue?
# TODO find why pc_align exit code

#Convert aligned PC to LAZ
# point2las -c ${pc_align_out}-trans_source.tif


#Create new DEMs from aligned PC
# asp_stats="weighted_average count nmad"
# parallel --delay 1 -v "point2dem --filter {} --t_srs \"${proj}\" --t_projwin $projwin --tr $res ${pc_align_out}-trans_source.tif -o ${pc_align_out}-trans_source_${res}m" ::: $asp_stats
# point2dem --errorimage --no-dem --t_srs "${proj}" --t_projwin $projwin --tr $res ${pc_align_out}-trans_source.tif -o ${pc_align_out}-trans_source_${res}m 


#Can also do dem_align.py the rasters, but better to stick with PC
#dem_align.py -outdir dem_align ${out_fn%.*}_idw.tif 

#Apply transformation to cameras using bundle_adjust
#https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#applying-the-pc-align-transform-to-cameras
cd $topdir/$pair #TODO switched
#Use -t rpc here, as that will be used by mapproject
ba_out=$outdir/$site/${pc_align_out}_ba_rpc
#latest ASP build has --apply-initial-transform-only, skips IP detection/matching


# TODO bundle adjustment is done elsewhere
# bundle_adjust *r100.tif *r100.xml -t rpc --datum WGS_1984 --initial-transform $outdir/$site/${pc_align_out}-transform.txt --apply-initial-transform-only -o $ba_out
# each camera
#metashape saves global transform
#in ASP, each camera has a camera model that specifies its transform
# 
# bundle_adjust *r100.tif *r100.xml -t rpc --datum WGS_1984 --initial-transform $outdir/$site/${pc_align_out}-transform.txt --rotation-weight 1 --translation-weight 1000 --ip-per-tile 50 -o $ba_out # Shashank suggestion to adjust cameras among themselves, see https://sheanlab.slack.com/archives/G015UCB378Q/p1645211599895269?thread_ts=1645048078.776139&cid=G015UCB378Q
# ip-per-tile 50 was another shashank suggestion to reduce runtime. Have to look up what this does

#bundle_adjust *r100.tif *r100.xml -t rpc --datum WGS_1984 --initial-transform $outdir/$site/${pc_align_out}-transform.txt --num-iterations 0 -o $ba_out

#Create new clipped orthoimages using the aligned DEM
#Use 3DEP grid as reference DEM
# TODO checking if this makes much improvement

# refdem=$outdir/$site/${laz_fn%.*}_${res}m_pdal_all_idw.tif
#Should use aligned ASP output as reference DEM, I think, but good to test

# TODO Should fill gaps in aligned stereo DEM at this point
# Because the next step is projecting the orthoimages
# and there will be holes in the results
# Seems reasonable to just 
# Do we want to fill the holes



fn_3dep_dem_filled=${fn_3dep_dem%.*}_holes_filled.tif
python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_3dep_dem $fn_3dep_dem_filled
fn_asp_dem_filled=${fn_asp_dem%.*}_holes_filled.tif
python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_asp_dem $fn_asp_dem_filled

refdem=$outdir/$site/${pc_align_out}-trans_source_${res}m-DEM_holes_filled.tif
#refdem=$outdir/$site/${pc_align_out}-trans_source_${res}m-DEM.tif
parallel --delay 1 -v "mapproject $refdem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin --tr $res --bundle-adjust-prefix $ba_out {} {.}.xml $outdir/$site/{.}_ortho_${res}m_ba.tif" ::: *r100.tif
# mapproject /mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/baker_2015_utm_m.tif --ot UInt16 --t_srs "EPSG:32610" --t_projwin 584000 5397000 585000 5396000 --tr 1 --bundle-adjust-prefix ba_from_usgs_all616_pc_align /mnt/1.0_TB_VOLUME/sethv/shashank_data/WV01_20150911_1020010042D39D00_1020010043455300/1020010042D39D00.r100.tif /mnt/1.0_TB_VOLUME/sethv/shashank_data/WV01_20150911_1020010042D39D00_1020010043455300/1020010042D39D00.r100.xml WV01_20150911_1020010042D39D00_test_ortho_ba.tif

#Should convert PAN DN values to TOA reflectance

#We should now have self-consistent orthoimages, ASP DEM and lidar DEM - identical UL origin, res, extent

#Compute higher-level products and package
cd $outdir/$site
#Aligned orthoimages
fn_ortho=$(ls *_ortho_${res}m_ba.tif)

#Triangulation error
fn_trierr=${pc_align_out}-trans_source_${res}m-IntersectionErr.tif

#3DEP intensity
fn_3dep_intensity=${laz_fn%.*}_${res}m_intensity_all_idw.tif
#3DEP DEM from PDAL
#fn_list+=${laz_fn%.*}_${res}m_pdal_all_idw.tif

#Compute difference map (use lidar to define grid, in case of any residual issues)
compute_diff.py $fn_3dep_dem $fn_asp_dem -te $fn_3dep_dem -tr $fn_3dep_dem
fn_diff=$(basename ${fn_3dep_dem%.*})_$(basename ${fn_asp_dem%.*})_diff.tif
#Generate shaded relief maps
hs.sh $fn_3dep_dem $fn_asp_dem
fn_hs=$(ls *hs.tif */*hs.tif)

#Package 3DEP LAZ original?
#3DEP LAZ projected, cropped, filtered
fn_3dep_laz=$laz_fn
#Aligned point cloud (LAZ)
fn_asp_laz=${pc_align_out}-trans_source.laz

mkdir files_to_zip
cp $fn_ortho $fn_trierr $fn_3dep_intensity $fn_3dep_dem $fn_3dep_dem_filled $fn_asp_dem $fn_asp_dem_filled $fn_diff $fn_hs $fn_3dep_laz $fn_asp_laz files_to_zip/

#Should export COG
# TODO not doing that
