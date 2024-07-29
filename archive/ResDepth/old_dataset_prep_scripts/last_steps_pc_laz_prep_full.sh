#! /bin/bash
# TODO Usage bash ./last_steps_pc_laz_prep_full.sh "USGS_LPC_WA_MtBaker_2015_10UEV8610_LAS_2017" "/mnt/Backups/sethv/USGS_LPC_WA_MtBaker_2015/USGS_LPC_WA_MtBaker_2015_10UEV8610_LAS_2017.laz" "586000 5410999 586999 5410000" "([586000, 586999], [5410000, 5410999])" "/mnt/1.0_TB_VOLUME/sethv/shashank_data/tile_stack_baker_128_global_coreg/pc_laz_prep_full_outputs_USGS_LPC_WA_MtBaker_2015_10UEV8610_LAS_2017"

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

topdir=/mnt/1.0_TB_VOLUME/sethv/shashank_data #/WV01_20150911_1020010042D39D00_1020010043455300
pair=WV01_20150911_1020010042D39D00_1020010043455300
mkdir -p $outdir
site=lower_easton3

cd $outdir
# Copy script for reference
echo "Saving copy of $(realpath $0) in output folder"
cp "$(realpath $0)" "pc_laz_prep_full_as_run$(date -Iminutes).sh"

$pwd
mkdir -p $site

gdal_opt='-co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER'

proj=EPSG:32610

#Output grid res
res=1.0

laz_fn=${laz_fn%.*}_32610_first_filt_v1.3.laz

#ASP PC 
#For parallel_stereo this PC.tif is just a vrt
pc_path="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/usgs_all616_laz_filtered_dem_mask_nlcd_rock_exclude_glaciers/pc_align_all/pc_align_all-trans_source.tif"
# pc_path=$(ls $topdir/$pair/*PC.tif) #$(ls *PC.tif)
#Create single full PC file from the large vrt
#gdal_translate $gdal_opt $pc ${pc%.*}_full.tif
echo "$pc_path"
pc="WV01_20150911_1020010042D39D00_1020010043455300_aligned.tif" # want a clearer name
# pc=$(basename $pc_path)
# TODO THIS PART DIDN"T WORK

# TODO modified 
pc_align_out="${pc%.*}_crop"

#Apply transformation to cameras using bundle_adjust
#https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#applying-the-pc-align-transform-to-cameras
cd $topdir/$pair #TODO switched
#Use -t rpc here, as that will be used by mapproject
# ba_out=$outdir/$site/${pc_align_out}_ba_rpc
ba_out="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/bundle_pc_transform_test/ba_from_usgs_all616_pc_align"

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

refdem=$outdir/$site/${pc_align_out}_${res}m-DEM_holes_filled.tif
# refdem=$outdir/$site/${pc_align_out}_${res}m-DEM.tif
# WV01_20150911_1020010042D39D00_1020010043455300_aligned_crop_1.0m-DEM.tif
parallel --delay 1 -v "mapproject $refdem -t rpc --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin --tr $res --bundle-adjust-prefix $ba_out {} {.}.xml $outdir/$site/{.}_ortho_${res}m_ba.tif" ::: *r100.tif
# mapproject /mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/baker_2015_utm_m.tif --ot UInt16 --t_srs "EPSG:32610" --t_projwin 584000 5397000 585000 5396000 --tr 1 --bundle-adjust-prefix ba_from_usgs_all616_pc_align /mnt/1.0_TB_VOLUME/sethv/shashank_data/WV01_20150911_1020010042D39D00_1020010043455300/1020010042D39D00.r100.tif /mnt/1.0_TB_VOLUME/sethv/shashank_data/WV01_20150911_1020010042D39D00_1020010043455300/1020010042D39D00.r100.xml WV01_20150911_1020010042D39D00_test_ortho_ba.tif

#Should convert PAN DN values to TOA reflectance

#We should now have self-consistent orthoimages, ASP DEM and lidar DEM - identical UL origin, res, extent

#Compute higher-level products and package
cd $outdir/$site
#Aligned orthoimages
fn_ortho=$(ls *_ortho_${res}m_ba.tif)
#fn_ortho1=$(ls *_ortho_${res}m_ba.tif | head -1)
#fn_ortho2=$(ls *_ortho_${res}m_ba.tif | tail -1)
#Triangulation error
fn_trierr=${pc_align_out}_${res}m-IntersectionErr.tif

#3DEP intensity
fn_3dep_intensity=${laz_fn%.*}_${res}m_intensity_all_idw.tif
#3DEP DEM from PDAL
#fn_list+=${laz_fn%.*}_${res}m_pdal_all_idw.tif
#3DEP DEM from point2dem
fn_3dep_dem=${laz_fn%.*}_${res}m-DEM.tif
#Aligned DEM
fn_asp_dem=${pc_align_out}_${res}m-DEM.tif
#Compute difference map (use lidar to define grid, in case of any residual issues)
compute_diff.py $fn_3dep_dem $fn_asp_dem -te $fn_3dep_dem -tr $fn_3dep_dem
fn_diff=$(basename ${fn_3dep_dem%.*})_$(basename ${fn_asp_dem%.*})_diff.tif
#Generate shaded relief maps
pwd
hs.sh $fn_3dep_dem $fn_asp_dem
echo "-------------------------------"
echo "-------------------------------"
echo "-------------------------------"
echo "-------------------------------"
#TODO why no hs_multi.tif? Do we need to pass hillshades as an input vs the DEM itself?
# fn_hs=$(ls *hs.tif *hs_multi.tif */*hs.tif */*hs_multi.tif)
mkdir -p files_to_zip
fn_hs=$(ls *hs.tif */*hs.tif)

#Package 3DEP LAZ original?
#3DEP LAZ projected, cropped, filtered
fn_3dep_laz=$laz_fn
#Aligned point cloud (LAZ)
fn_asp_laz=${pc_align_out}.laz

# TODO DEM inpainting
# fn_3dep_dem="USGS_LPC_WA_MtBaker_2015_10UEU8597_LAS_2017_32610_first_filt_v1.3_1.0m-DEM.tif"
# fn_asp_dem="pc_align_tr/pc_align_tr-trans_source_1.0m-DEM.tif"
# Just append a suffix to these filenames and save them also in the zip (if we are going that direction)
# TODO find predictable relative path for hole filling script
fn_3dep_dem_filled=${fn_3dep_dem%.*}_holes_filled.tif
python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_3dep_dem $fn_3dep_dem_filled
fn_asp_dem_filled=${fn_asp_dem%.*}_holes_filled.tif
python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_asp_dem $fn_asp_dem_filled

# No need for these zip files right now
# zip -v -j ${site}.zip $fn_ortho $fn_trierr $fn_3dep_intensity $fn_3dep_dem $fn_3dep_dem_filled $fn_asp_dem $fn_3dep_dem_filled $fn_diff $fn_hs $fn_3dep_laz $fn_asp_laz

cp $fn_ortho $fn_trierr $fn_3dep_intensity $fn_3dep_dem $fn_3dep_dem_filled $fn_asp_dem $fn_asp_dem_filled $fn_diff $fn_hs $fn_3dep_laz $fn_asp_laz files_to_zip/

# TODO Ross Beyer hole filling script works, ran separately, do it here instead
# python hole_fill.py files_to_zip../../shashank_data/torchgeo_dataset/pc_laz_prep_full_outputs_20220314_easton_original_bounds/lower_easton3/files_to_zip/pc_align_tr-trans_source_1.0m-DEM.tif ../../shashank_data/torchgeo_dataset/pc_laz_prep_full_outputs_20220314_easton_original_bounds/lower_easton3/files_to_zip/pc_align_tr-trans_source_1.0m-DEM_hole_fill.tif 

#scpput ${site}.zip /tmp/

#Should export COG
# TODO not doing that
