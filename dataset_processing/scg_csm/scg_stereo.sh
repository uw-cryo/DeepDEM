# Stereo processing and dataset preparation script
# Specific to South Cascade May 2019
# TODO if adapting to new regions, need to remove hard-coded paths, filenames, resolutions

# Set up the stereo directory:
stereo_dir="/mnt/1.0_TB_VOLUME/sethv/shashank_data/20190505_south_cascade_stereo"
mkdir -p $stereo_dir

# Stereo input directory with L1B images/xml
pair_dir="/mnt/1.0_TB_VOLUME/sethv/shashank_data/20190505_south_cascade_stereo/WV03_20190505_104001004C8CF300_104001004CBC0600"

# Stereo output dir
outdir="/mnt/1.0_TB_VOLUME/sethv/shashank_data/20190505_south_cascade_stereo/output_crop_fullres_test"
mkdir -p $outdir

# Copy to dataset directory
dataset_dir="/mnt/1.0_TB_VOLUME/sethv/shashank_data/SCG_EXAMPLE_STACK"
mkdir -p $dataset_dir

### Stereo processing ###

cd $stereo_dir
# Prepare L1B images
gdal_translate -ot UInt16 -co NBITS=16 -co bigtiff=if_safer -co tiled=yes -co compress=lzw WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.ntf WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif
gdal_translate -ot UInt16 -co NBITS=16 -co bigtiff=if_safer -co tiled=yes -co compress=lzw WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.ntf WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif

# Visualize stereo geometry
dg_geom_plot.py $pair_dir

refdem="output_COP30.tif"
proj="+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs"
res=0.386 # meters, determined from L1B smaller GSD

# Define a 9km x 9km box for testing purposes, slightly smaller than the entire scene
ulx=639750
uly=5359750
lrx=648750
lry=5350750

projwin_mediumsize="$ulx $uly $lrx $lry"
output_res=1.0 # meters

debug_res=1.0 # meters, show intermediate otuputs of workflow in lower resolution for quick debugging

# Bundle adjustment
# Run initial bundle_adjust step BEFORE stereo, as suggested in ASP documentation

ba_out="$outdir/ba_rpc_oleg_args"

bundle_adjust                               \
  -t dg                                     \
  --ip-per-image 10000                      \
  --tri-weight 0.1                          \
  --tri-robust-threshold 0.1                \
  --camera-weight 0                         \
  --remove-outliers-params '75.0 3.0 20 20' \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  -o $ba_out

# Create highest resolution mapprojected images to use as stereo inputs
# GNU parallel {/} means remove the path and only keep the filename itself
parallel --progress --delay 1 -v "mapproject -t rpc $refdem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin_mediumsize --tr $res --bundle-adjust-prefix $ba_out {} {.}.xml $outdir/{/}_ortho_${res}m_ba_mediumsize.tif" ::: $pair_dir/*P001.tif 

# Create lower-res version of input images for debugging
# GNU parallel {/} means remove the path and only keep the filename itself
parallel --progress --delay 1 -v "mapproject -t rpc $refdem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin_mediumsize --tr $debug_res --bundle-adjust-prefix $ba_out {} {.}.xml $outdir/{/}_ortho_${debug_res}m_ba_mediumsize_debug_version.tif" ::: $pair_dir/*P001.tif 

# Run stereo (best parameters found in Feb 2023 processing)
parallel_stereo                         \
  --corr-kernel 7 7                     \
  --cost-mode 3                         \
  --subpixel-kernel 15 15               \
  --subpixel-mode 9                     \
  --stereo-algorithm asp_mgm            \
  --alignment-method none               \
  --bundle-adjust-prefix $ba_out        \
  $outdir/WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001_ortho_${res}m_ba_mediumsize.tif \
  $outdir/WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001_ortho_${res}m_ba_mediumsize.tif \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  mgm_scg_david_shashank_params_crop_mediumsize_area_run3 \
  output_COP30.tif

### Coregistration ### 

# TODO point cloud coregistration in the lidar directory: 
lidar_dir="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET"
mkdir -p $lidar_dir
cd lidar_dir

# Align

# Set up ASP & reference point clouds

unaligned_stereo_pc="/mnt/1.0_TB_VOLUME/sethv/shashank_data/20190505_south_cascade_stereo/mgm_scg_david_shashank_params_crop_mediumsize_area_run3-PC.tif"
lidar_pc="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/merged_tiles_deduplicated.laz"

lidar_dem_prefix="scg_merged_lidar_dsm_${output_res}m"
stereo_dem_prefix="scg_aligned_asp_dsm_${output_res}m"
fn_lidar_dem="${lidar_dem_prefix}-DEM.tif"
fn_aligned_stereo_dem="${stereo_dem_prefix}-DEM.tif"
fn_aligned_stereo_trierror="${stereo_dem_prefix}-IntersectionErr.tif"

# pc_align (translation only) with the lidar as reference
pc_align_run="pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30"
pc_align --max-displacement 30 --save-transformed-source-points --compute-translation-only "$lidar_pc" "$unaligned_stereo_pc" -o $pc_align_run

# gives a translation vector (North-East-Down, meters): Vector3(18.811995,2.7807767,2.5954425)

# Create the DEM and triangulation error rasters

# lidar
point2dem --tr $output_res --t_srs "$proj" --t_projwin $projwin_mediumsize $lidar_dir/merged_tiles_deduplicated.laz -o $lidar_dem_prefix

# DEM and triangulation error
point2dem --tr $output_res --errorimage --t_srs "$proj" --t_projwin $projwin_mediumsize ${pc_align_run}-trans_source.tif -o $stereo_dem_prefix

# TODO make a Holes filled version
point2dem --tr $output_res --errorimage --t_srs "$proj" --t_projwin $projwin_mediumsize ${pc_align_run}-trans_source.tif -o $stereo_dem_prefix

# Orthorectify the L1B images
# following https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#applying-the-pc-align-transform-to-cameras
# because stereo was done with were already bundle-adjusted

cd $stereo_dir

initial_ba_out=$ba_out
ba_out_aligned="ba_initial_ba_and_align/ba_initial_ba_and_align"
# fn_aligned_stereo_dem="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/scg_merged_dsm_aligned_1.0m-DEM.tif"

bundle_adjust \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  -t rpc \
  --initial-transform /mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-transform.txt \
  --input-adjustments-prefix $initial_ba_out \
  --apply-initial-transform-only \
  -o $ba_out_aligned

parallel --progress --delay 1 -v "mapproject -t rpc $fn_aligned_stereo_dem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin_mediumsize --tr $output_res --bundle-adjust-prefix $ba_out_aligned {} {.}.xml {/.}_ortho_${output_res}m_ba_aligned_with_pc_align_and_initial_ba.tif" ::: $pair_dir/*P001.tif 

############## 
### TODO Not working to reduce offsets, retry

# NOTE Try running bundle adjust a SECOND time,
# unlike ASP docs and pc_laz_prep_full.sh, removing --apply-initial-transform-only this time.
# When not specifying --tri-weight, this does not decrease the offsets vs --apply-initial-transform-only !
# TODO revisit these final steps, perhaps possible to do better
bundle_adjust \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  -t rpc \
  --initial-transform /mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-transform.txt \
  --input-adjustments-prefix $initial_ba_out \
  -o $ba_out_aligned_with_second_ba

parallel --progress --delay 1 -v "mapproject -t rpc $fn_aligned_stereo_dem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin_mediumsize --tr $output_res --bundle-adjust-prefix $ba_out_aligned_with_second_ba {} {.}.xml {/.}_ortho_${output_res}m_ba_aligned_with_pc_align_and_initial_ba_AND_post_stereo_ba.tif" ::: $pair_dir/*P001.tif 

## Prepare output dataset folder for training/inference

# TODO clean up paths... possibly broken from moving folders
coregistered_dir="$lidar_dir/pc_align_testing"
cd $coregistered_dir

# Create interpolated DEMs at desired postings
# TODO this script crashes with large rasters! Find an alternative
fn_lidar_dem_filled=${fn_lidar_dem%.*}_holes_filled.tif
fn_asp_dem_filled=${fn_aligned_stereo_dem%.*}_holes_filled.tif

python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_lidar_dem $fn_lidar_dem_filled
python /mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/ResDepth/torchgeo_experiments/hole_fill.py $fn_aligned_stereo_dem $fn_asp_dem_filled

# Produces bad results but may avoid NaN issues
gdal_fillnodata.py -md 500 $fn_lidar_dem $fn_lidar_dem_filled
gdal_fillnodata.py -md 500 $fn_aligned_stereo_dem $fn_asp_dem_filled

# dem_mosaic is slow when large holes are considered
dem_mosaic --hole-fill-length 400 $fn_lidar_dem -o $fn_lidar_dem_filled
dem_mosaic --hole-fill-length 400 $fn_aligned_stereo_dem -o $fn_asp_dem_filled

# TODO orthos at the moment are a pair of geotiffs
fn_all_orthos=`ls *_ortho_${output_res}m_ba_aligned_with_pc_align_and_initial_ba.tif`
#_AND_post_stereo_ba.tif`
cp $fn_lidar_dem $fn_lidar_dem_filled $fn_aligned_stereo_dem $fn_asp_dem_filled $fn_aligned_stereo_trierror $fn_all_orthos $dataset_dir

### BONUS ###

# copy earlier results to google drive


# Copy to Google Drive for debugging
# rclone copy $dataset_dir "sethv1_gdrive:/south_cascade_stereo_lidar_debug"
attempt=1
rclone_dir="sethv1_gdrive:/south_cascade_stereo_lidar_debug"
stack_dir="$rclone_dir/stack_attempt$attempt/"
for fn in $fn_lidar_dem $fn_aligned_stereo_dem $fn_aligned_stereo_trierror $fn_all_orthos
do
  rclone copy $fn $stack_dir
done

# rclone copy $fn_lidar_dem $stack_dir
# rclone copy $fn_aligned_stereo_dem $stack_dir
# rclone copy $fn_aligned_stereo_trierror $stack_dir
# rclone copy $fn_all_orthos $stack_dir

cd $stereo_dir
stereo_share_dir="$rclone_dir/stereo_outputs/"

aligned_stereo_pc="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-trans_source.tif"

for fn in $unaligned_stereo_pc $aligned_stereo_pc $lidar_pc
do
  rclone copy $fn $stereo_share_dir
done

# had to merge the unaligned stereo point cloud with GDAL ?
# moved the many folders of tiles into separate directory
# had to then
find mgm_tiles/ -name "*-PC.tif" | grep run3| xargs gdal_merge.py {} -o MERGED_MGM_TILES.tif

### upload some logs of interest
logs_dir="${stereo_share_dir}logs/"
fn_pc_align_log="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-log-pc_align-02-17-1422-19698.txt"
for fn in $fn_pc_align_log 
do
  rclone copy $fn $logs_dir
done


find mgm_tiles/ -name "*-PC.tif" | grep run3 | xargs -I % cp % run3_pc/
# pc_merge does not work
pc_merge -o unaligned_run3_pc_merge-PC run3_pc/*


# bundle_adjust files

initial_transform="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-transform.txt"

for fn in $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
$pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
$pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
$pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
$initial_transform \
$refdem
do
  rclone copy $fn $stereo_share_dir
done


$initial_ba_files

--input-adjustments-prefix $initial_ba_out \
--apply-initial-transform-only \
-o $ba_out_aligned


script_dir="$rclone_dir/draft_scripts/"
fn_scg_script="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/deep-elevation-refinement/pleiades/scg_stereo.sh"
rclone copy $fn_scg_script $script_dir





###
# retry but with -t rpc

initial_ba_with_rpc_this_time="$outdir/initial_ba_with_rpc_this_time"

bundle_adjust                               \
  -t rpc                                     \
  --ip-per-image 10000                      \
  --tri-weight 0.1                          \
  --tri-robust-threshold 0.1                \
  --camera-weight 0                         \
  --remove-outliers-params '75.0 3.0 20 20' \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  WV03_20190505_104001004C8CF300_104001004CBC0600/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  -o $initial_ba_with_rpc_this_time

ba_out_aligned_with_rpc_for_first_ba="$stereo_dir/ba_out_aligned_with_rpc_for_first_ba/ba_out_aligned_with_rpc_for_first_ba"

bundle_adjust \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.tif \
  $pair_dir/WV03_20190505191051_104001004CBC0600_19MAY05191051-P1BS-503126480010_01_P001.xml \
  $pair_dir/WV03_20190505191140_104001004C8CF300_19MAY05191140-P1BS-503126480010_01_P001.xml \
  -t rpc \
  --initial-transform /mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/pc_align_SCG_2019_05_05_stereo_to_lidar_maxdisp30-transform.txt \
  --input-adjustments-prefix $initial_ba_with_rpc_this_time \
  --apply-initial-transform-only \
  -o $ba_out_aligned_with_rpc_for_first_ba

cd $dataset_dir
fn_aligned_stereo_dem="/mnt/1.0_TB_VOLUME/sethv/resdepth_all/data/20190506_SOUTH_CASCADE_DATASET/pc_align_testing/scg_merged_dsm_aligned_1.0m-DEM.tif"
parallel --progress --delay 1 -v "mapproject -t rpc $fn_aligned_stereo_dem --ot UInt16 --t_srs \"${proj}\" --t_projwin $projwin_mediumsize --tr $output_res --bundle-adjust-prefix $ba_out_aligned_with_rpc_for_first_ba {} {.}.xml {/.}_ortho_${output_res}m_ba_aligned_with_pc_align_and_initial_ba_actually_using_rpc.tif" ::: $pair_dir/*P001.tif 

for fn in *_ortho_${output_res}m_ba_aligned_with_pc_align_and_initial_ba_actually_using_rpc.tif
do
  rclone copy $fn $stack_dir
  echo "Copied $fn to Google Drive"
done

