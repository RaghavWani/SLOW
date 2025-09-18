#!/bin/bash

BASE_DIR="/lustre_scratch/spotlight/data/SPLT_20250918_003105/RawVisi"
band=4
ant=0x738ffff

echo "Started making dat file from visibility ..." 
./multi_make_pc_beam.sh $BASE_DIR $band $ant

echo "Started making PC beam from dat file ..." 
./multi_dat_to_beam.sh $BASE_DIR $band

echo "Started RFI mitigation ..." 
./multi_IQRM.sh $BASE_DIR

echo "Started plotting filterbank ..." 
python multi_plot_fil.py $BASE_DIR
