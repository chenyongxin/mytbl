#!/bin/bash
# Generate VTK files from HDF5 files with 4-digit file names.
# SYNOPSIS:
# 1. hdf2vtk.sh
# 2. hdf2vtk.sh start_id, e.g. hdf2vtk.sh 100

start_id=1
end_id=9999

if [ $# -eq 1 ]; then
  if [[ $1 =~ ^[0-9]+$ ]]; then
    start_id=$1
    else
    echo "Error: $1 is not an integer number!"
    exit 1
  fi
fi

for i in `seq $start_id $end_id`
do 
  if [ $i -lt 10 ]; then
    fname=000$i
  elif [ $i -lt 100 ]; then
    fname=00$i
  elif [ $i -lt 1000 ]; then
    fname=0$i
  else
    fname=$i
  fi
  if [ -e HDF/$fname.h5 ]; then
    echo Converting HDF/$fname.h5
    python "${TBLHOME}/mytbl/hdf2vtk.py" $fname
  fi
done
