#!/bin/bash
# Generate VTK files from HDF5 files with 4-digit file names a single field or vector field.
# SYNOPSIS:
# 1. make_field.sh vort
# 2. make_field.sh q start_id, e.g. make_field.sh q 100

start_id=1
end_id=9999

if [ $# -lt 1 ]; then
  echo "SYNOPSIS: make_field vort/q; make_field vort/q start_id"
  exit 1
fi

if ! [[ $1 == "vort" || $1 == "q" ]]; then
  echo "SYNOPSIS: make_field vort/q; make_field vort/q start_id"
  exit 1
fi

if [ $# -eq 2 ]; then
  if [[ $2 =~ ^[0-9]+$ ]]; then
    start_id=$2
    else
    echo "Error: $2 is not an integer number!"
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
    echo Making $1 field for HDF/$fname.h5
    python "${TBLHOME}/mytbl/make_field.py" $1 $fname
  fi
done
