#!/bin/bash

export OMP_NUM_THREADS=8

echo "Starting run at: `date`"

mpirun -np 4 -npersocket 1 sfmpiewem adj=y H=y \
     ux=ux.rsf uz=uz.rsf vp=vp_smooth.rsf vs=vs_smooth.rsf wav=wav.rsf \
     mpp=mpp.rsf mps=mps.rsf \
     verbose=y \
     nz=500 oz=0 dz=2 \
     nhx=101 dhx=2 ohx=-100 \
     npx=201 dpx=0.01 opx=-1 \
     gz=2 sz=2 \
     fmin=0 fmax=100

