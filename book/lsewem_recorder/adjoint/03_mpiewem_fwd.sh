#!/bin/bash

export OMP_NUM_THREADS=8

echo "Starting run at: `date`"

mpirun -np 4 -npersocket 1 sfmpiewem adj=n \
     ux=ux_fwd.rsf uz=uz_fwd.rsf vp=vp_smooth.rsf vs=vs_smooth.rsf wav=wav.rsf \
     mpp=mpp.rsf mps=mps.rsf \
     verbose=y \
     nz=500 oz=0 dz=2 \
     nhx=1 dhx=1 ohx=0 \
     npx=1 dpx=1 opx=0 \
     gz=2 sz=2 \
     nt=1500 dt=0.001 ot=0 \
     nsx=34 dsx=100 osx=100 \
     fmin=0 fmax=100

