#!/bin/bash

export OMP_NUM_THREADS=8
echo "Starting run at: `date`"
sflsewem niter=5 \
     np=4 numthreads=8 \
     reg=n \
     ux=ux.rsf uz=uz.rsf vp=vp_smooth.rsf vs=vs_smooth.rsf wav=wav.rsf \
     mpp=mpp_inv.rsf mps=mps_inv.rsf \
     misfit=misfit.rsf \
     verbose=y \
     nt=1500 dt=0.001 ot=0 \
     nsx=34 dsx=100 osx=100 \
     nz=500 oz=0 dz=2 \
     nhx=1 dhx=1 ohx=0 \
     npx=1 dpx=1 opx=0 \
     gz=2 sz=2 \
     fmin=0 fmax=100

echo "finished with exit code $? at: `date`"

