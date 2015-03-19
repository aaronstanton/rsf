#!/bin/bash
export OMP_NUM_THREADS=16

echo "Starting run at: `date`"

sflsewem niter=5 \
     np=1 numthreads=16 \
     reg=y pa=-100 pb=-50 pc=50 pd=100 \
     ux=ux_test2.rsf uz=uz_test2.rsf vp=vp_smooth_test2.rsf vs=vs_smooth_test2.rsf wav=wav_test2.rsf \
     mpp=mpp_inv_test2.rsf mps=mps_inv_test2.rsf \
     misfit=misfit_test2.rsf \
     verbose=y \
     nt=1750 dt=0.002 ot=0 \
     nsx=1 dsx=25 osx=2000 \
     nz=2437 oz=0 dz=1.249 \
     nhx=101 dhx=3.747 ohx=-187.35 \
     npx=201 dpx=0.01 opx=-1 \
     gz=2 sz=2 \
     fmin=0 fmax=100

echo "finished with exit code $? at: `date`"
