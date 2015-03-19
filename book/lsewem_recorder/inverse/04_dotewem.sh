#!/bin/bash

export OMP_NUM_THREADS=8
echo "Starting run at: `date`"

sfwindow < ux.rsf n3=1 f3=0 > ux_tmp.rsf
sfwindow < uz.rsf n3=1 f3=0 > uz_tmp.rsf
sfcp < mpp_inv.rsf > mpp_tmp.rsf
sfcp < mps_inv.rsf > mps_tmp.rsf
sfdotewem ux=ux_tmp.rsf uz=uz_tmp.rsf mpp=mpp_tmp.rsf mps=mps_tmp.rsf \
          vp=vp_smooth.rsf vs=vs_smooth.rsf wav=wav.rsf \
          nz=500 dz=2 oz=0 \
          nt=1500 dt=0.001 ot=0 \
          nhx=1 dhx=1 ohx=0 \
          npx=1 dpx=1 opx=0 \
          nsx=1 dsx=100 osx=100 \
          fmin=0 fmax=100 \
          sz=2 gz=2 \
          reg=n

echo "finished with exit code $? at: `date`"

