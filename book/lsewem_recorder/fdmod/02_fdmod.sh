#!/bin/sh

export OMP_NUM_THREADS=8
echo "Starting run at: `date`"

for isx in {0..35}; do
   echo "modelling shot number $isx"
   dsx=100
   osx=100.0
   sx=$(echo $osx+$isx*$dsx | bc)

   # shot positions
   sfmath output=2 < s_.rsf > zs_$isx.rsf
   sfmath output=$sx < s_.rsf > xs_$isx.rsf
   sfmath output=1 < s_.rsf > rs_$isx.rsf
   sfcat axis=2 space=n xs_$isx.rsf zs_$isx.rsf rs_$isx.rsf | sftransp > src_$isx.rsf
   # Isotropic Elastic Finite-difference modeling
   sfewefd2d < ewav.rsf \
          den=den.rsf rec=rec.rsf sou=src_$isx.rsf ccc=ccc.rsf \
          dabc=y snap=n verb=y jdata=2 \
          ssou=y nb=250 nbell=5 \
          > d_fd_$isx.rsf
   sfmath < xr.rsf output="input-$sx" > offset.rsf
   sfwindow n2=1 f2=1 < d_fd_$isx.rsf | 
         sftransp | 
         sfput label1=t unit1=s label2=x unit2=m title='ux' |
         sfmath output='(1e12)*input' |
         sfmutter half=n abs=y t0=0.05 v0=2000 offset=offset.rsf |
         sfput d3=100 o3=$sx \
         > ux_$isx.rsf 
   sfwindow n2=1 f2=0 < d_fd_$isx.rsf | 
         sftransp | 
         sfput label1=t unit1=s label2=x unit2=m title='uz' |
         sfmath output='(1e12)*input' |
         sfmutter half=n abs=y t0=0.05 v0=2000 offset=offset.rsf |
         sfput d3=100 o3=$sx \
         > uz_$isx.rsf 
   echo "shot $isx finished at: `date`"
done
