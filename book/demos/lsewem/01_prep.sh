#!/bin/sh

sfsegyread tape=vp_marmousi-ii.segy |
      sfput d1=1.249 d2=1.249 unit1='m' unit2='m' label1='Z' label2='X' |
      sfmath output='input*1000' |
      sfwindow min1=455 min2=0 max2=7000 |
      sfput o1=0 > vp.rsf 

sfsegyread tape=vs_marmousi-ii.segy |
      sfput d1=1.249 d2=1.249 unit1='m' unit2='m' label1='Z' label2='X' |
      sfmath output='input*1000' |
      sfwindow min1=455 min2=0 max2=7000 |
      sfput o1=0 > vs.rsf 

sfwindow < vp.rsf j2=3 | sfsmooth rect1=20 rect2=20 > vp_smooth_test2.rsf
sfwindow < vs.rsf j2=3 | sfsmooth rect1=20 rect2=20 > vs_smooth_test2.rsf
sfwindow < $RSFSRC/book/astanton/lsewem/marmousi2/data/ux.rsf j1=2 j2=3 n3=1 f3=0 > ux_test2.rsf
sfwindow < $RSFSRC/book/astanton/lsewem/marmousi2/data/uz.rsf j1=2 j2=3 n3=1 f3=0 > uz_test2.rsf
sfspike mag=1 n1=1750 d1=0.002 k1=25 | sfricker1 frequency=40 > wav_test2.rsf

