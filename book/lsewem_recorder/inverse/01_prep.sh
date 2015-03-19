#!/bin/sh

sfsmooth < ../model/vp.rsf rect1=50 rect2=50 > vp_smooth.rsf
sfsmooth < ../model/vs.rsf rect1=50 rect2=50 > vs_smooth.rsf
sfspike mag=1 n1=1500 d1=0.001 k1=25 | sfricker1 frequency=40 > wav.rsf
sfmath < ../cat/ux.rsf W=../cat/W.rsf output='input*W' > ux.rsf
sfmath < ../cat/uz.rsf output='input*1' > uz.rsf
