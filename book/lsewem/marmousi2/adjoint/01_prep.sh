#!/bin/sh

sfwindow < ../model/vp.rsf j2=3 | sfsmooth rect1=20 rect2=20 > vp_smooth.rsf
sfwindow < ../model/vs.rsf j2=3 | sfsmooth rect1=20 rect2=20 > vs_smooth.rsf
sfwindow < ../data/ux.rsf j1=2 j2=3 n3=1 f3=60 > ux.rsf
sfwindow < ../data/uz.rsf j1=2 j2=3 n3=1 f3=60 > uz.rsf
sfspike mag=1 n1=1750 d1=0.002 k1=25 | sfricker1 frequency=40 > wav.rsf

