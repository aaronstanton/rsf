#!/bin/sh

sfmath < ../model/vp.rsf output=1000 > den.rsf

sfmath vp=../model/vp.rsf den=den.rsf output='den*vp^2' | sfsmooth rect1=4 rect2=10 repeat=3 > c11.rsf
sfmath vs=../model/vs.rsf den=den.rsf output='den*vs^2' | sfsmooth rect1=4 rect2=10 repeat=3 > c55.rsf
sfcp < c11.rsf > c33.rsf
sfmath c11=c11.rsf c55=c55.rsf output='c11-2*c55' > c13.rsf
sfcat axis=3 c11.rsf c33.rsf c55.rsf c13.rsf > ccc.rsf
sfmath n1=1750 d1=2 o1=0 output=0 > r_.rsf
sfmath n1=1   d1=0  o1=0 output=0 > s_.rsf
# receiver positions
sfmath output=2 < r_.rsf > zr.rsf
sfmath output='x1' < r_.rsf > xr.rsf
sfcat axis=2 space=n xr.rsf zr.rsf | sftransp > rec.rsf
# Wavelet
sfspike mag=1 n1=3000 d1=0.0005 k1=50 | sfricker1 frequency=40 > wav.rsf
# Elastic wavelet
sfspray axis=1 n=2 < wav.rsf | sftransp plane=12 | sftransp plane=13 > ewav.rsf

