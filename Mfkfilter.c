/* FK fan filtering.
*/
/*
  Copyright (C) 2014 University of Alberta
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <rsf.h>
#include <fftw3.h>
#include "myfree.h"
void fkfilter(float **d, float dt, int nt, float dx, int nx, float pa, float pb, float pc, float pd);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
int main(int argc, char* argv[])
{
    int n1,n2;
    float **d,d1,o1,d2,o2,pa,pb,pc,pd;
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");
    if (!sf_getfloat("pa",&pa)) pa = -2; /* minimum slowness (value=0) */
    if (!sf_getfloat("pb",&pb)) pb = -1; /* minimum slowness taper (value=1) */
    if (!sf_getfloat("pc",&pc)) pc =  1; /* maximum slowness taper (value=1) */
    if (!sf_getfloat("pd",&pd)) pd =  2; /* maximum slowness (value=0) */
    /* read input file parameters */
    if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&o1)) o1=0.;
    if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&d2)) d2=1;
    if (!sf_histfloat(in,"o2",&o2)) o2=0.;
    d = sf_floatalloc2(n1,n2);
    sf_floatread(d[0],n1*n2,in);
    fkfilter(d,d1,n1,d2,n2,pa,pb,pc,pd);
    sf_floatwrite(d[0],n1*n2,out);
    free2float(d);
    exit (0);
}

void fkfilter(float **d, float dt, int nt, float dx, int nx, float pa, float pb, float pc, float pd)
{
  int iw,nw,ntfft,nk,ik,padt,padx;
  float k,w,dk,dw,p;
  sf_complex **m;
  sf_complex czero;
  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nx;
  dk = (float) 1/nk/dx;
  dw = (float) 1/ntfft/dt;
  m = sf_complexalloc2(nw,nk);
  fk_op(m,d,nw,nk,nt,nx,1);

  for (iw=1;iw<nw;iw++){
    w = dw*iw;
    for (ik=0;ik<nk;ik++){
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      p = k/w;
      if (p<pa)                m[ik][iw] = czero; 
      else if (p>=pa && p<pb)  m[ik][iw] = m[ik][iw]*((p-pa)/(pb-pa)); 
      else if (p>=pb && p<=pc) m[ik][iw] = m[ik][iw]; 
      else if (p>pc && p<=pd)  m[ik][iw] = m[ik][iw]*(1-(p-pc)/(pd-pc)); 
      else                     m[ik][iw] = czero;
    }
  }
  fk_op(m,d,nw,nk,nt,nx,0);
  free2complex(m);
  return;
}

void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj)
{
  sf_complex **cpfft,*out1a,*in1b,*in2a,*in2b,czero;
  float *in1a,*out1b;
  int *n,ntfft,ix,it,iw,ik;
  fftwf_plan p1a,p1b,p2a,p2b;

  ntfft = (nw-1)*2;
  __real__ czero = 0;
  __imag__ czero = 0;
  cpfft = sf_complexalloc2(nw,nk);
  out1a = sf_complexalloc(nw);
  in1a = sf_floatalloc(ntfft);
  p1a = fftwf_plan_dft_r2c_1d(ntfft, in1a, (fftwf_complex*)out1a, FFTW_ESTIMATE);
  out1b = sf_floatalloc(ntfft);
  in1b = sf_complexalloc(ntfft);
  p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
  n = sf_intalloc(1); n[0] = nk;
  in2a = sf_complexalloc(nk);
  in2b = sf_complexalloc(nk);
  p2a = fftwf_plan_dft(1, n, (fftwf_complex*)in2a, (fftwf_complex*)in2a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2b = fftwf_plan_dft(1, n, (fftwf_complex*)in2b, (fftwf_complex*)in2b, FFTW_BACKWARD, FFTW_ESTIMATE);

  if (adj){ /* data --> model */
    for (ix=0;ix<nx;ix++){
      for(it=0;it<nt;it++) in1a[it] = d[ix][it];
      for(it=nt;it<ntfft;it++) in1a[it] = 0;
      fftwf_execute(p1a); 
      for(iw=0;iw<nw;iw++) cpfft[ix][iw] = out1a[iw]; 
    }
    fftwf_destroy_plan(p1a);
    fftwf_free(in1a); fftwf_free(out1a);
    for (iw=0;iw<nw;iw++){  
      for (ik=0;ik<nk;ik++) in2a[ik] = cpfft[ik][iw];
      fftwf_execute(p2a); /* FFT x to k */
      for (ik=0;ik<nk;ik++) m[ik][iw] = in2a[ik]/sqrtf((float) ntfft);
    }
    fftwf_destroy_plan(p2a);
    fftwf_free(in2a); 
  }
  else{ /* model --> data */
    for (iw=0;iw<nw;iw++){  
      for (ik=0;ik<nk;ik++) in2b[ik] = m[ik][iw];
      fftwf_execute(p2b); /* FFT k to x */
      for (ik=0;ik<nx;ik++) cpfft[ik][iw] = in2b[ik]/nk;
    }
    fftwf_destroy_plan(p2b);
    fftwf_free(in2b);
    for (ix=0;ix<nx;ix++){
      for(iw=0;iw<nw;iw++) in1b[iw] = cpfft[ix][iw];
      for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
      fftwf_execute(p1b); 
      for(it=0;it<nt;it++) d[ix][it] = out1b[it]/sqrtf((float) ntfft); 
    }
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }
  free2complex(cpfft);
  return;
}

