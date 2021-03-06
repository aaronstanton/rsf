/* static preserving basis persuit denoising 
*/
/*
  Copyright (C) 2013 University of Alberta
  
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

#ifndef PI
#define PI (3.141592653589793)
#endif

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif 

void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
float find_lag(float *x, float *y, int n, float dt, float maxdelay);
void time_shift(float *d,float dt,int nt,float tshift);
void mysoft(sf_complex **m,int nw,int nk,float p,float lambda);

int main(int argc, char* argv[])
{
  int ix,it,nt,nx,seed,iw,ik;
  int n1,n2;
  int mode;
  float *trace,*trace1,*trace2;
  float **d;
  sf_complex **m;
  float d1,o1,d2,o2;
  int padt,padx;
  int ntfft,nw,nk;
  int iter,Niter; 
  float lag;
  float *tshift;
  float **s,**Q;
  sf_complex **grad;
  float Mmax,Gradmax,p;
  float alpha,lambda;
  float maxlag;
  bool verbose;
  sf_complex czero;
  sf_file in,out;
  
  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");

  __real__ czero = 0;
  __imag__ czero = 0;

  if (!sf_getfloat("maxlag",&maxlag)) maxlag = 0.02; /* maximum static shift in the data (seconds) */
  if (!sf_getint("iter",&Niter)) Niter = 20; /* number of iterations */
  if (!sf_getfloat("alpha",&alpha)) alpha = 0.03; /* step size */
  if (!sf_getfloat("lambda",&lambda)) lambda = 0.03; /* hyperparameter for the inversion */
  if (!sf_getint("mode",&mode)) mode = 1; /* flag for what to do with statics on output: 1=denoise only (leave statics in data), 2=denoise and correct, or 3=static correct only*/
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  /* read input file parameters */
  if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;

  nx = n2;
  nt = n1;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nx;

  trace = sf_floatalloc (n1);
  trace1 = sf_floatalloc (n1);
  trace2 = sf_floatalloc (n1);
  d = sf_floatalloc2(nt,nx);
  m = sf_complexalloc2(nw,nk);
  tshift = sf_floatalloc (nx);
  grad = sf_complexalloc2(nw,nk);
  s = sf_floatalloc2(nt,nx);
  Q = sf_floatalloc2(nt,nx);

  for (ix=0; ix<n2; ix++) {	
    sf_floatread(trace,n1,in);
    for (it=0; it<n1; it++) d[ix][it] = trace[it];
    tshift[ix] = 0.0;
  }

  fk_op(m,d,nw,nk,nt,nx,1); /* adjoint: d to m */

  for (ik=0;ik<nk;ik++){
    for (iw=0;iw<nw;iw++){
      m[ik][iw] = czero;
    }
  }

  for (iter=1;iter<=Niter;iter++){
    if (verbose) fprintf(stderr,"iter=%d/%d\n",iter,Niter);
    /* apply time shift */
    for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++) trace1[it] = d[ix][it];
      time_shift(trace1,d1,nt,tshift[ix]);
      for (it=0;it<nt;it++) s[ix][it] = trace1[it];
    }
    fk_op(grad,s,nw,nk,nt,nx,1); /* adjoint: s to grad */
    p = 0.0;
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        if (sf_cabs(grad[ik][iw]) > p) p = sf_cabs(grad[ik][iw]);
      }
    }
    mysoft(grad,nw,nk,p,lambda);
    Mmax = 0.0;
    Gradmax = 0.0;
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        if (sf_cabs(m[ik][iw]) > Mmax) Mmax = sf_cabs(m[ik][iw]);
        if (sf_cabs(grad[ik][iw]) > Gradmax) Gradmax = sf_cabs(grad[ik][iw]);
      }
    }
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        m[ik][iw] = m[ik][iw] - alpha*Mmax/(Gradmax*grad[ik][iw]);
      }
    }
    fk_op(m,Q,nw,nk,nt,nx,0); /* foward: m to Q */
    /* calculate time shift */
    for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++) trace1[it] = Q[ix][it];
      for (it=0;it<nt;it++) trace2[it] = d[ix][it];
      lag = find_lag(trace1,trace2,nt,d1,maxlag);
      tshift[ix] = lag;
    }

  }
  if (mode != 3) fk_op(m,d,nw,nk,nt,nx,0); /* foward: m to d */

  for (ix=0; ix<nx; ix++) {	
    for (it=0; it<nt; it++) trace[it] = d[ix][it];
    if (mode==1 || mode==3) time_shift(trace,d1,nt,-tshift[ix]);
    sf_floatwrite(trace,n1,out);
  }

  exit (0);
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
    for (ik=0;ik<nk;ik++) m[ik][iw] = in2a[ik];
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
    for(it=0;it<nt;it++) d[ix][it] = out1b[it]/ntfft; 
  }
  fftwf_destroy_plan(p1b);
  fftwf_free(in1b); fftwf_free(out1b);
}
  free2complex(cpfft);

  return;

}

float find_lag(float *x, float *y, int n, float dt, float maxdelay)
{
  int i,j;
  float mx,my,sx,sy,sxy,denom,r;
  int delay,rmax_delay,imaxdelay;
  float rmax;
  rmax = 0;
  rmax_delay = 0;
  mx = 0;
  my = 0;   
  for (i=0;i<n;i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  sx = 0;
  sy = 0;
  for (i=0;i<n;i++) {
    sx += (x[i] - mx) * (x[i] - mx);
    sy += (y[i] - my) * (y[i] - my);
  }
  denom = sqrt(sx*sy);

  imaxdelay = (int) trunc(maxdelay/dt);    

  for (delay=-imaxdelay;delay<imaxdelay;delay++) {
    sxy = 0;
    for (i=0;i<n;i++) {
      j = i + delay;
      if (j < 0 || j >= n)
        continue;
      else
        sxy += (x[i] - mx) * (y[j] - my);
    }
    r = sxy / denom; /* correlation coeff at delay */
    if (fabsf(r) > rmax){ 
      rmax = fabsf(r);
      rmax_delay = delay;
    }
  }

  return dt*rmax_delay;
}

void time_shift(float *d,float dt,int nt,float tshift)
/* apply a time shift to one trace */
{
  float *in1, *out2, omega;
  sf_complex *out1, *in2, shift_op, *D;
  fftwf_plan p1, p2;
  int padfactor,ntfft,nw,it,iw,ifmin,ifmax;

  padfactor = 2;
  ntfft = padfactor*nt;
  nw = (int) ntfft/2 + 1; 
  D = sf_complexalloc(ntfft);
  ifmin = 0;
  ifmax = nw;
  in1 = sf_floatalloc(ntfft);
  out1 = sf_complexalloc(nw);
  p1 = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  for(it=0;it<nt;it++)     in1[it] = d[it];
  for(it=nt;it<ntfft;it++) in1[it] = 0;
  fftwf_execute(p1);
  for(iw=0;iw<nw;iw++) D[iw] = out1[iw]; 
  for (iw=ifmin;iw<ifmax;iw++){
    omega = (float) 2*PI*iw/ntfft/dt;
    __real__ shift_op = cos(omega*tshift);
    __imag__ shift_op =-sin(omega*tshift);
    D[iw] = D[iw]*shift_op;
  } 
  in2 = sf_complexalloc(ntfft);
  out2 = sf_floatalloc(ntfft);
  p2 = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for(iw=0;iw<nw;iw++){
    in2[iw] = D[iw];
  }
  fftwf_execute(p2);
  for(it=0;it<nt;it++){
    d[it] = out2[it]; 
  }
  for (it=0; it<nt; it++) d[it]=d[it]/ntfft;
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);

  fftwf_free(in1);
  fftwf_free(in2);
  fftwf_free(out1);
  fftwf_free(out2);
  fftwf_free(D);

  return;
}

void mysoft(sf_complex **m,int nw,int nk,float p,float lambda)
{
  int ik,iw;
  float a,ang;

  for (ik=0;ik<nk;ik++){
    for (iw=0;iw<nw;iw++){
      a = sf_cabs(m[ik][iw]/p) - lambda;
      ang = cargf(m[ik][iw]/p);
      if (a<0.0) a = 0.0;
      __real__ m[ik][iw] = p*a*cos(ang);
      __imag__ m[ik][iw] = p*a*sin(ang);
    }
  }
  
  return;
}




