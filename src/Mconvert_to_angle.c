/* Convert 1 gather from time/offset to time/ray parameter
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

#ifndef MARK 
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif 

#ifndef PI
#define PI (3.141592653589793)
#endif

#include <rsf.h>
#include <fftw3.h>
#include "myfree.h"

void offset_to_angle(float **d_h, float **d_a,
                     int nt, float ot, float dt, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx, 
                     float fmin, float fmax,
                     bool adj, bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);

int main(int argc, char* argv[])
{
  sf_file in,out;
  int   nt,nhx,np;
  int   it,ix;
  float ot,ohx,op;
  float dt,dhx,dp;
  float **d_h,**d_a;
  bool adj;
  bool verbose;
  float fmin,fmax;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (adj){
    if (!sf_getint("np",&np)) np=181; /* number of angle samples */
    if (!sf_getfloat("op",&op)) op = -90; /* angle origin in degrees */
    if (!sf_getfloat("dp",&dp)) dp = 1; /* angle increment in degrees */
  }
  else{
    if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
    if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
    if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");
  }

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&nt)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&dt)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&ot)) ot=0.;

  if (adj){
    if (!sf_histint(  in,"n2",&nhx)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&dhx)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"o2",&ohx)) ohx=0.;
  }
  else{
    if (!sf_histint(  in,"n2",&np)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&dp)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"o2",&op)) op=0.;
  }
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  sf_putfloat(out,"o1",ot);
  sf_putfloat(out,"d1",dt);
  sf_putfloat(out,"n1",nt);

  if (adj){
    sf_putfloat(out,"o2",op);
    sf_putfloat(out,"d2",dp);
    sf_putfloat(out,"n2",np);
    sf_putstring(out,"label2","Angle");
    sf_putstring(out,"unit2"," Degrees");
  }
  else {
    sf_putfloat(out,"o2",ohx);
    sf_putfloat(out,"d2",dhx);
    sf_putfloat(out,"n2",nhx);
    sf_putstring(out,"label2","Offset");
    sf_putstring(out,"unit2","m");
  }

  if (verbose) fprintf(stderr,"nt=%d nhx=%d np=%d \n",nt,nhx,np);
  d_h = sf_floatalloc2(nt,nhx);
  d_a = sf_floatalloc2(nt,np);

  if (adj){
    sf_floatread(d_h[0],nt*nhx,in);
    for (ix=0;ix<np;ix++) for (it=0;it<nt;it++) d_a[ix][it] = 0.0;
  }
  else{
    sf_floatread(d_a[0],nt*np,in);
    for (ix=0;ix<nhx;ix++) for (it=0;it<nt;it++) d_h[ix][it] = 0.0;
  }

  offset_to_angle(d_h,d_a,
                  nt,ot,dt, 
                  nhx,ohx,dhx, 
                  np,op,dp, 
                  fmin,fmax,
                  adj,verbose);

  if (adj){
    sf_floatwrite(d_a[0],nt*np,out);
  }
  else{
    sf_floatwrite(d_h[0],nt*nhx,out);
  }
 
  free2float(d_h);
  free2float(d_a);
  exit (0);
}

void offset_to_angle(float **d_h, float **d_a,
                     int nt, float ot, float dt, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx, 
                     float fmin, float fmax,
                     bool adj, bool verbose)
/*< Convert from offset to angle (adj=true) or from angle to offset (adj=false) >*/
{

  int it,ihx,ipx,ik,iw,nw,nk,padt,padx,ntfft,ifmin,ifmax;
  float dw,dk,px;
  sf_complex czero;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_wh;
  sf_complex **d_wa;
  fftwf_complex *a,*b;
  sf_complex L;
  int *n;
  fftwf_plan p1,p2;
  float w,k;

  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  if(fmin*dt*ntfft+1<ifmax) ifmin = trunc(fmin*dt*ntfft);
  else ifmin = 0;

  nk = padx*nhx;
  dk = 2*PI/((float) nk)/dhx;
  dw = 2*PI/((float) ntfft)/dt;
  d_wh = sf_complexalloc2(nw,nhx);
  d_wa = sf_complexalloc2(nw,npx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  

  n = sf_intalloc(1); 
  n[0] = nk;

if (adj){
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  p1 = fftwf_plan_dft(1, n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
  for (iw=0;iw<nw;iw++) for (ipx=0;ipx<npx;ipx++) d_wa[ipx][iw] = czero;
  /* transform the time axis to the frequency domain */
  for (ihx=0;ihx<nhx;ihx++){
    for (it=0;it<nt;it++) d_t[it] = d_h[ihx][it];
    f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_wh[ihx][iw] = d_w[iw]/sqrtf((float) ntfft);
  }

  /* transform the offset axis to the frequency domain */
  for (iw=ifmin;iw<ifmax;iw++){
    w = iw*dw;
    for (ihx=0;ihx<nhx;ihx++) a[ihx] = d_wh[ihx][iw];
    for (ihx=nhx;ihx<nk;ihx++) a[ihx] = czero;
    fftwf_execute_dft(p1,a,a); 
    /* compute Ray Parameters */
    for (ipx=0;ipx<npx;ipx++){
      px = ipx*dpx + opx;
      k = -tanf((PI/180)*px)*w;
      if (k>0){ 
        ik = truncf(k/dk);
      }
      else{ 
        ik = truncf((dk*nk + k)/dk);
      }
      __real__ L = cos(-k*ohx);
      __imag__ L = sin(-k*ohx);
      if (ik < nk && ik >= 0){
        d_wa[ipx][iw] += L*a[ik]/sqrtf((float) nk);
      }
    }
  }
      
  /* transform the frequency axis to the depth domain */
  for (ipx=0;ipx<npx;ipx++){
    for (iw=0;iw<nw;iw++) d_w[iw] = d_wa[ipx][iw];
    f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
    for (it=0;it<nt;it++) d_a[ipx][it] = d_t[it]/sqrtf((float) ntfft);
  }
  fftwf_destroy_plan(p1);
  fftwf_free(a);
}
else{
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  p2 = fftwf_plan_dft(1, n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (iw=0;iw<nw;iw++) for (ihx=0;ihx<nhx;ihx++) d_wh[ihx][iw] = czero;
  /* transform the time axis to the frequency domain */
  for (ipx=0;ipx<npx;ipx++){
    for (it=0;it<nt;it++) d_t[it] = d_a[ipx][it];
    f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_wa[ipx][iw] = d_w[iw]/sqrtf((float) ntfft);
  }

  /* transform the offset axis to the frequency domain */
  for (iw=ifmin;iw<ifmax;iw++){
    w = iw*dw;
    for (ihx=0;ihx<nk;ihx++) b[ihx] = czero;
    /* compute wavenumbers from ray parameters */
    for (ipx=0;ipx<npx;ipx++){
      px = ipx*dpx + opx;
      k = -tanf((PI/180)*px)*w;
      if (k>0){ 
        ik = truncf(k/dk);
      }
      else{ 
        ik = truncf((dk*nk + k)/dk);
      }
      __real__ L = cos(k*ohx);
      __imag__ L = sin(k*ohx);
      if (ik < nk && ik >= 0){
        b[ik] += L*d_wa[ipx][iw];
      }
    }
    fftwf_execute_dft(p2,b,b); 
    for (ihx=0;ihx<nhx;ihx++) d_wh[ihx][iw] = b[ihx]/sqrtf((float) nk);
  }
      
  /* transform the frequency axis to the depth domain */
  for (ihx=0;ihx<nhx;ihx++){
    for (iw=0;iw<nw;iw++) d_w[iw] = d_wh[ihx][iw];
    f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
    for (it=0;it<nt;it++) d_h[ihx][it] = d_t[it]/sqrtf((float) ntfft);
  }
  fftwf_destroy_plan(p2);
  fftwf_free(b);
}

  free2complex(d_wh);
  free2complex(d_wa);
  free1float(d_t);
  free1complex(d_w);
  return;
}

void f_op(sf_complex *m,float *d,int nw,int nt,bool adj)
{
  fftwf_complex *out1a,*in1b;
  sf_complex czero;
  float *in1a,*out1b;
  int ntfft,it,iw;
  fftwf_plan p1a,p1b;

  ntfft = (nw-1)*2;
  __real__ czero = 0;
  __imag__ czero = 0;

  if (adj){ /* data --> model */
    out1a = fftwf_malloc(sizeof(fftwf_complex) * nw);
    in1a = sf_floatalloc(ntfft);
    p1a = fftwf_plan_dft_r2c_1d(ntfft, in1a, (fftwf_complex*)out1a, FFTW_ESTIMATE);
    for(it=0;it<nt;it++) in1a[it] = d[it];
    for(it=nt;it<ntfft;it++) in1a[it] = 0;
    fftwf_execute(p1a); 
    for(iw=0;iw<nw;iw++) m[iw] = out1a[iw];
    fftwf_destroy_plan(p1a);
    fftwf_free(in1a); fftwf_free(out1a);
  }

  else{ /* model --> data */
    out1b = sf_floatalloc(ntfft);
    in1b = fftwf_malloc(sizeof(fftwf_complex) * ntfft);
    p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
    for(iw=0;iw<nw;iw++) in1b[iw] = m[iw];
    for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
    fftwf_execute(p1b); 
    for(it=0;it<nt;it++) d[it] = out1b[it];
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }

  return;
}

