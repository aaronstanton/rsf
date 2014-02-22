/* Common Shot Wave Equation Migration with a 2D PSPI operator.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fftw3.h>
#include "myfree.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

void wem2dop(float **d, float **dmig, float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmin, float fmax,
                 int isx, 
                 int nref,
                 int numthreads,
                 bool adj, bool verbose);
void extrap1f(float **dmig,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,int nref,
              float **c,float **vref,int **iref1,int **iref2,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose);
void pspiop(sf_complex *d_x,
            float w,float dk,int nk,int nmx,int nref,float dz,int iz,
            float **c,float **vref,int **iref1,int **iref2,
            sf_complex i,sf_complex czero,
            fftwf_plan p1,fftwf_plan p2,
            bool adj, bool verbose);
void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
float linear_interp(float y1,float y2,float x1,float x2,float x);
void progress_msg(float progress);

int main(int argc, char* argv[])
{

  sf_file in,out,velp,source_wavelet;
  int n1,n2;
  int nt,nmx,nz;
  int it,ix,iz;
  float o1,o2;
  float d1,d2;
  float ot,omx,oz;
  float dt,dmx,dz;
  float **d,**dmig,**vp,*trace,*wd,*wav;
  bool adj;
  bool verbose;
  float sum;
  float fmin,fmax;
  int sum_wd;
  int op;  
  int nref;
  int numthreads;
  bool dottest;
/*  float **d_1,**d_2,**dmig_1,**dmig_2,tmp_sum1,tmp_sum2;
  unsigned long mseed, dseed;*/
  int isx;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  source_wavelet = sf_input("wav");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getint("op",&op)) op = 1; /* extrapolation operator to be used 1=pspi */
  if (!sf_getint("nref",&nref)) nref = 2; /* number of reference velocities for pspi. */
  if (!sf_getint("numthreads",&numthreads)) numthreads = 1; /* number of threads to be used for parallel processing (if pspi selected). */
  if (!sf_getbool("dottest",&dottest)) dottest = false; /* flag for dot product test, input should be the unmigrated data */
  if (adj || dottest){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
  }

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;
 
  nmx=n2;  
  dmx=d2;  
  omx=o2;
  if (adj || dottest){
    nt=n1;  
    dt=d1;  
    ot=o1;
  }
  else{
    nz=n1;  
    dz=d1;  
    oz=o1;  
  }
  if (!sf_getint("isx",&isx)) isx = (int) nmx/2.0; /* index of position of source */ 
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putfloat(out,"o1",oz);
    sf_putfloat(out,"d1",dz);
    sf_putfloat(out,"n1",nz);
    sf_putstring(out,"label1","Depth");
    sf_putstring(out,"unit1","m");
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"n1",nt);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"unit1","s");
  }
  sf_putfloat(out,"o2",omx);
  sf_putfloat(out,"d2",dmx);
  sf_putfloat(out,"n2",nmx);
  sf_putstring(out,"label2","x");
  sf_putstring(out,"unit2","m");
  if (adj){
    sf_putstring(out,"title","Migrated data");
  }
  else{
    sf_putstring(out,"title","Data");
  }
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc( nt > nz ? nt : nz  );
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,nz,velp);
    for (iz=0;iz<nz;iz++) vp[ix][iz] = trace[iz];
  }
  d = sf_floatalloc2(nt,nmx);
  dmig = sf_floatalloc2(nz,nmx);
  wd = sf_floatalloc(nmx);
  sum_wd = 0;
  if (adj){
    for (ix=0;ix<nmx;ix++){
      sf_floatread(trace,n1,in);
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += trace[it]*trace[it];
        d[ix][it] = trace[it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/(nmx));
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++){
        dmig[ix][iz] = 0.0;
      }
    }
  }

  if (dottest){
    fprintf(stderr,"dottest hasn't been added yet.\n");
    exit (0);
  }
  if (!adj){
    fprintf(stderr,"forward operator hasn't been added yet.\n");
    exit (0);
  }

  if (op==1){
    wem2dop(d,dmig,wav,
               nt,ot,dt, 
               nmx,omx,dmx,
               nz,oz,dz,
               vp,fmin,fmax,
               isx,
               nref,
               numthreads,
               adj,verbose);
  }

  if (adj){
    for (ix=0; ix<nmx; ix++) {
      for (iz=0; iz<nz; iz++) trace[iz] = dmig[ix][iz];	
      sf_floatwrite(trace,nz,out);
    }
  }
  else{
    for (ix=0; ix<nmx; ix++) {
      for (it=0; it<nt; it++) trace[it] = d[ix][it];	
      sf_floatwrite(trace,nt,out);
    }
  }
  exit (0);
}

void wem2dop(float **d, float **dmig,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmin, float fmax,
                 int isx,
                 int nref,
                 int numthreads,
                 bool adj, bool verbose)
/*< Phase Shift Plus Interpolation depth migration operator >*/
{
  int iz,ix,ik,iw,it,iref,nw,nk,padt,padx,ntfft;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  float **vref,vmin,vmax,v;
  int *n,**iref1,**iref2;
  fftwf_plan p1,p2;
  float progress;

  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  if(fmin*dt*ntfft+1<ifmax) ifmin = trunc(fmin*dt*ntfft);
  else ifmin = 0;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  d_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) d_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) d_g_wx[ix][iw] = czero;
      for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
    }
    for (it=0;it<nt;it++) d_t[it] = wav[it];
    f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_s_wx[isx][iw] = d_w[iw];
  }

  /* generate reference velocities for each layer */
  vref = sf_floatalloc2(nz,nref); /* reference velocities for each layer */
  iref1 = sf_intalloc2(nz,nmx); /* index of nearest lower reference velocity for each subsurface point */
  iref2 = sf_intalloc2(nz,nmx); /* index of nearest upper reference velocity for each subsurface point */
  
  for (iz=0;iz<nz;iz++){
    vmin=c[0][iz];
    for (ix=0;ix<nmx;ix++) if (c[ix][iz] < vmin) vmin = c[ix][iz];
    vmax=c[nmx-1][iz];
    for (ix=0;ix<nmx;ix++) if (c[ix][iz] > vmax) vmax = c[ix][iz];
    for (iref=0;iref<nref;iref++) vref[iref][iz] = vmin + (float) iref*(vmax-vmin)/((float) nref-1);
    for (ix=0;ix<nmx;ix++){
      v = c[ix][iz];
      if (vmax>vmin+10){
        iref = (int) truncf((nref-1)*(v-vmin)/(vmax-vmin));
        iref1[ix][iz] = iref;
        iref2[ix][iz] = iref+1;
        if (iref>nref-2){
          iref1[ix][iz] = nref-1;
          iref2[ix][iz] = nref-1;
        }
      }
      else{
        iref1[ix][iz] = 0;
        iref2[ix][iz] = 0;
      }
    }
  }

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a); 
  fftwf_execute_dft(p2,b,b); 

  progress = 0.0;

omp_set_num_threads(numthreads);
#ifdef _OPENMP
#pragma omp parallel for \
        private(iw) \
        shared(dmig,d_g_wx,d_s_wx,progress)
#endif
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    extrap1f(dmig,d_g_wx,d_s_wx,iw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,nref,c,vref,iref1,iref2,i,czero,p1,p2,adj,verbose);
  }
  if (verbose) fprintf(stderr,"\r                   \n");

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);

  return;
} 

void extrap1f(float **dmig,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,int nref,
              float **c,float **vref,int **iref1,int **iref2,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w;
  int iz,ix; 
  sf_complex *d_x;
  d_x = sf_complexalloc(nmx);

  w = iw*dw;
  if (adj){
    for (iz=0;iz<nz;iz++){
      /*########## extrapolate receiver wavefield ##########*/
      for (ix=0;ix<nmx;ix++) d_x[ix] = d_g_wx[ix][iw];
      pspiop(d_x,w,dk,nk,nmx,nref,dz,iz,c,vref,iref1,iref2,i,czero,p1,p2,adj,verbose);
      for (ix=0;ix<nmx;ix++) d_g_wx[ix][iw] = d_x[ix];
      /*########## extrapolate shot wavefield ##########*/
      for (ix=0;ix<nmx;ix++) d_x[ix] = d_s_wx[ix][iw];
      pspiop(d_x,w,dk,nk,nmx,nref,-dz,iz,c,vref,iref1,iref2,i,czero,p1,p2,adj,verbose); 
      for (ix=0;ix<nmx;ix++) d_s_wx[ix][iw] = d_x[ix];
      /*########## zero-lag cross-correlation imaging condition ##########*/
      for (ix=0;ix<nmx;ix++) dmig[ix][iz] += 2*crealf(conjf(d_s_wx[ix][iw])*d_g_wx[ix][iw])/ntfft;
    }
  }

  free1complex(d_x);

  return;
}

void pspiop(sf_complex *d_x,
            float w,float dk,int nk,int nmx,int nref,float dz,int iz,
            float **c,float **vref,int **iref1,int **iref2,
            sf_complex i,sf_complex czero,
            fftwf_plan p1,fftwf_plan p2,
            bool adj, bool verbose)
{
  float v,k,s,vref1,vref2;
  sf_complex L;
  int ix,ik,iref; 
  sf_complex *d_k,**dref;
  fftwf_complex *a,*b;

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  dref = sf_complexalloc2(nref,nmx);
  d_k = sf_complexalloc(nk);

  /************* d_x --> d_k *********/
  for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
  for(ix=nmx;ix<nk;ix++) a[ix] = czero;
  fftwf_execute_dft(p1,a,a); 
  /***********************************/
  for (iref=0;iref<nref;iref++){
    for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
    for (ik=0;ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      s = (w*w)/(vref[iref][iz]*vref[iref][iz]) - (k*k);
      if (s>0) L = cexpf(i*sqrtf(s)*dz);
      else     L = czero;
      d_k[ik] = d_k[ik]*L;
    }
    /************* d_k1 --> d_x1 *******/
    for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
    fftwf_execute_dft(p2,b,b);  
    for(ix=0; ix<nmx;ix++) dref[ix][iref] = b[ix]/nk; 
    /***********************************/
  }
  for (ix=0;ix<nmx;ix++){
    v = c[ix][iz];
    vref1 = vref[iref1[ix][iz]][iz];
    vref2 = vref[iref2[ix][iz]][iz];
    if (vref2 - vref1 > 10.0){
      __real__ d_x[ix] = linear_interp(crealf(dref[ix][iref1[ix][iz]]),crealf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
      __imag__ d_x[ix] = linear_interp(cimagf(dref[ix][iref1[ix][iz]]),cimagf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
    }
    else{
      d_x[ix] = dref[ix][iref1[ix][iz]];
    }
  }

  free2complex(dref);
  free1complex(d_k);
  fftwf_free(a);
  fftwf_free(b);

  return;
}


void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj)
{
  sf_complex *a,*b,czero;
  int *n,ix,ik;
  fftwf_plan p1,p2;
  __real__ czero = 0;
  __imag__ czero = 0;
  a  = sf_complexalloc(nk);
  b  = sf_complexalloc(nk);
  n = sf_intalloc(1); n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);

  if (adj){ /* x --> k */
    for(ix=0 ;ix<nx;ix++) a[ix] = d[ix];
    for(ix=nx;ix<nk;ix++) a[ix] = czero;
    fftwf_execute(p1); 
    for(ik=0 ;ik<nk;ik++) m[ik] = a[ik]; 
  }
  else{ /* k --> x */
    for(ik=0; ik<nk;ik++) b[ik] = m[ik];
    fftwf_execute(p2); 
    for(ix=0; ix<nx;ix++) d[ix] = b[ix]/nk; 
    for(ix=nx;ix<nk;ix++) d[ix] = czero;
  }
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  fftwf_free(a);fftwf_free(b);
  return;
}

void f_op(sf_complex *m,float *d,int nw,int nt,bool adj)
{
  sf_complex *out1a,*in1b,czero;
  float *in1a,*out1b;
  int ntfft,it,iw;
  fftwf_plan p1a,p1b;

  ntfft = (nw-1)*2;
  __real__ czero = 0;
  __imag__ czero = 0;

  if (adj){ /* data --> model */
    out1a = sf_complexalloc(nw);
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
    in1b = sf_complexalloc(ntfft);
    p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
    for(iw=0;iw<nw;iw++) in1b[iw] = m[iw];
    for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
    fftwf_execute(p1b); 
    for(it=0;it<nt;it++) d[it] = out1b[it]/ntfft; 
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }

  return;
}

float linear_interp(float y1,float y2,float x1,float x2,float x)
/*< linear interpolation between two points. x2-x1 must be nonzero. >*/
{
  return  y1 + (y2-y1)*(x-x1)/(x2-x1);
}

void progress_msg(float progress)
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

void my_ricker(float *w, float f0,float dt)
{
  int iw, nw, nc;
  float alpha, beta;  
  nw = (int) 2*trunc((float) (2.2/f0/dt)/2) + 1;
  nc = (int) trunc((float) nw/2);
 
  for (iw=0;iw<nw-2;iw++){
    alpha = (nc-iw+1)*f0*dt*PI;
    beta = alpha*alpha;
    w[iw] = (1-beta*2)*exp(-beta);
  }

  return;
}


