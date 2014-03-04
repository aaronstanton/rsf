/* Common Shot Wave Equation Migration of Elastic 2C-2D data.
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

void ewem2dop(float **dp, float **ds, float **dmigpp, float **dmigps, float *wav,
              int nt, float ot, float dt,
              int nmx, float omx, float dmx,
              int nz, float oz, float dz,
              float **vp, float **vs, float fmin, float fmax,
              int isx,
              int numthreads,
              bool adj, bool verbose);
void extrap1f(float **dmigpp,float **dmigps,
              sf_complex **dp_g_wx,sf_complex **ds_g_wx,sf_complex **d_s_wx,
              int iw,int ifmax,int ntfft,float dw,float dk,int nk,
              float dz,int nz,int nmx,
              float **vp,float **vs,float *po_p,float **pd_p,float *po_s,float **pd_s,
              sf_complex i,sf_complex czero,fftwf_plan p1,fftwf_plan p2,bool adj,bool verbose);
void ssop(sf_complex *d_x,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *p0,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool verbose);
void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
void progress_msg(float progress);

int main(int argc, char* argv[])
{

  sf_file in1,in2,out1,out2,velp,vels,source_wavelet;
  int n1,n2;
  int nt,nmx,nz;
  int it,ix,iz;
  float o1,o2;
  float d1,d2;
  float ot,omx,oz;
  float dt,dmx,dz;
  float **dp,**ds,**dmigpp,**dmigps,**vp,**vs,*trace,*wd,*wav;
  bool adj;
  bool verbose;
  float sum;
  float fmin,fmax;
  int sum_wd;
  int numthreads;
  bool dottest;
  float sx;
  int isx;

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  source_wavelet = sf_input("wav");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
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
  if (!sf_histint(  in1,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in1,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in1,"o1",&o1)) o1=0.;
  if (!sf_histint(  in1,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in1,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in1,"o2",&o2)) o2=0.;
 
  nmx=n2;  
  dmx=d2;  
  omx=o2;
  nt=n1;  
  dt=d1;  
  ot=o1;

  if (!sf_getfloat("sx",&sx)) sx = dmx*((float) nmx)/2.0; /* index of position of source */
  isx = (int) truncf((sx - omx)/dmx); 
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  sf_putfloat(out1,"o1",oz);
  sf_putfloat(out1,"d1",dz);
  sf_putfloat(out1,"n1",nz);
  sf_putstring(out1,"label1","Depth");
  sf_putstring(out1,"unit1","m");
  sf_putfloat(out1,"o2",omx);
  sf_putfloat(out1,"d2",dmx);
  sf_putfloat(out1,"n2",nmx);
  sf_putstring(out1,"label2","x");
  sf_putstring(out1,"unit2","m");
  sf_putstring(out1,"title","PP Image");

  sf_putfloat(out2,"o1",oz);
  sf_putfloat(out2,"d1",dz);
  sf_putfloat(out2,"n1",nz);
  sf_putstring(out2,"label1","Depth");
  sf_putstring(out2,"unit1","m");
  sf_putfloat(out2,"o2",omx);
  sf_putfloat(out2,"d2",dmx);
  sf_putfloat(out2,"n2",nmx);
  sf_putstring(out2,"label2","x");
  sf_putstring(out2,"unit2","m");
  sf_putstring(out2,"title","PS Image");

  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  vs = sf_floatalloc2(nz,nmx);
  trace = sf_floatalloc( nt > nz ? nt : nz  );
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,nz,velp);
    for (iz=0;iz<nz;iz++) vp[ix][iz] = trace[iz];
    sf_floatread(trace,nz,vels);
    for (iz=0;iz<nz;iz++) vs[ix][iz] = trace[iz];
  }
  dp = sf_floatalloc2(nt,nmx);
  ds = sf_floatalloc2(nt,nmx);
  dmigpp = sf_floatalloc2(nz,nmx);
  dmigps = sf_floatalloc2(nz,nmx);
  wd = sf_floatalloc(nmx);
  sum_wd = 0;
  if (adj){
    for (ix=0;ix<nmx;ix++){
      sf_floatread(trace,n1,in1); /* read component 1 */
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += trace[it]*trace[it];
        dp[ix][it] = trace[it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
      sf_floatread(trace,n1,in2); /* read component 2 */
      for (it=0;it<nt;it++) ds[ix][it] = trace[it];
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/(nmx));
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++){
        dmigpp[ix][iz] = 0.0;
        dmigps[ix][iz] = 0.0;
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

  ewem2dop(dp,ds,dmigpp,dmigps,wav,
           nt,ot,dt,
           nmx,omx,dmx,
           nz,oz,dz,
           vp,vs,fmin,fmax,
           isx,
           numthreads,
           adj,verbose);
  if (adj){
    for (ix=0; ix<nmx; ix++) {
      for (iz=0; iz<nz; iz++) trace[iz] = dmigpp[ix][iz];	
      sf_floatwrite(trace,nz,out1);
      for (iz=0; iz<nz; iz++) trace[iz] = dmigps[ix][iz];	
      sf_floatwrite(trace,nz,out2);
    }
  }
  exit (0);
}

void ewem2dop(float **dp, float **ds, float **dmigpp, float **dmigps, float *wav,
              int nt, float ot, float dt,
              int nmx, float omx, float dmx,
              int nz, float oz, float dz,
              float **vp, float **vs, float fmin, float fmax,
              int isx,
              int numthreads,
              bool adj, bool verbose)
/*< 2C-2D scalar elastic depth migration operator >*/
{
  int iz,ix,ik,iw,it,nw,nk,padt,padx,ntfft;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **dp_g_wx,**ds_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float progress;
  float *po_p,**pd_p,*po_s,**pd_s;

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
  dp_g_wx = sf_complexalloc2(nw,nmx);
  ds_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  if (adj){
    for (ix=0;ix<nmx;ix++){

      for (it=0;it<nt;it++) d_t[it] = dp[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) dp_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dp_g_wx[ix][iw] = czero;

      for (it=0;it<nt;it++) d_t[it] = ds[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) ds_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) ds_g_wx[ix][iw] = czero;

      for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
    }
    for (it=0;it<nt;it++) d_t[it] = wav[it];
    f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_s_wx[isx][iw] = d_w[iw];
  }

  /* Split-Step operator */
  /* decompose slowness into layer average, and layer purturbation */
  po_p = sf_floatalloc(nz); 
  pd_p = sf_floatalloc2(nz,nmx); 
  po_s = sf_floatalloc(nz); 
  pd_s = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po_p[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_p[iz] += 1.0/vp[ix][iz];
    po_p[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++) pd_p[ix][iz] = 1.0/vp[ix][iz] - po_p[iz];
    po_s[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_s[iz] += 1.0/vs[ix][iz];
    po_s[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++) pd_s[ix][iz] = 1.0/vs[ix][iz] - po_s[iz];
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
        shared(dmigpp,dmigps,dp_g_wx,ds_g_wx,d_s_wx,progress)
#endif
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    extrap1f(dmigpp,dmigps,
             dp_g_wx,ds_g_wx,d_s_wx,
             iw,ifmax,ntfft,dw,dk,nk,
             dz,nz,nmx,
             vp,vs,po_p,pd_p,po_s,pd_s,
             i,czero,p1,p2,adj,verbose);
  }
  if (verbose) fprintf(stderr,"\r                   \n");

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);

  return;
} 

void extrap1f(float **dmigpp,float **dmigps,
              sf_complex **dp_g_wx,sf_complex **ds_g_wx,sf_complex **d_s_wx,
              int iw,int ifmax,int ntfft,float dw,float dk,int nk,
              float dz,int nz,int nmx,
              float **vp,float **vs,float *po_p,float **pd_p,float *po_s,float **pd_s,
              sf_complex i,sf_complex czero,fftwf_plan p1,fftwf_plan p2,bool adj,bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w;
  int iz,ix; 
  sf_complex *d_x;
  d_x = sf_complexalloc(nmx);

  w = iw*dw;
  if (adj){
    for (iz=0;iz<nz;iz++){
      /*########## extrapolate receiver P wavefield ##########*/
      for (ix=0;ix<nmx;ix++) d_x[ix] = dp_g_wx[ix][iw];
      ssop(d_x,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,adj,verbose);
      for (ix=0;ix<nmx;ix++) dp_g_wx[ix][iw] = d_x[ix];
      /*########## extrapolate receiver S wavefield ##########*/
      for (ix=0;ix<nmx;ix++) d_x[ix] = ds_g_wx[ix][iw];
      ssop(d_x,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,adj,verbose);
      for (ix=0;ix<nmx;ix++) ds_g_wx[ix][iw] = d_x[ix];
      /*########## extrapolate shot wavefield ##########*/
      for (ix=0;ix<nmx;ix++) d_x[ix] = d_s_wx[ix][iw];
      ssop(d_x,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,adj,verbose); 
      for (ix=0;ix<nmx;ix++) d_s_wx[ix][iw] = d_x[ix];
      /*########## zero-lag cross-correlation imaging condition ##########*/
      for (ix=0;ix<nmx;ix++){
        dmigpp[ix][iz] += 2*crealf(conjf(d_s_wx[ix][iw])*dp_g_wx[ix][iw])/ntfft;
        dmigps[ix][iz] += 2*crealf(conjf(d_s_wx[ix][iw])*ds_g_wx[ix][iw])/ntfft;
      }
    }
  }

  free1complex(d_x);

  return;
}

void ssop(sf_complex *d_x,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *po,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool verbose)
{
  float k,s;
  sf_complex L;
  int ix,ik; 
  sf_complex *d_k;
  fftwf_complex *a,*b;

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  d_k = sf_complexalloc(nk);

  /************* d_x --> d_k *********/
  for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
  for(ix=nmx;ix<nk;ix++) a[ix] = czero;
  fftwf_execute_dft(p1,a,a); 
  /***********************************/
  for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
  for (ik=0;ik<nk;ik++){ 
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    s = (w*w)*(po[iz]*po[iz]) - (k*k);
    if (s>0){
      L = cexpf(i*sqrtf(s)*dz);
      d_k[ik] = d_k[ik]*L;
    }
    else {
      d_k[ik] = czero;
    }
  }
  /************* d_k1 --> d_x1 *******/
  for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
  fftwf_execute_dft(p2,b,b);
  for(ix=0; ix<nmx;ix++){
    d_x[ix] = b[ix]*cexpf(i*w*pd[ix][iz]*dz)/nk;
  }

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

void progress_msg(float progress)
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}


