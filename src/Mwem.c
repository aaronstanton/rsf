/* Wave Equation Migration.
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
#include "myfree.h"
#include <fftw3.h>

void wem_op(float **d,
         int nt,float ot,float dt,
         int nz,float oz,float dz,
         int nmx,float omx,float dmx,
         int nhx,float ohx,float dhx,
         float **m,
         float **vp,float *wav,
         float fmin, float fmax,
         int numthreads,
         bool adj,bool verbose);
void wem_setup_vel(float **c,float *po,float **pd,int nz,int nmx);
void wem_setup_src(sf_complex **d_s_wx,float *w,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,int nt,int nw,int ifmin, int ifmax);
void wem_setup_rec(sf_complex **d_g_wx,float **d,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,int nt,int nw,int ifmin, int ifmax);
void wem_migration(float **m,sf_complex **d_s_wx,sf_complex **d_g_wx,int nt,int ntfft,float ot,float dt,int iw,int nw,float dw,int nz,float oz,float dz,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,float *po,float **pd,fftwf_plan p1,fftwf_plan p2,bool verbose);
void wem_modelling(float **m,sf_complex **d_s_wx,sf_complex **d_g_wx,int nt,int ntfft,float ot,float dt,int iw,int nw,float dw,int nz,float oz,float dz,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,float *po,float **pd,fftwf_plan p1,fftwf_plan p2,bool verbose);
void wem_extrap_src(sf_complex **d, 
                    int iw, int nw, float dw,
                    int iz, int nz, float oz, float dz,
                    int nmx, float omx, float dmx,
                    int nhx, float ohx, float dhx,
                    float *po, float **pd,
                    fftwf_plan p1, fftwf_plan p2,
                    bool verbose);
void wem_extrap_rec(sf_complex **d, 
                    int iw, int nw, float dw,
                    int iz, int nz, float oz, float dz,
                    int nmx, float omx, float dmx,
                    int nhx, float ohx, float dhx,
                    float *po, float **pd,
                    fftwf_plan p1, fftwf_plan p2,
                    bool adj,
                    bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
void progress_msg(float progress);

int main(int argc, char* argv[])
{
  sf_file in,out,velp,source_wavelet;
  int nz,nt,nmx,nhx;
  float oz,ot,omx,ohx;
  float dz,dt,dmx,dhx;
  int it,ix;
  float **m,**d,**vp,*wd,*wav;
  bool adj;
  bool verbose;
  float sum;
  float fmin,fmax;
  int sum_wd;
  int numthreads;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  source_wavelet = sf_input("wav");
  if (!sf_getint("numthreads",&numthreads)) numthreads = 1; /* number of threads for omp */
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (adj){
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
  if (adj){
    if (!sf_histint(  in,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&ot)) ot=0.;
  }
  else{
    if (!sf_histint(  in,"n1",&nz)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&dz)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&oz)) oz=0.;
  }
  if (!sf_histint(  in,"n2",&nmx)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&dmx)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&omx)) omx=0.;
  if (!sf_histint(  in,"n3",&nhx)) sf_error("No n3= in input");
  if (!sf_histfloat(in,"d3",&dhx)) sf_error("No d3= in input");
  if (!sf_histfloat(in,"o3",&ohx)) ohx=0.;
 
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putfloat(out,"o1",oz);
    sf_putfloat(out,"d1",dz);
    sf_putfloat(out,"n1",nz);
    sf_putstring(out,"label1","Depth");
    sf_putstring(out,"unit1","m");
    sf_putstring(out,"title","Migrated data");
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"n1",nt);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"title","Data");
  }
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  d = sf_floatalloc2(nt,nmx*nhx);
  m = sf_floatalloc2(nz,nmx*nhx);
  wd = sf_floatalloc(nmx*nhx);
  sum_wd = 0;

  if (adj){
    for (ix=0;ix<nmx*nhx;ix++) sf_floatread(d[ix],nt,in);
    for (ix=0;ix<nmx*nhx;ix++){
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += d[ix][it]*d[ix][it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/((float) nmx*nhx));
  }
  else{
    for (ix=0;ix<nmx*nhx;ix++) sf_floatread(m[ix],nz,in);
  }
  wem_op(d,
         nt,ot,dt,
         nz,oz,dz,
         nmx,omx,dmx,
         nhx,ohx,dhx,
         m,
         vp,wav,
         fmin,fmax,
         numthreads,
         adj,verbose);
  if (adj){
    for (ix=0;ix<nmx*nhx;ix++) sf_floatwrite(m[ix],nz,out);
  }
  else{ 
    for (ix=0;ix<nmx*nhx;ix++) sf_floatwrite(d[ix],nt,out);
  }
  free2float(m);
  free2float(d);
  free2float(vp);
  free1float(wd);
  free1float(wav);
 
  exit (0);
}

void wem_op(float **d,
         int nt,float ot,float dt,
         int nz,float oz,float dz,
         int nmx,float omx,float dmx,
         int nhx,float ohx,float dhx,
         float **m,
         float **vp,float *wav,
         float fmin, float fmax,
         int numthreads,
         bool adj,bool verbose)
/*< wave equation depth migration operator >*/
{
  int it,iz,ix,iw,ik;
  int padt,ntfft,nw,ifmin,ifmax;
  sf_complex **d_s_wx,**d_g_wx,*d_w,czero;
  float *po,**pd,progress,*d_t,dw;
  int *n,padmx,padhx,nkmx,nkhx,nk;
  fftwf_complex *a,*b;
  fftwf_plan p1,p2;

  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  if(fmax*dt*ntfft+1<nw) ifmax = truncf(fmax*dt*ntfft)+1;
  else ifmax = nw;
  if(fmin*dt*ntfft+1<ifmax) ifmin = truncf(fmin*dt*ntfft);
  else ifmin = 0;
  dw = 2*PI/((float) ntfft)/dt;
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  d_s_wx = sf_complexalloc2(nw,(int)nmx*nhx);
  d_g_wx = sf_complexalloc2(nw,(int)nmx*nhx);
  po = sf_floatalloc(nz); 
  pd = sf_floatalloc2(nz,nmx); 

  if (adj) for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = 0.0;
  else     for (ix=0;ix<nmx*nhx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;

  wem_setup_vel(vp,po,pd,nz,nmx);
  wem_setup_src(d_s_wx,wav,nmx,omx,dmx,nhx,ohx,dhx,nt,nw,ifmin,ifmax);
  wem_setup_rec(d_g_wx,d,nmx,omx,dmx,nhx,ohx,dhx,nt,nw,ifmin,ifmax);

  padmx = 2;
  padhx = 2;
  nkmx = padmx*nmx;
  nkhx = padhx*nhx;
  nk = nkmx*nkhx;
  a = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b = fftwf_malloc(sizeof(fftwf_complex) * nk);
  n = sf_intalloc(2);
  n[0] = nkmx;
  n[1] = nkhx;
  p1 = fftwf_plan_dft(2,n,a,a,FFTW_FORWARD,FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(2,n,b,b,FFTW_BACKWARD,FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a); 
  fftwf_execute_dft(p2,b,b); 

  progress = 0.0;
  //omp_set_num_threads(numthreads);
  #pragma omp parallel for private(iw) shared(m,d_s_wx,d_g_wx,progress)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) (ifmax - ifmin)); 
    if (verbose) progress_msg(progress);
    if (adj){
      wem_migration(m,d_s_wx,d_g_wx,nt,ntfft,ot,dt,iw,nw,dw,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,verbose);
    }
    else{ 
      wem_modelling(m,d_s_wx,d_g_wx,nt,ntfft,ot,dt,iw,nw,dw,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,verbose);  
    }
  }
  if (!adj){
    for (ix=0;ix<nmx*nhx;ix++){ 
      for (iw=0;iw<ifmin;iw++)      d_w[iw] = czero;  
      for (iw=ifmin;iw<ifmax;iw++)  d_w[iw] = d_g_wx[ix][iw];  
      for (iw=ifmax;iw<nw;iw++)     d_w[iw] = czero;  
      f_op(d_w,d_t,nw,nt,0);
      for (it=0;it<nt;it++)         d[ix][it] = d_t[it]/sqrtf((float) ntfft);  
    }
  }
  if (verbose) fprintf(stderr,"\r                   \n");

  free2complex(d_s_wx);
  free2complex(d_g_wx);
  free1float(d_t);
  free1complex(d_w);
  fftwf_free(a); fftwf_free(b);
  fftwf_destroy_plan(p1); fftwf_destroy_plan(p2);

  return;
}

void wem_setup_vel(float **c,float *po,float **pd,int nz,int nmx)
/*< decompose slowness into layer average, and layer purturbation >*/
{
  int iz,ix;

  for (iz=0;iz<nz;iz++){
    po[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po[iz] += 1.0/c[ix][iz];
    po[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd[ix][iz] = 1.0/c[ix][iz] - po[iz];
    }
  }
  return;
}

void wem_setup_src(sf_complex **d_s_wx,float *w,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,int nt,int nw,int ifmin, int ifmax)
/*< load sources into array >*/
{
  int ix,iw,imin_hx,imx,ihx;
  sf_complex czero,*W;
  W = sf_complexalloc(nw);
  __real__ czero = 0;
  __imag__ czero = 0;
  for (ix=0;ix<nmx*nhx;ix++) for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
  f_op(W,w,nw,nt,1);
  /* find index of min offset to place the source */
  imin_hx = 0;
  for (ihx=0;ihx<nhx;ihx++){
      if (fabs(dhx*ihx + ohx) < fabs(dhx*imin_hx + ohx)) imin_hx = ihx;
  }
  for (imx=0;imx<nmx;imx++){
    for (iw=ifmin;iw<ifmax;iw++) d_s_wx[imin_hx*nmx + imx][iw] = W[iw];
  }
  free1complex(W);
  return;
}

void wem_setup_rec(sf_complex **d_g_wx,float **d,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,int nt,int nw,int ifmin, int ifmax)
/*< load data into array >*/
{
  int ix,it,iw;
  float *w;
  sf_complex czero,*W;
  w = sf_floatalloc(nt);
  W = sf_complexalloc(nw);
  __real__ czero = 0;
  __imag__ czero = 0;
  for (iw=0;iw<nw;iw++)  W[iw] = czero;  
  for (ix=0;ix<nmx*nhx;ix++) for (iw=0;iw<nw;iw++) d_g_wx[ix][iw] = czero;


  for (ix=0;ix<nmx*nhx;ix++){
    for (it=0;it<nt;it++) w[it] = d[ix][it];
    f_op(W,w,nw,nt,1);
    for (iw=0;iw<ifmin;iw++)     d_g_wx[ix][iw] = czero;
    for (iw=ifmin;iw<ifmax;iw++) d_g_wx[ix][iw] = W[iw];
    for (iw=ifmax;iw<nw;iw++)    d_g_wx[ix][iw] = czero;
  }
  free1float(w);
  free1complex(W);
  return;
}


void wem_migration(float **m,sf_complex **d_s_wx,sf_complex **d_g_wx,int nt,int ntfft,float ot,float dt,int iw,int nw,float dw,int nz,float oz,float dz,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,float *po,float **pd,fftwf_plan p1,fftwf_plan p2,bool verbose)
/*< migration of 1 frequency slice >*/
{ 
  int imx,ix,iz;
  float factor;
  if (iw==0) factor = 1;
  else factor = 2;
  for (iz=0;iz<nz;iz++){
    wem_extrap_src(d_s_wx,iw,nw,dw,iz,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,verbose);
    wem_extrap_rec(d_g_wx,iw,nw,dw,iz,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,1,verbose);
    for (imx=0;imx<nmx;ix++){


here I need to compute subsurface offset (x-s) and make the code image individual offsets!


      #pragma omp atomic
      m[ihx*nmx + imx][iz] += factor*crealf(conjf(d_s_wx[ix][iw])*d_g_wx[ix][iw]);
    }
  }
  return;
}

void wem_modelling(float **m,sf_complex **d_s_wx,sf_complex **d_g_wx,int nt,int ntfft,float ot,float dt,int iw,int nw,float dw,int nz,float oz,float dz,int nmx,float omx,float dmx,int nhx,float ohx,float dhx,float *po,float **pd,fftwf_plan p1,fftwf_plan p2,bool verbose)
/*< modelling of 1 frequency slice >*/
{
  int ix,iz;
  sf_complex **smig,czero;
  smig = sf_complexalloc2(nz,nmx*nhx);
  __real__ czero = 0;
  __imag__ czero = 0;
  for (iz=0;iz<nz;iz++){
    wem_extrap_src(d_s_wx,iw,nw,dw,iz,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,verbose);
    for (ix=0;ix<nmx*nhx;ix++) smig[ix][iz] = d_s_wx[ix][iw];
  }
  for (ix=0;ix<nmx*nhx;ix++) d_g_wx[ix][iw] = czero;
  for (iz=nz-1;iz>=0;iz--){
    for (ix=0;ix<nmx*nhx;ix++) d_g_wx[ix][iw] += smig[ix][iz]*m[ix][iz];
    wem_extrap_rec(d_g_wx,iw,nw,dw,iz,nz,oz,dz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,p1,p2,0,verbose);
  }
  free2complex(smig);
  return;
}

void wem_extrap_src(sf_complex **d, 
                    int iw, int nw, float dw,
                    int iz, int nz, float oz, float dz,
                    int nmx, float omx, float dmx,
                    int nhx, float ohx, float dhx,
                    float *po, float **pd,
                    fftwf_plan p1, fftwf_plan p2,
                    bool verbose)
/*< extrapolate sources >*/
{
  int padmx,padhx,nkmx,nkhx,nk;
  int ix,imx,ihx;
  sf_complex L,czero;
  float dkmx,dkhx,kmx,khx,k,s,w;
  fftwf_complex *a,*b;

  w = iw*dw;
  __real__ czero = 0;
  __imag__ czero = 0;
  padmx = 2;
  padhx = 2;
  nkmx = padmx*nmx;
  nkhx = padhx*nhx;
  dkmx = 2*PI/((float) nkmx)/dmx;
  dkhx = 2*PI/((float) nkhx)/dhx;
  nk = nkmx*nkhx;
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  for(ix=0;ix<nk;ix++) a[ix] = (fftwf_complex) czero;
  for (imx=0;imx<nkmx;imx++){
    for (ihx=0;ihx<nkhx;ihx++){
      if (imx<nmx && ihx<nhx){
        a[ihx*nkmx + imx] = d[ihx*nmx + imx][iw];
      }
    }
  }  
  fftwf_execute_dft(p1,a,a); 
  for (imx=0;imx<nkmx;imx++){
    if (imx<nkmx/2) kmx = dkmx*imx;
    else kmx = -(dkmx*nkmx - dkmx*imx);
    for (ihx=0;ihx<nkhx;ihx++){
      if (ihx<nkhx/2) khx = dkhx*ihx;
      else khx = -(dkhx*nkhx - dkhx*ihx);
      k = (khx - kmx)/2;
      s = (w*w)*(po[iz]*po[iz]) - (k*k);
      if (s>=0){ 
        __real__ L = cos(sqrt(s)*(-dz));
        __imag__ L = sin(sqrt(s)*(-dz));
      }
      else L = czero;
      b[ihx*nmx + imx] = ((sf_complex) a[ihx*nmx + imx])*L/sqrtf((float) nk);        
    }
  }  
  fftwf_execute_dft(p2,b,b); 
  for (imx=0;imx<nkmx;imx++){
    for (ihx=0;ihx<nkhx;ihx++){
      if (imx<nmx && ihx<nhx){
       // __real__ L = cos(w*pd[imx][iz]*(-dz));
       // __imag__ L = sin(w*pd[imx][iz]*(-dz)); 
        d[ihx*nmx + imx][iw] = b[ihx*nkmx + imx]/sqrtf((float) nk);
      }
    }
  }  
  fftwf_free(a);
  fftwf_free(b);
  return;
}

void wem_extrap_rec(sf_complex **d, 
                    int iw, int nw, float dw,
                    int iz, int nz, float oz, float dz,
                    int nmx, float omx, float dmx,
                    int nhx, float ohx, float dhx,
                    float *po, float **pd,
                    fftwf_plan p1, fftwf_plan p2,
                    bool adj,
                    bool verbose)
/*< extrapolate receivers >*/
{
  int padmx,padhx,nkmx,nkhx,nk;
  int ix,imx,ihx;
  sf_complex L,czero;
  float dkmx,dkhx,kmx,khx,k,s,w;
  fftwf_complex *a,*b;

  w = iw*dw;
  __real__ czero = 0;
  __imag__ czero = 0;
  padmx = 2;
  padhx = 2;
  nkmx = padmx*nmx;
  nkhx = padhx*nhx;
  dkmx = 2*PI/((float) nkmx)/dmx;
  dkhx = 2*PI/((float) nkhx)/dhx;
  nk = nkmx*nkhx;
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  for(ix=0;ix<nk;ix++) a[ix] = (fftwf_complex) czero;
  if (adj){
    for (imx=0;imx<nkmx;imx++){
      for (ihx=0;ihx<nkhx;ihx++){
        if (imx<nmx && ihx<nhx){
          a[ihx*nkmx + imx] = d[ihx*nmx + imx][iw];
        }
      }
    }
  }  
  else{
    for (imx=0;imx<nkmx;imx++){
      for (ihx=0;ihx<nkhx;ihx++){
        if (imx<nmx && ihx<nhx){
          //__real__ L = cos(w*pd[imx][iz]*dz);
          //__imag__ L = sin(w*pd[imx][iz]*dz); 
          a[ihx*nkmx + imx] = d[ihx*nmx + imx][iw];
        }
      }
    }
  }
  fftwf_execute_dft(p1,a,a); 
  for (imx=0;imx<nkmx;imx++){
    if (imx<nkmx/2) kmx = dkmx*imx;
    else kmx = -(dkmx*nkmx - dkmx*imx);
    for (ihx=0;ihx<nkhx;ihx++){
      if (ihx<nkhx/2) khx = dkhx*ihx;
      else khx = -(dkhx*nkhx - dkhx*ihx);
      k = (khx + kmx)/2;
      s = (w*w)*(po[iz]*po[iz]) - (k*k);
      if (s>=0){ 
        __real__ L = cos(sqrt(s)*dz);
        __imag__ L = sin(sqrt(s)*dz);
      }
      else L = czero;
      a[ihx*nmx + imx] = ((sf_complex) a[ihx*nmx + imx])*L/sqrtf((float) nk);        
    }
  }  
  fftwf_execute_dft(p2,b,b); 
  if (adj){
    for (imx=0;imx<nkmx;imx++){
      for (ihx=0;ihx<nkhx;ihx++){
        if (imx<nmx && ihx<nhx){
          //__real__ L = cos(w*pd[imx][iz]*dz);
          //__imag__ L = sin(w*pd[imx][iz]*dz); 
          d[ihx*nmx + imx][iw] = b[ihx*nkmx + imx]/sqrtf((float) nk);
        }
      }
    }
  }  
  else{
    for (imx=0;imx<nkmx;imx++){
      for (ihx=0;ihx<nkhx;ihx++){
        if (imx<nmx && ihx<nhx){
          d[ihx*nmx + imx][iw] = b[ihx*nkmx + imx]/sqrtf((float) nk);
        }
      }
    }
  }  
  fftwf_free(a);
  fftwf_free(b);
  return;
}

void f_op(sf_complex *m,float *d,int nw,int nt,bool adj)
/*< 1D Forward and Adjoint Fourier transform >*/
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

void progress_msg(float progress)
/*< print progress message to stderr >*/
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

