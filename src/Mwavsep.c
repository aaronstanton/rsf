/* Common Shot Isotropic Wavefield Separation of P and SV waves using Helmholtz operator in Fourier domain.
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

void wesep2dop(float **dx_1shot,float **dz_1shot,float **dp_1shot,float **ds_1shot,
               int nt,float ot,float dt,int nmx,float omx,float dmx,int nz,float oz,float dz,
               float **vp,float **vs,float fmin,float fmax,float gz,
               bool H, bool ss,
               bool verbose);
float signf(float a);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);

int main(int argc, char* argv[])
{

  sf_file in1,in2,out1,out2,velp,vels;
  int n1,n2,n3;
  int nt,nmx,nz,nsx;
  int it,ix,iz,isx;
  float o1,o2,o3;
  float d1,d2,d3;
  float ot,omx,osx,oz;
  float dt,dmx,dsx,dz;
  float **d_x,**d_z,**d_p,**d_s,**vp,**vs,*trace;
  float gz;
  bool H;
  bool ss;
  bool verbose;
  float fmin,fmax;

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("H",&H)) H = true; /* Helmholtz decomposition to p and s = y, Helmholtz recomposition to x and z = n */
  if (!sf_getbool("ss",&ss)) ss = false; /* flag for split-step correctoion */
  if (!sf_getfloat("gz",&gz)) gz=0.; /* depth of receivers */
  /* read input file parameters */
  if (!sf_histint(  in1,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in1,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in1,"o1",&o1)) o1=0.;
  if (!sf_histint(  in1,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in1,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in1,"o2",&o2)) o2=0.;
  if (!sf_histint(  velp,"n1",&nz)) sf_error("No n1= in vp");
  if (!sf_histfloat(velp,"d1",&dz)) sf_error("No d1= in vp");
  if (!sf_histfloat(velp,"o1",&oz)) oz=0.;
  if (!sf_histint(  in1,"n3",&n3)) n3=1;
  if (!sf_histfloat(in1,"d3",&d3)) d3=1;
  if (!sf_histfloat(in1,"o3",&o3)) o3=0.;

  nmx=n2;
  dmx=d2;  
  omx=o2;
  nt=n1;  
  dt=d1;  
  ot=o1;
  nsx=n3;
  osx=o3;
  dsx=d3;
  
  if (!sf_getfloat("fmin",&fmin)) fmin = 0;      /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;
  sf_putfloat(out1,"o1",ot);
  sf_putfloat(out1,"d1",dt);
  sf_putfloat(out1,"n1",nt);
  sf_putstring(out1,"label1","Time");
  sf_putstring(out1,"unit1","s");
  sf_putfloat(out1,"o2",omx);
  sf_putfloat(out1,"d2",dmx);
  sf_putfloat(out1,"n2",nmx);
  sf_putstring(out1,"label2","x");
  sf_putstring(out1,"unit2","m");
  sf_putstring(out1,"title","P");
  sf_putfloat(out1,"o3",osx);
  sf_putfloat(out1,"d3",dsx);
  sf_putfloat(out1,"n3",nsx);
  sf_putstring(out1,"label3","sx");
  sf_putstring(out1,"unit3","m");
  sf_putfloat(out2,"o1",ot);
  sf_putfloat(out2,"d1",dt);
  sf_putfloat(out2,"n1",nt);
  sf_putstring(out2,"label1","Time");
  sf_putstring(out2,"unit1","s");
  sf_putfloat(out2,"o2",omx);
  sf_putfloat(out2,"d2",dmx);
  sf_putfloat(out2,"n2",nmx);
  sf_putstring(out2,"label2","x");
  sf_putstring(out2,"unit2","m");
  sf_putstring(out2,"title","SV");
  sf_putfloat(out2,"o3",osx);
  sf_putfloat(out2,"d3",dsx);
  sf_putfloat(out2,"n3",nsx);
  sf_putstring(out2,"label3","sx");
  sf_putstring(out2,"unit3","m");
  vp = sf_floatalloc2(nz,nmx);
  vs = sf_floatalloc2(nz,nmx);
  trace = sf_floatalloc( nt > nz ? nt : nz  );
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,nz,velp);
    for (iz=0;iz<nz;iz++) vp[ix][iz] = trace[iz];
    sf_floatread(trace,nz,vels);
    for (iz=0;iz<nz;iz++) vs[ix][iz] = trace[iz];
  }
  d_x = sf_floatalloc2(nt,nmx);
  d_z = sf_floatalloc2(nt,nmx);
  d_p = sf_floatalloc2(nt,nmx);
  d_s = sf_floatalloc2(nt,nmx);
  if (H){
    for (isx=0;isx<nsx;isx++){
      for (ix=0;ix<nmx;ix++){
        sf_floatread(trace,n1,in1);
        for (it=0;it<nt;it++) d_x[ix][it] = trace[it];
        sf_floatread(trace,n1,in2);
        for (it=0;it<nt;it++) d_z[ix][it] = trace[it];
        for (it=0;it<nt;it++){
          d_p[ix][it] = 0.0;
          d_s[ix][it] = 0.0;
        }
      }
      wesep2dop(d_x,d_z,d_p,d_s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,vs,fmin,fmax,gz,H,ss,verbose);
      for (ix=0; ix<nmx; ix++) {
        for (it=0; it<nt; it++) trace[it] = d_p[ix][it];	
        sf_floatwrite(trace,nt,out1);
        for (it=0; it<nt; it++) trace[it] = d_s[ix][it];
        sf_floatwrite(trace,nt,out2);
      }
    }
  }
  else{
    for (isx=0;isx<nsx;isx++){
      for (ix=0;ix<nmx;ix++){
        sf_floatread(trace,n1,in1);
        for (it=0;it<nt;it++) d_p[ix][it] = trace[it];
        sf_floatread(trace,n1,in2);
        for (it=0;it<nt;it++) d_s[ix][it] = trace[it];
        for (it=0;it<nt;it++){
          d_x[ix][it] = 0.0;
          d_z[ix][it] = 0.0;
        }
      }
      wesep2dop(d_x,d_z,d_p,d_s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,vs,fmin,fmax,gz,H,ss,verbose);
      for (ix=0; ix<nmx; ix++) {
        for (it=0; it<nt; it++) trace[it] = d_x[ix][it];	
        sf_floatwrite(trace,nt,out1);
        for (it=0; it<nt; it++) trace[it] = d_z[ix][it];
        sf_floatwrite(trace,nt,out2);
      }
    }
  }

  exit (0);
}

void wesep2dop(float **dx_1shot,float **dz_1shot,float **dp_1shot,float **ds_1shot,
               int nt,float ot,float dt,int nmx,float omx,float dmx,int nz,float oz,float dz,
               float **vp,float **vs,float fmin,float fmax,float gz,
               bool H, bool ss,
               bool verbose)
{
  int iz,ix,ik,iw,it,nw,nk,padt,padx,ntfft;
  int igz;
  float dw,dk,w,kx,s1,s2;
  sf_complex czero,i;
  float kzp,kzs,denom;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w,*d_x,*d_z,*d_p,*d_s;
  sf_complex **dx_g_wx,**dz_g_wx;
  sf_complex **dp_g_wx,**ds_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po_p,**pd_p,*po_s,**pd_s;

  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;

  igz = (int) truncf(gz/dz);
  padt = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  if(fmin*dt*ntfft+1<ifmax) ifmin = trunc(fmin*dt*ntfft);
  else ifmin = 0;

  dx_g_wx = sf_complexalloc2(nw,nmx);
  dz_g_wx = sf_complexalloc2(nw,nmx);
  dp_g_wx = sf_complexalloc2(nw,nmx);
  ds_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);

  d_x = sf_complexalloc(nk);
  d_z = sf_complexalloc(nk);
  d_p = sf_complexalloc(nk);
  d_s = sf_complexalloc(nk);

  po_p = sf_floatalloc(nz); 
  pd_p = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po_p[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_p[iz] += 1.0/vp[ix][iz];
    po_p[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd_p[ix][iz] = 1.0/vp[ix][iz] - po_p[iz];
    }
  }
  po_s = sf_floatalloc(nz); 
  pd_s = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po_s[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_s[iz] += 1.0/vs[ix][iz];
    po_s[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd_s[ix][iz] = 1.0/vs[ix][iz] - po_s[iz];
    }
  }

  /* set up fftw plans */
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a);
  fftwf_execute_dft(p2,b,b);
  /**********************************************************************/

  if (H){
    for (ix=0;ix<nmx;ix++){
      // x component 
      for (it=0;it<nt;it++)        d_t[it] = dx_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     dx_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dx_g_wx[ix][iw] = d_w[iw]/sqrtf((float) ntfft);
      for (iw=ifmax;iw<nw;iw++)    dx_g_wx[ix][iw] = czero;
      // z component 
      for (it=0;it<nt;it++)        d_t[it] = dz_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     dz_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dz_g_wx[ix][iw] = d_w[iw]/sqrtf((float) ntfft);
      for (iw=ifmax;iw<nw;iw++)    dz_g_wx[ix][iw] = czero;
    }
    for (iw=0;iw<ifmax;iw++){
      w = iw*dw;
      for (ix=0;ix<nmx;ix++)   a[ix] = dx_g_wx[ix][iw];
      for (ix=nmx;ix<nk;ix++ ) a[ix] = czero;
      fftwf_execute_dft(p1,a,a);
      for (ik=0;ik<nk;ik++)    d_x[ik] = a[ik]/sqrtf(nk);
      for (ix=0;ix<nmx;ix++)   a[ix] = dz_g_wx[ix][iw];
      for (ix=nmx;ix<nk;ix++ ) a[ix] = czero;
      fftwf_execute_dft(p1,a,a);
      for (ik=0;ik<nk;ik++)    d_z[ik] = a[ik]/sqrtf(nk);
      for (ik=0;ik<nk;ik++){
        if (ik<nk/2) kx = dk*ik;
        else         kx = -(dk*nk - dk*ik);
        s1 = w*w*po_p[igz]*po_p[igz] - kx*kx;
        s2 = w*w*po_s[igz]*po_s[igz] - kx*kx;
        if (s1>0) kzp = sqrtf(s1);
        else kzp = 0;
        if (s2>0) kzs = sqrtf(s2);
        else kzs = 0;
        denom = kx*kx + kzp*kzs;
        d_p[ik] = i*kx*d_x[ik] + i*kzs*d_z[ik]; 
        d_s[ik] =-i*kzp*d_x[ik] + i*kx*d_z[ik];
      }
      for (ik=0;ik<nk;ik++)    b[ik] = d_p[ik];
      fftwf_execute_dft(p2,b,b);
      for (ix=0;ix<nmx;ix++)   dp_g_wx[ix][iw] = b[ix]/sqrtf(nk);
      for (ik=0;ik<nk;ik++)    b[ik] = d_s[ik];
      fftwf_execute_dft(p2,b,b);
      for (ix=0;ix<nmx;ix++)   ds_g_wx[ix][iw] = b[ix]/sqrtf(nk);
    }  
    for (ix=0;ix<nmx;ix++){
      // p component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++){ 
        w = iw*dw;
        if (ss) d_w[iw] = dp_g_wx[ix][iw] + i*w*pd_s[ix][igz]*dz_g_wx[ix][iw];
        else    d_w[iw] = dp_g_wx[ix][iw];
      }
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dp_1shot[ix][it] = d_t[it]/sqrtf((float) ntfft);
      // s component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++){ 
        w = iw*dw;
        if (ss) d_w[iw] = ds_g_wx[ix][iw] - i*w*pd_p[ix][igz]*dx_g_wx[ix][iw];
        else    d_w[iw] = ds_g_wx[ix][iw];
      }
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) ds_1shot[ix][it] = d_t[it]/sqrtf((float) ntfft);
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      // x component 
      for (it=0;it<nt;it++)        d_t[it] = dp_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     dp_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dp_g_wx[ix][iw] = d_w[iw]/sqrtf((float) ntfft);
      for (iw=ifmax;iw<nw;iw++)    dp_g_wx[ix][iw] = czero;
      // z component 
      for (it=0;it<nt;it++)        d_t[it] = ds_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     ds_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) ds_g_wx[ix][iw] = d_w[iw]/sqrtf((float) ntfft);
      for (iw=ifmax;iw<nw;iw++)    ds_g_wx[ix][iw] = czero;
    }
    for (iw=0;iw<ifmax;iw++){
      w = iw*dw;
      for (ix=0;ix<nmx;ix++)   a[ix] = dp_g_wx[ix][iw];
      for (ix=nmx;ix<nk;ix++ ) a[ix] = czero;
      fftwf_execute_dft(p1,a,a);
      for (ik=0;ik<nk;ik++)    d_p[ik] = a[ik]/sqrtf(nk);
      for (ix=0;ix<nmx;ix++)   a[ix] = ds_g_wx[ix][iw];
      for (ix=nmx;ix<nk;ix++ ) a[ix] = czero;
      fftwf_execute_dft(p1,a,a);
      for (ik=0;ik<nk;ik++)    d_s[ik] = a[ik]/sqrtf(nk);
      for (ik=0;ik<nk;ik++){
        if (ik<nk/2) kx = dk*ik;
        else         kx = -(dk*nk - dk*ik);
        s1 = w*w*po_p[igz]*po_p[igz] - kx*kx;
        s2 = w*w*po_s[igz]*po_s[igz] - kx*kx;
        if (s1>0) kzp = sqrtf(s1);
        else kzp = 0;
        if (s2>0) kzs = sqrtf(s2);
        else kzs = 0;
        denom = kx*kx + kzp*kzs;
        if (fabsf(denom)>=0.00001){
          d_x[ik] = (i*kx*d_p[ik] - i*kzs*d_s[ik])/denom; 
          d_z[ik] = (i*kzp*d_p[ik] + i*kx*d_s[ik])/denom;
        } 
        else{
          d_z[ik] = d_p[ik]; 
          d_x[ik] = d_s[ik]; 
        } 
      }
      for (ik=0;ik<nk;ik++)    b[ik] = d_x[ik];
      fftwf_execute_dft(p2,b,b);
      for (ix=0;ix<nmx;ix++)   dx_g_wx[ix][iw] = b[ix]/sqrtf(nk);
      for (ik=0;ik<nk;ik++)    b[ik] = d_z[ik];
      fftwf_execute_dft(p2,b,b);
      for (ix=0;ix<nmx;ix++)   dz_g_wx[ix][iw] = b[ix]/sqrtf(nk);
    }  
    for (ix=0;ix<nmx;ix++){
      // p component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dx_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dx_1shot[ix][it] = d_t[it]/sqrtf((float) ntfft);
      // s component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dz_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dz_1shot[ix][it] = d_t[it]/sqrtf((float) ntfft);
    }
  }

  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  free1float(d_t);
  free1complex(d_w);
  free2complex(dx_g_wx);
  free2complex(dz_g_wx);
  free2complex(dp_g_wx);
  free2complex(ds_g_wx);
  free2complex(d_s_wx);
  free1float(po_p);
  free2float(pd_p);
  free1float(po_s);
  free2float(pd_s);
  free1complex(d_x);
  free1complex(d_z);
  free1complex(d_p);
  free1complex(d_s);

  return;
}

float signf(float a)
{
 float b;
 if (a>0)      b = 1.0;
 else if (a<0) b =-1.0;
 else          b = 0.0;
 return b;
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
    free1float(in1a); fftwf_free(out1a);
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
    fftwf_free(in1b); free1float(out1b);
  }

  return;
}


