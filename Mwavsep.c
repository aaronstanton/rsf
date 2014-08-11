/* Common Shot Isotropic Wavefield Separation of P and SV waves using Helmholtz or Christoffel formulations.
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

void wesep2dop(float **d_x,float **d_z,float **d_p,float **d_sv,
               int nt,float ot,float dt,int nmx,float omx,float dmx,
               float **vp,float **vs,float fmin,float fmax,
               bool adj, bool inv,
               int mode, 
               bool verbose);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
float signf(float a);

int main(int argc, char* argv[])
{

  sf_file in1,in2,out1,out2,velp,vels;
  int n1,n2;
  int nt,nmx,nz;
  int it,ix,iz;
  float o1,o2;
  float d1,d2;
  float ot,omx;
  float dt,dmx;
  float **d_x,**d_z,**d_p,**d_sv,**vp,**vs,*trace;
  bool adj;
  bool inv;
  bool verbose;
  float fmin,fmax;
  int mode;

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getbool("inv",&inv)) inv = false; /* flag for inverse */
  if (inv) adj = false;
  if (!sf_getint("mode",&mode)) mode = 1; /* 1=Helmholtz based method, 2=Christoffel based method */
  /* read input file parameters */
  if (!sf_histint(  in1,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in1,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in1,"o1",&o1)) o1=0.;
  if (!sf_histint(  in1,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in1,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in1,"o2",&o2)) o2=0.;

  if (!sf_histint(  velp,"n1",&nz)) sf_error("No n1= in vp");
 
  nmx=n2;
  dmx=d2;  
  omx=o2;
  nt=n1;  
  dt=d1;  
  ot=o1;

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
  d_sv = sf_floatalloc2(nt,nmx);
  if (adj || inv){
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,in1);
    for (it=0;it<nt;it++) d_x[ix][it] = trace[it];
    sf_floatread(trace,n1,in2);
    for (it=0;it<nt;it++) d_z[ix][it] = trace[it];
  }
  for (ix=0;ix<nmx;ix++){
    for (it=0;it<nt;it++){
      d_p[ix][it] = 0.0;
      d_sv[ix][it] = 0.0;
    }
  }
  }
  else{
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,in1);
    for (it=0;it<nt;it++) d_p[ix][it] = trace[it];
    sf_floatread(trace,n1,in2);
    for (it=0;it<nt;it++) d_sv[ix][it] = trace[it];
  }
  for (ix=0;ix<nmx;ix++){
    for (it=0;it<nt;it++){
      d_x[ix][it] = 0.0;
      d_z[ix][it] = 0.0;
    }
  }
  }
  wesep2dop(d_x,d_z,d_p,d_sv,nt,ot,dt,nmx,omx,dmx,vp,vs,fmin,fmax,adj,inv,mode,verbose);
  if (adj || inv){
  for (ix=0; ix<nmx; ix++) {
    for (it=0; it<nt; it++) trace[it] = d_p[ix][it];	
    sf_floatwrite(trace,nt,out1);
    for (it=0; it<nt; it++) trace[it] = d_sv[ix][it];
    sf_floatwrite(trace,nt,out2);
  }
  }
  else{
  for (ix=0; ix<nmx; ix++) {
    for (it=0; it<nt; it++) trace[it] = d_x[ix][it];	
    sf_floatwrite(trace,nt,out1);
    for (it=0; it<nt; it++) trace[it] = d_z[ix][it];
    sf_floatwrite(trace,nt,out2);
  }
  }

  exit (0);
}

void wesep2dop(float **d_x,float **d_z,float **d_p,float **d_sv,
               int nt,float ot,float dt,int nmx,float omx,float dmx,
               float **vp,float **vs,float fmin,float fmax,
               bool adj,bool inv, 
               int mode,
               bool verbose)
{

  sf_complex **D_z,**D_x,**D_p,**D_sv;
  int iw,ik,nw,nk,padt,padx,ntfft;
  float w,sp,ss,kx,kzp,kzs,dw,dk,k;
  sf_complex czero,i,i_neg;
  int ifmax;
  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  __real__ i_neg =  0;
  __imag__ i_neg = -1;

  padt = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nmx;

  D_x = sf_complexalloc2(nw,nk);
  D_z = sf_complexalloc2(nw,nk);
  D_p = sf_complexalloc2(nw,nk);
  D_sv = sf_complexalloc2(nw,nk);

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  if (adj || inv){
    fk_op(D_x,d_x,nw,nk,nt,nmx,1);
    fk_op(D_z,d_z,nw,nk,nt,nmx,1);
  }
  else{
    fk_op(D_p,d_p,nw,nk,nt,nmx,1);
    fk_op(D_sv,d_sv,nw,nk,nt,nmx,1);
  }
  for (ik=0;ik<nk;ik++){
    if (ik<nk/2) kx = dk*ik;
    else         kx = -(dk*nk - dk*ik);
    for (iw=0;iw<ifmax;iw++){ 
      w = dw*iw;
      sp = (w*w)/(vp[0][0]*vp[0][0]) - (kx*kx);
      ss = (w*w)/(vs[0][0]*vs[0][0]) - (kx*kx);
      if (sp>0) kzp = sqrtf(sp);
      else      kzp = 0;  
      if (ss>0) kzs = sqrtf(ss);
      else      kzs = 0;
      if (mode==1){
        if (adj){
          D_p[ik][iw]  =             kx*D_x[ik][iw] +          kzs*D_z[ik][iw];
          D_sv[ik][iw] = -signf(kx)*kzp*D_x[ik][iw] + signf(kx)*kx*D_z[ik][iw];
        }
        else{
          D_x[ik][iw]  =  kx*D_p[ik][iw] - signf(kx)*kzp*D_sv[ik][iw];
          D_z[ik][iw]  = kzs*D_p[ik][iw] +  signf(kx)*kx*D_sv[ik][iw];
        }
      }
      else if (mode==2){
        if (adj){
          if (w>0){
            D_p[ik][iw]  = ( vp[0][0]*kx*D_x[ik][iw]  + vp[0][0]*kzp*D_z[ik][iw])/w;
            D_sv[ik][iw] = (-signf(kx)*vs[0][0]*kzs*D_x[ik][iw] + signf(kx)*vs[0][0]*kx*D_z[ik][iw])/w;
          }
          else{
            D_p[ik][iw]  = czero;
            D_sv[ik][iw] = czero;
          }
        }
        else{
          if (w>0){
            D_x[ik][iw] = (vp[0][0]*kx*D_p[ik][iw]  - signf(kx)*vs[0][0]*kzs*D_sv[ik][iw])/w;
            D_z[ik][iw] = (vp[0][0]*kzp*D_p[ik][iw] + signf(kx)*vs[0][0]*kx*D_sv[ik][iw])/w;
          }
          else{
            D_x[ik][iw] = czero;
            D_z[ik][iw] = czero;
          }
        }
      }   
    }  
  }
  if (adj || inv){
    fk_op(D_p,d_p,nw,nk,nt,nmx,0);
    fk_op(D_sv,d_sv,nw,nk,nt,nmx,0);
  }
  else {
    fk_op(D_x,d_x,nw,nk,nt,nmx,0);
    fk_op(D_z,d_z,nw,nk,nt,nmx,0);
  }
  free2complex(D_x);
  free2complex(D_z);
  free2complex(D_p);
  free2complex(D_sv);
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
    for(iw=0;iw<nw;iw++) cpfft[ix][iw] = out1a[iw]/sqrtf(ntfft); 
  }
  fftwf_destroy_plan(p1a);
  fftwf_free(in1a); fftwf_free(out1a);
  for (iw=0;iw<nw;iw++){  
    for (ik=0;ik<nk;ik++) in2a[ik] = cpfft[ik][iw];
    fftwf_execute(p2a); /* FFT x to k */
    for (ik=0;ik<nk;ik++) m[ik][iw] = in2a[ik]/sqrtf(nk);
  }
  fftwf_destroy_plan(p2a);
  fftwf_free(in2a); 
}

else{ /* model --> data */
  for (iw=0;iw<nw;iw++){  
    for (ik=0;ik<nk;ik++) in2b[ik] = m[ik][iw];
    fftwf_execute(p2b); /* FFT k to x */
    for (ik=0;ik<nx;ik++) cpfft[ik][iw] = in2b[ik]/sqrtf(nk);
  }
  fftwf_destroy_plan(p2b);
  fftwf_free(in2b);
  for (ix=0;ix<nx;ix++){
    for(iw=0;iw<nw;iw++) in1b[iw] = cpfft[ix][iw];
    for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
    fftwf_execute(p1b); 
    for(it=0;it<nt;it++) d[ix][it] = out1b[it]/sqrtf(ntfft); 
  }
  fftwf_destroy_plan(p1b);
  fftwf_free(in1b); fftwf_free(out1b);
}
  free2complex(cpfft);

  return;

}

