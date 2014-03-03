/* Common Shot Isotropic Wavefield Separation of P and SV waves.
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

void wesep2dop(float **d_z,float **d_x,float **d_p,float **d_sv,
               int nt,float ot,float dt,int nmx,float omx,float dmx,
               float **vp,float **vs,float fmin,float fmax,
               bool verbose);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);

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
  float **d_z,**d_x,**d_p,**d_sv,**vp,**vs,*trace;
  bool adj;
  bool verbose;
  float fmin,fmax;
  float sx;
  int isx;

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */

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
  if (!sf_getfloat("sx",&sx)) sx = dmx*((float) nmx)/2.0; /* index of position of source */
  isx = (int) truncf((sx - omx)/dmx); 
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
  vp = sf_floatalloc2(nt,nmx);
  vs = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc( nt > nz ? nt : nz  );
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,nz,velp);
    for (iz=0;iz<nz;iz++) vp[ix][iz] = trace[iz];
    sf_floatread(trace,nz,vels);
    for (iz=0;iz<nz;iz++) vs[ix][iz] = trace[iz];
  }
  d_z = sf_floatalloc2(nt,nmx);
  d_x = sf_floatalloc2(nt,nmx);
  d_p = sf_floatalloc2(nt,nmx);
  d_sv = sf_floatalloc2(nt,nmx);
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,in1);
    for (it=0;it<nt;it++) d_z[ix][it] = trace[it];
    sf_floatread(trace,n1,in2);
    for (it=0;it<nt;it++) d_x[ix][it] = trace[it];
  }
  for (ix=0;ix<nmx;ix++){
    for (it=0;it<nt;it++){
      d_p[ix][it] = 0.0;
      d_sv[ix][it] = 0.0;
    }
  }

  if (!adj){
    fprintf(stderr,"forward operator hasn't been added yet.\n");
    exit (0);
  }

  wesep2dop(d_z,d_x,d_p,d_sv,nt,ot,dt,nmx,omx,dmx,vp,vs,fmin,fmax,verbose);

  for (ix=0; ix<nmx; ix++) {
    for (it=0; it<nt; it++) trace[it] = d_p[ix][it];	
    sf_floatwrite(trace,nt,out1);
    if (ix<isx){
      for (it=0; it<nt; it++) trace[it] = d_sv[ix][it];
    }
    else{
      for (it=0; it<nt; it++) trace[it] = -d_sv[ix][it];
    }	
    sf_floatwrite(trace,nt,out2);
  }

  exit (0);
}

void wesep2dop(float **d_z,float **d_x,float **d_p,float **d_sv,
               int nt,float ot,float dt,int nmx,float omx,float dmx,
               float **vp,float **vs,float fmin,float fmax,
               bool verbose)
{

  sf_complex **D_z,**D_x,**D_p,**D_sv;
  int iw,ik,nw,nk,padt,padx,ntfft;
  float w,s1,s2,k,vel,dw,dk;
  sf_complex czero,i;
  int ifmax;
  int ix,it;
  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;

  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  D_z = sf_complexalloc2(nw,nk);
  D_x = sf_complexalloc2(nw,nk);
  D_p = sf_complexalloc2(nw,nk);
  D_sv = sf_complexalloc2(nw,nk);
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;

  fk_op(D_z,d_z,nw,nk,nt,nmx,1);
  fk_op(D_x,d_x,nw,nk,nt,nmx,1);
  for (ik=0;ik<nk;ik++){
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    for (iw=0;iw<ifmax;iw++){ 
      w = dw*iw;
      s1 = (w*w)/(vs[0][0]*vs[0][0]) - (k*k);
      s2 = (w*w)/(vp[0][0]*vp[0][0]) - (k*k);
      if (s1 > 0 && s2 > 0){ 
        D_p[ik][iw]  = i*k*D_x[ik][iw] - i*sqrtf(s1)*D_z[ik][iw];
        D_sv[ik][iw] = i*k*D_z[ik][iw] + i*sqrtf(s2)*D_x[ik][iw];
      }
      else{
        D_p[ik][iw]  = czero;
        D_sv[ik][iw] = czero;
      }
    }  
  }
  fk_op(D_p,d_p,nw,nk,nt,nmx,0);
  fk_op(D_sv,d_sv,nw,nk,nt,nmx,0);

  free2complex(D_z);
  free2complex(D_x);
  free2complex(D_p);
  free2complex(D_sv);
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

