/* Convert a gather from depth/offset to depth/angle
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

void convert_to_angle(float **d_zh, float **d_za,
                      int nz, float oz, float dz, 
                      int nmx, float omx, float dmx, 
                      int nhx, float ohx, float dhx, 
                      int np, float op, float dp, 
                      bool adj, bool verbose);
void offset_to_angle(float **d_zh, float **d_za,
                     int nz, float oz, float dz, 
                     int nhx, float ohx, float dhx, 
                     int np, float op, float dp, 
                     bool adj, bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);

int main(int argc, char* argv[])
{
  sf_file in,out;
  int   nz,nmx,nhx,np;
  int   iz,ix;
  float oz,omx,ohx,op;
  float dz,dmx,dhx,dp;
  float **d_zh,**d_za;
  bool adj;
  bool verbose;
  float fmin,fmax;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (adj){
    if (!sf_getint("np",&np)) np=91; /* number of angle samples */
    if (!sf_getfloat("op",&op)) op = 0; /* angle origin in degrees */
    if (!sf_getfloat("dp",&dp)) dp = 1; /* angle increment in degrees */
  }
  else{
    if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
    if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
    if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");
  }

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&nz)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&dz)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&oz)) oz=0.;
  if (!sf_histint(  in,"n2",&nmx)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&dmx)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&omx)) omx=0.;

  if (adj){
    if (!sf_histint(  in,"n3",&nhx)) sf_error("No n3= in input");
    if (!sf_histfloat(in,"d3",&dhx)) sf_error("No d3= in input");
    if (!sf_histfloat(in,"o3",&ohx)) ohx=0.;
  }
  else{
    if (!sf_histint(  in,"n3",&np)) sf_error("No n3= in input");
    if (!sf_histfloat(in,"d3",&dp)) sf_error("No d3= in input");
    if (!sf_histfloat(in,"o3",&op)) op=0.;
  }
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dz; /* max frequency to process */
  if (fmax > 0.5/dz) fmax = 0.5/dz;

  sf_putfloat(out,"o1",oz);
  sf_putfloat(out,"d1",dz);
  sf_putfloat(out,"n1",nz);
  sf_putstring(out,"label1","Depth");
  sf_putstring(out,"unit1","m");
  sf_putfloat(out,"o2",omx);
  sf_putfloat(out,"d2",dmx);
  sf_putfloat(out,"n2",nmx);
  sf_putstring(out,"label2","x");
  sf_putstring(out,"unit2","m");

  if (adj){
    sf_putfloat(out,"o3",op);
    sf_putfloat(out,"d3",dp);
    sf_putfloat(out,"n3",np);
    sf_putstring(out,"label3","Angle");
    sf_putstring(out,"unit3","Degrees");
  }
  else {
    sf_putfloat(out,"o3",ohx);
    sf_putfloat(out,"d3",dhx);
    sf_putfloat(out,"n3",nhx);
    sf_putstring(out,"label3","Offset");
    sf_putstring(out,"unit3","m");
  }

  if (verbose) fprintf(stderr,"nz=%d nmx=%d nhx=%d np=%d \n",nz,nmx,nhx,np);
  d_zh = sf_floatalloc2(nz,nmx*nhx);
  d_za = sf_floatalloc2(nz,nmx*np);

  if (adj){
    sf_floatread(d_zh[0],nz*nmx*nhx,in);
    for (ix=0;ix<nmx*np;ix++) for (iz=0;iz<nz;iz++) d_za[ix][iz] = 0.0;
  }
  else{
    sf_floatread(d_za[0],nz*nmx*np,in);
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) d_zh[ix][iz] = 0.0;
  }

  convert_to_angle(d_zh,d_za,
                   nz,oz,dz,
                   nmx,omx,dmx,
                   nhx,ohx,dhx, 
                   np,op,dp, 
                   adj,verbose);

  if (adj){
    sf_floatwrite(d_za[0],nz*nmx*np,out);
  }
  else{
    sf_floatwrite(d_zh[0],nz*nmx*nhx,out);
  }
 
  free2float(d_zh);
  free2float(d_za);
  exit (0);
}

void convert_to_angle(float **d_zh, float **d_za,
                      int nz, float oz, float dz, 
                      int nmx, float omx, float dmx, 
                      int nhx, float ohx, float dhx, 
                      int np, float op, float dp, 
                      bool adj, bool verbose)
{
  float **a,**b;
  int imx,ihx,ip,iz;
  a = sf_floatalloc2(nz,nhx); 
  b = sf_floatalloc2(nz,np); 
  for (imx=0;imx<nmx;imx++){
    for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) a[ihx][iz] = d_zh[ihx*nmx + imx][iz];
    offset_to_angle(a,b,nz,oz,dz,nhx,ohx,dhx,np,op,dp,adj,verbose);
    for (ip=0;ip<np;ip++) for (iz=0;iz<nz;iz++) d_za[ip*nmx + imx][iz] = b[ip][iz];
  }
  free2float(a);
  free2float(b);
  return;
}

void offset_to_angle(float **d_zh, float **d_za,
                     int nz, float oz, float dz, 
                     int nhx, float ohx, float dhx, 
                     int np, float op, float dp, 
                     bool adj, bool verbose)
{

  int iz,ihx,ip,ik,iw,nw,nk,padz,padx,nzfft;
  float dw,dk,p;
  sf_complex czero;
  float *d_z;
  sf_complex *d_w;
  sf_complex **d_wh;
  sf_complex **d_wa;
  fftwf_complex *a;
  int *n;
  fftwf_plan p1;
  float w,k;

  __real__ czero = 0;
  __imag__ czero = 0;
  padz = 2;
  padx = 2;
  nzfft = padz*nz;
  nw=nzfft/2+1;
  nk = padx*nhx;
  dk = 2*PI/((float) nk)/dhx;
  dw = 2*PI/((float) nzfft)/dz;
  d_wh = sf_complexalloc2(nw,nhx);
  d_wa = sf_complexalloc2(nw,np);
  d_z = sf_floatalloc(nz);
  d_w = sf_complexalloc(nw);
  for (iz=0;iz<nz;iz++)  d_z[iz] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  

  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
  } 

  for (iw=0;iw<nw;iw++) for (ip=0;ip<np;ip++) d_wa[ip][iw] = czero;
  /* transform the depth axis to the frequency domain */
  for (ihx=0;ihx<nhx;ihx++){
    for (iz=0;iz<nz;iz++) d_z[iz] = d_zh[ihx][iz];
    f_op(d_w,d_z,nw,nz,1); /* d_z to d_w */
    for (iw=0;iw<nw;iw++) d_wh[ihx][iw] = d_w[iw];
  }

  /* transform the offset axis to the frequency domain */
  for (iw=0;iw<nw;iw++){
    w = iw*dw;
    for (ihx=0;ihx<nhx;ihx++) a[ihx] = d_wh[ihx][iw];
    for (ihx=nhx;ihx<nk;ihx++) a[ihx] = czero;
    fftwf_execute_dft(p1,a,a); 
    /* compute Ray Parameters */
    for (ik=0;ik<nk;ik++){
      if (ik<nk/2){ 
        k = dk*ik;
      }
      else{ 
        k = -(dk*nk - dk*ik);
      }
      if (w>0){ 
        p = (180/PI)*atanf(fabs(k/w));
        ip = (int) truncf((p - op)/dp);
        if (verbose) if (iw==50) fprintf(stderr,"p=%f ip=%d\n",p,ip);
        if (ip < np && p >= op){
          d_wa[ip][iw] += a[ik]/sqrtf((float) nk);
        }
      }
    }
  }      
  /* transform the frequency axis to the depth domain */
  for (ip=0;ip<np;ip++){
    for (iw=0;iw<nw;iw++) d_w[iw] = d_wa[ip][iw];
    f_op(d_w,d_z,nw,nz,0); /* d_w to d_z */
    for (iz=0;iz<nz;iz++) d_za[ip][iz] = d_z[iz];
  }

  free2complex(d_wh);
  free2complex(d_wa);
  free1float(d_z);
  free1complex(d_w);
  fftwf_destroy_plan(p1);
  fftwf_free(a);

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

