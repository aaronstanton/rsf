/* Convert a depth velocity section to an RMS velocity section
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

#include <rsf.h>
#include <fftw3.h>
#include "myfree.h"

void depth_to_rms(float **v_depth, float **v_rms,
                  int nz, float dz, float oz,
                  int nt, float dt, float ot,
                  int nx, float dx, float ox,
                  bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);

int main(int argc, char* argv[])
{
  sf_file in,out;
  int   nz,nx,nt;
  int   ix,it;
  float oz,ox,ot;
  float dz,dx,dt;
  float **v_depth,**v_rms;
  bool verbose;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&nz)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&dz)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&oz)) oz=0.;
  if (!sf_histint(  in,"n2",&nx)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&dx)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&ox)) ox=0.;

  if (!sf_getint("nt",&nt)) nt = nz; /* number of time steps of rms section */
  if (!sf_getfloat("dt",&dt)) dt = 0; /* time step of rms section */
  if (!sf_getfloat("ot",&ot)) ot = 0; /* time origin of rms section */

  sf_putfloat(out,"o1",ot);
  sf_putfloat(out,"d1",dt);
  sf_putfloat(out,"n1",nt);
  sf_putstring(out,"label1","Time");
  sf_putstring(out,"unit1","s");
  sf_putstring(out,"title","RMS velocity");

  fprintf(stderr,"nz=%d nx=%d nt=%d\n",nz,nx,nt);
  v_depth = sf_floatalloc2(nz,nx);
  v_rms   = sf_floatalloc2(nt,nx);

  sf_floatread(v_depth[0],nz*nx,in);
  for (ix=0;ix<nx;ix++) for (it=0;it<nt;it++) v_rms[ix][it] = 0.0;
  depth_to_rms(v_depth,v_rms,nz,dz,oz,nt,dt,ot,nx,dx,ox,verbose);
  sf_floatwrite(v_rms[0],nt*nx,out);
 
  free2float(v_depth);
  free2float(v_rms);
  exit (0);
}

void depth_to_rms(float **v_depth, float **v_rms,
                  int nz, float dz, float oz,
                  int nt, float dt, float ot,
                  int nx, float dx, float ox,
                  bool verbose)
{
  int ix,iz,it,j,k,iw,padt,ntfft,nw,*wd;
  float a,t,*trace;
  sf_complex *Trace,czero;

  padt = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  trace = sf_floatalloc(nt);
  Trace = sf_complexalloc(nw);
  __real__ czero = 0;
  __imag__ czero = 0;

  wd = sf_intalloc(nt);
  for (ix=0;ix<nx;ix++){
    for (it=0;it<nt;it++) wd[it] = 0;
    for (iz=0;iz<nz*5;iz++){
      a = 0.0; t = 0.0;
      for (j=0;j<iz;j++){
        if (iz<nz){
          a += v_depth[ix][iz]*v_depth[ix][iz]*2*(dz/v_depth[ix][iz]);
          t += 2*(dz/v_depth[ix][iz]);
        }
        else {
          a += v_depth[ix][nz-1]*v_depth[ix][nz-1]*2*(dz/v_depth[ix][nz-1]);
          t += 2*(dz/v_depth[ix][nz-1]);
        }
      }
      it = (int) truncf((t - ot)/dt);
      if (it>0 && it<nt){ 
        v_rms[ix][it] = sqrtf(a/t);
        wd[it] = 1;
        if (verbose && ix==0) fprintf(stderr,"v_rms[%d][%d]=%f\n",ix,it,v_rms[ix][it]);
      } 
    }
    /* interpolate the missing values on this trace */
    /* apply iteration of low pass filtering, then if wd[it]=1 reset to original value, and repeat. */
/*
    for (it=0;it<nt;it++) trace[it] = 0.0;
    for (k=0;k<50;k++){
      for (it=0;it<nt;it++) if (wd[it]>0) trace[it] = v_rms[ix][it];
      f_op(Trace,trace,nw,nt,1);
      for (iw=20;iw<nw;iw++) Trace[iw] = czero;
      f_op(Trace,trace,nw,nt,0);
      for (it=0;it<nt;it++) trace[it] = trace[it]/(float) ntfft;
    }
    for (it=0;it<nt;it++) v_rms[ix][it] = trace[it];
*/
  }
  free1int(wd);
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


