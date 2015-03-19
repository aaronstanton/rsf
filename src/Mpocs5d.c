/* Reconstruction of 5d seismic data by Projection Onto Convex Sets (POCS).
*/
/*
  Copyright (C) 2015 University of Alberta
  
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
#ifndef PI
#define PI (3.141592653589793)
#endif
#include "myfree.h"

void pocs(float **d,int nt,int nx,float dt,int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int niter,float alpha,float fmax,float p, 
          int aa2, int aa3, int aa4, int aa5,
          bool debias,bool verbose);
int compare (const void * a, const void * b);
void smooth4d(float *d, float *wd,
              int nx1,int smooth1,
              int nx2,int smooth2,
              int nx3,int smooth3,
              int nx4,int smooth4,
              int nrepeat,bool verbose);
void mean_filter(float *trace, int nt, int ntw, int nrepeat);
void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           int axis,int L);
void radial_filter(float **d, int nt, int nx, int nL);
float signf(float a);
int main(int argc, char* argv[])
{ 
    int ix,nx;
    int n1,n2,n3,n4,n5;
    int i1,i2;
    float *wd,*trace,**d;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float sum;
    sf_file in,out;
    sf_init (argc,argv);
    int niter;
    float alpha, fmax;
    int sum_wd;
    bool verbose;
    float p;
    bool debias;
    int aa2,aa3,aa4,aa5;

    in = sf_input("in");
    out = sf_output("out");

    /* read input file parameters */
    if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&o1)) o1=0.;
    if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&d2)) d2=1;
    if (!sf_histfloat(in,"o2",&o2)) o2=0.;
    if (!sf_histint(in,"n3",&n3))   n3=1;
    if (!sf_histfloat(in,"d3",&d3)) d3=1;
    if (!sf_histfloat(in,"o3",&o3)) o3=0.;
    if (!sf_histint(in,"n4",&n4))   n4=1;
    if (!sf_histfloat(in,"d4",&d4)) d4=1;
    if (!sf_histfloat(in,"o4",&o4)) o4=0.;
    if (!sf_histint(in,"n5",&n5))   n5=1;
    if (!sf_histfloat(in,"d5",&d5)) d5=1;
    if (!sf_histfloat(in,"o5",&o5)) o5=0.;

    if (!sf_getint("niter",&niter)) niter = 100; /* number of iterations */
    if (!sf_getfloat("alpha",&alpha)) alpha = 1; /* denoising parameter (1=no denoise) */
    if (!sf_getbool("verbose",&verbose)) verbose = false; /* flag for verbosity */
    if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/d1; /* max frequency to process */
    if (fmax > 0.5/d1) fmax = 0.5/d1;
    if (!sf_getfloat("p",&p)) p = 8.0; /* Exponent for thresholding, 1=>soft 2=>stein large=>hard thresholding */
    if (!sf_getbool("debias",&debias)) debias = false; /* flag for amplitude compensation of interpolated traces */
    if (!sf_getint("aa2",&aa2)) aa2 = 0; /* radial filter length in FK for anti-aliasing; dimension 2 */
    if (!sf_getint("aa3",&aa3)) aa3 = 0; /* radial filter length in FK for anti-aliasing; dimension 3 */
    if (!sf_getint("aa4",&aa4)) aa4 = 0; /* radial filter length in FK for anti-aliasing; dimension 4  */
    if (!sf_getint("aa5",&aa5)) aa5 = 0; /* radial filter length in FK for anti-aliasing; dimension 5  */
    sf_putfloat(out,"o1",o1);
    sf_putfloat(out,"o2",o2);
    sf_putfloat(out,"o3",o3);
    sf_putfloat(out,"o4",o4);
    sf_putfloat(out,"o5",o5);
    sf_putfloat(out,"d1",d1);
    sf_putfloat(out,"d2",d2);
    sf_putfloat(out,"d3",d3);
    sf_putfloat(out,"d4",d4);
    sf_putfloat(out,"d5",d5);
    sf_putfloat(out,"n1",n1);
    sf_putfloat(out,"n2",n2);
    sf_putfloat(out,"n3",n3);
    sf_putfloat(out,"n4",n4);
    sf_putfloat(out,"n5",n5);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","ix1");
    sf_putstring(out,"label3","ix2");
    sf_putstring(out,"label4","ix3");
    sf_putstring(out,"label5","ix4");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","index");
    sf_putstring(out,"unit3","index");
    sf_putstring(out,"unit4","index");
    sf_putstring(out,"unit5","index"); 

    nx = n2*n3*n4*n5;

    trace = sf_floatalloc (n1);
    d = sf_floatalloc2 (n1,nx);
    wd    = sf_floatalloc (nx);

    for (i2=0; i2<nx; i2++){
      for (i1=0; i1<n1; i1++) d[i2][i1] = 0; 
      wd[i2] = 0;
    }

    sum_wd = 0;
    for (ix=0; ix<nx; ix++) {
      sf_floatread(trace,n1,in);
      sum = 0;
      for (i1=0; i1<n1; i1++){
        sum      += trace[i1]*trace[i1]; 
        d[ix][i1] = trace[i1];
      }
      if (sum){ 
        wd[ix] = 1;
        sum_wd++;
      }      
    }
 
    if (verbose) fprintf(stderr,"the block has %6.2f %% missing traces.\n", (float) 100 - 100*sum_wd/(n2*n3*n4*n5));
    
    if ((float) sum_wd/(n2*n3*n4*n5) > 0.05){
      pocs(d,n1,nx,d1,n2,n3,n4,n5,wd,niter,alpha,fmax,p,aa2,aa3,aa4,aa5,debias,verbose);
    }
        
    for (ix=0; ix<nx; ix++) {
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }

    exit (0);
}


void pocs(float **d,int nt,int nx,float dt,int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int niter,float alpha,float fmax,float p, 
          int aa2, int aa3, int aa4, int aa5,
          bool debias,bool verbose)
{  
  int it,ix,iw,ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk,if_low,if_high,padfactor,ix_no_pad,ix1,ix2,ix3,ix4,iter,*n,nzero;
  float perc,perci,percf,*wd,thres,*in1,*out2,f_low,f_high,*amp,*b1,*b2,median,bias,**M,**Mshift;
  sf_complex czero,*freqslice,*out1,*in2,**D,**Dobs;
  fftwf_plan p1,p2,p3,p4; 
  int ix_shift,ix1_shift,ix2_shift,ix3_shift,ix4_shift;
  float phi;
  
  __real__ czero = 0;
  __imag__ czero = 0;
  perci = 1.0;
  percf = 0.0;
  padfactor = 2;
  /* copy data from input to FFT array and pad with zeros */
  ntfft = padfactor*nt;
  nx1fft = padfactor*nx1;
  nx2fft = padfactor*nx2;
  nx3fft = padfactor*nx3;
  nx4fft = padfactor*nx4;
  if(nx1==1) nx1fft = 1;
  if(nx2==1) nx2fft = 1;
  if(nx3==1) nx3fft = 1;
  if(nx4==1) nx4fft = 1;
  nw=ntfft/2+1;
  nk=nx1fft*nx2fft*nx3fft*nx4fft;

  wd = sf_floatalloc(nk);
  D = sf_complexalloc2(nw,nk);
  Dobs = sf_complexalloc2(nw,nk);
  M = sf_floatalloc2(nw,nk);
  Mshift = sf_floatalloc2(nw,nk);
  
  for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) D[ix][iw] = czero;
  for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) Dobs[ix][iw] = czero;
  for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) M[ix][iw] = 0.0;
  for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) Mshift[ix][iw] = 0.0;
  
  in1 = sf_floatalloc(ntfft);
  out1 = sf_complexalloc(nw);
  p1 = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
    if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
      ix_no_pad = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4;
      ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
      for (it=0; it<nt; it++) in1[it] = d[ix_no_pad][it];
      for (it=nt;it<ntfft;it++) in1[it] = 0.0;       
      fftwf_execute(p1);
      for(iw=0;iw<nw;iw++) D[ix][iw] = Dobs[ix][iw] = out1[iw]; 
    }
  }}}}

  for (ix=0;ix<nk;ix++) wd[ix] = 0.0;
  for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
    if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
      ix_no_pad = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4;
      ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
      wd[ix] = wd_no_pad[ix_no_pad];
    }
  }}}}
	
  f_low = 0.1; 
  f_high = fmax;

  if(f_low>0) if_low = trunc(f_low*dt*ntfft);
  else if_low = 0;
  if(f_high*dt*ntfft<nw) if_high = trunc(f_high*dt*ntfft);
  else if_high = 0;

  freqslice = sf_complexalloc(nk);
  n = sf_intalloc(4); n[0] = nx1fft; n[1] = nx2fft; n[2] = nx3fft; n[3] = nx4fft;
  p2 = fftwf_plan_dft(4, n, (fftwf_complex*)freqslice, (fftwf_complex*)freqslice, FFTW_FORWARD, FFTW_ESTIMATE);  
  p3 = fftwf_plan_dft(4, n, (fftwf_complex*)freqslice, (fftwf_complex*)freqslice, FFTW_BACKWARD, FFTW_ESTIMATE);

  amp = sf_floatalloc(nk*nw);

  for (iter=0;iter<niter;iter++){
    for (iw=if_low;iw<if_high;iw++){
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
          freqslice[ix] = D[ix][iw];
        }
        else{
          freqslice[ix] = czero;
        }
      }}}}
      fftwf_execute(p2);
      for (ix=0;ix<nk;ix++) D[ix][iw] = freqslice[ix];
    }
      
    for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) M[ix][iw] = cabsf(D[ix][iw]);        
    // radial smoothing to dealias the spectrum
    if (aa2>0 || aa3>0 || aa4>0 || aa5>0){
    // map M to Mshift
    for (iw=if_low;iw<if_high;iw++){
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        if (ix1 < (int) truncf(nx1fft/2)) ix1_shift = ix1 + (int) truncf(nx1fft/2);
        else ix1_shift = ix1 - ((int) truncf(nx1fft/2));
        if (ix2 < (int) truncf(nx2fft/2)) ix2_shift = ix2 + (int) truncf(nx2fft/2);
        else ix2_shift = ix2 - ((int) truncf(nx2fft/2));
        if (ix3 < (int) truncf(nx3fft/2)) ix3_shift = ix3 + (int) truncf(nx3fft/2);
        else ix3_shift = ix3 - ((int) truncf(nx3fft/2));
        if (ix4 < (int) truncf(nx4fft/2)) ix4_shift = ix4 + (int) truncf(nx4fft/2);
        else ix4_shift = ix4 - ((int) truncf(nx4fft/2));
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        ix_shift = ix1_shift*nx2fft*nx3fft*nx4fft + ix2_shift*nx3fft*nx4fft + ix3_shift*nx4fft + ix4_shift;
        Mshift[ix_shift][iw] = M[ix][iw];
      }}}}
    }
    
    if (aa2>0){
      radial_filter_gathers(Mshift,0,1,nw,0,1,nx1fft,0,1,nx2fft,0,1,nx3fft,0,1,nx4fft,2,aa2);
    }
    if (aa3>0){
      radial_filter_gathers(Mshift,0,1,nw,0,1,nx1fft,0,1,nx2fft,0,1,nx3fft,0,1,nx4fft,3,aa3);
    }
    if (aa4>0){
      radial_filter_gathers(Mshift,0,1,nw,0,1,nx1fft,0,1,nx2fft,0,1,nx3fft,0,1,nx4fft,4,aa4);
    }
    if (aa5>0){
      radial_filter_gathers(Mshift,0,1,nw,0,1,nx1fft,0,1,nx2fft,0,1,nx3fft,0,1,nx4fft,5,aa5);
    }
    
    // map Mshift to M
    for (iw=if_low;iw<if_high;iw++){
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        if (ix1 < (int) truncf(nx1fft/2)) ix1_shift = ix1 + (int) truncf(nx1fft/2);
        else ix1_shift = ix1 - ((int) truncf(nx1fft/2));
        if (ix2 < (int) truncf(nx2fft/2)) ix2_shift = ix2 + (int) truncf(nx2fft/2);
        else ix2_shift = ix2 - ((int) truncf(nx2fft/2));
        if (ix3 < (int) truncf(nx3fft/2)) ix3_shift = ix3 + (int) truncf(nx3fft/2);
        else ix3_shift = ix3 - ((int) truncf(nx3fft/2));
        if (ix4 < (int) truncf(nx4fft/2)) ix4_shift = ix4 + (int) truncf(nx4fft/2);
        else ix4_shift = ix4 - ((int) truncf(nx4fft/2));
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        ix_shift = ix1_shift*nx2fft*nx3fft*nx4fft + ix2_shift*nx3fft*nx4fft + ix3_shift*nx4fft + ix4_shift;
        M[ix][iw] = Mshift[ix_shift][iw];
      }}}}
    }
    }
    
    // sort non-zero amplitudes
    nzero = 0;
    for (iw=0;iw<nw;iw++){  
      for (ix=0;ix<nk;ix++){ 
        amp[ix*nw + iw] = M[ix][iw];
        if (amp[ix*nw + iw] > 0.000001) nzero++;
      }
    }
    qsort (amp,nk*nw, sizeof(*amp), compare);
    perc = perci + iter*((percf-perci)/(niter-1));
    thres = amp[(int) truncf(nk*nw - 1 - (1-perc)*nzero)];
    for (iw=if_low;iw<if_high;iw++){
      for (ix=0;ix<nk;ix++){
        if (M[ix][iw]<thres || cabsf(D[ix][iw])<thres) D[ix][iw] = czero;
        else D[ix][iw] = D[ix][iw]*(1 - powf(thres/(cabsf(D[ix][iw]) + 0.0000001),p));
      }
    }
        
    for (iw=if_low;iw<if_high;iw++){
      for (ix=0;ix<nk;ix++) freqslice[ix] = D[ix][iw];
      fftwf_execute(p3);
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
          D[ix][iw] = freqslice[ix]/nk;
        }
        else{
          D[ix][iw] = czero;
        }
      }}}}
    }
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) D[ix][iw] = alpha*Dobs[ix][iw] + (1-alpha*wd[ix])*D[ix][iw];
  }
  
  free1complex(freqslice);
  in2 = sf_complexalloc(ntfft);
  out2 = sf_floatalloc(ntfft);
  p4 = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
    if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
      ix_no_pad = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4;
      ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
      for (iw=0; iw<ntfft; iw++) in2[iw] = D[ix][iw];
      fftwf_execute(p4);
      for(it=0;it<nt;it++) d[ix_no_pad][it] = out2[it]/ntfft; 
    }
  }}}}

  if (debias){
    // scale RMS amplitude to median value of observed traces
    b1 = sf_floatalloc(nx);
    b2 = sf_floatalloc(nx);
    nzero = 0;
    for (ix=0;ix<nx;ix++){
      b1[ix] = b2[ix] = 0.0;
      if (wd_no_pad[ix]){
        for(it=0;it<nt;it++) b1[ix] += d[ix][it]*d[ix][it];
        nzero++;
      }
      for(it=0;it<nt;it++) b2[ix] += d[ix][it]*d[ix][it];
      b1[ix] = sqrtf(b1[ix]);
      b2[ix] = sqrtf(b2[ix]);
    }
    qsort (b1,nx, sizeof(*b1), compare);
    median = b1[(int) truncf(nx - 1 - 0.5*nzero)];
    for (ix=0;ix<nx;ix++){
      if (wd_no_pad[ix]==0 && b2[ix] > 1e-8) bias = median/b2[ix];
      else bias = 1.0;
      if (bias < 2.0) for(it=0;it<nt;it++) d[ix][it] *= bias;
      else            for(it=0;it<nt;it++) d[ix][it] *= 2.0;
    }
    // smooth RMS amplitude laterally
    for (ix=0;ix<nx;ix++){
      b1[ix] = b2[ix] = 0.0;
      for(it=0;it<nt;it++) b1[ix] += d[ix][it]*d[ix][it];
      for(it=0;it<nt;it++) b2[ix] += d[ix][it]*d[ix][it];
      b1[ix] = sqrtf(b1[ix]);
      b2[ix] = sqrtf(b2[ix]);
    }

    int smooth1,smooth2,smooth3,smooth4;
    if (nx1>50) smooth1 = 5;
    else        smooth1 = 0;
    if (nx2>50) smooth2 = 5;
    else        smooth2 = 0;
    if (nx3>50) smooth3 = 5;
    else        smooth3 = 0;
    if (nx4>50) smooth4 = 5;
    else        smooth4 = 0;
    smooth4d(b1,wd,nx1,smooth1,nx2,smooth2,nx3,smooth3,nx4,smooth4,5,verbose);    
    for (ix=0;ix<nx;ix++){
      if (wd_no_pad[ix]==0 && b2[ix] > 1e-8) for(it=0;it<nt;it++) d[ix][it] *= b1[ix]/b2[ix];
    }
    
    free1float(b1);
    free1float(b2);
  }

  free1float(wd);
  free1float(amp);
  free1float(in1);
  free1float(out2);
  free1complex(out1);
  free1complex(in2);
  free2complex(D);
  free2complex(Dobs);
  free2float(M);
  free2float(Mshift);
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  fftwf_destroy_plan(p4);

  return;

}

int compare (const void * a, const void * b)
{
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

void smooth4d(float *d, float *wd,
              int nx1,int smooth1,
              int nx2,int smooth2,
              int nx3,int smooth3,
              int nx4,int smooth4,
              int nrepeat,bool verbose)
{
  float *din,*d_gather1,*d_gather2,*d_gather3,*d_gather4;
  int irepeat,ix,ix1,ix2,ix3,ix4;
  
  din = sf_floatalloc(nx1*nx2*nx3*nx4);
  d_gather1 = sf_floatalloc(nx1);
  d_gather2 = sf_floatalloc(nx2);
  d_gather3 = sf_floatalloc(nx3);
  d_gather4 = sf_floatalloc(nx4);
  
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++) din[ix] =  d[ix];
  for (irepeat=0;irepeat<nrepeat;irepeat++){
  if (smooth1>1){
    for (ix2=0;ix2<nx2;ix2++){ for (ix3=0;ix3<nx3;ix3++){ for (ix4=0;ix4<nx4;ix4++){
      for (ix1=0;ix1<nx1;ix1++) d_gather1[ix1] = d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1];
      mean_filter(d_gather1,nx1,smooth1,1); 
      for (ix1=0;ix1<nx1;ix1++) d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1] = d_gather1[ix1]; 
    }}}
  }
  if (smooth2>1){
    for (ix1=0;ix1<nx1;ix1++){ for (ix3=0;ix3<nx3;ix3++){ for (ix4=0;ix4<nx4;ix4++){
      for (ix2=0;ix2<nx2;ix2++) d_gather2[ix2] = d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1];
      mean_filter(d_gather2,nx2,smooth2,1); 
      for (ix2=0;ix2<nx2;ix2++) d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1] = d_gather2[ix2]; 
    }}}
  }
  if (smooth3>1){
    for (ix1=0;ix1<nx1;ix1++){ for (ix2=0;ix2<nx2;ix2++){ for (ix4=0;ix4<nx4;ix4++){
      for (ix3=0;ix3<nx3;ix3++) d_gather3[ix3] = d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1];
      mean_filter(d_gather3,nx3,smooth3,1); 
      for (ix3=0;ix3<nx3;ix3++) d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1] = d_gather3[ix3]; 
    }}}
  }
  if (smooth4>1){
    for (ix1=0;ix1<nx1;ix1++){ for (ix2=0;ix2<nx2;ix2++){ for (ix3=0;ix3<nx3;ix3++){
      for (ix4=0;ix4<nx4;ix4++) d_gather4[ix4] = d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1];
      mean_filter(d_gather4,nx4,smooth4,1); 
      for (ix4=0;ix4<nx4;ix4++) d[ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1] = d_gather4[ix4]; 
    }}}
  }
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++) d[ix] =  wd[ix]*din[ix] + (1-wd[ix])*d[ix];
  }

  free1float(din);
  free1float(d_gather1);
  free1float(d_gather2);
  free1float(d_gather3);
  free1float(d_gather4);
                           
  return;
}

void mean_filter(float *trace, int nt, int ntw, int nrepeat)
/*< mean filter in 1d. ntw is window length and nrepeat allows to repeat the smoothing >*/
{
  int irepeat,it,itw,index1;
  float sum,nsum;
  float *tracein;
  tracein = sf_floatalloc(nt); 
  /* calculate mean value of a window and assign it to the central index */
    for (irepeat=0;irepeat<nrepeat;irepeat++){
      for (it=0;it<nt;it++) tracein[it] = trace[it];
      for (it=0;it<nt;it++){
        sum  = 0.0;
        nsum = 0.0;
        for (itw=0;itw<ntw;itw++){
          index1 = it - (int) truncf(ntw/2) + itw;
          if (index1>=0 && index1<nt){
            sum += tracein[index1];
            nsum += 1.0;
          }
        }
        trace[it] = sum/nsum;
      }
    }
  free1float(tracein);
  return;
}

void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           int axis,int L)
{
  float **d_gather;
  int i1,i2,i3,i4,i5;
  // process gathers
  if (axis==2){
    d_gather = sf_floatalloc2(n1,n2);
    for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d_gather[i2][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1];
      radial_filter(d_gather,n1,n2,L); 
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i2][i1]; 
    }}}
  }
  else if (axis==3){
    d_gather = sf_floatalloc2(n1,n3);
    for (i2=0;i2<n2;i2++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d_gather[i3][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,n1,n3,L); 
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i3][i1]; 
    }}}
  }
  else if (axis==4){
    d_gather = sf_floatalloc2(n1,n4);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i5=0;i5<n5;i5++){
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d_gather[i4][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,n1,n4,L); 
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i4][i1]; 
    }}}
  }
  else if (axis==5){
    d_gather = sf_floatalloc2(n1,n5);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d_gather[i5][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,n1,n5,L); 
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i5][i1]; 
    }}}
  }
                           
  free2float(d_gather);
  return;
}

void radial_filter(float **d, int nt, int nx, int nL)
{
  int iL,it0,ix0,it,ix;
  float dx,dt,oL,dL,L,x0,t0,x,t,theta,t1,t2,x1,x2,w1,w2,w3,w4;
  float **dout;
  
  dt = 1/(float) nt;
  dx = 2/(float) nx;  
  dL = dt/4;//(dt < dx) ? dt : dx;
  oL = -dL*(((float) nL-1)/2);
  dout = sf_floatalloc2(nt,nx);
  //fprintf(stderr,"oL=%f dL=%f\n",oL,dL);
  for (ix0=0;ix0<nx;ix0++){
    for (it0=0;it0<nt;it0++){
      dout[ix0][it0] = 0.0;
      x0 = ix0*dx - 1; 
      t0 = it0*dt; 
      if (t0>0) theta = atanf(fabsf(x0)/t0);
      else theta = 90;
      for (iL=0;iL<nL;iL++){
        L = iL*dL + oL;       
        //bilinear interpolation to get 1 value from 4 surrounding points.
        t = t0 + L*cosf(theta);
        it = (int) truncf(t/dt);
        t1 = it*dt;
        t2 = t1 + dt;
        x = x0 + signf(x0)*L*sinf(theta);
        ix = (int) truncf((x + 1)/dx);
        x1 = ix*dx - 1;
        x2 = x1 + dx;
        w1 = (t2-t)*(x2-x)/((t2-t1)*(x2-x1));
        w2 = (t-t1)*(x2-x)/((t2-t1)*(x2-x1));
        w3 = (t2-t)*(x-x1)/((t2-t1)*(x2-x1));
        w4 = (t-t1)*(x-x1)/((t2-t1)*(x2-x1));
        if (it >= 0 && it+1 < nt && ix >= 0 && ix+1 < nx && w1 >= 0 && w2 >= 0 && w3 >= 0 && w4 >= 0){
          dout[ix0][it0] += (w1*d[ix][it] + w2*d[ix][it+1] + w3*d[ix+1][it] + w4*d[ix+1][it+1])/nL;
        }
        //MARK
        //fprintf(stderr,"ix0=%d it0=%d ix=%d it=%d \n",ix0,it0,ix,it);
        //dout[ix0][it0] += d[ix][it]/nL;
        //MARK
      }
    }
  }
  for (ix=0;ix<nx;ix++) for (it=0;it<nt;it++) d[ix][it] = dout[ix][it];

  free2float(dout);
  return;
}

float signf(float a)
/*< sign of a float >*/
{
 float b;
 if (a>0)      b = 1.0;
 else if (a<0) b =-1.0;
 else          b = 0.0;
 return b;
}
