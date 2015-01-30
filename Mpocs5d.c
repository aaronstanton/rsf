/* Reconstruction of 5d seismic data by Projection Onto Convex Sets (POCS).
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
#ifndef PI
#define PI (3.141592653589793)
#endif
#include "myfree.h"

void pocs(float **d,
	      int nt,int nx,float dt,
          int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int niter,
          float alpha,
          float fmax,
          float p, bool debias,
          bool verbose);
int compare (const void * a, const void * b);
void cg(sf_complex *m,float *wm,int nm,
	    sf_complex *d,float *wd,int nd,
	    int *n,
	    int niter,
	    bool verbose);
float cgdot(sf_complex *x,int nm);

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
    bool verbose, debias;
    float p;
    
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
    if (!sf_getfloat("p",&p)) p = 8.0; /* Exponent for thresholding, 1=>soft 2=>stein large=>hard */
    if (!sf_getbool("debias",&debias)) debias = false; /* flag for debias */

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
      pocs(d,
           n1,nx,d1,
           n2,n3,n4,n5,
           wd,niter,alpha,
           fmax,
           p,debias,
           verbose);
    }
        
    for (ix=0; ix<nx; ix++) {
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }

    exit (0);
}

void pocs(float **d,
	      int nt,int nx,float dt,
          int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int niter,
          float alpha,
          float fmax,
          float p, bool debias,
          bool verbose)
{  
  int it,ix,iw,ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk,if_low,if_high,padfactor,ix_no_pad,ix1,ix2,ix3,ix4,iter,*n,nzero;
  float perc,perci,percf,*wd,thres,*in1,*out2,f_low,f_high,*amp,*trace,**T,*wm;
  sf_complex czero,*freqslice,*out1,*in2,**D,**Dobs,*m,**M,**R,*r;
  fftwf_plan p1,p2,p3,p4; 
  bool debug;
  
  __real__ czero = 0;
  __imag__ czero = 0;
  perci = 0.99;
  percf = 0.90;
  padfactor = 2;
  
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
  D    = sf_complexalloc2(nw,nk);
  R    = sf_complexalloc2(nw,nk);
  M    = sf_complexalloc2(nw,nk);
  m    = sf_complexalloc(nk);
  r    = sf_complexalloc(nk);
  wm   = sf_floatalloc(nk);
  T    = sf_floatalloc2(nw,nk);
  Dobs = sf_complexalloc2(nw,nk);
  for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) M[ix][iw] = R[ix][iw] = D[ix][iw] = Dobs[ix][iw] = czero;
  for (ix=0;ix<nk;ix++) m[ix] = r[ix] =  czero;

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
      for(iw=0;iw<nw;iw++) R[ix][iw] = D[ix][iw] = Dobs[ix][iw] = out1[iw]; 
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

  trace = sf_floatalloc(nw); 
  amp = sf_floatalloc(nk*nw);

  for (iter=0;iter<niter;iter++){
    for (iw=0;iw<nw;iw++) for (ix=0;ix<nk;ix++) T[ix][iw] = 0; 
    for (iw=if_low;iw<if_high;iw++){
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
          freqslice[ix] = R[ix][iw];
        }
        else{
          freqslice[ix] = czero;
        }
      }}}}
      fftwf_execute(p2);
      for (ix=0;ix<nk;ix++) D[ix][iw] = freqslice[ix]/sqrt((float) nk);
    }
    // Projection 1: Thresholding
    nzero = 0;
    for (iw=0;iw<nw;iw++){  
      for (ix=0;ix<nk;ix++){ 
        amp[ix*nw + iw] = sf_cabs(D[ix][iw]);
        if (amp[ix*nw + iw] > 0.000001) nzero++;
      }
    }
    qsort (amp,nk*nw, sizeof(*amp), compare);
    perc = perci + ((float) iter)*((percf-perci)/((float) niter-1));
    thres = amp[(int) truncf(nk*nw - 1 - (1-perc)*nzero)];
    //fprintf(stderr,"perc=%f thres=%f\n",perc,thres);
    for (iw=if_low;iw<if_high;iw++){
      for (ix=0;ix<nk;ix++){
        if (sf_cabs(D[ix][iw])<thres) D[ix][iw]  = czero;
        else{ 
          D[ix][iw] = D[ix][iw]*(1 - powf(thres/(sf_cabs(D[ix][iw]) + 0.0000001),p));
          T[ix][iw] = 1.0;
        }
      }
    }
    
    if (0){
      for (iw=if_low;iw<if_high;iw++){
        if (iw==20) debug=true;
        else debug=false;
        for (ix=0;ix<nk;ix++) r[ix] = Dobs[ix][iw];
        for (ix=0;ix<nk;ix++) m[ix] = D[ix][iw];
        for (ix=0;ix<nk;ix++) wm[ix] = T[ix][iw];
        cg(m,wm,nk,r,wd,nk,n,10,debug);
        for (ix=0;ix<nk;ix++) M[ix][iw] = m[ix];
      }
    }
    else{
      for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) M[ix][iw] += D[ix][iw];
    }
    
    // transform D from w-k to w-x (afterwards re-zero the zero pad regions of the array)
    for (iw=if_low;iw<if_high;iw++){
      for (ix=0;ix<nk;ix++) freqslice[ix] = M[ix][iw];
      fftwf_execute(p3);
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
          D[ix][iw] = freqslice[ix]/sqrt((float) nk);
        }
        else{
          D[ix][iw] = czero;
        }
      }}}}
    }
    // Projection 2: Subtract from original traces
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) R[ix][iw] = Dobs[ix][iw] - wd[ix]*D[ix][iw];
  }

  if (debias){
      for (iw=if_low;iw<if_high;iw++){
        if (iw==20) debug=true;
        else debug=false;
        for (ix=0;ix<nk;ix++) r[ix] = Dobs[ix][iw];
        for (ix=0;ix<nk;ix++) m[ix] = M[ix][iw];
        for (ix=0;ix<nk;ix++) wm[ix] = T[ix][iw];
        cg(m,wm,nk,r,wd,nk,n,10,debug);
        for (ix=0;ix<nk;ix++) M[ix][iw] = m[ix];
      }

    // transform D from w-k to w-x (afterwards re-zero the zero pad regions of the array)
    for (iw=if_low;iw<if_high;iw++){
      for (ix=0;ix<nk;ix++) freqslice[ix] = M[ix][iw];
      fftwf_execute(p3);
      for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
        ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
        if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
          D[ix][iw] = freqslice[ix]/sqrt((float) nk);
        }
        else{
          D[ix][iw] = czero;
        }
      }}}}
    }

  }

  for (iw=if_low;iw<if_high;iw++){
    for (ix=0;ix<nk;ix++) freqslice[ix] = M[ix][iw];
    fftwf_execute(p3);
    for (ix1=0;ix1<nx1fft;ix1++){ for (ix2=0;ix2<nx2fft;ix2++){ for (ix3=0;ix3<nx3fft;ix3++){ for (ix4=0;ix4<nx4fft;ix4++){
      ix = ix1*nx2fft*nx3fft*nx4fft + ix2*nx3fft*nx4fft + ix3*nx4fft + ix4;
      if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
        D[ix][iw] = freqslice[ix]/sqrt((float) nk);
      }
      else{
        D[ix][iw] = czero;
      }
    }}}}
  }
  //for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) D[ix][iw] = alpha*Dobs[ix][iw] + (1 - alpha*wd[ix])*D[ix][iw];

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

  free1float(trace);
  free1float(amp);
  free1float(in1);
  free1float(out2);
  free1complex(out1);
  free1complex(in2);
  free1complex(m);
  free1float(wd);
  free1float(wm);
  free2complex(D);
  free2complex(Dobs);
  free2float(T);
  free2complex(R);
  free1complex(r);
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

void cg(sf_complex *m,float *wm,int nm,
	    sf_complex *d,float *wd,int nd,
	    int *n,
	    int niter,
	    bool verbose)
/* 
   Quadratic regularization with CG-LS Algorithm 2 from Scales, 1987.
*/

{
  sf_complex *p,*q,*r,*s;
  float alpha,beta,r_dot,q_dot; 
  int i,iter;
  fftwf_plan s_to_r,p_to_q;

  p = sf_complexalloc(nm);
  r = sf_complexalloc(nm);
  q = sf_complexalloc(nd);
  s = sf_complexalloc(nd);

  s_to_r   = fftwf_plan_dft(4,n,(fftwf_complex*)s,(fftwf_complex*)r,FFTW_FORWARD,FFTW_ESTIMATE);
  p_to_q = fftwf_plan_dft(4,n,(fftwf_complex*)p,(fftwf_complex*)q,FFTW_BACKWARD,FFTW_ESTIMATE);

  for (i=0;i<nm;i++) p[i] = m[i];
  fftwf_execute(p_to_q);
  for (i=0;i<nd;i++) s[i] = d[i] - q[i]*wd[i]/sqrt((float) nm);

  fftwf_execute(s_to_r);
  for (i=0;i<nm;i++) r[i] = r[i]*wm[i]/sqrt((float) nm);
  for (i=0;i<nm;i++) p[i] = r[i];

  r_dot = cgdot(r,nm);
  if (r_dot < 0.0000001) return; 

  fftwf_execute(p_to_q);
  for (i=0;i<nm;i++) q[i] = q[i]*wd[i]/sqrt((float) nm);

  if (verbose) fprintf(stderr,"misfit=");
  for (iter=0;iter<niter;iter++){
    if (verbose) fprintf(stderr,"%6.4f,",cgdot(s,nd));
    q_dot = cgdot(q,nd);
    if (q_dot > 0.00001) alpha = cgdot(r,nm)/q_dot; 
    else break;
    for (i=0;i<nm;i++) m[i] = m[i] + alpha*p[i];
    for (i=0;i<nd;i++) s[i] = s[i] - alpha*q[i];
    r_dot = cgdot(r,nm);
    fftwf_execute(s_to_r);
    for (i=0;i<nm;i++) r[i] = r[i]*wm[i]/sqrt((float) nm);
    if (r_dot > 0.0000001) beta = cgdot(r,nm)/r_dot; 
    else  break;
    for (i=0;i<nm;i++) p[i] = r[i] + beta*p[i];
    fftwf_execute(p_to_q);
    for (i=0;i<nd;i++) q[i] = q[i]*wd[i]/sqrt((float) nm);
  }
  if (verbose) fprintf(stderr,"\n");

  fftwf_destroy_plan(s_to_r);
  fftwf_destroy_plan(p_to_q);
  free1complex(p);
  free1complex(q);
  free1complex(r);
  free1complex(s);

  return;
}

float cgdot(sf_complex *x,int nm)
{
  /*     Compute the inner product */
  /*     dot=(x,x) for complex x */     
  int i;
  float  cgdot; 
  sf_complex val;
  
  __real__ val = 0;
  __imag__ val = 0;
  for (i=0;i<nm;i++){  
    val = val + conjf(x[i])*x[i];
  }
  cgdot= crealf(val);
  return(cgdot);
}

