/* Reconstruction of 5d seismic data.
*/
/*
  Copyright (C) 2013 University of Alberta
  
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
#include "perturb4d.h"

void process1c(float **d,
	           int verbose,int nt,int nx,float dt,
               int nx1,int nx2,int nx3,int nx4,
               float *wd_no_pad,int iter,int iter_e,float alphai,float alphaf,int ranki,int rankf,float fmax,int method);
void pocs5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,float *k1,float *k2,float *k3,float *k4,int Iter,float perci,float percf,float alphai,float alphaf);
void mwni5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int itmax_external,int itmax_internal,int verbose);
float cgdot(sf_complex *x,int nm);
float max_abs(sf_complex *x, int nm);
void cg_irls(sf_complex *m,int nm,
	     sf_complex *d,int nd,
	     sf_complex *m0,int nm0,
	     float *wm,int nwm,
	     float *wd,int nwd,
	     int *N,int rank,
	     int itmax_external,
	     int itmax_internal,
	     int verbose);  
void cgesdd_(const char *jobz,const int *M,const int *N,sf_complex *Avec,const int *lda,float *Svec,sf_complex *Uvec,const int *ldu,sf_complex *VTvec,const int *ldvt,sf_complex *work,const int *lwork,float *rwork,int *iwork,int *info);	
void unfold(sf_complex *in, sf_complex **out,int *n,int a);
void fold(sf_complex **in, sf_complex *out,int *n,int a);
void csvd(sf_complex **A,sf_complex **U, float *S,sf_complex **VT,int M,int N);
void mult_svd(sf_complex **A, sf_complex **U, float *S, sf_complex **VT, int M, int N, int rank);
void seqsvd5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft, int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float alphai,float alphaf,int ranki, int rankf);
void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           float fa,float fb,float fc,float fd,
                           int axis);
void radial_filter(float **d,float ot, float dt, int nt,float ox, float dx, int nx,float fa,float fb,float fc,float fd);
void radial_op(float **d,float **m,int nt,int nx,int np,float op,float dp,bool adj);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile);

int main(int argc, char* argv[])
{ 
    int ix,nx,method;
    int n1,n2,n3,n4,n5;
    int i1,i2;
    float *wd,*trace,**d;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float sum;
    sf_file in,out;
    sf_init (argc,argv);
    int tw_length, tw_overlap, iter, iter_e;
    float alphai, alphaf, fmax;
    int ranki, rankf;
    int sum_wd;
    int verbose;
 
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

    if (!sf_getint("method",&method)) method = 1; /* reconstruction algorithm to choose (1=POCS,2=MWNI,3=SEQSVD) */
    if (!sf_getint("tw_length",&tw_length)) tw_length = n1; /* length of time windows in number of samples */
    if (tw_length>n1) tw_length = n1;
    if (!sf_getint("tw_overlap",&tw_overlap)) tw_overlap = 10; /* length of time window overlap in number of samples */
    if (tw_length==n1) tw_overlap=0;
    if (!sf_getint("iter",&iter)) iter = 10; /* number of iterations */
    if (!sf_getint("iter_e",&iter_e)) iter_e = 3; /* number of external iterations for sparsity promotion if MWNI is used */
    if (!sf_getfloat("alphai",&alphai)) alphai = 1; /* denoising parameter for 1st iteration 1=no denoise */
    if (!sf_getfloat("alphaf",&alphaf)) alphaf = 1; /* denoising parameter for last iteration 1=no denoise */
    if (!sf_getint("ranki",&ranki)) ranki = 8; /* rank for first iteration (if using SEQSVD) */
    if (!sf_getint("rankf",&rankf)) rankf = 8; /* rank for last iteration (if using SEQSVD) */
    if (!sf_getint("verbose",&verbose)) verbose = 0; /* verbosity 0=quiet 1=loud */
    if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/d1; /* max frequency to process */
    if (fmax > 0.5/d1) fmax = 0.5/d1;

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

    process1c(d,
              verbose,n1,nx,d1,
              n2,n3,n4,n5,
              wd,iter,iter_e,alphai,alphaf,ranki,rankf,fmax,method);

    for (ix=0; ix<nx; ix++) {
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }

    exit (0);
}


void process1c(float **d,
	           int verbose,int nt,int nx,float dt,
               int nx1,int nx2,int nx3,int nx4,
               float *wd_no_pad,int iter,int iter_e,float alphai,float alphaf,int ranki,int rankf,float fmax,int method)
{  
  int it, ix, iw;
  sf_complex czero;
  int ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk;
  float perci;
  float percf;
  int padfactor;
  float *wd;
  float **pfft; 
  sf_complex **cpfft;
  int N; 
  sf_complex *out;
  fftwf_plan p1;
  int ix_no_pad;
  int ix1,ix2,ix3,ix4;
  float  f_low;
  float  f_high;
  int if_low;
  int if_high;
  float *out2;
  fftwf_plan p4;
  sf_complex* in2;
  sf_complex* freqslice;
  float* in;
  sf_complex* freqslice2;
  float dk1,dk2,dk3,dk4;
  float *k1,*k2,*k3,*k4;
  float min_k1,max_k1,min_k2,max_k2,min_k3,max_k3,min_k4,max_k4;
  __real__ czero = 0;
  __imag__ czero = 0;

  perci = 0.999;
  percf = 0.001;
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

  wd = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  freqslice = sf_complexalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  
  if (nx > nx1*nx2*nx3*nx4) {
  pfft  = sf_floatalloc2(ntfft,nx); /* trace oriented (Hale's reversed convention for alloc)*/
  cpfft = sf_complexalloc2(nw,nx);     /* trace oriented*/
  }
  else{
  pfft  = sf_floatalloc2(ntfft,nx1*nx2*nx3*nx4); /* trace oriented (Hale's reversed convention for alloc)*/
  cpfft = sf_complexalloc2(nw,nx1*nx2*nx3*nx4);     /* trace oriented*/
  }
  /* copy data from input to FFT array and pad with zeros in time dimension*/
  for (ix=0;ix<nx;ix++){
    for (it=0; it<nt; it++) pfft[ix][it]=d[ix][it];
    for (it=nt; it< ntfft;it++) pfft[ix][it] = 0.0;
  }
  /******************************************************************************************** TX to FX
  transform data from t-x to w-x using FFTW */
  N = ntfft; 
  out = sf_complexalloc(nw);
  in = sf_floatalloc(N);
  p1 = fftwf_plan_dft_r2c_1d(N, in, (fftwf_complex*)out, FFTW_ESTIMATE);

  for (ix=0;ix<nx;ix++){
    for(it=0;it<ntfft;it++){
      in[it] = pfft[ix][it];
    }
    fftwf_execute(p1); /* take the FFT along the time dimension */
    for(iw=0;iw<nw;iw++){
      cpfft[ix][iw] = out[iw]; 
    }
  }
  fftwf_destroy_plan(p1);
  fftwf_free(in); fftwf_free(out);
  /********************************************************************************************/


  freqslice2= sf_complexalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  for (ix=0;ix<nk;ix++) wd[ix] = 0.0;
  ix=0;
  for (ix1=0;ix1<nx1fft;ix1++){
    for (ix2=0;ix2<nx2fft;ix2++){
      for (ix3=0;ix3<nx3fft;ix3++){
        for (ix4=0;ix4<nx4fft;ix4++){
          if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
            ix_no_pad = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1;
            wd[ix] = wd_no_pad[ix_no_pad];
          }
    	  ix++;
        }
      }
    }
  }
	
  f_low = 0.1;   /* min frequency to process */
  f_high = fmax; /* max frequency to process */

  if(f_low>0){ 
    if_low = trunc(f_low*dt*ntfft);
  }
  else{
    if_low = 0;
  }
  if(f_high*dt*ntfft<nw){ 
    if_high = trunc(f_high*dt*ntfft);
  }
  else{
    if_high = 0;
  }

  k1 = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  k2 = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  k3 = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  k4 = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);

  /* define normalized wavenumber vectors (range between -0.5 to 0.5) */
  dk1 = (float) 1/nx1fft;
  dk2 = (float) 1/nx2fft;
  dk3 = (float) 1/nx3fft;
  dk4 = (float) 1/nx4fft;
  ix = 0;
  for (ix4=0;ix4<nx4fft;ix4++){
    for (ix3=0;ix3<nx3fft;ix3++){
      for (ix2=0;ix2<nx2fft;ix2++){
        for (ix1=0;ix1<nx1fft;ix1++){
    	  if (ix1<truncf(nx1fft/2)) k1[ix] = dk1*ix1;
    	  else k1[ix] =  -(dk1*nx1fft - dk1*ix1);
    	  if (ix2<truncf(nx2fft/2)) k2[ix] = dk2*ix2;
    	  else k2[ix] =  -(dk2*nx2fft - dk2*ix2);
    	  if (ix3<truncf(nx3fft/2)) k3[ix] = dk3*ix3;
    	  else k3[ix] =  -(dk3*nx3fft - dk3*ix3);
    	  if (ix4<truncf(nx4fft/2)) k4[ix] = dk4*ix4;
    	  else k4[ix] =  -(dk4*nx4fft - dk4*ix4);
    	  if (nx1fft==1) k1[ix] = 0.0;
    	  if (nx2fft==1) k2[ix] = 0.0;
    	  if (nx3fft==1) k3[ix] = 0.0;
    	  if (nx4fft==1) k4[ix] = 0.0;
    	  ix++;
        }
      }
    }
  }

  min_k1 = k1[0]; max_k1 = k1[0];min_k2 = k2[0]; max_k2 = k2[0];min_k3 = k3[0]; max_k3 = k3[0];min_k4 = k4[0]; max_k4 = k4[0];
  for (ix=0;ix<nk;ix++){
    if (k1[ix]<min_k1) min_k1 = k1[ix];
    if (k1[ix]>max_k1) max_k1 = k1[ix];
    if (k2[ix]<min_k2) min_k2 = k2[ix];
    if (k2[ix]>max_k2) max_k2 = k2[ix];
    if (k3[ix]<min_k3) min_k3 = k3[ix];
    if (k3[ix]>max_k3) max_k3 = k3[ix];
    if (k4[ix]<min_k4) min_k4 = k4[ix];
    if (k4[ix]>max_k4) max_k4 = k4[ix];
  }
  fprintf(stderr,"nx1fft=%d min_k1=%f max_k1=%f\n",nx1fft,min_k1,max_k1);
  fprintf(stderr,"nx2fft=%d min_k2=%f max_k2=%f\n",nx2fft,min_k2,max_k2);
  fprintf(stderr,"nx3fft=%d min_k3=%f max_k3=%f\n",nx3fft,min_k3,max_k3);
  fprintf(stderr,"nx4fft=%d min_k4=%f max_k4=%f\n",nx4fft,min_k4,max_k4);
  /* process frequency slices */
  for (iw=if_low;iw<if_high;iw++){
    if (verbose) fprintf(stderr,"\r                                         ");
    if (verbose) fprintf(stderr,"\rfrequency slice %d of %d",iw-if_low+1,if_high-if_low);
	  
    for (ix=0;ix<nk;ix++) freqslice[ix] = freqslice2[ix] = czero;
    ix=0;
    for (ix1=0;ix1<nx1fft;ix1++){
      for (ix2=0;ix2<nx2fft;ix2++){
        for (ix3=0;ix3<nx3fft;ix3++){
          for (ix4=0;ix4<nx4fft;ix4++){
            if (ix1 < nx1 && ix2 < nx2 && ix3 < nx3 && ix4 < nx4){ 
              ix_no_pad = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4;
	          freqslice[ix] = freqslice2[ix] = cpfft[ix_no_pad][iw];
            }
    	    ix++;
          }
        }
      }
    }

    /* The reconstruction engine: */
    if (method==1) pocs5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,k1,k2,k3,k4,iter,perci,percf,alphai,alphaf);
    else if (method==2) mwni5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter_e,iter,verbose);
    else if (method==3) seqsvd5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,alphai,alphaf,ranki,rankf);

    ix        = 0;
    ix_no_pad = 0;
    for (ix1=0;ix1<nx1fft;ix1++){
      for (ix2=0;ix2<nx2fft;ix2++){
    	for (ix3=0;ix3<nx3fft;ix3++){
    	  for (ix4=0;ix4<nx4fft;ix4++){
    	    if (ix1<nx1 && ix2<nx2 && ix3<nx3 && ix4<nx4){
              ix_no_pad = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4;
    	      cpfft[ix_no_pad][iw] = freqslice2[ix];
    	    }
            ix++;
    	  }
    	}
      }
    }
  }

  /* zero all other frequencies */
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++){
    for (iw=if_high;iw<nw;iw++){
      cpfft[ix][iw] = czero;
    }
  }

  /******************************************************************************************** FX to TX
  transform data from w-x to t-x using IFFTW*/
  N = ntfft; 
  out2 = sf_floatalloc(ntfft);
  in2 = sf_complexalloc(N);
  p4 = fftwf_plan_dft_c2r_1d(N, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++){
    /*    fprintf(stderr,"ix=%d\n", ix); */
    for(iw=0;iw<nw;iw++){

      in2[iw] = cpfft[ix][iw];
    }
    fftwf_execute(p4); /* take the FFT along the time dimension */
    for(it=0;it<nt;it++){
      pfft[ix][it] = out2[it]; 
    }
  }
  if (verbose) fprintf(stderr,"\n");

  fftwf_destroy_plan(p4);
  fftwf_free(in2); fftwf_free(out2);
  /********************************************************************************************/
  /* Fourier transform w to t */

  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++) for (it=0; it<nt; it++) d[ix][it]=pfft[ix][it]/ntfft;
  
  free1float(k1);
  free1float(k2);
  free1float(k3);
  free1float(k4);

  return;

}

void pocs5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,float *k1,float *k2,float *k3,float *k4,int Iter,float perci,float percf,float alphai,float alphaf)
{  

  sf_complex czero;
  int ix;
  float *mabs;  
  float *mabsiter;
  float sigma;
  float alpha;
  int rank;
  int *n;
  int count;
  int iter;
  int nclip;
  float pclip;
  fftwf_plan p2;
  fftwf_plan p3;
  __real__ czero = 0;
  __imag__ czero = 0;


  mabs = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);  
  mabsiter = sf_floatalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  /********************************************************************************************/
  rank = 4;
  n = sf_intalloc(4);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  p2 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_FORWARD, FFTW_ESTIMATE);
  /********************************************************************************************/
  
  /********************************************************************************************/
  p3 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_BACKWARD, FFTW_ESTIMATE);
  /********************************************************************************************/

  fftwf_execute(p2); /* FFT x to k */
  
  /* threshold in k */
  for (ix=0;ix<nk;ix++) mabs[ix]=sf_cabs(freqslice2[ix]);
  fftwf_execute(p3); /* FFT k to x */
  
  for (ix=0;ix<nk;ix++) freqslice2[ix]=freqslice2[ix]*(1/(float) nk);
  for (iter=0;iter<Iter;iter++){  /* loop for thresholding */
    fftwf_execute(p2); /* FFT x to k */
    
    count = 0;
    
    /* This is to increase the thresholding within each internal iteration */
    /* Shoplifted from plot/lib/gainpar.c: */
    pclip = 100*(perci - (iter-1)*((perci-percf)/(Iter-1)));
    nclip = SF_MAX(SF_MIN(nk*pclip/100. + .5,nk-1),0);
    sigma=sf_quantile(nclip,nk,mabs);
    /* This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    for (ix=0;ix<nk;ix++) mabsiter[ix]=sf_cabs(freqslice2[ix]);
    for (ix=0;ix<nk;ix++){
      /* thresholding */
      if (mabsiter[ix]<sigma) freqslice2[ix] = czero;
      /* band limitation */
      //if (fabsf(k1[ix])>0.4) freqslice2[ix] = czero;
      //if (fabsf(k2[ix])>0.4) freqslice2[ix] = czero;
      //if (fabsf(k3[ix])>0.4) freqslice2[ix] = czero;
      //if (fabsf(k4[ix])>0.4) freqslice2[ix] = czero;      
    }
    
    fftwf_execute(p3); /* FFT k to x */
    
    for (ix=0;ix<nk;ix++) freqslice2[ix]=freqslice2[ix]*(1/((float) nx1fft*nx2fft*nx3fft*nx4fft));
    
	/* reinsertion into original data */
    for (ix=0;ix<nk;ix++) freqslice2[ix]=freqslice[ix]*alpha + freqslice2[ix]*(1-alpha*wd[ix]); /* x,w */
    
  }
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  
  return;

}

void mwni5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int itmax_external,int itmax_internal,int verbose)
{  
  sf_complex czero;
  sf_complex *x1;
  sf_complex *x2;
  int rank = 4;
  int *n;
  float *wm;
  fftwf_plan prv;
  int i;
  sf_complex* m0;

  __real__ czero = 0;
  __imag__ czero = 0;
  wm = sf_floatalloc(nk);
  x1 = sf_complexalloc(nk);
  x2 = sf_complexalloc(nk);
  for (i=0; i<nk; i++) wm[i] = 1;
  /*********************************************************************/
  rank = 4;
  n = sf_intalloc(rank);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  prv = fftwf_plan_dft(rank,n,(fftwf_complex*)x2,(fftwf_complex*)x2,FFTW_BACKWARD,FFTW_ESTIMATE);
  /*********************************************************************/
  m0 = sf_complexalloc(nk);
  for (i=0; i<nk; i++){ 
	  x1[i] = freqslice[i];
	  x2[i] = freqslice2[i];
	  m0[i] = czero;
  }
  
  cg_irls(x2,nk,
	  x1,nk,
	  m0,nk,
	  wm,nk,
	  wd,nk,
	  n,rank,
	  itmax_external,
	  itmax_internal,
	  verbose);
   
  fftwf_execute(prv); /* FFT (k to x)  (x2 ---> x2 (in place)) */
  for (i=0; i<nk; i++){ 
    freqslice2[i]=x2[i]*(1/sqrt((float) nk));
  }	
  fftwf_destroy_plan(prv);
  return;
}

float max_abs(sf_complex *x, int nm)
{   
  /* Compute Mx = max absolute value of complex vector x */	
  int i;
  float Mx;
  
  Mx = 0;
  for (i=0;i<nm;i++){  
    if(Mx<sf_cabs(x[i])) Mx=sf_cabs(x[i]);
  }
  return(Mx);
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

void cg_irls(sf_complex *m,int nm,
	     sf_complex *d,int nd,
	     sf_complex *m0,int nm0,
	     float *wm,int nwm,
	     float *wd,int nwd,
	     int *N,int rank,
	     int itmax_external,
	     int itmax_internal,
	     int verbose)
/* 
   Non-quadratic regularization with CG-LS. The inner CG routine is taken from
   Algorithm 2 from Scales, 1987. Make sure linear operator passes the dot product.
   In this case (MWNI), the linear operator is the FFT implemented using the FFTW package.
*/

{
  sf_complex *v,*Pv,*Ps,*s,*ss,*g,*r;
  float alpha,beta,delta,gamma,gamma_old,*P,Max_m; 
  int i,j,k;
  fftwf_plan Pv_to_r,r_to_g,Ps_to_ss;
  v  = sf_complexalloc(nm);
  P  = sf_floatalloc(nm);
  Pv = sf_complexalloc(nm);
  Ps = sf_complexalloc(nm);
  g  = sf_complexalloc(nm);
  r  = sf_complexalloc(nd);
  s  = sf_complexalloc(nm);
  ss = sf_complexalloc(nd);
  Pv_to_r  = fftwf_plan_dft(rank,N,(fftwf_complex*)Pv,(fftwf_complex*)r,FFTW_BACKWARD,FFTW_ESTIMATE);
  r_to_g   = fftwf_plan_dft(rank,N,(fftwf_complex*)r,(fftwf_complex*)g,FFTW_FORWARD,FFTW_ESTIMATE);
  Ps_to_ss = fftwf_plan_dft(rank,N,(fftwf_complex*)Ps,(fftwf_complex*)ss,FFTW_BACKWARD,FFTW_ESTIMATE);


  for (i=0;i<nm;i++){
    m[i] = m0[i];				
    P[i] = 1;
    v[i] = m[i];
  }

  for (i=0;i<nd;i++){
    r[i] = d[i];				
  }

  for (j=1;j<=itmax_external;j++){
    for (i=0;i<nm;i++) Pv[i] = v[i]*P[i];

    /* fft_op(Pv,nm,r,nd,wm,nm,wd,nd,p_fwd,fwd,verbose); */
    for (i=0;i<nm;i++)  Pv[i] = Pv[i]*wm[i];
    fftwf_execute(Pv_to_r);
    for (i=0;i<nd;i++)  r[i] = r[i]*wd[i]/sqrt((float) nm);

    for (i=0;i<nd;i++) r[i] = d[i] - r[i];

    /* fft_op(g,nm,r,nd,wm,nm,wd,nd,p_adj,adj,verbose); */
    for (i=0;i<nd;i++)  r[i] = r[i]*wd[i];
    fftwf_execute(r_to_g);
    for (i=0;i<nm;i++)  g[i] = g[i]*wm[i]/sqrt((float) nm);

    for (i=0;i<nm;i++){
      g[i] = g[i]*P[i];
      s[i] = g[i];
    }

    gamma = cgdot(g,nm);
    gamma_old = gamma;

    for (k=1;k<=itmax_internal;k++){
      for (i=0;i<nm;i++) Ps[i] = s[i]*P[i];

      /* fft_op(Ps,nm,ss,nd,wm,nm,wd,nd,p_fwd,fwd,verbose); */
      for (i=0;i<nm;i++)  Ps[i] = Ps[i]*wm[i];
      fftwf_execute(Ps_to_ss);
      for (i=0;i<nd;i++)  ss[i] = ss[i]*wd[i]/sqrt((float) nm);

      delta = cgdot(ss,nd);
      alpha = gamma/(delta + 0.00000001);

      for (i=0;i<nm;i++) v[i] = v[i] +  s[i]*alpha;
      for (i=0;i<nd;i++) r[i] = r[i] - ss[i]*alpha;

      /* fft_op(g,nm,r,nd,wm,nm,wd,nd,p_adj,adj,verbose); */   
      for (i=0;i<nd;i++)  r[i] = r[i]*wd[i];
      fftwf_execute(r_to_g);
      for (i=0;i<nm;i++)  g[i] = g[i]*wm[i]/sqrt((float) nm);


      for (i=0;i<nm;i++) g[i] = g[i]*P[i];
      gamma = cgdot(g,nm);
      beta = gamma/(gamma_old + 0.00000001);

      gamma_old = gamma;
      for (i=0;i<nm;i++) s[i] = g[i] + s[i]*beta;
    }
    for (i=0;i<nm;i++) m[i] = v[i]*P[i];
    Max_m = max_abs(m,nm);
    for (i=0;i<nm;i++) P[i] = sf_cabs(m[i]*(1/Max_m));
  }

  fftwf_destroy_plan(Pv_to_r);
  fftwf_destroy_plan(r_to_g);
  fftwf_destroy_plan(Ps_to_ss);

  return;
  
}

void seqsvd5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft, int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float alphai,float alphaf,int ranki, int rankf)
{  

  int ix, rank; 
  int M, N;
  int *n;
  sf_complex **uf1;  
  sf_complex **uf2;  
  sf_complex **uf3;
  sf_complex **uf4;
  sf_complex **U1;
  sf_complex **U2;
  sf_complex **U3;
  sf_complex **U4;
  sf_complex **VT1;
  sf_complex **VT2;
  sf_complex **VT3;
  sf_complex **VT4;
  float *S1;
  float *S2;
  float *S3;
  float *S4;
  float alpha;
  int iter;
  float var;

  n = sf_intalloc(4);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  uf1 = sf_complexalloc2(nx2fft*nx3fft*nx4fft,nx1fft);  
  uf2 = sf_complexalloc2(nx1fft*nx3fft*nx4fft,nx2fft);  
  uf3 = sf_complexalloc2(nx1fft*nx2fft*nx4fft,nx3fft);  
  uf4 = sf_complexalloc2(nx1fft*nx2fft*nx3fft,nx4fft);
  M = nx1fft;
  N = nx2fft*nx3fft*nx4fft;
  U1 = sf_complexalloc2(M,M);
  S1 = sf_floatalloc(M);
  VT1 = sf_complexalloc2(N,M);
  M = nx2fft;
  N = nx1fft*nx3fft*nx4fft;
  U2 = sf_complexalloc2(M,M);
  S2 = sf_floatalloc(M);
  VT2 = sf_complexalloc2(N,M);
  M = nx3fft;
  N = nx1fft*nx2fft*nx4fft;
  U3 = sf_complexalloc2(M,M);
  S3 = sf_floatalloc(M);
  VT3 = sf_complexalloc2(N,M);
  M = nx4fft;
  N = nx1fft*nx2fft*nx3fft;
  U4 = sf_complexalloc2(M,M);
  S4 = sf_floatalloc(M);
  VT4 = sf_complexalloc2(N,M);
  for (iter=1;iter<=Iter;iter++){  /* loop for iteration */
    var = (float) 3* (float) (Iter - iter)/ (float) Iter;
    perturb4d(freqslice2,nx1fft,nx2fft,nx3fft,nx4fft,var,var,var,var);
    /*  This is to increase rank at each iteration */
    rank=ranki + (int) trunc((iter-1)*((rankf-ranki)/(Iter-1)));

    if (nx1fft > rank){    
      unfold(freqslice2,uf1,n,1);
      M = nx1fft;
      N = nx2fft*nx3fft*nx4fft;
      csvd(uf1,U1,S1,VT1,M,N);
      mult_svd(uf1,U1,S1,VT1,M,N,rank);
      fold(uf1,freqslice2,n,1);
    }
    
    if (nx2fft > rank){    
      unfold(freqslice2,uf2,n,2);
      M = nx2fft;
      N = nx1fft*nx3fft*nx4fft;
      csvd(uf2,U2,S2,VT2,M,N);
      mult_svd(uf2,U2,S2,VT2,M,N,rank);
      fold(uf2,freqslice2,n,2);
    }    

    if (nx3fft > rank){    
      unfold(freqslice2,uf3,n,3);
      M = nx3fft;
      N = nx1fft*nx2fft*nx4fft;
      csvd(uf3,U3,S3,VT3,M,N);
      mult_svd(uf3,U3,S3,VT3,M,N,rank);
      fold(uf3,freqslice2,n,3);
    }   

    if (nx4fft > rank){    
      unfold(freqslice2,uf4,n,4);
      M = nx4fft;
      N = nx1fft*nx2fft*nx3fft;
      csvd(uf4,U4,S4,VT4,M,N);    
      mult_svd(uf4,U4,S4,VT4,M,N,rank);
      fold(uf4,freqslice2,n,4);
    }

    /*  This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    /*  reinsertion into original data */
    for (ix=0;ix<nk;ix++) freqslice2[ix]=freqslice[ix]*alpha + freqslice2[ix]*(1-alpha*wd[ix]);

  }

  free2complex(uf1);
  free2complex(uf2);
  free2complex(uf3);
  free2complex(uf4);

  free2complex(U1);
  free1float(S1);
  free2complex(VT1);

  free2complex(U2);
  free1float(S2);
  free2complex(VT2);

  free2complex(U3);
  free1float(S3);
  free2complex(VT3);

  free2complex(U4);
  free1float(S4);
  free2complex(VT4);

  return;
}

void unfold(sf_complex *in, sf_complex **out,int *n,int a)
{   
  /* 
  unfold a long-vector representing a 4D tensor into a matrix.	
  in  = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4])
  out = matrix that is the unfolding of "in" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;

  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];
  
  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1][ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


 
  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix2][ix1*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
            out[ix3][ix1*nx2*nx4+ix2*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix4][ix1*nx2*nx3+ix2*nx3+ix3]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  return;
}

void fold(sf_complex **in, sf_complex *out,int *n,int a)
{   
  /* 
  fold a matrix back to a long-vector representing a 4D tensor.	
  in  = the unfolding of "out" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  out = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4]) 
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;
  
  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];

  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1][ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix2][ix1*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix3][ix1*nx2*nx4+ix2*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix4][ix1*nx2*nx3+ix2*nx3+ix3];
	  }
	}
      }
    }
  }    

  return;
}

void csvd(sf_complex **A,sf_complex **U,float *S,sf_complex **VT,int M,int N)
{

  sf_complex czero;
  sf_complex* Avec;
  int i,j,n;
  float* Svec;  
  char jobz;
  int lda;
  int ldu;
  int ldvt;
  int lwork; 
  int info;
  sf_complex *work;
  int lrwork;
  float *rwork;  
  float *iwork;  
  sf_complex* Uvec; 
  sf_complex* VTvec; 
  
  __real__ czero = 0;
  __imag__ czero = 0;

  Avec = sf_complexalloc(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      Avec[n] = A[j][i];
      n++;
    }
  }
  Uvec = sf_complexalloc(M*M); 
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      Uvec[n] = czero;
      n++;
    }
  }
  VTvec = sf_complexalloc(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VTvec[n] = czero;
      n++;
    }
  }
  Svec = sf_floatalloc(M);  
  
  jobz = 'S';
  lda   = M;
  ldu   = M;
  ldvt  = M;
  lwork = 4*(5*M + N); 
  work = sf_complexalloc(lwork); 

  /* make float array rwork with dimension lrwork (where lrwork >= min(M,N)*max(5*min(M,N)+7,2*max(M,N)+2*min(M,N)+1)) */
  lrwork = 4*(5*M*M+ 7*M);
  rwork = sf_floatalloc(lrwork);  

  /* make float array iwork with dimension 8*M (where A is MxN) */
  iwork = sf_floatalloc(4*8*M);  
  cgesdd_(&jobz, &M, &N, (sf_complex*)Avec, &lda, Svec, (sf_complex*)Uvec, &ldu, (sf_complex*)VTvec, &ldvt, (sf_complex*)work, &lwork, (float*)rwork, (int*)iwork, &info);  
  if (info != 0)  fprintf(stderr,"Error in cgesdd: info = %d\n",info);

  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VT[j][i] = VTvec[n];
      n++;
    }
  }
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      U[j][i] = Uvec[n];
      n++;
    }
  }

  for (i=0;i<M;i++){
    S[i] = Svec[i];
  }
  

  free1complex(Avec);
  free1complex(VTvec);
  free1complex(Uvec);
  free1float(Svec);
  free1complex(work);
  free1float(rwork);
  free1float(iwork);

  return;
}

void mult_svd(sf_complex **A, sf_complex **U,float *S,sf_complex **VT,int M,int N,int rank)
{   
  int i,j,k;
  sf_complex czero;
  sf_complex **SVT;
  sf_complex sum;

  SVT = sf_complexalloc2(N,M);
  __real__ czero = 0;
  __imag__ czero = 0;
  sum = czero;
  
  for (i=0;i<rank;i++){
    for (j=0;j<N;j++){
      SVT[i][j] = czero;
      if (i < rank) SVT[i][j] = VT[i][j]*S[i];
    }
  }
  for (i=0;i<M;i++){
    for (j=0;j<N;j++){
      sum = czero;
      for (k=0;k<rank;k++){
      sum = sum + SVT[k][j]*U[i][k];
      }
      A[i][j] = sum;
    }
  }
 
  free2complex(SVT);

  return;
}

void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           float fa,float fb,float fc,float fd,
                           int axis)
{
  float **d_gather;
  int i1,i2,i3,i4,i5;
  // process gathers
  if (axis==2){
    d_gather = sf_floatalloc2(n1,n2);
    for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d_gather[i2][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1];
      radial_filter(d_gather,o1,d1,n1,o2,d2,n2,fa,fb,fc,fd); 
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i2][i1]; 
    }}}
  }
  else if (axis==3){
    d_gather = sf_floatalloc2(n1,n3);
    for (i2=0;i2<n2;i2++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d_gather[i3][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o3,d3,n3,fa,fb,fc,fd); 
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i3][i1]; 
    }}}
  }
  else if (axis==4){
    d_gather = sf_floatalloc2(n1,n4);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i5=0;i5<n5;i5++){
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d_gather[i4][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o4,d4,n4,fa,fb,fc,fd); 
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i4][i1]; 
    }}}
  }
  else if (axis==5){
    d_gather = sf_floatalloc2(n1,n5);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d_gather[i5][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o5,d5,n5,fa,fb,fc,fd); 
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i5][i1]; 
    }}}
  }
                           
  free2float(d_gather);
  return;
}


void radial_filter(float **d,float ot, float dt, int nt,float ox, float dx, int nx,float fa,float fb,float fc,float fd)
{
  int np,ip,it;
  float op,dp;
  float *trace,**m;
//  sf_file outtmp;
//  char tmpname[256];
  
//  sprintf(tmpname, "tmp_radial_transform.rsf");
//  outtmp = sf_output(tmpname);

  np=1801;
  op=-90;
  dp=0.1;
  m = sf_floatalloc2(nt,np);
  trace = sf_floatalloc(nt);  
  radial_op(d,m,nt,nx,np,op,dp,true); 
  
//  write5d(m,
//          nt,ot,dt,"Time","s",  
//          np,op,dp,"Angle","Degrees",
//          1,0,1," "," ",
//          1,0,1," "," ",
//          1,0,1," "," ",
//          "Radial Domain",outtmp);
//  sf_fileclose(outtmp);
  
  for (ip=0;ip<np;ip++){  
    for (it=0;it<nt;it++) trace[it] = m[ip][it];    
    bpfilter(trace,dt,nt,fa,fb,fc,fd);
    for (it=0;it<nt;it++) m[ip][it] = trace[it];    
  }
  radial_op(d,m,nt,nx,np,op,dp,false); 

  free1float(trace);
  free2float(m);
  return;
}

void radial_op(float **d,float **m,int nt,int nx,int np,float op,float dp,bool adj)
{
  int ip,it,ix;
  float ox,dx,ot,dt,p,p_floor,x,t,alpha,beta;
  
  ox=-1;
  dx=2/(float) nx;
  ot=0;
  dt=1/(float) nt;
  
  if (adj){
    for (it=0;it<nt;it++) for (ip=0;ip<np;ip++) m[ip][it] = 0.0; 
  }
  else{
    for (it=0;it<nt;it++) for (ix=0;ix<nx;ix++) d[ix][it] = 0.0; 
  }
  
  for (it=0;it<nt;it++){ 
    for (ix=0;ix<nx;ix++){
      x = ix*dx + ox;
      t = it*dt + ot;
      p = (180/PI)*atanf(x/t);
      //fprintf(stderr,"p=%f\n",p);
      ip = (int) truncf((p - op)/dp);
      p_floor = truncf((p - op)/dp)*dp + op;
      alpha = (p-p_floor)/dp;
      beta = 1-alpha;
      if (ip >= 0 && ip+1 < np){
        if (adj){
          m[ip][it]   +=  beta*d[ix][it];
          m[ip+1][it] += alpha*d[ix][it];
        }
        else{
          d[ix][it] += (1/(beta*beta + alpha*alpha))*(beta*m[ip][it] + alpha*m[ip+1][it]);
	    }
	  }
    }
  }
  return;
}

void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
/*< bandpass filter >*/
{
  int iw,nw,ntfft,ia,ib,ic,id,it;
  float *in1, *out2;
  sf_complex *in2,*out1;
  sf_complex czero;
  fftwf_plan p1;
  fftwf_plan p2;

  __real__ czero = 0;
  __imag__ czero = 0;
  ntfft = 4*nt;
  nw=ntfft/2+1;
  if(a>0) ia = trunc(a*dt*ntfft);
  else ia = 0;
  if(b>0) ib = trunc(b*dt*ntfft);
  else ib = 1;
  if(c*dt*ntfft<nw) ic = trunc(c*dt*ntfft);
  else ic = nw-1;
  if(d*dt*ntfft<nw) id = trunc(d*dt*ntfft);
  else id = nw;

  out1 = sf_complexalloc(nw);
  in1  = sf_floatalloc(ntfft);
  p1   = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  out2 = sf_floatalloc(ntfft);
  in2  = sf_complexalloc(ntfft);
  p2   = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);

  for (it=0; it<nt; it++) in1[it]=trace[it];
  for (it=nt; it< ntfft;it++) in1[it] = 0.0;
  fftwf_execute(p1);
  for(iw=0;iw<ia;iw++)  in2[iw] = czero; 
  for(iw=ia;iw<ib;iw++) in2[iw] = out1[iw]*((float) (iw-ia)/(ib-ia))/sqrtf((float) ntfft); 
  for(iw=ib;iw<ic;iw++) in2[iw] = out1[iw]/sqrtf((float) ntfft); 
  for(iw=ic;iw<id;iw++) in2[iw] = out1[iw]*(1 - (float) (iw-ic)/(id-ic))/sqrtf((float) ntfft); 
  for(iw=id;iw<nw;iw++) in2[iw] = czero; 
  fftwf_execute(p2); /* take the FFT along the time dimension */
  for(it=0;it<nt;it++) trace[it] = out2[it]/sqrtf((float) ntfft); 
  
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_free(in1); fftwf_free(out1);
  fftwf_free(in2); fftwf_free(out2);
  return;
}

void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile)
/*< write a 5d array of floats to disk >*/
{
  sf_putfloat(outfile,"o1",o1);
  sf_putfloat(outfile,"d1",d1);
  sf_putint(outfile,"n1",n1);
  sf_putstring(outfile,"label1",label1);
  sf_putstring(outfile,"unit1",unit1);
  sf_putfloat(outfile,"o2",o2);
  sf_putfloat(outfile,"d2",d2);
  sf_putint(outfile,"n2",n2);
  sf_putstring(outfile,"label2",label2); 
  sf_putstring(outfile,"unit2",unit2);
  sf_putfloat(outfile,"o3",o3);
  sf_putfloat(outfile,"d3",d3);
  sf_putint(outfile,"n3",n3);
  sf_putstring(outfile,"label3",label3);
  sf_putstring(outfile,"unit3",unit3);
  sf_putfloat(outfile,"o4",o4);
  sf_putfloat(outfile,"d4",d4);
  sf_putint(outfile,"n4",n4);
  sf_putstring(outfile,"label4",label4);
  sf_putstring(outfile,"unit4",unit4);
  sf_putfloat(outfile,"o5",o5);
  sf_putfloat(outfile,"d5",d5);
  sf_putint(outfile,"n5",n5);
  sf_putstring(outfile,"label5",label5);
  sf_putstring(outfile,"unit5",unit5);
  sf_putstring(outfile,"title",title);
  sf_floatwrite(data[0],n1*n2*n3*n4*n5,outfile);
  sf_fileclose(outfile);
  return;
}







