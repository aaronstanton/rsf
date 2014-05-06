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

void process_time_windows(float **d,
			  int nt,float dt,int nx1,int nx2,int nx3,int nx4,
                          int Ltw,int Dtw,
                          int *ix1_in,int *ix2_in,int *ix3_in,int *ix4_in,
                          float *wd,int iter,int iter_e,float alphai,float alphaf,int ranki,int rankf,
                          float fmax,int method,int verbose);
void process1c(float **d,
	       int verbose,int nt,int nx,float dt,
               int *x1h,int *x2h,int *x3h,int *x4h,
               int nx1,int nx2,int nx3,int nx4,
               float *wd_no_pad,int iter,int iter_e,float alphai,float alphaf,int ranki,int rankf,float fmax,int method);
void pocs5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf);
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

int main(int argc, char* argv[])
{ 
    int ix,nx,method;
    int n1,n2,n3,n4,n5;
    int i1,i2,i3,i4,i5;
    int nx1,nx2,nx3,nx4;
    int *ix1,*ix2,*ix3,*ix4;
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
    if (!sf_histint(in,"n2",&n5)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&d5)) d5=1;
    if (!sf_histfloat(in,"o2",&o5)) o5=0.;
    if (!sf_histint(in,"n3",&n4))   n4=1;
    if (!sf_histfloat(in,"d3",&d4)) d4=1;
    if (!sf_histfloat(in,"o3",&o4)) o4=0.;
    if (!sf_histint(in,"n4",&n3))   n3=1;
    if (!sf_histfloat(in,"d4",&d3)) d3=1;
    if (!sf_histfloat(in,"o4",&o3)) o3=0.;
    if (!sf_histint(in,"n5",&n2))   n2=1;
    if (!sf_histfloat(in,"d5",&d2)) d2=1;
    if (!sf_histfloat(in,"o5",&o2)) o2=0.;

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
    sf_putfloat(out,"o2",o5);
    sf_putfloat(out,"o3",o4);
    sf_putfloat(out,"o4",o3);
    sf_putfloat(out,"o5",o2);
    sf_putfloat(out,"d1",d1);
    sf_putfloat(out,"d2",d5);
    sf_putfloat(out,"d3",d4);
    sf_putfloat(out,"d4",d3);
    sf_putfloat(out,"d5",d2);
    sf_putfloat(out,"n1",n1);
    sf_putfloat(out,"n2",n5);
    sf_putfloat(out,"n3",n4);
    sf_putfloat(out,"n4",n3);
    sf_putfloat(out,"n5",n2);
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

    nx1 = n2; nx2 = n3; nx3 = n4; nx4 = n5;
    nx = nx1*nx2*nx3*nx4;

    ix1 = sf_intalloc (nx1*nx2*nx3*nx4);
    ix2 = sf_intalloc (nx1*nx2*nx3*nx4);
    ix3 = sf_intalloc (nx1*nx2*nx3*nx4);
    ix4 = sf_intalloc (nx1*nx2*nx3*nx4);

    /* 
      RSF stores long vectors in an order opposite to FFTW: 
      RSF:  ix = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1 
      FFTW: ix = ix1*nx2*nx3*nx4 + ix2*nx3*nx4 + ix3*nx4 + ix4
      here we fetch traces from RSF order
    */
    ix = 0;
    for (i5=0; i5<n5; i5++) {	
      for (i4=0; i4<n4; i4++) {	
        for (i3=0; i3<n3; i3++) {	
          for (i2=0; i2<n2; i2++) {	
            ix1[ix] = i2;
            ix2[ix] = i3;
            ix3[ix] = i4;
            ix4[ix] = i5;
            ix++;
          }
        }
      }
    }

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
 
    if (verbose) fprintf(stderr,"the block has %6.2f %% missing traces.\n", (float) 100 - 100*sum_wd/(nx1*nx2*nx3*nx4));

    process_time_windows(d,
			 n1,d1,nx1,nx2,nx3,nx4,
                         tw_length,tw_overlap,
                         ix1,ix2,ix3,ix4,
                         wd,iter,iter_e,alphai,alphaf,ranki,rankf,
                         fmax,method,verbose);

    for (ix=0; ix<nx; ix++) {
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }

    exit (0);
}

void process_time_windows(float **d,
			  int nt,float dt,int nx1,int nx2,int nx3,int nx4,
                          int Ltw,int Dtw,
                          int *ix1_in,int *ix2_in,int *ix3_in,int *ix4_in,
                          float *wd,int iter,int iter_e,float alphai,float alphaf,int ranki,int rankf,
                          float fmax,int method, int verbose)
/***********************************************************************/
/* process with overlapping time windows */
/***********************************************************************/
{
  int ix,itw,Itw,Ntw,nx,twstart,taper;
  float **d_tw;

  Ntw = 9999;	
  nx = nx1*nx2*nx3*nx4;
  twstart = 0;
  taper = 0;
  d_tw = sf_floatalloc2 (nt,nx);

  for (Itw=0;Itw<Ntw;Itw++){	
    if (Itw == 0){
      Ntw = trunc(nt/(Ltw-Dtw));
      if ( (float) nt/(Ltw-Dtw) - (float) Ntw > 0) Ntw++;
    }		
    twstart = Itw*(Ltw-Dtw);
    if (twstart+Ltw-1 >nt) twstart=nt-Ltw;
    if (Itw*(Ltw-Dtw+1) > nt){
      Ltw = Ltw + nt - Itw*(Ltw-Dtw+1);
    }
    for (ix=0;ix<nx;ix++){ 
      for (itw=0;itw<Ltw;itw++){
        d_tw[ix][itw] = d[ix][twstart+itw]*wd[ix];
      } 
    }

    if (verbose) fprintf(stderr,"processing time window %d of %d\n",Itw+1,Ntw);

    process1c(d_tw,
              verbose,Ltw,nx,dt,
              ix1_in,ix2_in,ix3_in,ix4_in,
              nx1,nx2,nx3,nx4,
              wd,iter,iter_e,alphai,alphaf,ranki,rankf,fmax,method);

    if (Itw==0){ 
      for (ix=0;ix<nx;ix++){ 
        for (itw=0;itw<Ltw;itw++){   
	  d[ix][twstart+itw] = d_tw[ix][itw];
        }
      }	 	 
    }
    else{ 
      for (ix=0;ix<nx;ix++){ 
        for (itw=0;itw<Dtw;itw++){   
	  taper = (float) ((Dtw-1) - itw)/(Dtw-1); 
	  d[ix][twstart+itw] = d[ix][twstart+itw]*(taper) 
                             + d_tw[ix][itw]*(1-taper);
        }
        for (itw=Dtw;itw<Ltw;itw++){   
	  d[ix][twstart+itw] = d_tw[ix][itw];
        }
      }	 	 
    }
  }
 
  return;
}

void process1c(float **d,
	       int verbose,int nt,int nx,float dt,
               int *x1h,int *x2h,int *x3h,int *x4h,
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

  __real__ czero = 0;
  __imag__ czero = 0;

  perci = 0.999;
  percf = 0.001;
  padfactor = 2;
  /* copy data from input to FFT array and pad with zeros */
  ntfft = padfactor*nt;
  /* DANGER: YOU MIGHT WANT TO PAD THE SPATIAL DIRECTIONS TOO.*/
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
  for (ix=0;ix<nk;ix++){
	wd[ix] = 0;  
  }
  for (ix_no_pad=0;ix_no_pad<nx;ix_no_pad++){
	ix = x1h[ix_no_pad]*(nx2fft*nx3fft*nx4fft) + x2h[ix_no_pad]*(nx3fft*nx4fft) + x3h[ix_no_pad]*(nx4fft) + x4h[ix_no_pad];
      if (wd_no_pad[ix_no_pad] > 0){
	wd[ix] = wd_no_pad[ix_no_pad];
      }	
  }
	
  freqslice2= sf_complexalloc(nx1fft*nx2fft*nx3fft*nx4fft);
  
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

  /* process frequency slices */
  for (iw=if_low;iw<if_high;iw++){
    if (verbose) fprintf(stderr,"\r                                         ");
    if (verbose) fprintf(stderr,"\rfrequency slice %d of %d",iw-if_low+1,if_high-if_low);
	  
    for (ix=0;ix<nk;ix++){
      freqslice[ix] = freqslice2[ix] = czero;	
    }

    for (ix_no_pad=0;ix_no_pad<nx;ix_no_pad++){
      ix = x1h[ix_no_pad]*(nx2fft*nx3fft*nx4fft) + x2h[ix_no_pad]*(nx3fft*nx4fft) + x3h[ix_no_pad]*(nx4fft) + x4h[ix_no_pad];
      if (wd_no_pad[ix_no_pad] > 0){
	freqslice[ix] = freqslice2[ix] = cpfft[ix_no_pad][iw];
      }	
    }

    /* The reconstruction engine: */
    if (method==1) pocs5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,perci,percf,alphai,alphaf);
    else if (method==2) mwni5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter_e,iter,verbose);
    else if (method==3) seqsvd5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,alphai,alphaf,ranki,rankf);

    ix        = 0;
    ix_no_pad = 0;
    for (ix1=0;ix1<nx1fft;ix1++){
      for (ix2=0;ix2<nx2fft;ix2++){
    	for (ix3=0;ix3<nx3fft;ix3++){
    	  for (ix4=0;ix4<nx4fft;ix4++){
    	    if (ix1<nx1 && ix2<nx2 && ix3<nx3 && ix4<nx4){
              ix_no_pad = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1;
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
  
  return;

}

void pocs5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf)
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
  /******************************************************************************************** FX1X2 to FK1K2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  rank = 4;
  n = sf_intalloc(4);
  n[0] = nx1fft;
  n[1] = nx2fft;
  n[2] = nx3fft;
  n[3] = nx4fft;
  p2 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_FORWARD, FFTW_ESTIMATE);
  /********************************************************************************************/
  
  /******************************************************************************************** FK1K2 to FX1X2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  p3 = fftwf_plan_dft(rank, n, (fftwf_complex*)freqslice2, (fftwf_complex*)freqslice2, FFTW_BACKWARD, FFTW_ESTIMATE);
  /********************************************************************************************/

  fftwf_execute(p2); /* FFT x to k */
  
  /* threshold in k */
  for (ix=0;ix<nk;ix++) mabs[ix]=sf_cabs(freqslice2[ix]);
  fftwf_execute(p3); /* FFT k to x */
  
  for (ix=0;ix<nk;ix++) freqslice2[ix]=freqslice2[ix]*(1/(float) nk);
  for (iter=1;iter<Iter;iter++){  /* loop for thresholding */
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
      else{ count++; }
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








