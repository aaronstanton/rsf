/* POCS reconstruction of 5d seismic data.
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

#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>

#include <stdio.h>

#include <time.h>

#include <rsf.h>
#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif


void process_time_windows(float **d,
			  int nt,int nx1,int nx2,int nx3,int nx4,
                          int Ltw,int Dtw,
                          int *ix1_in,int *ix2_in,int *ix3_in,int *ix4_in,
                          float *wd_no_pad,int iter,float alphai,float alphaf,
                          float fmax, int verbose);

int main(int argc, char* argv[])
{ 
    int ix,nx;
    int n1,n2,n3,n4,n5;
    int i1,i2,i3,i4,i5;
    int nx1,nx2,nx3,nx4;
    int *ix1_in,*ix2_in,*ix3_in,*ix4_in;
    float *wd,*trace,**d;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float sum;
    sf_file in,out;
    sf_init (argc,argv);
    int tw_length, tw_overlap, iter;
    float alphai, alphaf, fmax;
    int verbose;
 
    in = sf_input("in");
    out = sf_output("out");

    if (!sf_getint("tw_length",&tw_length)) tw_length = 100; /* length of time windows in number of samples */
    if (!sf_getint("tw_overlap",&tw_overlap)) tw_overlap = 10; /* length of time window overlap in number of samples */
    if (!sf_getint("iter",&iter)) iter = 10; /* number of iterations */
    if (!sf_getfloat("alphai",&alphai)) alphai = 1; /* denoising parameter for 1st iteration 1=no denoise */
    if (!sf_getfloat("alphaf",&alphaf)) alphaf = 1; /* denoising parameter for last iteration 1=no denoise */
    if (!sf_getint("verbose",&verbose)) verbose = 0; /* verbosity 0=quiet 1=loud */

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

    ix1_in = sf_intalloc (n2*n3*n4*n5);
    ix2_in = sf_intalloc (n2*n3*n4*n5);
    ix3_in = sf_intalloc (n2*n3*n4*n5);
    ix4_in = sf_intalloc (n2*n3*n4*n5);

    ix = 0;
    for (i5=0; i5<n5; i5++) {	
      for (i4=0; i4<n4; i4++) {	
        for (i3=0; i3<n3; i3++) {	
          for (i2=0; i2<n2; i2++) {	
            ix1_in[ix] = i2;
            ix2_in[ix] = i3;
            ix3_in[ix] = i4;
            ix4_in[ix] = i5;
            ix++;
          }
        }
      }
    }
    nx = ix;
    nx1 = n2; nx2 = n3; nx3 = n4; nx4 = n5;

    trace = sf_floatalloc (n1);
    d = sf_floatalloc2 (n1,nx);
    wd    = sf_floatalloc (n1);
    for (i2=0; i2<nx; i2++) {
      sf_floatread(trace,n1,in);
      ix =   ix1_in[i2]*nx2*nx3*nx4 
           + ix2_in[i2]*nx3*nx4 
           + ix3_in[i2]*nx4 
           + ix4_in[i2];
      sum = 0;
      for (i1=0; i1<n1; i1++){
        sum      += trace[i1]*trace[i1]; 
        d[ix][i1] = trace[i1];
      }
      if (sum>0.001) wd[i2] = 1;
      else           wd[i2] = 0; 	
    }

    process_time_windows(d,
			 n1,nx1,nx2,nx3,nx4,
                         tw_length,tw_overlap,
                         ix1_in,ix2_in,ix3_in,ix4_in,
                         wd,iter,alphai,alphaf,
                         fmax,verbose);

    for (i2=0; i2<nx; i2++) {
      ix =   ix4_in[i2]*nx3*nx2*nx1 
           + ix3_in[i2]*nx2*nx1 
           + ix2_in[i2]*nx1 
           + ix1_in[i2];
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];	
      sf_floatwrite(trace,n1,out);
    }


    exit (0);
}

void process_time_windows(float **d,
			  int nt,int nx1,int nx2,int nx3,int nx4,
                          int Ltw,int Dtw,
                          int *ix1_in,int *ix2_in,int *ix3_in,int *ix4_in,
                          float *wd_no_pad,int iter,float alphai,float alphaf,
                          float fmax, int verbose)
/***********************************************************************/
/* process with overlapping time windows */
/***********************************************************************/
{
  int ix,itw,Itw,Ntw,nx,twstart,taper;
  float **din_tw, **dout_tw;

  Ntw = 9999;	
  nx = nx1*nx2*nx3*nx4;
  twstart = 0;
  taper = 0;

  din_tw = sf_floatalloc2 (nt,nx);
  dout_tw = sf_floatalloc2 (nt,nx);

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
        din_tw[ix][itw] = d[ix][twstart+itw]*wd_no_pad[ix];
      } 
    }
    fprintf(stderr,"processing time window %d of %d\n",Itw+1,Ntw);

    process1c(din_tw,dout_tw,
              verbose,Ltw,nx,dt,
              ix1_in,ix2_in,ix3_in,ix4_in,
              nx1,nx2,nx3,nx4,
              wd_no_pad,iter,alphai,alphaf,fmax);

    if (Itw==0){ 
      for (ix=0;ix<nx;ix++){ 
        for (itw=0;itw<Ltw;itw++){   
	  d[ix][twstart+itw] = dout_tw[ix][itw];
        }
      }	 	 
    }
    else{ 
      for (ix=0;ix<nx;ix++){ 
        for (itw=0;itw<Dtw;itw++){   
	  taper = (float) ((Dtw-1) - itw)/(Dtw-1); 
	  d[ix][twstart+itw] = d[ix][twstart+itw]*(taper) 
                             + dout_tw[ix][itw]*(1-taper);
        }
        for (itw=Dtw;itw<Ltw;itw++){   
	  d[ix][twstart+itw] = dout_tw[ix][itw];
        }
      }	 	 
    }
  }
 
  return;
}

void process1c(float **datain,float **dataout,int verbose,int nt,int nx,float dt,int *x1h,int *x2h,int *x3h,int *x4h,int nx1,int nx2,int nx3,int nx4,float *wd_no_pad,int iter,int iter_external,float alphai,float alphaf,float bandi,float bandf,float fmax,int method,int ranki,int rankf)
{  
  int it, ix, iw;
  complex czero;
  int ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk;
  float perci;
  float percf;
  int padfactor;
  float *wd;
  float **pfft; 
  complex **cpfft;
  int N; 
  complex *out;
  fftwf_plan p1;
  int *mapping_vector;
  int ix_no_pad;
  float *k_n_1;  
  float *k_n_2;  
  float *k_n_3;  
  float *k_n_4; 
  int sum_wd;
  int ix1,ix2,ix3,ix4;
  float  f_low;
  float  f_high;
  int if_low;
  int if_high;
  float *out2;
  fftwf_plan p4;
  complex* in2;
  complex* freqslice;
  float* in;
  complex* freqslice2;

  czero.r=czero.i=0;
  perci = 0.999;
  percf = 0.001;
  padfactor = 2;
  /* copy data from input to FFT array and pad with zeros */
  ntfft = npfar(padfactor*nt);
  /* DANGER: YOU MIGHT WANT TO PAD THE SPATIAL DIRECTIONS TOO.*/
  nx1fft = nx1;
  nx2fft = nx2;
  nx3fft = nx3;
  nx4fft = nx4;
  if(nx1==1) nx1fft = 1;
  if(nx2==1) nx2fft = 1;
  if(nx3==1) nx3fft = 1;
  if(nx4==1) nx4fft = 1;
  nw=ntfft/2+1;
  nk=nx1fft*nx2fft*nx3fft*nx4fft;
  wd = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);
  freqslice = ealloc1complex(nx1fft*nx2fft*nx3fft*nx4fft);
  
  if (nx > nx1*nx2*nx3*nx4) {
  pfft  = ealloc2float(ntfft,nx); 
  cpfft = ealloc2complex(nw,nx);
  }
  else{
  pfft  = ealloc2float(ntfft,nx1*nx2*nx3*nx4);
  cpfft = ealloc2complex(nw,nx1*nx2*nx3*nx4);
  }
  /* copy data from input to FFT array and pad with zeros in time dimension*/
  for (ix=0;ix<nx;ix++){
    for (it=0; it<nt; it++) pfft[ix][it]=datain[ix][it];
    for (it=nt; it< ntfft;it++) pfft[ix][it] = 0.0;
  }
  /******************************************************************************************** TX to FX
  transform data from t-x to w-x using FFTW */
  N = ntfft; 
  out = ealloc1complex(nw);
  in = ealloc1float(N);
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
  mapping_vector = ealloc1int(nk+1);
  for (ix=0;ix<nk;ix++){
	wd[ix] = 0;  
	mapping_vector[ix] = 0;
  }
	
  ix_no_pad = 0;
	
  for (ix_no_pad=0;ix_no_pad<nx;ix_no_pad++){
	ix = x1h[ix_no_pad]*(nx2fft*nx3fft*nx4fft) + x2h[ix_no_pad]*(nx3fft*nx4fft) + x3h[ix_no_pad]*(nx4fft) + x4h[ix_no_pad];
      if (wd_no_pad[ix_no_pad] > 0){
	wd[ix] = wd_no_pad[ix_no_pad];  
	mapping_vector[ix] = ix_no_pad+1;
      }	
  }
	
  /* algorithm starts*/
  freqslice2= ealloc1complex(nx1fft*nx2fft*nx3fft*nx4fft);
  
  /* make long vectors of normalized wavenumbers (for each of the 4 dimensions). */
  /* (to be used in the band limitation) */
  k_n_1 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_2 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_3 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  k_n_4 = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft); 
  sum_wd = 0;
  ix        = 0;
  for (ix1=0;ix1<nx1fft;ix1++){
    for (ix2=0;ix2<nx2fft;ix2++){
      for (ix3=0;ix3<nx3fft;ix3++){
	for (ix4=0;ix4<nx4fft;ix4++){
	  if (ix1>nx1fft/2) {
	    k_n_1[ix] = 1 - (float) (ix1-nx1fft/2)/(nx1fft/2);
	  }
	  else {
	    k_n_1[ix] = 1 - (float) (nx1fft/2-ix1)/(nx1fft/2);
	  }
	  if (ix2>nx2fft/2) {
	    k_n_2[ix] = 1 - (float) (ix2-nx2fft/2)/(nx2fft/2);
	  }
	  else {
	    k_n_2[ix] = 1 - (float) (nx2fft/2-ix2)/(nx2fft/2);
	  }
	  if (ix3>nx3fft/2) {
	    k_n_3[ix] = 1 - (float) (ix3-nx3fft/2)/(nx3fft/2);
	  }
	  else {
	    k_n_3[ix] = 1 - (float) (nx3fft/2-ix3)/(nx3fft/2);
	  }
	  if (ix4>nx4fft/2) {
	    k_n_4[ix] = 1 - (float) (ix4-nx4fft/2)/(nx4fft/2);
	  }
	  else {
	    k_n_4[ix] = 1 - (float) (nx4fft/2-ix4)/(nx4fft/2);
	  }
	  if (wd[ix]>0) sum_wd++;
	  ix++;
	}
      }
    }
  }
  if (verbose) fprintf(stderr,"out of %d input traces %d traces fall in repeated bins (%f %%).\n",nx,nx-sum_wd,(float) 100*(nx-sum_wd)/nx);
  if (verbose) fprintf(stderr,"the block has %f %% missing traces.\n", (float) 100 - 100*sum_wd/(nx1fft*nx2fft*nx3fft*nx4fft));

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

  /* freq by freq algorithm */
  for (iw=if_low;iw<if_high;iw++){
    fprintf(stderr,"\r                                         ");
    fprintf(stderr,"\rfrequency slice %d of %d",iw-if_low+1,if_high-if_low);
	  
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
      pocs5d(freqslice,freqslice2,wd,nx1fft,nx2fft,nx3fft,nx4fft,nk,iter,perci,percf,alphai,alphaf,bandi,bandf,k_n_1,k_n_2,k_n_3,k_n_4);

    ix        = 0;
    ix_no_pad = 0;
    for (ix1=0;ix1<nx1fft;ix1++){
      for (ix2=0;ix2<nx2fft;ix2++){
    	for (ix3=0;ix3<nx3fft;ix3++){
    	  for (ix4=0;ix4<nx4fft;ix4++){
    	    if (ix1<nx1 && ix2<nx2 && ix3<nx3 && ix4<nx4){
    	      cpfft[ix_no_pad][iw] = freqslice2[ix];
    	      ix_no_pad++;
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

  free1complex(freqslice2);
  free1complex(freqslice);
  free1float(wd);
  /* algorithm ends */


  /******************************************************************************************** FX to TX
  transform data from w-x to t-x using IFFTW*/
  N = ntfft; 
  out2 = ealloc1float(ntfft);
  in2 = ealloc1complex(N);
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
  fprintf(stderr,"\n");

  fftwf_destroy_plan(p4);
  fftwf_free(in2); fftwf_free(out2);
  /********************************************************************************************/
  /* Fourier transform w to t */

  for (ix=0;ix<nx1*nx2*nx3*nx4;ix++) for (it=0; it<nt; it++) dataout[ix][it]=pfft[ix][it]/ntfft;
  
  free2float(pfft);
  free2complex(cpfft);
  return;

}

void pocs5d(complex *freqslice,complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,int Iter,float perci,float percf,float alphai,float alphaf,float bandi,float bandf,float *k_n_1,float *k_n_2,float *k_n_3,float *k_n_4)
{  

  complex czero;
  int ix;
  float *mabs;  
  float *mabsiter;
  float sigma;
  float alpha;
  float band;
  int rank;
  int *n;
  int count;
  int iter;
  fftwf_plan p2;
  fftwf_plan p3;

  czero.r=czero.i=0;
  mabs = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);  
  mabsiter = ealloc1float(nx1fft*nx2fft*nx3fft*nx4fft);
  /******************************************************************************************** FX1X2 to FK1K2
  make the plan that will be used for each frequency slice
  written as a 4D transform with length =1 for two of the dimensions. 
  This is to make it easier to upgrade to reconstruction of 4 spatial dimensions. */
  rank = 4;
  n = ealloc1int(4);
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
  for (ix=0;ix<nk;ix++) mabs[ix]=rcabs(freqslice2[ix]);
  fftwf_execute(p3); /* FFT k to x */
  
  for (ix=0;ix<nk;ix++) freqslice2[ix]=crmul(freqslice2[ix],1/(float) nk);
  for (iter=1;iter<Iter;iter++){  /* loop for thresholding */
    fftwf_execute(p2); /* FFT x to k */
    
    count = 0;
    
    /* This is to increase the thresholding within each internal iteration */
    sigma=quest(perci - (iter-1)*((perci-percf)/(Iter-1)),nk,mabs);
    
    /* This is to increase alpha at each iteration */
    alpha=alphai + (iter-1)*((alphaf-alphai)/(Iter-1));
    
    /* This is to increase band at each iteration */
    band=bandi + (iter-1)*((bandf-bandi)/(Iter-1));
    
    for (ix=0;ix<nk;ix++) mabsiter[ix]=rcabs(freqslice2[ix]);
    for (ix=0;ix<nk;ix++){
      /* thresholding */
      if (mabsiter[ix]<sigma) freqslice2[ix] = czero;
      /* band limitation */
      if ((k_n_1[ix] > band) || (k_n_2[ix] > band) || (k_n_3[ix] > band) || (k_n_4[ix] > band)) freqslice2[ix] = czero;
      else{ count++; }
    }
    
    fftwf_execute(p3); /* FFT k to x */
    
    for (ix=0;ix<nk;ix++) freqslice2[ix]=crmul(freqslice2[ix],1/((float) nx1fft*nx2fft*nx3fft*nx4fft));
    
	/* reinsertion into original data */
    for (ix=0;ix<nk;ix++) freqslice2[ix]=cadd(crmul(freqslice[ix],alpha),crmul(freqslice2[ix],1-alpha*wd[ix])); /* x,w */
    
  }
  
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  
  return;

}




