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
	      int verbose,int nt,int nx,float dt,
          int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int iter,float alphai,float alphaf,float fmax);
void pocs5d(sf_complex *freqslice,sf_complex *freqslice2,float *wd,int nx1fft,int nx2fft,int nx3fft,int nx4fft,int nk,float *k1,float *k2,float *k3,float *k4,int Iter,float perci,float percf,float alphai,float alphaf);
void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           float fa,float fb,float fc,float fd,
                           int axis);
void radial_filter(float **d,float ot, float dt, int nt,float ox, float dx, int nx,float fa,float fb,float fc,float fd);
void radial_op(float **d,float **m,int nt,int nx,int ntp,int np,float op,float dp,bool adj);
void mean_filter(float *trace, int nt, int ntw, int nrepeat);
void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile);

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
    float alphai, alphaf, fmax;
    int sum_wd;
    bool verbose;
 
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

    if (!sf_getint("niter",&niter)) niter = 10; /* number of iterations */
    if (!sf_getfloat("alphai",&alphai)) alphai = 1; /* denoising parameter for 1st iteration 1=no denoise */
    if (!sf_getfloat("alphaf",&alphaf)) alphaf = 1; /* denoising parameter for last iteration 1=no denoise */
    if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity 0=quiet 1=loud */
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

    pocs(d,
         verbose,n1,nx,d1,
         n2,n3,n4,n5,
         wd,niter,alphai,alphaf,fmax);

    for (ix=0; ix<nx; ix++) {
      for (i1=0; i1<n1; i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }

    exit (0);
}


void pocs(float **d,
	      int verbose,int nt,int nx,float dt,
          int nx1,int nx2,int nx3,int nx4,
          float *wd_no_pad,int niter,float alphai,float alphaf,float fmax)
{  
  int it,ix,iw,ntfft,nx1fft,nx2fft,nx3fft,nx4fft,nw,nk,if_low,if_high,padfactor,ix_no_pad,ix1,ix2,ix3,ix4,nclip,iter,*n,ix1_shift,ix2_shift,ix3_shift,ix4_shift,ix_shift;
  float perci,percf,*wd,**M,**Mshift,*thres,*in1,*out2,f_low,f_high,alpha,pclip,*amp,*trace;
  sf_complex czero,*freqslice,*out1,*in2,**D,**Dobs;
  fftwf_plan p1,p2,p3,p4; 
  
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

/*
  fprintf(stderr,"nt=%d\n",nt);
  fprintf(stderr,"nx1=%d\n",nx1);
  fprintf(stderr,"nx2=%d\n",nx2);
  fprintf(stderr,"nx3=%d\n",nx3);
  fprintf(stderr,"nx4=%d\n",nx4);
  fprintf(stderr,"ntfft=%d\n",ntfft);
  fprintf(stderr,"nx1fft=%d\n",nx1fft);
  fprintf(stderr,"nx2fft=%d\n",nx2fft);
  fprintf(stderr,"nx3fft=%d\n",nx3fft);
  fprintf(stderr,"nx4fft=%d\n",nx4fft);
  fprintf(stderr,"nx=%d\n",nx);
  fprintf(stderr,"nw=%d\n",nw);
  fprintf(stderr,"nk=%d\n",nk);
*/

  wd = sf_floatalloc(nk);
  D    = sf_complexalloc2(nw,nk);
  Dobs = sf_complexalloc2(nw,nk);
  M    = sf_floatalloc2(nw,nk);
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
  for (iw=0;iw<nw;iw++){  
    for (ix=0;ix<nk;ix++) freqslice[ix] = D[ix][iw];
    fftwf_execute(p2);
    for (ix=0;ix<nk;ix++) amp[ix*nw + iw] = sf_cabs(freqslice[ix]);
  }

  // build iteration dependent threshold schedule
  thres = sf_floatalloc(niter);
  for (iter=0;iter<niter;iter++){
    pclip = 100*(perci - (iter-1)*((perci-percf)/(niter-1)));
    nclip = SF_MAX(SF_MIN(nw*nk*pclip/100. + .5,nw*nk-1),0);
    thres[iter]=sf_quantile(nclip,nw*nk,amp);
    //fprintf(stderr,"thres[%d]=%f\n",iter,thres[iter]);
  }

  trace = sf_floatalloc(nw); 

  for (iter=0;iter<niter;iter++){
    for (ix=0;ix<nk;ix++) for (iw=0;iw<nw;iw++) M[ix][iw] = 1.0;
    // *****************************************************************************
    // transform D from w-x to w-k (first re-zero the zero pad regions of the array)
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
    // *****************************************************************************
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) if (sf_cabs(D[ix][iw])<thres[iter]) M[ix][iw] = 0.0;
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) D[ix][iw] = D[ix][iw]*M[ix][iw]; 
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) M[ix][iw] = cabsf(D[ix][iw]);
    for (iw=0;iw<if_low;iw++)   for (ix=0;ix<nk;ix++) M[ix][iw] = 0.0;
    for (iw=if_high;iw<nw;iw++) for (ix=0;ix<nk;ix++) M[ix][iw] = 0.0;

    // smooth M along the frequency axis
    for (ix=0;ix<nk;ix++){
      for (iw=0;iw<nw;iw++) trace[iw] = M[ix][iw];
      mean_filter(trace,nw,5,3);
      for (iw=0;iw<nw;iw++) M[ix][iw] = trace[iw];    
    }

    // replace original values
    for (iw=if_low;iw<if_high;iw++) for (ix=0;ix<nk;ix++) if (sf_cabs(D[ix][iw])>=thres[iter]) M[ix][iw] = cabsf(D[ix][iw]);



/*
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

//if (iter==0){    
//sf_file outtmp1;
//char tmpname1[256];
//sprintf(tmpname1, "01_fk_raw.rsf");
//outtmp1 = sf_output(tmpname1);  
//write5d(Mshift,
//        nw,0,1,"w","index",  
//        nx1fft,0,1,"k1","Index",
//        nx2fft,0,1,"k2","Index",
//        nx3fft,0,1,"k3","Index",
//        nx4fft,0,1,"k4","Index",
//        "F-K Unfiltered",outtmp1);
//sf_fileclose(outtmp1);
//}
    
    radial_filter_gathers(Mshift,0,1,nw,0,1,nx1fft,0,1,nx2fft,0,1,nx3fft,0,1,nx4fft,0,0,0.05,0.15,2);

//if (iter==0){    
//sf_file outtmp2;
//char tmpname2[256];
//sprintf(tmpname2, "02_fk_filtered.rsf");
//outtmp2 = sf_output(tmpname2);  
//write5d(Mshift,
//        nw,0,1,"w","index",  
//        nx1fft,0,1,"k1","Index",
//        nx2fft,0,1,"k2","Index",
//        nx3fft,0,1,"k3","Index",
//        nx4fft,0,1,"k4","Index",
//        "F-K Filtered",outtmp2);
//sf_fileclose(outtmp2);
//}

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
*/

    for (iw=if_low;iw<if_high;iw++){ for (ix=0;ix<nk;ix++){ 
      __real__ D[ix][iw] = M[ix][iw]*cosf(cargf(D[ix][iw]));
      __imag__ D[ix][iw] = M[ix][iw]*sinf(cargf(D[ix][iw]));
    }}
    
    // *****************************************************************************
    // transform D from w-k to w-x (afterwards re-zero the zero pad regions of the array)
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
    alpha=alphai + (iter-1)*((alphaf-alphai)/(niter-1));
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

  free1float(trace);
  free1float(wd);
  free1float(amp);
  free2float(M);
  free2float(Mshift);
  free1float(thres);
  free1float(in1);
  free1float(out2);
  free1complex(out1);
  free1complex(in2);
  free2complex(D);
  free2complex(Dobs);
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_destroy_plan(p3);
  fftwf_destroy_plan(p4);

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
  int np,ip,it,ix,ntp;
  float op,dp,p;
  float *trace,**m;
  float dmax1,dmax2;
//  sf_file outtmp;
//  char tmpname[256];  
//  sprintf(tmpname, "tmp_radial_transform.rsf");
//  outtmp = sf_output(tmpname);

  np=1801;
  op=-90;
  dp=0.1;
  ntp = truncf(1.4142*nt);

  m = sf_floatalloc2(ntp,np);
  trace = sf_floatalloc(ntp);  
  radial_op(d,m,nt,nx,ntp,np,op,dp,true); 
  //dmax1 = 0.0;
  //for (ix=0;ix<nx;ix++) for (it=0;it<nt;it++) if (fabsf(d[ix][it]) > dmax1) dmax1 = fabsf(d[ix][it]);

  for (ip=0;ip<np;ip++){
    p = ip*dp + op;
    if (fabsf(p)>60){
      for (it=0;it<nt;it++) m[ip][it] *= cosf(fabsf(p)*PI/180)/cosf(60*PI/180);
    }
  }

/*
  write5d(m,
          ntp,ot,dt,"Time","s",  
          np,op,dp,"Angle","Degrees",
          1,0,1," "," ",
          1,0,1," "," ",
          1,0,1," "," ",
          "Radial Domain",outtmp);
  sf_fileclose(outtmp); 
*/

  for (ip=0;ip<np;ip++){
    for (it=0;it<ntp;it++) trace[it] = m[ip][it];
    mean_filter(trace,ntp,10,2);
    for (it=0;it<ntp;it++) m[ip][it] = trace[it];    
  }

  radial_op(d,m,nt,nx,ntp,np,op,dp,false); 
  //for (ix=0;ix<nx;ix++) for (it=0;it<nt;it++) if (d[ix][it] > 1.0) d[ix][it] = 1.0;
  //for (ix=0;ix<nx;ix++) for (it=0;it<nt;it++) if (d[ix][it] < 1.0) d[ix][it] = 1.0;
  
  free1float(trace);
  free2float(m);
  return;
}

void radial_op(float **d,float **m,int nt,int nx,int ntp,int np,float op,float dp,bool adj)
{
  int ip,it,ix,itp;
  float ox,dx,ot,dt,p,t_floor,x_floor,x,t,tp,t1,t2,x1,x2,w1,w2,w3,w4,norm;
  
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
  
  for (itp=0;itp<ntp;itp++){ 
    for (ip=0;ip<np;ip++){
      tp = itp*dt + ot;
      p = ip*dp + op;
      t = tp*cosf(p*PI/180);
      it = (int) truncf((t - ot)/dt);
      t1 = it*dt + ot;
      t2 = t1 + dt;
      x = tp*sinf(p*PI/180);
      ix = (int) truncf((x - ox)/dx);
      x1 = ix*dx + ox;
      x2 = x1 + dx;
      w1 = (t2-t)*(x2-x)/((t2-t1)*(x2-x1));
      w2 = (t-t1)*(x2-x)/((t2-t1)*(x2-x1));
      w3 = (t2-t)*(x-x1)/((t2-t1)*(x2-x1));
      w4 = (t-t1)*(x-x1)/((t2-t1)*(x2-x1));
      // bilinear interpolation into bins
      if (it >= 0 && it+1 < nt && ix >= 0 && ix+1 < nx && x>-1 && x<1 && t>0 && t<ntp*dt+ot){
      //if (w1<0 || w1>1 || w2<0 || w2>1 || w3<0 || w3>1 || w4<0 || w4>1) fprintf(stderr,"w1=%f w2=%f w3=%f w4=%f t=%f t1=%f t2=%f x=%f x1=%f x2=%f\n",w1,w2,w3,w4,t,t1,t2,x,x1,x2);
        if (adj){
          m[ip][itp] += w1*d[ix][it] + w2*d[ix][it+1] + w3*d[ix+1][it] + w4*d[ix+1][it+1];
        }
        else{
          d[ix][it]     += w1*m[ip][itp]*0.1;
          d[ix][it+1]   += w2*m[ip][itp]*0.1;
          d[ix+1][it]   += w3*m[ip][itp]*0.1;
          d[ix+1][it+1] += w4*m[ip][itp]*0.1;
	    }
	  }
    }
  }
  return;
}

void mean_filter(float *trace, int nt, int ntw, int nrepeat)
/*< mean filter in 1d. ntw is window length and nrepeat allows to repeat the smoothing >*/
{
  int irepeat,it,itw,index1;
  float sum,nsum;

  /* calculate mean value of a window and assign it to the central index */
    for (irepeat=0;irepeat<nrepeat;irepeat++){
      for (it=0;it<nt;it++){
        sum  = 0.0;
        nsum = 0.0;
        for (itw=0;itw<ntw;itw++){
          index1 = it - (int) truncf(ntw/2) + itw;
          if (index1>=0 && index1<nt){
            sum += trace[index1];
            nsum += 1.0;
          }
        }
        trace[it] = sum/nsum;
      }
    }

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







