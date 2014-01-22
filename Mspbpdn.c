/* static preserving basis persuit denoising 
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

#include <rsf.h>
#include <fftw3.h>
#include "myfree.h"

#ifndef PI
#define PI (3.141592653589793)
#endif

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif 

void my_op(sf_complex **m,float **d,int nw,int nk,int nt,float dt,int nx,float dx,float ox,int np,float dp,float op,float fmin,float fmax,bool adj,int operator);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
void radon_op(float **m,float **d,int ntau,int np,float dp,float op,int nt,float dt,int nx,float dx,float ox,float fmin,float fmax,bool adj);
float find_lag(float *x, float *y, int n, float dt, float maxdelay);
void time_shift(float *d,float dt,int nt,float tshift);
void mysoft(sf_complex **m,int nw,int nk,float alpha);
float power_method(int nw, int nk,int nt,float dt,int nx,float dx,float ox,int np, float dp, float op,float fmin, float fmax, int operator, int itmax_power);

int main(int argc, char* argv[])
{
  int ix,it,nt,nx,iw,ik;
  int n1,n2;
  int mode;
  float *trace,*trace1,*trace2;
  float **d;
  sf_complex **m;
  float d1,o1,d2,o2;
  int padt,padx;
  int ntfft,nw,nk;
  int iter,Niter; 
  float lag;
  float *tshift;
  float **s,**r;
  sf_complex **g;
  float alpha,lambda;
  float maxlag;
  float maxeig,t;
  int Niterpower;
  sf_file in,out,costfile;
  float *cost,costsum1,costsum2;
  char *costname;
  bool verbose;
  float p;
  float dt,dx,ox,dp,op,fmin,fmax,pmin,pmax;
  int operator,np,ip;
  bool debug;
  float **tm1,**tm2,**td1,**td2;
  float tmp_sum1,tmp_sum2;
  float l2norm_in,l2norm_out;
  bool powermethod;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  if (!sf_getfloat("maxlag",&maxlag)) maxlag = 0.02; /* maximum static shift in the data (seconds) */
  if (!sf_getint("iter",&Niter)) Niter = 20; /* number of iterations for ISTA*/
  if (!sf_getint("iterpower",&Niterpower)) Niterpower = 20; /* number of iterations for power method used to approximate max eigenvalue (used in ISTA) */
  if (!sf_getfloat("lambda",&lambda)) lambda = 0.2; /* hyperparameter for the inversion */
  if (!sf_getint("mode",&mode)) mode = 1; /* flag for what to do with statics on output: 1=denoise only (leave statics in data), 2=denoise and correct, or 3=static correct only 0=denoise ignoring statics */
  if (!sf_getint("operator",&operator)) operator = 1; /* flag for linear operator to be used 1= FK, 2=Linear Radon*/
  if (!sf_getfloat("pmin",&pmin)) pmin = -0.001; /* slowness min */
  if (!sf_getfloat("pmax",&pmax)) pmax = 0.001; /* slowness max */
  if (!sf_getint("np",&np)) np = 101; /* slowness length, should be odd number to ensure it is 0 centered.*/
  op = pmin;
  dp = (pmax-pmin)/((float) np-1);
  if (!sf_getbool("debug",&debug)) debug = false; /* debugging flag */
  if (!sf_getbool("powermethod",&powermethod)) powermethod = false; /* if not using FK then should do powermethod to find max eig value of operator  */
  if (!sf_getfloat("maxeig",&maxeig)) maxeig = 1; /* max eigenvalue (note for orthog op maxeig=1) */

  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag */
    costname = sf_getstring("cost");
  /* read input file parameters */
  if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;

  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/d1; /* max frequency to process */
  if (fmax > 0.5/d1) fmax = 0.5/d1;
  if (!sf_getfloat("fmin",&fmin)) fmin = 0.1; /* min frequency to process */
  dt = d1;
  dx = d2;
  ox = o2;
  nx = n2;
  nt = n1;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nx;

  costname = sf_getstring("cost");
  costfile = sf_output(costname);
  sf_putint(costfile,"n1",Niter);
  sf_putfloat(costfile,"d1",1);
  sf_putfloat(costfile,"o1",1);
  sf_putstring(costfile,"label1","Iteration Number");
  sf_putstring(costfile,"label2","cost");
  sf_putstring(costfile,"unit1"," ");
  sf_putstring(costfile,"unit2"," ");
  sf_putstring(costfile,"title","Cost");
  sf_putint(costfile,"n2",1);
  sf_putint(costfile,"n3",1);
  sf_putint(costfile,"n4",1);
  sf_putint(costfile,"n5",1);
  cost = sf_floatalloc(Niter);

  trace = sf_floatalloc (n1);
  trace1 = sf_floatalloc (n1);
  trace2 = sf_floatalloc (n1);
  d = sf_floatalloc2(nt,nx);
  if (operator==1){
    m = sf_complexalloc2(nw,nk);
    g = sf_complexalloc2(nw,nk);
  }
  else{
    m = sf_complexalloc2(nt,np);
    g = sf_complexalloc2(nt,np);
  }
  tshift = sf_floatalloc (nx);
  s = sf_floatalloc2(nt,nx);
  r = sf_floatalloc2(nt,nx);

  l2norm_in = 0.0;
  for (ix=0; ix<n2; ix++) {	
    sf_floatread(trace,n1,in);
    for (it=0; it<n1; it++){ 
      d[ix][it] = trace[it];
      l2norm_in = l2norm_in + trace[it]*trace[it];
    }
    tshift[ix] = 0.0;
  }
  l2norm_in = sqrt(l2norm_in);

  for (ix=0; ix<n2; ix++) {	
    for (it=0; it<n1; it++) d[ix][it] = d[ix][it];
  }

 
  if (operator==2) nk=np;

  my_op(m,d,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,1,operator); /* adjoint: d to m */ 
  my_op(m,s,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,0,operator); /* forward: m to s */ 
/*
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        __real__ m[ik][iw] = 0.0;
        __imag__ m[ik][iw] = 0.0;
      }
    }
*/
  p = 0.0;

  if (debug){ 
    /* for testing, just output the adjoint radon model to ensure it is ok */
    if (operator==1){
      sf_putfloat(out,"o2",-PI/dx);
      sf_putfloat(out,"d2",2*PI/nk/dx);
      sf_putint(out,"n2",nk);
      sf_putstring(out,"label1","F");
      sf_putstring(out,"label2","K");
      sf_putstring(out,"unit1","1/s");
      sf_putstring(out,"unit2","1/m");
      sf_putstring(out,"title","FK Panel");
      for (ik=nk/2; ik<nk; ik++) {	
        for (iw=0; iw<nw; iw++) trace[iw] = cabsf(m[ik][iw]);
        sf_floatwrite(trace,nw,out);
      }
      for (ik=0; ik<nk/2; ik++) {	
        for (iw=0; iw<nw; iw++) trace[iw] = cabsf(m[ik][iw]);
        sf_floatwrite(trace,nw,out);
      }
    }
    else{

      init_genrand((unsigned long) 1); /* initialize with seed */

      td1 = sf_floatalloc2(nt,nx);
      td2 = sf_floatalloc2(nt,nx);
      tm1 = sf_floatalloc2(nt,np);
      tm2 = sf_floatalloc2(nt,np);
      for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++){
        td2[ix][it] = genrand_real1();
      }
      }
      for (ip=0;ip<np;ip++){
      for (it=0;it<nt;it++){
        tm1[ip][it] = genrand_real1();
      }
      }
      radon_op(tm1,td1,nt,np,dp,op,nt,dt,nx,dx,ox,fmin,fmax,0); 
      radon_op(tm2,td2,nt,np,dp,op,nt,dt,nx,dx,ox,fmin,fmax,1); 
      tmp_sum1=0;
      for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++){
        tmp_sum1 += td1[ix][it]*td2[ix][it];
      }
      }
      tmp_sum2=0;
      for (ip=0;ip<np;ip++){
      for (it=0;it<nt;it++){
        tmp_sum2 += tm1[ip][it]*tm2[ip][it];
      }
      }
      fprintf(stderr,"dot product check if match: %f and %f\n",tmp_sum1,tmp_sum2);

      sf_putfloat(out,"o2",pmin);
      sf_putfloat(out,"d2",dp);
      sf_putint(out,"n2",np);
      sf_putstring(out,"label1","Time");
      sf_putstring(out,"label2","Slowness");
      sf_putstring(out,"unit1","s");
      sf_putstring(out,"unit2","s/m");
      sf_putstring(out,"title","Radon Panel");
      for (ip=0; ip<np; ip++) {	
        for (it=0; it<nt; it++) trace[it] = crealf(m[ip][it]);
        sf_floatwrite(trace,nt,out);
      }
    }
    exit (0);
  }

  /*
  for (ik=0;ik<nk;ik++){
    for (iw=0;iw<nw;iw++){
      if (sf_cabs(m[ik][iw]) > p) p = cabsf(m[ik][iw]);
    }
  }
  lambda = lambda*p;
  */

  if (operator>1 && powermethod){
    /* if operator=1 (FK) the max eigenvalue is 1 (it is orthogonal), 
    otherwise use power method to find the max. */
    maxeig = power_method(nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,operator,Niterpower);
    if (verbose) fprintf(stderr,"maxeig after %d iterations of the power method = %f\n",Niterpower,maxeig);
    if (verbose) fprintf(stderr,"re-run the program setting maxeig=%f\n",maxeig);
    exit (0);    
  }

  t = 1/(2*1.2*maxeig);
  alpha = lambda*t;
  if (verbose) fprintf(stderr,"alpha=%f t=%f \n",alpha,t);

  for (iter=1;iter<=Niter;iter++){

/* ----------------------------------------------------------- */
    if (mode!=0){
      /* add static shifts to s */
      for (ix=0;ix<nx;ix++){
        for (it=0;it<nt;it++) trace1[it] = s[ix][it];
        time_shift(trace1,d1,nt,tshift[ix]);
        for (it=0;it<nt;it++) s[ix][it] = trace1[it];
      }
    }
/* ----------------------------------------------------------- */

    for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++) r[ix][it] = s[ix][it] - d[ix][it];
    }
    /* calculate cost */
    costsum1 = 0.0; 
    for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++) costsum1 += r[ix][it]*r[ix][it];
    }
    costsum2 = 0.0; 
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++) costsum2 += sf_cabs(m[ik][iw]);
    }
    cost[iter-1] = costsum1 + lambda*costsum2;

    if (verbose) fprintf(stderr,"iter=%d/%d: cost=%f\n",iter,Niter,cost[iter-1]);

/* ----------------------------------------------------------- */
    if (mode!=0){
      /* remove static shifts from r */
      for (ix=0;ix<nx;ix++){
        for (it=0;it<nt;it++) trace1[it] = r[ix][it];
        time_shift(trace1,d1,nt,-tshift[ix]);
        for (it=0;it<nt;it++) r[ix][it] = trace1[it];
      }
    }
/* ----------------------------------------------------------- */
    my_op(g,r,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,1,operator); /* adjoint: r to g */
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        m[ik][iw] = m[ik][iw] - 2*t*g[ik][iw];
      }
    }
    mysoft(m,nw,nk,alpha);
    my_op(m,s,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,0,operator); /* forward: m to s */

/* ----------------------------------------------------------- */
    /* calculate time shift */
    for (ix=0;ix<nx;ix++){
      for (it=0;it<nt;it++) trace1[it] = s[ix][it];
      for (it=0;it<nt;it++) trace2[it] = d[ix][it];
      lag = find_lag(trace1,trace2,nt,d1,maxlag);
      tshift[ix] = lag;
    }
/* ----------------------------------------------------------- */
  }

  if (mode!=3) my_op(m,d,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,0,operator); /* forward: m to d */

  l2norm_out = 0.0;
  for (ix=0; ix<nx; ix++) {	
    for (it=0; it<nt; it++){ 
      l2norm_out = l2norm_out + d[ix][it]*d[ix][it];
    }
  }
  l2norm_out = sqrt(l2norm_out);


  for (ix=0; ix<nx; ix++) {	
    for (it=0; it<nt; it++) trace[it] = d[ix][it]*l2norm_in/l2norm_out;
    if (mode==1 || mode==4) time_shift(trace,d1,nt,tshift[ix]);
    if (mode==3) time_shift(trace,d1,nt,-tshift[ix]);
    sf_floatwrite(trace,n1,out);
  }

  sf_floatwrite(cost,Niter,costfile);

  free1float(trace);
  free1float(trace1);
  free1float(trace2);
  free2float(d);
  free2complex(m);
  free1float(tshift);
  free2complex(g);
  free2float(s);
  free2float(r);

  exit (0);
}

void my_op(sf_complex **m,float **d,int nw,int nk,int nt,float dt,int nx,float dx,float ox,int np,float dp,float op,float fmin,float fmax,bool adj,int operator)
{
  int ip,it;
  float **mradon;
  mradon = sf_floatalloc2(nt,np);

  if (operator==1){
    fk_op(m,d,nw,nk,nt,nx,adj);
  }
  else{
    if (!adj){
      for (ip=0;ip<np;ip++){
        for(it=0;it<nt;it++){ 
          mradon[ip][it] = crealf(m[ip][it]);
        }
      }
    }
    radon_op(mradon,d,nt,np,dp,op,nt,dt,nx,dx,ox,fmin,fmax,adj);   
    if (adj){
      for (ip=0;ip<np;ip++){
        for(it=0;it<nt;it++){ 
          __real__ m[ip][it] = mradon[ip][it];
          __imag__ m[ip][it] = 0.0;
        }
      }
    }
  }

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

void radon_op(float **m,float **d,int ntau,int np,float dp,float op,int nt,float dt,int nx,float dx,float ox,float fmin,float fmax,bool adj)
{
  int padfactor,nw;
  sf_complex **df,**mf,**cpfft,*out1a,*in1b,czero,L;
  float *in1a,*out1b;
  int ntfft,ix,it,iw;
  fftwf_plan p1a,p1b;
  float f_low,f_high,omega,p,x;
  int if_low,if_high,ip;

  if (adj){
    padfactor = 2;
    nw = nt*padfactor;
    ntfft = (nw-1)*2;
    __real__ czero = 0;
    __imag__ czero = 0;
    cpfft = sf_complexalloc2(nw,nx);
    df = sf_complexalloc2(nw,nx);
    mf = sf_complexalloc2(nw,np);
    out1a = sf_complexalloc(nw);
    in1a = sf_floatalloc(ntfft);
    p1a = fftwf_plan_dft_r2c_1d(ntfft, in1a, (fftwf_complex*)out1a, FFTW_ESTIMATE);
    out1b = sf_floatalloc(ntfft);
    in1b = sf_complexalloc(ntfft);
    p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
    
    for (ix=0;ix<nx;ix++){
      for(it=0;it<nt;it++) in1a[it] = d[ix][it];
      for(it=nt;it<ntfft;it++) in1a[it] = 0;
      fftwf_execute(p1a); 
      for(iw=0;iw<nw;iw++) cpfft[ix][iw] = out1a[iw]; 
    }
    fftwf_destroy_plan(p1a);
    fftwf_free(in1a); fftwf_free(out1a);
    
    f_low = fmin;   /* min frequency to process */
    f_high = fmax;  /* max frequency to process */

    if(f_low>0)if_low = trunc(f_low*dt*ntfft);
    else if_low = 0;
    if(f_high*dt*ntfft+1<nw) if_high = trunc(f_high*dt*ntfft)+1;
    else if_high = nw;
    /* process frequency slices */
    for (iw=if_low;iw<if_high;iw++){
      omega = 2*PI*iw/ntfft/dt;
      for (ip=0;ip<np;ip++){
	mf[ip][iw] = czero;
	p = op + ip*dp;
	for (ix=0;ix<nx;ix++){ 
	  x = ox + ix*dx;
	  __real__ L = cos(omega*x*p);
	  __imag__ L = -sin(omega*x*p);
          /*fprintf(stderr,"real(L)=%f imag(L)=%f omega=%f x=%f p=%f ox=%f ix=%d dx=%f op=%f ip=%d dp=%f \n",crealf(L),cimagf(L),omega,x,p,ox,ix,dx,op,ip,dp);*/
	  mf[ip][iw] = mf[ip][iw] + cpfft[ix][iw]*L;
	}
      }   
    }
    for (ip=0;ip<np;ip++){
      for(iw=0;iw<nw;iw++) in1b[iw] = mf[ip][iw];
      for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
      fftwf_execute(p1b); 
      for(it=0;it<nt;it++) m[ip][it] = out1b[it]/ntfft; 
    }
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }
  
  else{
    padfactor = 2;
    nw = nt*padfactor;
    ntfft = (nw-1)*2;
    __real__ czero = 0;
    __imag__ czero = 0;
    cpfft = sf_complexalloc2(nw,np);
    df = sf_complexalloc2(nw,nx);
    mf = sf_complexalloc2(nw,np);
    out1a = sf_complexalloc(nw);
    in1a = sf_floatalloc(ntfft);
    p1a = fftwf_plan_dft_r2c_1d(ntfft, in1a, (fftwf_complex*)out1a, FFTW_ESTIMATE);
    out1b = sf_floatalloc(ntfft);
    in1b = sf_complexalloc(ntfft);
    p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
    for (ip=0;ip<np;ip++){
      for(it=0;it<nt;it++) in1a[it] = m[ip][it];
      for(it=nt;it<ntfft;it++) in1a[it] = 0;
      fftwf_execute(p1a); 
      for(iw=0;iw<nw;iw++) cpfft[ip][iw] = out1a[iw]; 
    }
    fftwf_destroy_plan(p1a);
    fftwf_free(in1a); fftwf_free(out1a);
    f_low = 0.1;   /* min frequency to process */
    f_high = fmax; /* max frequency to process */
    if(f_low>0)if_low = trunc(f_low*dt*ntfft);
    else if_low = 0;
    if(f_high*dt*ntfft+1<nw) if_high = trunc(f_high*dt*ntfft)+1;
    else if_high = nw;
    /* process frequency slices */
    for (iw=if_low;iw<if_high;iw++){
      omega = 2*PI*iw/ntfft/dt;
      for (ix=0;ix<nx;ix++){
	df[ix][iw] = czero;
	x = ox + ix*dx;
	for (ip=0;ip<np;ip++){ 
	  p = op + ip*dp;
	  __real__ L = cos(omega*x*p);
	  __imag__ L = sin(omega*x*p);
	  df[ix][iw] = df[ix][iw] + cpfft[ip][iw]*L;
	}
      }   
    }
    for (ix=0;ix<nx;ix++){
      for(iw=0;iw<nw;iw++) in1b[iw] = df[ix][iw];
      for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
      fftwf_execute(p1b); 
      for(it=0;it<nt;it++) d[ix][it] = out1b[it]/ntfft; 
    }
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }
  return;

}

float find_lag(float *x, float *y, int n, float dt, float maxdelay)
{
  int i,j;
  float mx,my,sx,sy,sxy,denom,r;
  int delay,rmax_delay,imaxdelay;
  float rmax;
  rmax = 0;
  rmax_delay = 0;
  mx = 0;
  my = 0;   
  for (i=0;i<n;i++) {
    mx += x[i];
    my += y[i];
  }
  mx /= n;
  my /= n;
  sx = 0;
  sy = 0;
  for (i=0;i<n;i++) {
    sx += (x[i] - mx) * (x[i] - mx);
    sy += (y[i] - my) * (y[i] - my);
  }
  denom = sqrt(sx*sy);

  imaxdelay = (int) trunc(maxdelay/dt);    

  for (delay=-imaxdelay;delay<imaxdelay;delay++) {
    sxy = 0;
    for (i=0;i<n;i++) {
      j = i + delay;
      if (j < 0 || j >= n)
        continue;
      else
        sxy += (x[i] - mx) * (y[j] - my);
    }
    r = sxy / denom; /* correlation coeff at delay */
    if (fabsf(r) > rmax){ 
      rmax = fabsf(r);
      rmax_delay = delay;
    }
  }

  return dt*rmax_delay;
}

void time_shift(float *d,float dt,int nt,float tshift)
/* apply a time shift to one trace */
{
  float *in1, *out2, omega;
  sf_complex *out1, *in2, shift_op, *D;
  fftwf_plan p1, p2;
  int padfactor,ntfft,nw,it,iw,ifmin,ifmax;

  padfactor = 2;
  ntfft = padfactor*nt;
  nw = (int) ntfft/2 + 1; 
  D = sf_complexalloc(ntfft);
  ifmin = 0;
  ifmax = nw;
  in1 = sf_floatalloc(ntfft);
  out1 = sf_complexalloc(nw);
  p1 = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  for(it=0;it<nt;it++)     in1[it] = d[it];
  for(it=nt;it<ntfft;it++) in1[it] = 0;
  fftwf_execute(p1);
  for(iw=0;iw<nw;iw++) D[iw] = out1[iw]; 
  for (iw=ifmin;iw<ifmax;iw++){
    omega = (float) 2*PI*iw/ntfft/dt;
    __real__ shift_op = cos(omega*tshift);
    __imag__ shift_op =-sin(omega*tshift);
    D[iw] = D[iw]*shift_op;
  } 
  in2 = sf_complexalloc(ntfft);
  out2 = sf_floatalloc(ntfft);
  p2 = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for(iw=0;iw<nw;iw++){
    in2[iw] = D[iw];
  }
  fftwf_execute(p2);
  for(it=0;it<nt;it++){
    d[it] = out2[it]; 
  }
  for (it=0; it<nt; it++) d[it]=d[it]/ntfft;
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);

  fftwf_free(in1);
  fftwf_free(in2);
  fftwf_free(out1);
  fftwf_free(out2);
  fftwf_free(D);

  return;
}

void mysoft(sf_complex **m,int nw,int nk,float alpha)
{
  int ik,iw;
  float a,ang;

  for (ik=0;ik<nk;ik++){
    for (iw=0;iw<nw;iw++){
      a = sf_cabs(m[ik][iw]) - alpha;
      ang = cargf(m[ik][iw]);
      /*fprintf(stderr,"ang=%f\n",ang);*/
      if (a<0.0) a = 0.0;
      __real__ m[ik][iw] = a*cos(ang);
      __imag__ m[ik][iw] = a*sin(ang);
    }
  }

  return;
}

float power_method(int nw, int nk,int nt,float dt,int nx,float dx,float ox,int np, float dp, float op,float fmin, float fmax, int operator, int itmax_power)
{
  float a,b,maxeig;
  int ik,iw,iter;
  sf_complex **x,**ATAx;
  float **Ax; 
  sf_complex sum;

  if (operator>1) nk = np; nw = nt;

  maxeig = 0.0;
  x = sf_complexalloc2(nw,nk);
  ATAx = sf_complexalloc2(nw,nk);
  Ax = sf_floatalloc2(nt,nx);

  init_genrand((unsigned long) 1); /* initialize with seed */

  a = -1; b = 1;
  if (operator==1){
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        __real__ x[ik][iw] = a + (b-a)*genrand_real1();
        __imag__ x[ik][iw] = a + (b-a)*genrand_real1();
      }
    }
  }
  else{
     for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        __real__ x[ik][iw] = a + (b-a)*genrand_real1();
        __imag__ x[ik][iw] = 0.0;
      }
    }
  }

  for (iter=0;iter<itmax_power;iter++){
    my_op(x,Ax,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,0,operator); /* forward: x to Ax */
    my_op(ATAx,Ax,nw,nk,nt,dt,nx,dx,ox,np,dp,op,fmin,fmax,1,operator); /* adjoint: Ax to ATAx */
    __real__ sum = 0.0;
    __imag__ sum = 0.0;
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        sum = sum + conjf(ATAx[ik][iw])*ATAx[ik][iw];
      }
    }
    maxeig = (float) sqrt(crealf(sum));
    for (ik=0;ik<nk;ik++){
      for (iw=0;iw<nw;iw++){
        x[ik][iw] = ATAx[ik][iw]/maxeig;
      }
    } 
    fprintf(stderr,"iter=%d maxeig= %f\n",iter,maxeig);
  }

  free2complex(x);
  free2complex(ATAx);
  free2float(Ax);

  return maxeig;
}









