/* Zero Offset Wave Equation Migration with Stolt or Gazdag 2D operators.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fftw3.h>
#include "myfree.h"

void stolt_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float c, float fmax,
                 bool adj, bool verbose);
void gazdag_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmax,
                 bool adj, bool verbose);
void phase_shift(sf_complex **m, sf_complex **dmig_zk,
                 int iz, float dz,
                 int ifmax, float dw,
                 int nk, float dk,
                 float vel,
                 bool adj, bool verbose);
void pspi_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmax,
                 int nref,
                 bool adj, bool verbose);
void pspi1f(sf_complex **d_wx,int iw,
            int nmx,int nk,
            float dw,float dk,float dz,
            float velmin,float velmax,float *velref,float **c,
            sf_complex i, sf_complex czero);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
float linear_interp(float y1,float y2,float x1,float x2,float x);

int main(int argc, char* argv[])
{

  sf_file in,out,velp;
  int n1,n2;
  int nt,nmx,nz;
  int it,ix,iz;
  float o1,o2;
  float d1,d2;
  float ot,omx,oz;
  float dt,dmx,dz;
  float **d,**dmig,**vp,*trace,*wd;
  bool adj;
  bool verbose;
  float sum;
  float fmax;
  int sum_wd;
  int op;  
  int nref;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getint("op",&op)) op = 1; /* extrapolation operator to be used 1=stolt, 2=gazdag*/
  if (!sf_getint("nref",&nref)) nref = 2; /* number of reference velocities to be used if pspi is selected. */
  if (adj){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
  }

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;
 
  nmx=n2;  
  dmx=d2;  
  omx=o2;
  if (adj){
    nt=n1;  
    dt=d1;  
    ot=o1;
  }
  else{
    nz=n1;  
    dz=d1;  
    oz=o1;  
  }
 
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putfloat(out,"o1",oz);
    sf_putfloat(out,"d1",dz);
    sf_putfloat(out,"n1",nz);
    sf_putstring(out,"label1","Depth");
    sf_putstring(out,"unit1","m");
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"n1",nt);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"unit1","s");
  }
  sf_putfloat(out,"o2",omx);
  sf_putfloat(out,"d2",dmx);
  sf_putfloat(out,"n2",nmx);
  sf_putstring(out,"label2","x");
  sf_putstring(out,"unit2","m");
  if (adj){
    sf_putstring(out,"title","Migrated data");
  }
  else{
    sf_putstring(out,"title","Data");
  }
  vp = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc( nt > nz ? nt : nz  );
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,nz,velp);
    for (iz=0;iz<nz;iz++) vp[ix][iz] = trace[iz];
  }
  d = sf_floatalloc2(nt,nmx);
  dmig = sf_floatalloc2(nz,nmx);
  wd = sf_floatalloc(nmx);
  sum_wd = 0;
  if (adj){
    for (ix=0;ix<nmx;ix++){
      sf_floatread(trace,n1,in);
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += trace[it]*trace[it];
        d[ix][it] = trace[it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/(nmx));
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++){
        dmig[ix][iz] = 0.0;
      }
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      sf_floatread(trace,n1,in);
      sum = 0.0;
      for (iz=0;iz<nz;iz++){
        dmig[ix][iz] = trace[iz];
      }
    }
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++){
        d[ix][it] = 0.0;
      }
    }
  }
  if (op==1){
    stolt_2d_op(d,dmig,
                nt,ot,dt, 
                nmx,omx,dmx,
                nz,oz,dz,
                vp[0][0],fmax,
                adj,verbose);
  }
  else if (op==2){
    gazdag_2d_op(d,dmig,
                 nt,ot,dt, 
                 nmx,omx,dmx,
                 nz,oz,dz,
                 vp,fmax,
                 adj,verbose);
  }
  else if (op==3){
    pspi_2d_op(d,dmig,
               nt,ot,dt, 
               nmx,omx,dmx,
               nz,oz,dz,
               vp,fmax,
               nref,
               adj,verbose);
  }
  if (adj){
    for (ix=0; ix<nmx; ix++) {
      for (iz=0; iz<nz; iz++) trace[iz] = dmig[ix][iz];	
      sf_floatwrite(trace,nz,out);
    }
  }
  else{
    for (ix=0; ix<nmx; ix++) {
      for (it=0; it<nt; it++) trace[it] = d[ix][it];	
      sf_floatwrite(trace,nt,out);
    }
  }
  exit (0);
}

void stolt_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float c, float fmax,
                 bool adj, bool verbose)
/*< Stolt zero offset wave equation depth migration operator >*/
{
  int iw,iwz,ik,nw,nwz,nk,padt,padx,padz,ntfft,nzfft;
  float w,kz,k,vel,dw,dk,dkz;
  sf_complex **m,**mig;
  sf_complex czero;
  int ifmax;

  vel = c/2;
  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 4;
  padz = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;

  nzfft = padz*nz;
  nwz = nzfft/2+1;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  dkz = 2*PI/nzfft/dz;
  m = sf_complexalloc2(nw,nk);
  for (ik=0;ik<nk;ik++){
    for (iw=0;iw<nw;iw++){
      m[ik][iw] = czero;
    }
  }
  mig = sf_complexalloc2(nwz,nk);
  for (ik=0;ik<nk;ik++){
    for (iwz=0;iwz<nwz;iwz++){
      mig[ik][iwz] = czero;
    }
  }

  if (adj) fk_op(m,d,nw,nk,nt,nmx,1); /* adjoint: d to m */
  else     fk_op(mig,dmig,nwz,nk,nz,nmx,1); /* adjoint: dmig to mig */
  for (ik=0;ik<nk;ik++){
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    for (iw=0;iw<ifmax;iw++){ 
      w = dw*iw;
      if ((w*w)/(vel*vel) - (k*k) > 0){ 
        kz = sqrt((w*w)/(vel*vel) - (k*k));
        iwz = trunc(kz/dkz); 
       /* fprintf(stderr,"kz=%f, iwz=%d\n",kz,iwz);*/
      }
      else iwz = nwz;
      /*fprintf(stderr,"kz=%f, (kz*c/2)*sqrt(1 + (k*k)/(kz*kz))=%f, w=%f \n",kz,(kz*c/2)*sqrt(1 + (k*k)/(kz*kz)),w);*/
      if (iwz>=0 && iwz<nwz){ 
        if (adj) mig[ik][iwz] = m[ik][iw]/sqrt(1 + (k*k)/(kz*kz));
        else     m[ik][iw] = mig[ik][iwz]/sqrt(1 + (k*k)/(kz*kz));
      }
    }
  }
  if (adj) fk_op(mig,dmig,nwz,nk,nz,nmx,0); /* forward: mig to dmig */
  else     fk_op(m,d,nw,nk,nt,nmx,0); /* forward: m to d */
  return;
} 

void gazdag_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmax,
                 bool adj, bool verbose)
/*< Gazdag zero offset wave equation depth migration operator >*/
{
  int iw,ik,iz,ix,nw,nk,padt,padx,ntfft;
  float vel,dw,dk;
  sf_complex **m,*dmig_x,*dmig_k,**dmig_zk,czero;
  int ifmax;

  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;

  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  m = sf_complexalloc2(nw,nk);

  dmig_zk = sf_complexalloc2(nz,nk);
  dmig_x = sf_complexalloc(nmx);
  dmig_k = sf_complexalloc(nk);

  if (adj){
    fk_op(m,d,nw,nk,nt,nmx,1); /* d to m */
    for (ix=0;ix<nmx;ix++) dmig[ix][0] = d[ix][0]; 
    for (iw=ifmax;iw<nw;iw++) for (ik=0;ik<nk;ik++) m[ik][iw] = czero; 
  }
  else{
    for (iw=0;iw<nw;iw++) for (ik=0;ik<nk;ik++) m[ik][iw] = czero;
    for (iz=0;iz<nz;iz++){
      for (ix=0;ix<nmx;ix++){ 
        __real__ dmig_x[ix] = dmig[ix][iz];
        __imag__ dmig_x[ix] = 0.0;
      }
      k_op(dmig_k,dmig_x,nk,nmx,1); /* dmig_x to dmig_k */
      for (ik=0;ik<nk;ik++) dmig_zk[ik][iz] = dmig_k[ik];
    }
    for (ix=0;ix<nmx;ix++) d[ix][0] = dmig[ix][0];
  }

  for (iz=1;iz<nz;iz++){
    fprintf(stderr,"extrapolating depth step %d/%d\n",iz+1,nz);
    vel = c[0][iz]/2;
    if (adj){
      phase_shift(m,dmig_zk,iz,dz,ifmax,dw,nk,dk,vel,adj,verbose);
      fk_op(m,d,nw,nk,nt,nmx,0); /* m to d */
      for (ix=0;ix<nmx;ix++) dmig[ix][iz] = d[ix][0];  
    }
    else{ 
      phase_shift(m,dmig_zk,nz-iz,-dz,ifmax,dw,nk,dk,vel,adj,verbose);
    }
  }
  if (!adj){ 
    fk_op(m,d,nw,nk,nt,nmx,0); /* m to d */ 
  }
  return;
} 

void pspi_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmax,
                 int nref,
                 bool adj, bool verbose)
/*< Phase Shift Plus Interpolation zero offset wave equation depth migration operator >*/
{
  int iw,ik,iz,ix,it,nw,nk,padt,padx,ntfft,iref;
  float vel,velmin,velmax,*velref,dw,dk;
  sf_complex **m,*dmig_x,*dmig_k,**dmig_zk,czero,i,L,Lref;
  int ifmax;
  float w,k,kz,s,*d_t;
  sf_complex *d_w,*d_x,**dref,*d_k,*dkref;
  sf_complex **d_wx;
  sf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;

  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  m = sf_complexalloc2(nw,nk);
  dmig_zk = sf_complexalloc2(nz,nk);
  dmig_x = sf_complexalloc(nmx);
  dmig_k = sf_complexalloc(nk);
  a  = sf_complexalloc(nk);
  b  = sf_complexalloc(nk);
  n = sf_intalloc(1); n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);
  velref = sf_floatalloc(nref);
  d_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  d_x = sf_complexalloc(nmx);
  dref = sf_complexalloc2(nref,nmx);
  d_k = sf_complexalloc(nk);
  dkref = sf_complexalloc(nk);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  if (adj){
   for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) d_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) d_wx[ix][iw] = czero;
      dmig[ix][0] = d[ix][0]; 
    }
  }
  else{
    fprintf(stderr,"Sorry, PSPI forward operator not written yet!\n");
    return;
  }
  for (iw=0;iw<ifmax;iw++){ 
    fprintf(stderr,"extrapolating frequency %d/%d\n",iw+1,ifmax);
    pspi_extrap_1f();
  }
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  fftwf_free(a);fftwf_free(b);
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

void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj)
{
  sf_complex *a,*b,czero;
  int *n,ix,ik;
  fftwf_plan p1,p2;
  __real__ czero = 0;
  __imag__ czero = 0;
  a  = sf_complexalloc(nk);
  b  = sf_complexalloc(nk);
  n = sf_intalloc(1); n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);

  if (adj){ /* x --> k */
    for(ix=0 ;ix<nx;ix++) a[ix] = d[ix];
    for(ix=nx;ix<nk;ix++) a[ix] = czero;
    fftwf_execute(p1); 
    for(ik=0 ;ik<nk;ik++) m[ik] = a[ik]; 
  }
  else{ /* k --> x */
    for(ik=0; ik<nk;ik++) b[ik] = m[ik];
    fftwf_execute(p2); 
    for(ix=0; ix<nx;ix++) d[ix] = b[ix]/nk; 
    for(ix=nx;ix<nk;ix++) d[ix] = czero;
  }
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  fftwf_free(a);fftwf_free(b);
  return;
}

void f_op(sf_complex *m,float *d,int nw,int nt,bool adj)
{
  sf_complex *out1a,*in1b,czero;
  float *in1a,*out1b;
  int ntfft,it,iw;
  fftwf_plan p1a,p1b;

  ntfft = (nw-1)*2;
  __real__ czero = 0;
  __imag__ czero = 0;

  if (adj){ /* data --> model */
    out1a = sf_complexalloc(nw);
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
    in1b = sf_complexalloc(ntfft);
    p1b = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in1b, out1b, FFTW_ESTIMATE);
    for(iw=0;iw<nw;iw++) in1b[iw] = m[iw];
    for(iw=nw;iw<ntfft;iw++) in1b[iw] = czero;
    fftwf_execute(p1b); 
    for(it=0;it<nt;it++) d[it] = out1b[it]/ntfft; 
    fftwf_destroy_plan(p1b);
    fftwf_free(in1b); fftwf_free(out1b);
  }


  return;
}

void phase_shift(sf_complex **m, sf_complex **dmig_zk,
                 int iz, float dz,
                 int ifmax, float dw,
                 int nk, float dk,
                 float vel,
                 bool adj, bool verbose)
/*< phase shift >*/
{

  sf_complex L,i;
  float w,k,kz;
  int iw,ik;
  __real__ i = 0;
  __imag__ i = 1;

  for (iw=0;iw<ifmax;iw++){
    w = dw*iw;
    for (ik=0;ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      if ((w*w)/(vel*vel) - (k*k)>0){
        kz = -sqrt((w*w)/(vel*vel) - (k*k));
        L =  cexpf(-i*kz*dz);
        if (adj){
          m[ik][iw] = m[ik][iw]*L;
        }
        else{
          m[ik][iw] = m[ik][iw]*L + dmig_zk[ik][iz];
        }
      }
    }
  }
  return;
}

float linear_interp(float y1,float y2,float x1,float x2,float x)
/*< linear interpolation between two points. x2-x1 must be nonzero. >*/
{  
  return (y1*(x2-x)+y2*(x-x1))/(x2-x1);
}

void pspi_extrap_1f()
/*< extrapolate 1 frequency >*/
{
    w = iw*dw;
    for (iz=1;iz<nz;iz++){
      /*fprintf(stderr,"extrapolating depth step %d/%d\n",iz+1,nz);*/
      velmin=c[0][iz]/2;
      for (ix=0;ix<nmx;ix++) if (c[ix][iz]/2 < velmin) velmin = c[ix][iz]/2;
      velmax = c[nmx-1][iz]/2;
      for (ix=0;ix<nmx;ix++) if (c[ix][iz]/2 > velmax) velmax = c[ix][iz]/2;
      for (iref=0;iref<nref;iref++) velref[iref] = velmin + iref*(velmax-velmin)/(nref-1);
      for (ix=0;ix<nmx;ix++){ 
        vel = c[ix][iz]/2;
        L = cexpf(-i*w*dz/vel);
        d_x[ix] = d_wx[ix][iw]*L;
      }
      /************* d_x --> d_k *********/
      for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
      for(ix=nmx;ix<nk;ix++) a[ix] = czero;
      fftwf_execute(p1); 
      for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
      /***********************************/
      for (iref=0;iref<nref;iref++){
        for (ik=0;ik<nk;ik++){ 
          if (ik<nk/2) k = dk*ik;
          else         k = -(dk*nk - dk*ik);
          s = (w*w)/(velref[iref]*velref[iref]) - (k*k);
          if (s>0){
            kz = -sqrt(s);
            Lref = cexpf(-i*(kz-w/velref[iref])*dz);
            dkref[ik] = d_k[ik]*Lref;
          }
        }
        /************* d_k1 --> d_x1 *******/
        for(ik=0; ik<nk;ik++) b[ik] = dkref[ik];
        fftwf_execute(p2); 
        for(ix=0; ix<nmx;ix++) dref[ix][iref] = b[ix]/nk; 
        /***********************************/
      }
      for (ix=0;ix<nmx;ix++){
        vel = c[ix][iz]/2;
        for (iref=0;iref<nref;iref++){
          velref[iref] = velmin + iref*(velmax-velmin)/(nref-1);
          if (velref[iref] >=vel) break;
        }
        if (iref+1<nref && velmin<velmax){
          __real__ d_wx[ix][iw] = linear_interp(crealf(dref[ix][iref]),crealf(dref[ix][iref+1]),velref[iref],velref[iref+1],vel);
          __imag__ d_wx[ix][iw] = linear_interp(cimagf(dref[ix][iref]),cimagf(dref[ix][iref+1]),velref[iref],velref[iref+1],vel);
        }
        else d_wx[ix][iw] = dref[ix][iref];
      }
      for (ix=0;ix<nmx;ix++) dmig[ix][iz] += crealf(d_wx[ix][iw]);
    }
  return;
}


