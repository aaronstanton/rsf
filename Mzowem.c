/* Zero Offset Wave Equation Migration with Stolt, Gazdag, PSPI, and Split-Step 2D operators.
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

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

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
void phase_shift(sf_complex **m,
                 int iz, float dz,
                 int ifmax, float dw,
                 int nk, float dk,
                 float vel,
                 bool verbose);
void pspi_2d_op(float **d, float **dmig,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nz, float oz, float dz,
                 float **c, float fmax,
                 int nref,
                 int numthreads,
                 bool adj, bool verbose);
void ss_2d_op(float **d, float **dmig,
              int nt, float ot, float dt, 
              int nmx, float omx, float dmx,
              int nz, float oz, float dz,
              float **c, float fmax,
              int numthreads,
              bool adj, bool verbose);
void pspi_extrap_1f(float **dmig,
                    sf_complex **d_wx,
                    int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,int nref,
                    float **c,float **vref,int **iref1,int **iref2,
                    sf_complex i,sf_complex czero,
                    fftwf_plan p1,fftwf_plan p2,
                    bool adj, bool verbose);
void ss_extrap_1f(float **dmig,
                    sf_complex **d_wx,
                    int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,
                    float *po,float **pd,
                    sf_complex i,sf_complex czero,
                    fftwf_plan p1,fftwf_plan p2,
                    bool adj, bool verbose);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
void k_op(sf_complex *m,sf_complex *d,int nk,int nx,bool adj);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
float linear_interp(float y1,float y2,float x1,float x2,float x);
void progress_msg(float progress);
void ls_zowem(float **d,float **dmig,float *wd,
             int nt,float ot,float dt, 
             int nmx,float omx,float dmx,
             int nz,float oz,float dz,
             float **vp,float fmax,
             int nref,
             int numthreads,
             float *misfit,
             int op,int Niter,bool verbose);
float cgdot(float **x,int nt,int nm);

int main(int argc, char* argv[])
{

  sf_file in,out,velp,misfitfile;
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
  int numthreads;
  bool dottest;
  float **d_1,**d_2,**dmig_1,**dmig_2,tmp_sum1,tmp_sum2;
  unsigned long mseed, dseed;
  bool inv;
  int Niter;
  float *misfit;
  char *misfitname;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getint("op",&op)) op = 1; /* extrapolation operator to be used 1=stolt, 2=gazdag, 3=PSPI, 4=Split-Step */
  if (!sf_getint("nref",&nref)) nref = 2; /* number of reference velocities to be used if pspi is selected. */
  if (!sf_getint("numthreads",&numthreads)) numthreads = 1; /* number of threads to be used for parallel processing (if pspi selected). */
  if (!sf_getbool("dottest",&dottest)) dottest = false; /* flag for dot product test, input should be the unmigrated data */
  if (adj || dottest){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
  }

  if (!sf_getbool("inv",&inv)) inv = false; /* flag for LS migration*/
  if (!sf_getint("Niter",&Niter)) Niter = 20; /* number of CG iterations for LS migration */


  if (inv){ 
    adj = true; /* activate adjoint flags */
    misfitname = sf_getstring("misfit");
    misfitfile = sf_output(misfitname);
    sf_putint(misfitfile,"n1",Niter);
    sf_putfloat(misfitfile,"d1",1);
    sf_putfloat(misfitfile,"o1",1);
    sf_putstring(misfitfile,"label1","Iteration Number");
    sf_putstring(misfitfile,"label2","Misfit");
    sf_putstring(misfitfile,"unit1"," ");
    sf_putstring(misfitfile,"unit2"," ");
    sf_putstring(misfitfile,"title","Misfit");
    sf_putint(misfitfile,"n2",1);
    sf_putint(misfitfile,"n3",1);
    sf_putint(misfitfile,"n4",1);
    sf_putint(misfitfile,"n5",1);
    misfit = sf_floatalloc(Niter);
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
  if (adj || dottest){
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


  if (dottest){
    mseed = (unsigned long) time(NULL);
    init_genrand(mseed);
    dseed = genrand_int32();
    d_1 = sf_floatalloc2(nt,nmx);
    d_2 = sf_floatalloc2(nt,nmx);
    dmig_1 = sf_floatalloc2(nz,nmx);
    dmig_2 = sf_floatalloc2(nz,nmx);
    init_genrand(dseed);
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++){
        d_1[ix][it] = 0.0;
        d_2[ix][it] = genrand_real1();
      }
      for (iz=0;iz<nz;iz++){
        dmig_1[ix][iz] = genrand_real1();
        dmig_2[ix][iz] = 0.0;
      }
    }
    if (op==1){
      stolt_2d_op(d_1,dmig_1,
                  nt,ot,dt, 
                  nmx,omx,dmx,
                  nz,oz,dz,
                  vp[0][0],fmax,
                  false,verbose);
      stolt_2d_op(d_2,dmig_2,
                  nt,ot,dt, 
                  nmx,omx,dmx,
                  nz,oz,dz,
                  vp[0][0],fmax,
                  true,verbose);
    }
    else if (op==2){
      gazdag_2d_op(d_1,dmig_1,
                   nt,ot,dt, 
                   nmx,omx,dmx,
                   nz,oz,dz,
                   vp,fmax,
                   false,verbose);
      gazdag_2d_op(d_2,dmig_2,
                   nt,ot,dt, 
                   nmx,omx,dmx,
                   nz,oz,dz,
                   vp,fmax,
                   true,verbose);
    }
    else if (op==3){
      pspi_2d_op(d_1,dmig_1,
                 nt,ot,dt, 
                 nmx,omx,dmx,
                 nz,oz,dz,
                 vp,fmax,
                 nref,
                 numthreads,
                 false,verbose);
      pspi_2d_op(d_2,dmig_2,
                 nt,ot,dt, 
                 nmx,omx,dmx,
                 nz,oz,dz,
                 vp,fmax,
                 nref,
                 numthreads,
                 true,verbose);
    }
    else if (op==4){
      ss_2d_op(d_1,dmig_1,
               nt,ot,dt, 
               nmx,omx,dmx,
               nz,oz,dz,
               vp,fmax,
               numthreads,
               false,verbose);
      ss_2d_op(d_2,dmig_2,
               nt,ot,dt, 
               nmx,omx,dmx,
               nz,oz,dz,
               vp,fmax,
               numthreads,
               true,verbose);
    }
    tmp_sum1=0;
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) tmp_sum1 += d_1[ix][it]*d_2[ix][it];
    tmp_sum2=0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2 += dmig_1[ix][iz]*dmig_2[ix][iz];
    fprintf(stderr,"DOT PRODUCT: %6.2f and %6.2f\n",tmp_sum1,tmp_sum2);
    exit (0);
  }

  if (!inv){
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
               numthreads,
               adj,verbose);
  }
  else if (op==4){
    ss_2d_op(d,dmig,
             nt,ot,dt, 
             nmx,omx,dmx,
             nz,oz,dz,
             vp,fmax,
             numthreads,
             adj,verbose);
  }
  }
  else{
    ls_zowem(d,dmig,wd,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,nref,numthreads,misfit,op,Niter,verbose);
  }

  if (adj || inv){
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

  if (inv){
    sf_floatwrite(misfit,Niter,misfitfile);
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
  int iw,ikz,ik,nw,nkz,nk,padt,padx,padz,ntfft,nzfft;
  float w,kz,k,vel,dw,dk,dkz;
  sf_complex **m,**mig;
  sf_complex czero;
  int ifmax;
  int ix,it;
  vel = c/2;
  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  padz = 4;
  padx = 4;
  ntfft = padt*nt;
  nw=ntfft/2+1;

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;

  nzfft = padz*nz;
  nkz = nzfft/2+1;
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
  mig = sf_complexalloc2(nkz,nk);
  for (ik=0;ik<nk;ik++){
    for (ikz=0;ikz<nkz;ikz++){
      mig[ik][ikz] = czero;
    }
  }

  if (adj){ 
    fk_op(m,d,nw,nk,nt,nmx,1); /* adjoint: d to m */
    for (ik=0;ik<nk;ik++){ 
      for (iw=ifmax;iw<nw;iw++) m[ik][iw] = czero;
    }
  }
  else{
    fk_op(mig,dmig,nkz,nk,nz,nmx,1); /* adjoint: dmig to mig */
  }
  for (ik=0;ik<nk;ik++){
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    for (iw=0;iw<ifmax;iw++){ 
      w = dw*iw;
      if ((w*w)/(vel*vel) - (k*k) > 0){ 
        kz = sqrtf((w*w)/(vel*vel) - (k*k));
        ikz = (int) trunc((float) kz/dkz);
        if (ikz>=0 && ikz<nkz){
          if (adj){
            mig[ik][ikz] = m[ik][iw]/sqrtf(1 + (k*k)/(kz*kz));
          }
          else{
            m[ik][iw] = mig[ik][ikz]/sqrtf(1 + (k*k)/(kz*kz));
          }
        }
      }
    }
  }
  if (adj){ 
    for (iw=ifmax;iw<nw;iw++) for (ik=0;ik<nk;ik++) mig[ik][iw] = czero;
    fk_op(mig,dmig,nkz,nk,nz,nmx,0); /* forward: mig to dmig */
  }
  else{
    for (iw=ifmax;iw<nw;iw++) for (ik=0;ik<nk;ik++) m[ik][iw] = czero;
    fk_op(m,d,nw,nk,nt,nmx,0); /* forward: m to d */
    for (it=0;it<nt;it++) for (ix=0;ix<nmx;ix++) d[ix][it] = d[ix][it]/(2*((float) nz)/((float)nt));
  }
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
  int it,iw,ik,iz,ix,nw,nk,padt,padx,ntfft;
  float **d2,vel,dw,dk,progress;
  sf_complex **m,*dmig_x,*dmig_k,**dmig_zk,czero;
  int ifmax;

  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;

  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;

  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  m = sf_complexalloc2(nw,nk);

  d2 = sf_floatalloc2(nt,nmx);
  dmig_zk = sf_complexalloc2(nz,nk);
  dmig_x = sf_complexalloc(nmx);
  dmig_k = sf_complexalloc(nk);

  if (adj){
    for (it=0;it<nt;it++) for (ix=0;ix<nmx;ix++) d2[ix][it] = d[ix][it];
    fk_op(m,d2,nw,nk,nt,nmx,1); /* d2 to m */
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
  }

  progress = 0.0;
  if (adj){
    for (iz=0;iz<nz;iz++){
      progress += 1.0/((float) nz);
      if (verbose) progress_msg(progress);
      vel = c[0][iz]/2;
      phase_shift(m,iz,dz,ifmax,dw,nk,dk,vel,verbose);
      fk_op(m,d2,nw,nk,nt,nmx,0); /* m to d2 */
      for (ix=0;ix<nmx;ix++) dmig[ix][iz] = d2[ix][0];
    }
  }
  else{
    for (iz=nz-1;iz>=0;iz--){
      progress += 1.0/((float) nz);
      if (verbose) progress_msg(progress);
      vel = c[0][iz]/2;
      for (iw=0;iw<ifmax;iw++) for (ik=0;ik<nk;ik++) m[ik][iw] += dmig_zk[ik][iz];
      phase_shift(m,iz,-dz,ifmax,dw,nk,dk,vel,verbose);
    }
  }

  if (verbose) fprintf(stderr,"\r                   \n");
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
                 int numthreads,
                 bool adj, bool verbose)
/*< Phase Shift Plus Interpolation zero offset wave equation depth migration operator >*/
{
  int iz,ix,ik,iw,it,iref,nw,nk,padt,padx,ntfft;
  float dw,dk;
  sf_complex czero,i;
  int ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_wx;
  fftwf_complex *a,*b;
  float **vref,vmin,vmax,v;
  int *n,**iref1,**iref2;
  fftwf_plan p1,p2;
  float progress;

  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  d_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  if (adj){
   for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) d_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) d_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++) d_wx[ix][iw] = czero;
    }
  }

  /* generate reference velocities for each layer */
  vref = sf_floatalloc2(nz,nref); /* reference velocities for each layer */
  iref1 = sf_intalloc2(nz,nmx); /* index of nearest lower reference velocity for each subsurface point */
  iref2 = sf_intalloc2(nz,nmx); /* index of nearest upper reference velocity for each subsurface point */
  
  for (iz=0;iz<nz;iz++){
    vmin=c[0][iz]/2;
    for (ix=0;ix<nmx;ix++) if (c[ix][iz]/2 < vmin) vmin = c[ix][iz]/2;
    vmax=c[nmx-1][iz]/2;
    for (ix=0;ix<nmx;ix++) if (c[ix][iz]/2 > vmax) vmax = c[ix][iz]/2;
    for (iref=0;iref<nref;iref++) vref[iref][iz] = vmin + (float) iref*(vmax-vmin)/((float) nref-1);
    for (ix=0;ix<nmx;ix++){
      v = c[ix][iz]/2;
      if (vmax>vmin+10){
        iref = (int) truncf((nref-1)*(v-vmin)/(vmax-vmin));
        iref1[ix][iz] = iref;
        iref2[ix][iz] = iref+1;
        if (iref>nref-2){
          iref1[ix][iz] = nref-1;
          iref2[ix][iz] = nref-1;
        }
      }
      else{
        iref1[ix][iz] = 0;
        iref2[ix][iz] = 0;
      }
    }
  }

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a); 
  fftwf_execute_dft(p2,b,b); 

  progress = 0.0;

omp_set_num_threads(numthreads);
#ifdef _OPENMP
#pragma omp parallel for \
        private(iw) \
        shared(dmig,d_wx,progress)
#endif
  for (iw=0;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    pspi_extrap_1f(dmig,d_wx,iw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,nref,c,vref,iref1,iref2,i,czero,p1,p2,adj,verbose);
  }
  if (verbose) fprintf(stderr,"\r                   \n");
  if (!adj){
   for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmax;iw++) d_w[iw] = d_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) d[ix][it] = d_t[it];
    }
  }

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);

  return;
} 

void ss_2d_op(float **d, float **dmig,
              int nt, float ot, float dt, 
              int nmx, float omx, float dmx,
              int nz, float oz, float dz,
              float **c, float fmax,
              int numthreads,
              bool adj, bool verbose)
/*< Split Step zero offset wave equation depth migration operator >*/
{
  int iz,ix,ik,iw,it,nw,nk,padt,padx,ntfft;
  float dw,dk;
  sf_complex czero,i;
  int ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_wx;
  fftwf_complex *a,*b;
  float *po,**pd;
  int *n;
  fftwf_plan p1,p2;
  float progress;

  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  nk = padx*nmx;
  dk = 2*PI/nk/dmx;
  dw = 2*PI/ntfft/dt;
  d_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  if (adj){
   for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmax;iw++) d_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) d_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++) d_wx[ix][iw] = czero;
    }
  }

  /* decompose slowness into layer average, and layer purturbation */
  po = sf_floatalloc(nz); 
  pd = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po[iz] += 2.0/c[ix][iz];
    po[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++) pd[ix][iz] = 2.0/c[ix][iz] - po[iz];
  }

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, (fftwf_complex*)a, (fftwf_complex*)a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, (fftwf_complex*)b, (fftwf_complex*)b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a); 
  fftwf_execute_dft(p2,b,b); 

  progress = 0.0;

omp_set_num_threads(numthreads);
#ifdef _OPENMP
#pragma omp parallel for \
        private(iw) \
        shared(dmig,d_wx,progress)
#endif
  for (iw=0;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    ss_extrap_1f(dmig,d_wx,iw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,po,pd,i,czero,p1,p2,adj,verbose);
  }
  if (verbose) fprintf(stderr,"\r                   \n");
  if (!adj){
   for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmax;iw++) d_w[iw] = d_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) d[ix][it] = d_t[it];
    }
  }
  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);

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

void phase_shift(sf_complex **m,
                 int iz, float dz,
                 int ifmax, float dw,
                 int nk, float dk,
                 float vel,
                 bool verbose)
/*< phase shift >*/
{

  sf_complex L,i,czero;
  float w,k,kz,s;
  int iw,ik;
  __real__ i = 0;
  __imag__ i = 1;
  __real__ czero = 0;
  __imag__ czero = 0;

  for (iw=0;iw<ifmax;iw++){
    w = dw*iw;
    for (ik=0;ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      s = (w*w)/(vel*vel) - (k*k);
      if (s>0){
        kz = sqrtf(s);
        L =  cexpf(i*kz*dz);
      }
      else{
        L =  czero;
      }
      m[ik][iw] = m[ik][iw]*L;
    }
  }
  return;
}

void pspi_extrap_1f(float **dmig,
                    sf_complex **d_wx,
                    int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,int nref,
                    float **c,float **vref,int **iref1,int **iref2,
                    sf_complex i,sf_complex czero,
                    fftwf_plan p1,fftwf_plan p2,
                    bool adj, bool verbose)
/*< extrapolate 1 frequency using the PSPI method >*/
{
  float w,v,k,s,vref1,vref2;
  sf_complex L;
  int iz,ix,ik,iref; 
  sf_complex *d_k,*d_x,**dref;
  fftwf_complex *a,*b;

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  dref = sf_complexalloc2(nref,nmx);
  d_x = sf_complexalloc(nmx);
  d_k = sf_complexalloc(nk);

  w = iw*dw;

  if (adj){
    for (iz=0;iz<nz;iz++){
      for (ix=0;ix<nmx;ix++) d_x[ix] = d_wx[ix][iw]; 
      /************* d_x --> d_k *********/
      for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
      for(ix=nmx;ix<nk;ix++) a[ix] = czero;
      fftwf_execute_dft(p1,a,a); 
      /***********************************/
      for (iref=0;iref<nref;iref++){
        for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
        for (ik=0;ik<nk;ik++){ 
	  if (ik<nk/2) k = dk*ik;
	  else         k = -(dk*nk - dk*ik);
	 /* s = (w*w)/(vref[iref][iz]*vref[iref][iz]) - (k*k);
          if (s>=0.0) L = cexpf(i*sqrtf(s)*dz);
	  else     L = czero;*/
          s = (w*w)/(vref[iref][iz]*vref[iref][iz]) - (k*k);
          if (s>0){ 
            L = cexpf(i*sqrtf(s)*dz);
            d_k[ik] = d_k[ik]*L;
          }
          else{
            d_k[ik] = czero;
          }
        }
        /************* d_k1 --> d_x1 *******/
        for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
        fftwf_execute_dft(p2,b,b);  
        for(ix=0; ix<nmx;ix++) dref[ix][iref] = b[ix]/nk; 
        /***********************************/
      }
      for (ix=0;ix<nmx;ix++){
        v = c[ix][iz]/2;
        vref1 = vref[iref1[ix][iz]][iz];
        vref2 = vref[iref2[ix][iz]][iz];
        if (vref2 - vref1 > 10.0){
	  __real__ d_wx[ix][iw] = linear_interp(crealf(dref[ix][iref1[ix][iz]]),crealf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
	  __imag__ d_wx[ix][iw] = linear_interp(cimagf(dref[ix][iref1[ix][iz]]),cimagf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
        }
        else{
         d_wx[ix][iw] = dref[ix][iref1[ix][iz]];
        }
      }
      for (ix=0;ix<nmx;ix++) dmig[ix][iz] += 2*crealf(d_wx[ix][iw])/ntfft;
    }
  }
  else{
    for (iz=nz-1;iz>=0;iz--){
      for (ix=0;ix<nmx;ix++){ 
        __real__ d_x[ix] = crealf(d_wx[ix][iw]) + dmig[ix][iz]; 
        __imag__ d_x[ix] = cimagf(d_wx[ix][iw]); 
      }
      /************* d_x --> d_k *********/
      for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
      for(ix=nmx;ix<nk;ix++) a[ix] = czero;
      fftwf_execute_dft(p1,a,a); 
      /***********************************/
      for (iref=0;iref<nref;iref++){
        for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
        for (ik=0;ik<nk;ik++){ 
	  if (ik<nk/2) k = dk*ik;
	  else         k = -(dk*nk - dk*ik);
          s = (w*w)/(vref[iref][iz]*vref[iref][iz]) - (k*k);
          if (s>0){
            L = cexpf(-i*sqrtf(s)*dz);
            d_k[ik] = d_k[ik]*L;
          }
          else{
            d_k[ik] = czero;
          }
        }
        /************* d_k1 --> d_x1 *******/
        for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
        fftwf_execute_dft(p2,b,b);  
        for(ix=0; ix<nmx;ix++) dref[ix][iref] = b[ix]/nk; 
        /***********************************/
      }
      for (ix=0;ix<nmx;ix++){
        v = c[ix][iz]/2;
        vref1 = vref[iref1[ix][iz]][iz];
        vref2 = vref[iref2[ix][iz]][iz];
        if (vref2 - vref1 > 10.0){
	  __real__ d_wx[ix][iw] = linear_interp(crealf(dref[ix][iref1[ix][iz]]),crealf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
	  __imag__ d_wx[ix][iw] = linear_interp(cimagf(dref[ix][iref1[ix][iz]]),cimagf(dref[ix][iref2[ix][iz]]),vref1,vref2,v);
        }
        else{
         d_wx[ix][iw] = dref[ix][iref1[ix][iz]];
        }
      }
    }
  }

  free2complex(dref);
  free1complex(d_x);
  free1complex(d_k);
  fftwf_free(a);
  fftwf_free(b);

  return;
}

void ss_extrap_1f(float **dmig,
                    sf_complex **d_wx,
                    int iw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,
                    float *po,float **pd,
                    sf_complex i,sf_complex czero,
                    fftwf_plan p1,fftwf_plan p2,
                    bool adj, bool verbose)
/*< extrapolate 1 frequency using the Split Step method >*/
{
  float w,k,kz,s;
  sf_complex L;
  int iz,ix,ik; 
  sf_complex *d_k,*d_x;
  fftwf_complex *a,*b;

  a  = fftwf_malloc(sizeof(fftw_complex) * nk);
  b  = fftwf_malloc(sizeof(fftw_complex) * nk);
  d_x = sf_complexalloc(nmx);
  d_k = sf_complexalloc(nk);

  w = iw*dw;
  if (adj){
  for (iz=0;iz<nz;iz++){
    for (ix=0;ix<nmx;ix++){
      d_x[ix] = d_wx[ix][iw];
    }
    /************* d_x --> d_k *********/
    for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
    for(ix=nmx;ix<nk;ix++) a[ix] = czero;
    fftwf_execute_dft(p1,a,a); 
    /***********************************/
    for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
    for (ik=0;ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      s = (w*w)*(po[iz]*po[iz]) - (k*k);
      if (s>0){
        kz = sqrtf(s);
	L = cexpf(i*kz*dz);
	d_k[ik] = d_k[ik]*L;
      }
      else {
	d_k[ik] = czero;
      }
    }
    /************* d_k1 --> d_x1 *******/
    for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
    fftwf_execute_dft(p2,b,b);  
    /***********************************/
    for (ix=0;ix<nmx;ix++){
      d_wx[ix][iw] = b[ix]*cexpf(i*w*pd[ix][iz]*dz)/nk;
    }
    for (ix=0;ix<nmx;ix++) dmig[ix][iz] += 2*crealf(d_wx[ix][iw])/ntfft;
  }
  }
  else{
  for (iz=nz-1;iz>=0;iz--){
    for (ix=0;ix<nmx;ix++){ 
      __real__ d_x[ix] = crealf(d_wx[ix][iw]) + dmig[ix][iz]; 
      __imag__ d_x[ix] = cimagf(d_wx[ix][iw]); 
    }
    for (ix=0;ix<nmx;ix++){
      d_x[ix] = d_x[ix]*cexpf(-i*w*pd[ix][iz]*dz);
    }
    /************* d_x --> d_k *********/
    for(ix=0 ;ix<nmx;ix++) a[ix] = d_x[ix];
    for(ix=nmx;ix<nk;ix++) a[ix] = czero;
    fftwf_execute_dft(p1,a,a); 
    /***********************************/
    for(ik=0 ;ik<nk;ik++) d_k[ik] = a[ik]; 
    for (ik=0;ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      s = (w*w)*(po[iz]*po[iz]) - (k*k);
      if (s>0){
        kz = sqrtf(s);
	L = cexpf(-i*kz*dz);
	d_k[ik] = d_k[ik]*L;
      }
      else {
	d_k[ik] = czero;
      }
    }
    /************* d_k1 --> d_x1 *******/
    for(ik=0; ik<nk;ik++) b[ik] = d_k[ik];
    fftwf_execute_dft(p2,b,b);  
    /***********************************/
    for (ix=0;ix<nmx;ix++){
      d_wx[ix][iw] = b[ix]/nk;
    }
  }
  }

  free1complex(d_x);
  free1complex(d_k);
  fftwf_free(a);
  fftwf_free(b);

  return;
}

float linear_interp(float y1,float y2,float x1,float x2,float x)
/*< linear interpolation between two points. x2-x1 must be nonzero. >*/
{
  return  y1 + (y2-y1)*(x-x1)/(x2-x1);
}

void progress_msg(float progress)
/*< progress message (progress is the ratio between 0 and 1 for the progress you wish to print). >*/
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

void ls_zowem(float **d,float **dmig,float *wd,
             int nt,float ot,float dt, 
             int nmx,float omx,float dmx,
             int nz,float oz,float dz,
             float **vp,float fmax,
             int nref,
             int numthreads,
             float *misfit,
             int op,int Niter,bool verbose)
/*< Least squares migration. >*/
{
  int k,ix,it,iz;
  float progress,gamma,gamma_old,delta,alpha,beta,**r,**ss,**g,**s,**v;

  r = sf_floatalloc2(nt,nmx);
  ss = sf_floatalloc2(nt,nmx);
  g = sf_floatalloc2(nz,nmx);
  s = sf_floatalloc2(nz,nmx);
  v = sf_floatalloc2(nz,nmx);
 
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++){
      dmig[ix][iz] = 0.0;				
      v[ix][iz] = dmig[ix][iz];
      g[ix][iz] = 0.0;
    }
  }
  for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) r[ix][it] = d[ix][it];
  for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) r[ix][it] = r[ix][it]*wd[ix];
  if (op==1) stolt_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp[0][0],fmax,true,false);
  else if (op==2) gazdag_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,true,false);
  else if (op==3) pspi_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,nref,numthreads,true,false);
  else if (op==4) ss_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,numthreads,true,false);
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++){
      s[ix][iz] = g[ix][iz];
    }
  }
  gamma = cgdot(g,nz,nmx);
  gamma_old = gamma;
  progress = 0.0;
  for (k=0;k<Niter;k++){
    progress += 1.0/((float) Niter);
    if (verbose) progress_msg(progress);
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) ss[ix][it] = 0.0;
    if (op==1) stolt_2d_op(ss,s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp[0][0],fmax,false,false);
    else if (op==2) gazdag_2d_op(ss,s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,false,false);
    else if (op==3) pspi_2d_op(ss,s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,nref,numthreads,false,false);
    else if (op==4) ss_2d_op(ss,s,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,numthreads,false,false);
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) ss[ix][it] = ss[ix][it]*wd[ix];
    delta = cgdot(ss,nt,nmx);
    alpha = gamma/(delta + 0.00000001);
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) v[ix][iz] = v[ix][iz] +  s[ix][iz]*alpha;
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) r[ix][it] = (r[ix][it] -  ss[ix][it]*alpha)*wd[ix];
    misfit[k] = cgdot(r,nt,nmx);
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) g[ix][iz] = 0.0;
    if (op==1) stolt_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp[0][0],fmax,true,false);
    else if (op==2) gazdag_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,true,false);
    else if (op==3) pspi_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,nref,numthreads,true,false);
    else if (op==4) ss_2d_op(r,g,nt,ot,dt,nmx,omx,dmx,nz,oz,dz,vp,fmax,numthreads,true,false);
    gamma = cgdot(g,nz,nmx);
    beta = gamma/(gamma_old + 0.00000001);
    gamma_old = gamma;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) s[ix][iz] = g[ix][iz] + s[ix][iz]*beta;
  }
  if (verbose) fprintf(stderr,"\r                   \n");
  for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmig[ix][iz] = v[ix][iz];

  free2float(r);
  free2float(ss);
  free2float(g);
  free2float(s);
  free2float(v);

  return;
}

float cgdot(float **x,int nt,int nm)
/*< Compute the inner product for matrix of floats, x >*/
{
  int it,ix;
  float cgdot;
  cgdot = 0.0;
  for (ix=0;ix<nm;ix++) for (it=0;it<nt;it++) cgdot = cgdot + x[ix][it]*x[ix][it];
  return(cgdot);
}




