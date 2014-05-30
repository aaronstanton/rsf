/* Shot Profile Wave Equation Migration.
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

void wem_sp2d_op(float **d, float **dmig,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vel, float fmin, float fmax,
                 int numthreads,
                 bool adj, bool verbose);
void extrap1f(float **dmig,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
              int nmx,float omx, float dmx,
              int nhx,float ohx, float dhx,
              int npx, float opx, float dpx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,fftwf_plan p3,fftwf_plan p4,
              bool adj, bool verbose);
void ssop(sf_complex *d_x,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *p0,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
float linear_interp(float y1,float y2,float x1,float x2,float x);
void progress_msg(float progress);
void ls_shotwem(float **d, float **dmig,float *wav,float *wd,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp, float fmin, float fmax,
                 int numthreads,
                 float *misfit,
                 int Niter,
                 bool verbose);
float cgdot(float **x,int nt,int nm);

int main(int argc, char* argv[])
{

  sf_file in,out,velp,source_wavelet,misfitfile;
  int nt,nmx,nz,nsx,nhx,npx;
  int it,ix,iz;
  float ot,omx,oz,osx,ohx,opx;
  float dt,dmx,dz,dsx,dhx,dpx;
  float **dmig,**d,**vp,*wd,*wav;
  bool adj;
  bool verbose;
  float sum;
  float fmin,fmax;
  int sum_wd;
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
  source_wavelet = sf_input("wav");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getint("nref",&nref)) nref = 2; /* number of reference velocities for pspi. */
  if (!sf_getint("numthreads",&numthreads)) numthreads = 1; /* number of threads to be used for parallel processing. */
  if (!sf_getbool("dottest",&dottest)) dottest = false; /* flag for dot product test, input should be the unmigrated data */

  if (adj || dottest){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
    if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
    if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
    if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");
    if (!sf_getint("npx",&npx)) sf_error("npx must be specified");
    if (!sf_getfloat("opx",&opx)) sf_error("opx must be specified");
    if (!sf_getfloat("dpx",&dpx)) sf_error("dpx must be specified");
    opx = opx/1000000;
    dpx = dpx/1000000;
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
    if (!sf_getint("nsx",&nsx)) sf_error("nsx must be specified");
    if (!sf_getfloat("osx",&osx)) sf_error("osx must be specified");
    if (!sf_getfloat("dsx",&dsx)) sf_error("dsx must be specified");
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
  if (adj){
    if (!sf_histint(  in,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&ot)) ot=0.0;
    if (!sf_histint(  in,"n2",&nmx)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"o2",&omx)) omx=0.0;
    if (!sf_histint(  in,"n3",&nsx)) nsx=1;
    if (!sf_histfloat(in,"d3",&dsx)) dsx=1.0;
    if (!sf_histfloat(in,"o3",&osx)) osx=0.0;
  }
  else{
    if (!sf_histint(  in,"n1",&nz)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&dz)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&oz)) sf_error("No o1= in input");
    if (!sf_histint(  in,"n2",&nmx)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"o2",&omx)) sf_error("No o2= in input");
    if (!sf_histint(  in,"n3",&npx)) sf_error("No n3= in input");
    if (!sf_histfloat(in,"d3",&dpx)) sf_error("No d3= in input");
    if (!sf_histfloat(in,"o3",&opx)) sf_error("No o3= in input");
  }

  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putfloat(out,"o1",oz);
    sf_putfloat(out,"d1",dz);
    sf_putfloat(out,"n1",nz);
    sf_putstring(out,"label1","Depth");
    sf_putstring(out,"unit1","m");
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"n2",nmx);
    sf_putstring(out,"label2","X");
    sf_putstring(out,"unit2","m");
    sf_putfloat(out,"o3",opx*1000000);
    sf_putfloat(out,"d3",dpx*1000000);
    sf_putfloat(out,"n3",npx);
    sf_putstring(out,"label3","Ray Parameter");
    sf_putstring(out,"unit3"," micro-s/m");
    sf_putstring(out,"title","Migrated data");
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"n1",nt);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"unit1","s");
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"n2",nmx);
    sf_putstring(out,"label2","Receiver-X");
    sf_putstring(out,"unit2","m");
    sf_putfloat(out,"o3",osx);
    sf_putfloat(out,"d3",dsx);
    sf_putfloat(out,"n3",nsx);
    sf_putstring(out,"label3","Source-X");
    sf_putstring(out,"unit3","m");
    sf_putstring(out,"title","Data");
  }
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  d = sf_floatalloc2(nt,nmx*nsx);
  dmig = sf_floatalloc2(nz,nmx*npx);
  wd = sf_floatalloc(nmx*nsx);
  sum_wd = 0;
  if (adj){
    sf_floatread(d[0],nt*nmx*nsx,in);
    for (ix=0;ix<nmx*nsx;ix++){
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += d[ix][it]*d[ix][it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/((float) nmx*nsx));
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmig[ix][iz] = 0.0;
  }
  else{
    sf_floatread(dmig[0],nz*nmx*npx,in);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
  }

  if (dottest){
    mseed = (unsigned long) time(NULL);
    init_genrand(mseed);
    dseed = genrand_int32();
    d_1 = sf_floatalloc2(nt,nmx*nsx);
    d_2 = sf_floatalloc2(nt,nmx*nsx);
    dmig_1 = sf_floatalloc2(nz,nmx*npx);
    dmig_2 = sf_floatalloc2(nz,nmx*npx);
    init_genrand(dseed);
    for (ix=0;ix<nmx*nsx;ix++){
      for (it=0;it<nt;it++){
        d_1[ix][it] = 0.0;
        d_2[ix][it] = (float) 1.0*sf_randn_one_bm();
      }
    }
    for (ix=0;ix<nmx*npx;ix++){
      for (iz=0;iz<nz;iz++){
        dmig_1[ix][iz] = (float) 1.0*sf_randn_one_bm();
        dmig_2[ix][iz] = 0.0;
      }
    }
    wem_sp2d_op(d_1,dmig_1,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,false,verbose);
    wem_sp2d_op(d_2,dmig_2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,true,verbose);

    tmp_sum1=0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) tmp_sum1 += d_1[ix][it]*d_2[ix][it];
    tmp_sum2=0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2 += dmig_1[ix][iz]*dmig_2[ix][iz];
    fprintf(stderr,"DOT PRODUCT: %6.5f and %6.5f\n",tmp_sum1,tmp_sum2);
    free2float(d_1);
    free2float(d_2);
    free2float(dmig_1);
    free2float(dmig_2);
    exit (0);
  }

  if (!inv){
    wem_sp2d_op(d,dmig,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,adj,verbose);
  }
  else{
    ls_shotwem(d,dmig,wav,wd,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,misfit,Niter,verbose);
    sf_floatwrite(misfit,Niter,misfitfile);
  }

  if (adj || inv){
    sf_floatwrite(dmig[0],nz*nmx*npx,out);
  }
  else{
    sf_floatwrite(d[0],nt*nmx*nsx,out);
  }
 
  free2float(dmig);
  free2float(d);
  free2float(vp);
  free1float(wd);
  free1float(wav);
  
  exit (0);

}

void wem_sp2d_op(float **d, float **dmig,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nhx, float ohx, float dhx,
                 int npx,float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vel, float fmin, float fmax,
                 int numthreads,
                 bool adj, bool verbose)
/*< 2d shot profile wave equation depth migration operator >*/
{
  int iz,ix,isx,igx,ik,iw,it,nw,nk,padt,padx,ntfft,nkhx;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b,*c1,*c2;
  int *n,*n2;
  fftwf_plan p1,p2,p3,p4;
  float progress;
  float *po,**pd;

  if (adj){
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmig[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
  }

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
  if(fmin*dt*ntfft+1<ifmax) ifmin = trunc(fmin*dt*ntfft);
  else ifmin = 0;
  nk = padx*nmx;
  dk = 2*PI/((float) nk)/dmx;
  dw = 2*PI/((float) ntfft)/dt;
  d_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  /* decompose slowness into layer average, and layer purturbation */
  po = sf_floatalloc(nz); 
  pd = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po[iz] += 1.0/vel[ix][iz];
    po[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd[ix][iz] = 1.0/vel[ix][iz] - po[iz];
    }
  }

/* set up fftw plans and pass then to the parallel region of the code */
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  n = sf_intalloc(1); 
  n[0] = nk;
  p1 = fftwf_plan_dft(1, n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft(1, n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nk;ik++){
    a[ik] = czero;
    b[ik] = czero;
  } 
  fftwf_execute_dft(p1,a,a);
  fftwf_execute_dft(p2,b,b);
  
  nkhx = nhx;
  c1  = fftwf_malloc(sizeof(fftwf_complex) * nkhx);
  c2  = fftwf_malloc(sizeof(fftwf_complex) * nkhx);
  n2 = sf_intalloc(1); 
  n2[0] = nkhx;
  p3 = fftwf_plan_dft(1, n2, c1, c1, FFTW_FORWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nkhx;ik++) c1[ik] = czero;
  fftwf_execute_dft(p3,c1,c1); 
  p4 = fftwf_plan_dft(1, n2, c2, c2, FFTW_FORWARD, FFTW_ESTIMATE);
  for (ik=0;ik<nkhx;ik++) c2[ik] = czero;
  fftwf_execute_dft(p4,c2,c2); 
/**********************************************************************/


for (isx=0;isx<nsx;isx++){
  igx = (int) truncf(((float) isx*dsx + osx) - omx)/dmx;  
  /* source wavefield*/
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
  for (it=0;it<nt;it++) d_t[it] = wav[it];
  f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
  for (iw=0;iw<nw;iw++) d_s_wx[igx][iw] = d_w[iw];
  /* receiver wavefield*/
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[isx*nmx + ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) d_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) d_g_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++){
	d_g_wx[ix][iw] = czero;
      }
    }
  }
  progress = 0.0;
  omp_set_num_threads(numthreads);
  if (verbose && adj) fprintf(stderr,"\r migrating shot %d/%d:\n",isx+1,nsx);
  if (verbose && !(adj)) fprintf(stderr,"\r demigrating shot %d/%d:\n",isx+1,nsx);
  #pragma omp parallel for private(iw) shared(dmig,d_g_wx,d_s_wx,progress)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    extrap1f(dmig,d_g_wx,d_s_wx,iw,nw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,omx,dmx,nhx,ohx,dhx,npx,opx,dpx,po,pd,i,czero,p1,p2,p3,p4,adj,verbose);
  }
  if (!adj){
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = d_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) d[isx*nmx + ix][it] = d_t[it];
    }
  }
}
  if (verbose) fprintf(stderr,"\r                   \n");

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  free1float(d_t);
  free1complex(d_w);
  free2complex(d_g_wx);
  free2complex(d_s_wx);
  free1float(po);
  free2float(pd);
  free1int(n2); 
  fftwf_destroy_plan(p3);
  fftwf_free(c1);
  fftwf_destroy_plan(p4);
  fftwf_free(c2);

  return;
} 

void extrap1f(float **dmig,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
              int nmx,float omx, float dmx,
              int nhx,float ohx, float dhx,
              int npx,float opx, float dpx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,fftwf_plan p3,fftwf_plan p4,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor,x,hx,sx,gx,k,px;
  int iz,ix,ihx,isx,igx,ik,ipx,nkhx,nhx_half; 
  sf_complex *d_xg,*d_xs,**smig;
  fftwf_complex *c1,*c2;
  
  nkhx = nhx;
  nhx_half = (nhx -1)/2;
  c1  = fftwf_malloc(sizeof(fftwf_complex) * nkhx);
  c2  = fftwf_malloc(sizeof(fftwf_complex) * nkhx);
 
  d_xg = sf_complexalloc(nmx);
  d_xs = sf_complexalloc(nmx);

  if (iw==0) factor = 1;
  else factor = 2;

  w = iw*dw;
  if (adj){
    for (ix=0;ix<nmx;ix++){ 
      d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
      d_xg[ix] = d_g_wx[ix][iw]/sqrtf((float) ntfft);
    }
    for (iz=0;iz<nz;iz++){ /* extrapolate source and receiver wavefields */
      ssop(d_xs,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,verbose); 
      ssop(d_xg,w,dk,nk,nmx,dz,iz,po,pd,i,czero,p1,p2,true,verbose);
      for (ix=0;ix<nmx;ix++){
        for (ik=0;ik<nkhx;ik++) c1[ik] = c2[ik] = czero;
        for (ihx=0;ihx<nhx;ihx++){
          hx = ihx*dhx + ohx;
          sx = (ix*dmx + omx) - hx;
          gx = (ix*dmx + omx) + hx;
          isx = (int) truncf((sx - omx)/dmx);
          igx = (int) truncf((gx - omx)/dmx);
          if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
            if (hx < 0) c1[nhx_half - ihx] = -factor*crealf(d_xs[isx]*conjf(d_xg[igx]));
            else        c2[ihx - nhx_half] = factor*crealf(d_xs[isx]*conjf(d_xg[igx]));
          } 
        }
        fftwf_execute_dft(p3,c1,c1); 
        fftwf_execute_dft(p4,c2,c2); 
        for (ik=0;ik<nkhx;ik++){
          if (ik<nkhx/2) k = dk*ik;
          else k = -(dk*nkhx - dk*ik);
          if (w>0){
            px = -k/w;
            ipx = (int) truncf((px - opx)/dpx);
            if (ipx>=0 && ipx<npx){
              if (px>0){
                #pragma omp atomic 
                dmig[ipx*nmx + ix][iz] += c2[ik]/sqrtf((float) nkhx);
              }
              else{
                #pragma omp atomic 
                dmig[ipx*nmx + ix][iz] += c1[ik]/sqrtf((float) nkhx);
              }
            }
          }
        }
      }
    }
  }

  else{
    smig = sf_complexalloc2(nz,nmx);
    for (ix=0;ix<nmx;ix++) d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
    for (iz=0;iz<nz;iz++){ /* extrapolate source wavefield */
      ssop(d_xs,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,verbose); 
      for (ix=0;ix<nmx;ix++) smig[ix][iz] = d_xs[ix];
    }
    for (ix=0;ix<nmx;ix++) d_xg[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ /* extrapolate receiver wavefield */
      for (ix=0;ix<nmx;ix++){ 
        x = omx + ix*dmx;
        hx = x - sx; //FIX THIS PART OF THE CODE TO WHAT YOU HAVE IN THE ADJOINT!!
        ihx = (int) truncf((hx - ohx)/dhx);
        if (ihx >= 0 && ihx < nhx){
          d_xg[ix] = d_xg[ix] + smig[ix][iz]*dmig[ihx*nmx + ix][iz];
        }
      }
      ssop(d_xg,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,false,verbose);
    }
    for (ix=0;ix<nmx;ix++){
      d_g_wx[ix][iw] = d_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(d_xg);
  free1complex(d_xs);

  fftwf_free(c1);
  fftwf_free(c2);

  return;
}

void ssop(sf_complex *d_x,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *po,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool verbose)
{
  float k,s;
  sf_complex L;
  int ix,ik; 
  sf_complex *d_k;
  fftwf_complex *a,*b;

  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  d_k = sf_complexalloc(nk);

  if (adj){
    for(ix=0; ix<nmx;ix++) a[ix] = d_x[ix];
  }
  else{
    for(ix=0; ix<nmx;ix++){
      __real__ L = cos(w*pd[ix][iz]*dz);
      __imag__ L = sin(w*pd[ix][iz]*dz); 
      a[ix] = d_x[ix]*L;
    }
  }
  for(ix=nmx;ix<nk;ix++) a[ix] = (fftwf_complex) czero;

  fftwf_execute_dft(p1,a,a); 
  for (ik=0;ik<nk;ik++){ 
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    s = (w*w)*(po[iz]*po[iz]) - (k*k);
    if (s>=0){ 
      __real__ L = cos(sqrt(s)*dz);
      __imag__ L = sin(sqrt(s)*dz);
    }
    else L = czero;
    d_k[ik] = ((sf_complex) a[ik])*L/sqrtf((float) nk);        
  }
  for(ik=0; ik<nk;ik++) b[ik] = (fftwf_complex) d_k[ik];
  fftwf_execute_dft(p2,b,b);
  if (adj){
    for(ix=0; ix<nmx;ix++){ 
      __real__ L = cos(w*pd[ix][iz]*dz);
      __imag__ L = sin(w*pd[ix][iz]*dz);
      d_x[ix] = ((sf_complex) b[ix])*L/sqrtf((float) nk);
    }
  }
  else{
    for(ix=0; ix<nmx;ix++){ 
      d_x[ix] = ((sf_complex) b[ix])/sqrtf((float) nk);
    }
  }

  free1complex(d_k);
  fftwf_free(a);
  fftwf_free(b);

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

float linear_interp(float y1,float y2,float x1,float x2,float x)
/*< linear interpolation between two floats. x2-x1 must be nonzero. >*/
{
  return  y1 + (y2-y1)*(x-x1)/(x2-x1);
}

void progress_msg(float progress)
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

void ls_shotwem(float **d, float **dmig,float *wav,float *wd,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp, float fmin, float fmax,
                 int numthreads,
                 float *misfit,
                 int Niter,
                 bool verbose)
/*< Least squares migration. >*/
{
  int k,ix,it,iz;
  float progress,gamma,gamma_old,delta,alpha,beta,**r,**ss,**g,**s,**v;

  r = sf_floatalloc2(nt,nmx*nsx);
  ss = sf_floatalloc2(nt,nmx*nsx);
  g = sf_floatalloc2(nz,nmx*nhx);
  s = sf_floatalloc2(nz,nmx*nhx);
  v = sf_floatalloc2(nz,nmx*nhx);
 
  for (ix=0;ix<nmx*nhx;ix++){
    for (iz=0;iz<nz;iz++){
      dmig[ix][iz] = 0.0;				
      v[ix][iz] = dmig[ix][iz];
      g[ix][iz] = 0.0;
    }
  }
  for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r[ix][it] = d[ix][it];
  for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r[ix][it] = r[ix][it]*wd[ix];
  wem_sp2d_op(r,g,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,true,false);
  for (ix=0;ix<nmx*nhx;ix++){
    for (iz=0;iz<nz;iz++){
      s[ix][iz] = g[ix][iz];
    }
  }
  gamma = cgdot(g,nz,nmx*nhx);
  gamma_old = gamma;
  progress = 0.0;
  for (k=0;k<Niter;k++){
    progress += 1.0/((float) Niter);
    if (verbose) progress_msg(progress);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss[ix][it] = 0.0;
    wem_sp2d_op(ss,s,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,false,false);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss[ix][it] = ss[ix][it]*wd[ix];
    delta = cgdot(ss,nt,nmx*nsx);
    alpha = gamma/(delta + 0.00000001);
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) v[ix][iz] = v[ix][iz] +  s[ix][iz]*alpha;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r[ix][it] = (r[ix][it] -  ss[ix][it]*alpha)*wd[ix];
    misfit[k] = cgdot(r,nt,nmx*nhx);
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) g[ix][iz] = 0.0;
    wem_sp2d_op(r,g,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,fmin,fmax,numthreads,true,false);
    gamma = cgdot(g,nz,nmx*nhx);
    beta = gamma/(gamma_old + 0.00000001);
    gamma_old = gamma;
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) s[ix][iz] = g[ix][iz] + s[ix][iz]*beta;
  }
  if (verbose) fprintf(stderr,"\r                   \n");
  for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) dmig[ix][iz] = v[ix][iz];

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



