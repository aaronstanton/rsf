/* Shot Profile Elastic Wave Equation Migration.
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

void ewem_sp2d_op(float **dp,float **ds,float **dmigpp,float **dmigps,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float fmin, float fmax,
                 int numthreads,
                 bool adj, bool verbose);
void eextrap1f(float **dmigpp,float **dmigps,
              sf_complex **dp_g_wx, sf_complex **ds_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,
              float *po_p,float **pd_p,float *po_s,float **pd_s,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose);
void ssop(sf_complex *d_x,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *p0,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
void progress_msg(float progress);

void ls_shotewem(float **dp,float **ds,float **dmigpp,float **dmigps,float *wav,float *wd,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float fmin,float fmax,
                 int numthreads,
                 float *misfit1,float *misfit2,
                 int Niter,int Nextern,
                 bool verbose);
float cgdot(float **x,int nt,int nm);
float cgdotc(sf_complex **x,int nt,int nm);
void match_weights(float **w1, float **w2, float **m1, float **m2, int nz, int nmx, float e);
void triangle_filter(sf_complex **m,int nz,int nmx,bool adj);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void adapt(sf_complex **f, sf_complex **m, sf_complex **d, int nw, int nmx, int Niter);
void conv_op(sf_complex **filter,sf_complex **d,sf_complex **m,int nw,int nmx,bool adj);

int main(int argc, char* argv[])
{

  sf_file in1,in2,out1,out2,velp,vels,source_wavelet,misfitfile1,misfitfile2,weights1file,weights2file;
  int n1,n2;
  int nt,nmx,nz,nsx;
  int it,ix,iz;
  float o1,o2;
  float d1,d2;
  float ot,omx,oz,osx;
  float dt,dmx,dz,dsx;
  float **dmigpp,**dmigps,**dp,**ds,**vp,**vs,*wd,*wav;
  bool adj;
  bool verbose;
  float sum;
  float fmin,fmax;
  int sum_wd;
  int nref;
  int numthreads;
  bool dottest;
  float **dp_1,**dp_2,**dmigpp_1,**dmigpp_2,tmp_sum1_p,tmp_sum2_p;
  float **ds_1,**ds_2,**dmigps_1,**dmigps_2,tmp_sum1_s,tmp_sum2_s;
  unsigned long mseed, dseed;
  bool inv;
  int Niter,Nextern;
  float *misfit1;
  char *misfitname1;
  float *misfit2;
  char *misfitname2;
  bool debug;
  float **weights1,**weights2;
  char *weights1name,*weights2name;

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  source_wavelet = sf_input("wav");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getint("nref",&nref)) nref = 2; /* number of reference velocities for pspi. */
  if (!sf_getint("numthreads",&numthreads)) numthreads = 1; /* number of threads to be used for parallel processing. */
  if (!sf_getbool("dottest",&dottest)) dottest = false; /* flag for dot product test, input should be the unmigrated data */
  if (!sf_getbool("inv",&inv)) inv = false; /* flag for LS migration*/
  if (!sf_getint("Niter",&Niter)) Niter = 20; /* number of CG iterations for LS migration */
  if (!sf_getint("Nextern",&Nextern)) Nextern = 1; /* number of external CG iterations for joint regularization of PP and PS data */
  if (!sf_getbool("debug",&debug)) debug = false; /* flag for debugging */
  if (debug) adj = true; 

  if (adj || dottest || inv || debug){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
  }

  if (!(adj) || dottest){
    if (!sf_getint("nsx",&nsx)) sf_error("nsx must be specified");
    if (!sf_getfloat("osx",&osx)) sf_error("osx must be specified");
    if (!sf_getfloat("dsx",&dsx)) sf_error("dsx must be specified");
  }

  if (inv){ 
    adj = true; /* activate adjoint flags */
    misfitname1 = sf_getstring("misfit1");
    misfitfile1 = sf_output(misfitname1);
    sf_putint(misfitfile1,"n1",Niter*Nextern);
    sf_putfloat(misfitfile1,"d1",1);
    sf_putfloat(misfitfile1,"o1",1);
    sf_putstring(misfitfile1,"label1","Iteration Number");
    sf_putstring(misfitfile1,"label2","Misfit");
    sf_putstring(misfitfile1,"unit1"," ");
    sf_putstring(misfitfile1,"unit2"," ");
    sf_putstring(misfitfile1,"title","Misfit: PP");
    sf_putint(misfitfile1,"n2",1);
    sf_putint(misfitfile1,"n3",1);
    sf_putint(misfitfile1,"n4",1);
    sf_putint(misfitfile1,"n5",1);
    misfit1 = sf_floatalloc(Niter*Nextern);
    misfitname2 = sf_getstring("misfit2");
    misfitfile2 = sf_output(misfitname2);
    sf_putint(misfitfile2,"n1",Niter*Nextern);
    sf_putfloat(misfitfile2,"d1",1);
    sf_putfloat(misfitfile2,"o1",1);
    sf_putstring(misfitfile2,"label1","Iteration Number");
    sf_putstring(misfitfile2,"label2","Misfit");
    sf_putstring(misfitfile2,"unit1"," ");
    sf_putstring(misfitfile2,"unit2"," ");
    sf_putstring(misfitfile2,"title","Misfit: PS");
    sf_putint(misfitfile2,"n2",1);
    sf_putint(misfitfile2,"n3",1);
    sf_putint(misfitfile2,"n4",1);
    sf_putint(misfitfile2,"n5",1);
    misfit2 = sf_floatalloc(Niter*Nextern);
  }

  /* read input file parameters */
  if (!sf_histint(  in1,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in1,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in1,"o1",&o1)) o1=0.;
  if (!sf_histint(  in1,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in1,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in1,"o2",&o2)) o2=0.;

  if (adj){
    if (!sf_histint(  in1,"n3",&nsx)) sf_error("No n3= in input");
    if (!sf_histfloat(in1,"d3",&dsx)) sf_error("No d3= in input");
    if (!sf_histfloat(in1,"o3",&osx)) osx=0.;
  }

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
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putfloat(out1,"o1",oz);
    sf_putfloat(out1,"d1",dz);
    sf_putfloat(out1,"n1",nz);
    sf_putstring(out1,"label1","Depth");
    sf_putstring(out1,"unit1","m");
    sf_putfloat(out1,"o3",0);
    sf_putfloat(out1,"d3",0);
    sf_putfloat(out1,"n3",1);
    sf_putfloat(out2,"o1",oz);
    sf_putfloat(out2,"d1",dz);
    sf_putfloat(out2,"n1",nz);
    sf_putstring(out2,"label1","Depth");
    sf_putstring(out2,"unit1","m");
    sf_putfloat(out2,"o3",0);
    sf_putfloat(out2,"d3",0);
    sf_putfloat(out2,"n3",1);
  }
  else{
    sf_putfloat(out1,"o1",ot);
    sf_putfloat(out1,"d1",dt);
    sf_putfloat(out1,"n1",nt);
    sf_putstring(out1,"label1","Time");
    sf_putstring(out1,"unit1","s");
    sf_putfloat(out1,"o3",osx);
    sf_putfloat(out1,"d3",dsx);
    sf_putfloat(out1,"n3",nsx);
    sf_putfloat(out2,"o1",ot);
    sf_putfloat(out2,"d1",dt);
    sf_putfloat(out2,"n1",nt);
    sf_putstring(out2,"label1","Time");
    sf_putstring(out2,"unit1","s");
    sf_putfloat(out2,"o3",osx);
    sf_putfloat(out2,"d3",dsx);
    sf_putfloat(out2,"n3",nsx);
  }
  sf_putfloat(out1,"o2",omx);
  sf_putfloat(out1,"d2",dmx);
  sf_putfloat(out1,"n2",nmx);
  sf_putstring(out1,"label2","x");
  sf_putstring(out1,"unit2","m");
  sf_putfloat(out2,"o2",omx);
  sf_putfloat(out2,"d2",dmx);
  sf_putfloat(out2,"n2",nmx);
  sf_putstring(out2,"label2","x");
  sf_putstring(out2,"unit2","m");
  if (adj){
    sf_putstring(out1,"title","PP Migration");
    sf_putstring(out2,"title","PS Migration");
  }
  else{
    sf_putstring(out1,"title","PP data");
    sf_putstring(out2,"title","PS data");
  }
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  vs = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  sf_floatread(vs[0],nz*nmx,vels);
  dp = sf_floatalloc2(nt,nmx*nsx);
  ds = sf_floatalloc2(nt,nmx*nsx);
  dmigpp = sf_floatalloc2(nz,nmx);
  dmigps = sf_floatalloc2(nz,nmx);
  wd = sf_floatalloc(nmx*nsx);
  sum_wd = 0;
  if (adj){
    sf_floatread(dp[0],nt*nmx*nsx,in1);
    sf_floatread(ds[0],nt*nmx*nsx,in2);
    for (ix=0;ix<nmx*nsx;ix++){
      sum = 0.0;
      for (it=0;it<nt;it++){
        sum += dp[ix][it]*dp[ix][it];
      }
      if (sum){ 
        wd[ix] = 1.0;
        sum_wd++;
      }
      else wd[ix] = 0.0;
    }
    if (verbose && adj) fprintf(stderr,"There are %6.2f %% missing traces.\n", 
                         (float) 100 - 100*sum_wd/((float) nmx*nsx));
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = 0.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = 0.0;
  }
  else{
    sf_floatread(dmigpp[0],nz*nmx,in1);
    sf_floatread(dmigps[0],nz*nmx,in2);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) dp[ix][it] = 0.0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ds[ix][it] = 0.0;
  }

  if (debug){
    adj = true;
    fprintf(stderr,"DEBUG MODE: checking weights\n");
    weights1name = sf_getstring("weights1");
    weights1file = sf_output(weights1name);
    sf_putfloat(weights1file,"o1",oz);
    sf_putfloat(weights1file,"d1",dz);
    sf_putfloat(weights1file,"n1",nz);
    sf_putfloat(weights1file,"o2",omx);
    sf_putfloat(weights1file,"d2",dmx);
    sf_putfloat(weights1file,"n2",nmx);
    sf_putstring(weights1file,"label1","Depth");
    sf_putstring(weights1file,"unit1","m");
    sf_putfloat(weights1file,"o3",0);
    sf_putfloat(weights1file,"d3",0);
    sf_putfloat(weights1file,"n3",1);
    weights2name = sf_getstring("weights2");
    weights2file = sf_output(weights2name);
    sf_putfloat(weights2file,"o1",oz);
    sf_putfloat(weights2file,"d1",dz);
    sf_putfloat(weights2file,"n1",nz);
    sf_putfloat(weights2file,"o2",omx);
    sf_putfloat(weights2file,"d2",dmx);
    sf_putfloat(weights2file,"n2",nmx);
    sf_putstring(weights2file,"label1","Depth");
    sf_putstring(weights2file,"unit1","m");
    sf_putfloat(weights2file,"o3",0);
    sf_putfloat(weights2file,"d3",0);
    sf_putfloat(weights2file,"n3",1);
    ewem_sp2d_op(dp,ds,dmigpp,dmigps,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,adj,verbose);
    weights1 = sf_floatalloc2(nz,nmx);
    weights2 = sf_floatalloc2(nz,nmx);
    match_weights(weights1,weights2,dmigpp,dmigps,nz,nmx,1);
    sf_floatwrite(dmigpp[0],nz*nmx,out1);
    sf_floatwrite(dmigps[0],nz*nmx,out2);
    sf_floatwrite(weights1[0],nz*nmx,weights1file);
    sf_floatwrite(weights2[0],nz*nmx,weights2file);
    exit (0);
  }


  if (dottest){
    mseed = (unsigned long) time(NULL);
    init_genrand(mseed);
    dseed = genrand_int32();
    dp_1 = sf_floatalloc2(nt,nmx*nsx);
    dp_2 = sf_floatalloc2(nt,nmx*nsx);
    ds_1 = sf_floatalloc2(nt,nmx*nsx);
    ds_2 = sf_floatalloc2(nt,nmx*nsx);
    dmigpp_1 = sf_floatalloc2(nz,nmx);
    dmigpp_2 = sf_floatalloc2(nz,nmx);
    dmigps_1 = sf_floatalloc2(nz,nmx);
    dmigps_2 = sf_floatalloc2(nz,nmx);
    init_genrand(dseed);
    for (ix=0;ix<nmx*nsx;ix++){
      for (it=0;it<nt;it++){
        dp_1[ix][it] = 0.0;
        dp_2[ix][it] = (float) 1.0*sf_randn_one_bm();
        ds_1[ix][it] = 0.0;
        ds_2[ix][it] = (float) 1.0*sf_randn_one_bm();
      }
    }
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++){
        dmigpp_1[ix][iz] = (float) 1.0*sf_randn_one_bm();
        dmigpp_2[ix][iz] = 0.0;
        dmigps_1[ix][iz] = (float) 1.0*sf_randn_one_bm();
        dmigps_2[ix][iz] = 0.0;
      }
    }
    ewem_sp2d_op(dp_1,ds_1,dmigpp_1,dmigps_1,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,false,verbose);
    ewem_sp2d_op(dp_2,ds_2,dmigpp_2,dmigps_2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,true,verbose);
    tmp_sum1_p=0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) tmp_sum1_p += dp_1[ix][it]*dp_2[ix][it];
    tmp_sum2_p=0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2_p += dmigpp_1[ix][iz]*dmigpp_2[ix][iz];
    fprintf(stderr,"DOT PRODUCT (PP): %6.5f and %6.5f\n",tmp_sum1_p,tmp_sum2_p);
    tmp_sum1_s=0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) tmp_sum1_s += ds_1[ix][it]*ds_2[ix][it];
    tmp_sum2_s=0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2_s += dmigps_1[ix][iz]*dmigps_2[ix][iz];
    fprintf(stderr,"DOT PRODUCT (PS): %6.5f and %6.5f\n",tmp_sum1_s,tmp_sum2_s);
    free2float(dp_1);
    free2float(dp_2);
    free2float(ds_1);
    free2float(ds_2);
    free2float(dmigpp_1);
    free2float(dmigpp_2);
    free2float(dmigps_1);
    free2float(dmigps_2);
    exit (0);
  }

  if (!inv){
    ewem_sp2d_op(dp,ds,dmigpp,dmigps,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,adj,verbose);
  }
  else{
    ls_shotewem(dp,ds,dmigpp,dmigps,wav,wd,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,misfit1,misfit2,Niter,Nextern,verbose);
    sf_floatwrite(misfit1,Niter*Nextern,misfitfile1);
    sf_floatwrite(misfit2,Niter*Nextern,misfitfile2);
  }

  if (adj || inv){
    sf_floatwrite(dmigpp[0],nz*nmx,out1);
    sf_floatwrite(dmigps[0],nz*nmx,out2);
  }
  else{
    sf_floatwrite(dp[0],nt*nmx*nsx,out1);
    sf_floatwrite(ds[0],nt*nmx*nsx,out2);
  }
 
  free2float(dmigpp);
  free2float(dmigps);
  free2float(dp);
  free2float(ds);
  free2float(vp);
  free2float(vs);
  free1float(wd);
  free1float(wav);
  
  exit (0);

}

void ewem_sp2d_op(float **dp,float **ds,float **dmigpp,float **dmigps,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float fmin, float fmax,
                 int numthreads,
                 bool adj, bool verbose)
/*< 2d shot profile elastic wave equation depth migration operator >*/
{
  int iz,ix,isx,igx,ik,iw,it,nw,nk,padt,padx,ntfft;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **dp_g_wx,**ds_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float progress;
  float *po_p,**pd_p,*po_s,**pd_s;

  if (adj){
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = 0.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) dp[ix][it] = 0.0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ds[ix][it] = 0.0;
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
  dp_g_wx = sf_complexalloc2(nw,nmx);
  ds_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  /* decompose slowness into layer average, and layer purturbation */
  po_p = sf_floatalloc(nz); 
  pd_p = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po_p[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_p[iz] += 1.0/vp[ix][iz];
    po_p[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd_p[ix][iz] = 1.0/vp[ix][iz] - po_p[iz];
    }
  }
  po_s = sf_floatalloc(nz); 
  pd_s = sf_floatalloc2(nz,nmx); 
  for (iz=0;iz<nz;iz++){
    po_s[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po_s[iz] += 1.0/vs[ix][iz];
    po_s[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd_s[ix][iz] = 1.0/vs[ix][iz] - po_s[iz];
    }
  }

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
  
for (isx=0;isx<nsx;isx++){
  igx = (int) truncf(((float) isx*dsx + osx) - omx)/dmx;  
  /* source wavefield*/
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
  for (it=0;it<nt;it++) d_t[it] = wav[it];
  f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
  for (iw=0;iw<nw;iw++) d_s_wx[igx][iw] = d_w[iw];
  /* P and S receiver wavefields*/
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = dp[isx*nmx + ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) dp_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dp_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dp_g_wx[ix][iw] = czero;
    }
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = ds[isx*nmx + ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) ds_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) ds_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) ds_g_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++){
	dp_g_wx[ix][iw] = czero;
	ds_g_wx[ix][iw] = czero;
      }
    }
  }
  progress = 0.0;
  omp_set_num_threads(numthreads);
  if (verbose && adj) fprintf(stderr,"\r migrating shot %d/%d:\n",isx+1,nsx);
  if (verbose && !(adj)) fprintf(stderr,"\r demigrating shot %d/%d:\n",isx+1,nsx);
  #pragma omp parallel for private(iw) shared(dmigpp,dmigps,dp_g_wx,ds_g_wx,d_s_wx,progress)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    eextrap1f(dmigpp,dmigps,dp_g_wx,ds_g_wx,d_s_wx,iw,nw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,po_p,pd_p,po_s,pd_s,i,czero,p1,p2,adj,verbose);
  }
  if (!adj){
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dp_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dp[isx*nmx + ix][it] = d_t[it];
    }
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = ds_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) ds[isx*nmx + ix][it] = d_t[it];
    }
  }
  if (verbose) fprintf(stderr,"\r                     ");
}
  if (verbose) fprintf(stderr,"\r                   \n");

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  free1float(d_t);
  free1complex(d_w);
  free2complex(dp_g_wx);
  free2complex(ds_g_wx);
  free2complex(d_s_wx);
  free1float(po_p);
  free2float(pd_p);
  free1float(po_s);
  free2float(pd_s);

  return;
} 

void eextrap1f(float **dmigpp,float **dmigps,
              sf_complex **dp_g_wx, sf_complex **ds_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,int nmx,
              float *po_p,float **pd_p,float *po_s,float **pd_s,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor;
  int iz,ix; 
  sf_complex *dp_xg,*ds_xg,*d_xs,**smig;
  dp_xg = sf_complexalloc(nmx);
  ds_xg = sf_complexalloc(nmx);
  d_xs = sf_complexalloc(nmx);

  if (iw==0) factor = 1;
  else factor = 2;

  w = iw*dw;
  if (adj){
    for (ix=0;ix<nmx;ix++){ 
      d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
      dp_xg[ix] = dp_g_wx[ix][iw]/sqrtf((float) ntfft);
      ds_xg[ix] = ds_g_wx[ix][iw]/sqrtf((float) ntfft);
    }
    for (iz=0;iz<nz;iz++){ /* extrapolate source and receiver wavefields */
      ssop(d_xs,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose); 
      ssop(dp_xg,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose);
      ssop(ds_xg,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,true,verbose);
      for (ix=0;ix<nmx;ix++){
        #pragma omp atomic 
        dmigpp[ix][iz] += factor*crealf(conjf(d_xs[ix])*dp_xg[ix]);
      }
      for (ix=0;ix<nmx;ix++){
        #pragma omp atomic 
        dmigps[ix][iz] += factor*crealf(conjf(d_xs[ix])*ds_xg[ix]);
      }
    }
  }

  else{
    smig = sf_complexalloc2(nz,nmx);
    for (ix=0;ix<nmx;ix++) d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
    for (iz=0;iz<nz;iz++){ /* extrapolate source wavefield */
      ssop(d_xs,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose); 
      for (ix=0;ix<nmx;ix++) smig[ix][iz] = d_xs[ix];
    }
    for (ix=0;ix<nmx;ix++) dp_xg[ix] = czero;
    for (ix=0;ix<nmx;ix++) ds_xg[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ /* extrapolate receiver wavefield */
      for (ix=0;ix<nmx;ix++) dp_xg[ix] = dp_xg[ix] + smig[ix][iz]*dmigpp[ix][iz];
      for (ix=0;ix<nmx;ix++) ds_xg[ix] = ds_xg[ix] + smig[ix][iz]*dmigps[ix][iz];
      ssop(dp_xg,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,false,verbose);
      ssop(ds_xg,w,dk,nk,nmx,-dz,iz,po_s,pd_s,i,czero,p1,p2,false,verbose);
    }
    for (ix=0;ix<nmx;ix++){
      dp_g_wx[ix][iw] = dp_xg[ix]/sqrtf((float) ntfft);
      ds_g_wx[ix][iw] = ds_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(dp_xg);
  free1complex(ds_xg);
  free1complex(d_xs);

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

void progress_msg(float progress)
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

void ls_shotewem(float **dp,float **ds,float **dmigpp,float **dmigps,float *wav,float *wd,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float fmin,float fmax,
                 int numthreads,
                 float *misfit1, float *misfit2,
                 int Niter,int Nextern,
                 bool verbose)
/*< Least squares migration. >*/
{
  int k,k2,ix,it,iz;
  float progress;
  float gamma1,gamma1_old,delta1,alpha1,beta1,**r1,**ss1,**g1,**s1,**v1,**v1wm;
  float gamma2,gamma2_old,delta2,alpha2,beta2,**r2,**ss2,**g2,**s2,**v2,**v2wm;
  float **wm1,**wm2;

  r1 = sf_floatalloc2(nt,nmx*nsx);
  ss1 = sf_floatalloc2(nt,nmx*nsx);
  g1 = sf_floatalloc2(nz,nmx);
  s1 = sf_floatalloc2(nz,nmx);
  v1 = sf_floatalloc2(nz,nmx);

  r2 = sf_floatalloc2(nt,nmx*nsx);
  ss2 = sf_floatalloc2(nt,nmx*nsx);
  g2 = sf_floatalloc2(nz,nmx);
  s2 = sf_floatalloc2(nz,nmx);
  v2 = sf_floatalloc2(nz,nmx);

  v1wm = sf_floatalloc2(nz,nmx);
  v2wm = sf_floatalloc2(nz,nmx);

  wm1 = sf_floatalloc2(nz,nmx);
  wm2 = sf_floatalloc2(nz,nmx);

  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++){
      dmigpp[ix][iz] = 0.0;				
      dmigps[ix][iz] = 0.0;				
      v1[ix][iz] = 0.0;
      v2[ix][iz] = 0.0;
      wm1[ix][iz] = 1.0;
      wm2[ix][iz] = 1.0;
    }
  }
  progress = 0.0;


  for (k2=0;k2<Nextern;k2++){

    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) v1wm[ix][iz] = v1[ix][iz]*wm1[ix][iz];
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) v1wm[ix][iz] = v2[ix][iz]*wm2[ix][iz];
    ewem_sp2d_op(r1,r2,v1wm,v2wm,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,false,false);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r1[ix][it] = (dp[ix][it] -  r1[ix][it])*wd[ix];
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r2[ix][it] = (ds[ix][it] -  r2[ix][it])*wd[ix];
    ewem_sp2d_op(r1,r2,g1,g2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,true,false);
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) g1[ix][iz] = g1[ix][iz]*wm1[ix][iz];
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) g2[ix][iz] = g2[ix][iz]*wm2[ix][iz];
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) s1[ix][iz] = g1[ix][iz];
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) s2[ix][iz] = g2[ix][iz];
    
    gamma1 = cgdot(g1,nz,nmx);
    gamma1_old = gamma1;
    gamma2 = cgdot(g2,nz,nmx);
    gamma2_old = gamma2;
    
    for (k=0;k<Niter;k++){
      progress += 1.0/((float) Niter*Nextern);
      if (verbose) progress_msg(progress);
      ewem_sp2d_op(ss1,ss2,s1,s2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,false,false);
      for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss1[ix][it] = ss1[ix][it]*wd[ix];
      for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss2[ix][it] = ss2[ix][it]*wd[ix];
      delta1 = cgdot(ss1,nt,nmx*nsx);
      alpha1 = gamma1/(delta1 + 0.00000001);
      delta2 = cgdot(ss2,nt,nmx*nsx);
      alpha2 = gamma2/(delta2 + 0.00000001);
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) v1[ix][iz] = v1[ix][iz] +  s1[ix][iz]*alpha1;
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) v2[ix][iz] = v2[ix][iz] +  s2[ix][iz]*alpha2;
      for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r1[ix][it] = (r1[ix][it] -  ss1[ix][it]*alpha1)*wd[ix];
      for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r2[ix][it] = (r2[ix][it] -  ss2[ix][it]*alpha2)*wd[ix];
      misfit1[k2*Niter + k] = cgdot(r1,nt,nmx*nsx);
      misfit2[k2*Niter + k] = cgdot(r2,nt,nmx*nsx);
      ewem_sp2d_op(r1,r2,g1,g2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nz,oz,dz,vp,vs,fmin,fmax,numthreads,true,false);
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) g1[ix][iz] = g1[ix][iz]*wm1[ix][iz];
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) g2[ix][iz] = g2[ix][iz]*wm2[ix][iz];
      gamma1 = cgdot(g1,nz,nmx);
      gamma2 = cgdot(g2,nz,nmx);
      beta1 = gamma1/(gamma1_old + 0.00000001);
      beta2 = gamma2/(gamma2_old + 0.00000001);
      gamma1_old = gamma1;
      gamma2_old = gamma2;
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) s1[ix][iz] = g1[ix][iz] + s1[ix][iz]*beta1;
      for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) s2[ix][iz] = g2[ix][iz] + s2[ix][iz]*beta2;
    }
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = v1[ix][iz]*wm1[ix][iz];
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = v2[ix][iz]*wm2[ix][iz];
    match_weights(wm1,wm2,v1,v2,nz,nmx,0.001); 
  }
  
  if (verbose) fprintf(stderr,"\r                   \n");

  free2float(r1);
  free2float(ss1);
  free2float(g1);
  free2float(s1);
  free2float(v1);

  free2float(r2);
  free2float(ss2);
  free2float(g2);
  free2float(s2);
  free2float(v2);

  free2float(wm1);
  free2float(wm2);

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

float cgdotc(sf_complex **x,int nt,int nm)
{
  /*     Compute the inner product */
  /*     dot=(x,x) for complex x */     
  int it,ix;
  float  cgdot; 
  sf_complex val;
  
  __real__ val = 0;
  __imag__ val = 0;
  for (ix=0;ix<nm;ix++) for (it=0;it<nt;it++) val = val + conjf(x[ix][it])*x[ix][it];
  cgdot= crealf(val);
  return(cgdot);
}

void match_weights(float **w1, float **w2, float **m1, float **m2, int nz, int nmx, float e)
/* create weights to improve the similarity of m1 and m2 */
{
  float *d_z;
  sf_complex *d_w,**W1,**W2,**M1,**M2,czero;
  int padz,nzfft,nw,ix,iz,iw;
  float norm1,norm2;
  int ifmax;
  
  __real__ czero = 0;
  __imag__ czero = 0;
  padz = 2;
  nzfft = padz*nz;
  nw=nzfft/2+1;
  ifmax = (int) nw/8;
  d_z = sf_floatalloc(nz);
  d_w = sf_complexalloc(nw);
  M1 = sf_complexalloc2(nw,nmx);
  M2 = sf_complexalloc2(nw,nmx);

  for (iz=0;iz<nz;iz++)  d_z[iz] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  

  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++) d_z[iz] = m1[ix][iz];
    f_op(d_w,d_z,nw,nz,1); /* d_z to d_w */
    for (iw=0;iw<nw;iw++) M1[ix][iw] = d_w[iw];
  }
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++) d_z[iz] = m2[ix][iz];
    f_op(d_w,d_z,nw,nz,1); /* d_z to d_w */
    for (iw=0;iw<nw;iw++) M2[ix][iw] = d_w[iw];
  }
 
  W1 = sf_complexalloc2(nw,nmx);
  W2 = sf_complexalloc2(nw,nmx);

  adapt(W1,M2,M1,nw,nmx,5);
  adapt(W2,M1,M2,nw,nmx,5);

  for (ix=0;ix<nmx;ix++){
    for (iw=0;iw<nw;iw++) d_w[iw] = W1[ix][iw]*M2[ix][iw];
    f_op(d_w,d_z,nw,nz,0); /* d_w to d_z */
    for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz]/((float) nzfft);
  }

  for (ix=0;ix<nmx;ix++){
    for (iw=0;iw<nw;iw++) d_w[iw] = W2[ix][iw]*M1[ix][iw];
    f_op(d_w,d_z,nw,nz,0); /* d_w to d_z */
    for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz]/((float) nzfft);
  }

/* band limit the weights along the z axis */
/*
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++) d_z[iz] = w1[ix][iz];
    bpfilter(d_z,0.004,nz,0,0,10,20);
    for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz];
  }

  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++) d_z[iz] = w2[ix][iz];
    bpfilter(d_z,0.004,nz,0,0,10,20);
    for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz];
  }
*/
/*
  norm1 = 0.0;
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++){ 
      if (fabs(w[ix][iz]) > norm1) norm1 = fabs(w[ix][iz]);
    }
  }
  
  norm1 = 0.75*norm1;
 
  for (ix=0;ix<nmx;ix++){
    for (iz=0;iz<nz;iz++){
      if (fabs(w[ix][iz]) > norm1) w[ix][iz] = norm1;
      else w[ix][iz] = fabs(w[ix][iz])/norm1;
    }
  }

*/
  free1complex(d_w);
  free2complex(W1);
  free2complex(W2);
  free2complex(M1);
  free2complex(M2);

  return;
}


void triangle_filter(sf_complex **m,int nz,int nmx,bool adj)
/*< 5 point triangle filter forward and adjoint operator. It acts on the x axis. The operator does nothing if the x axis has a length less than or equal to 5. >*/
{

  int iz, ix;
  sf_complex **a;
  a = sf_complexalloc2(nz,nmx);

  if (nmx>5){ 
    if (!adj){
      for (ix=0;ix<nmx;ix++){
        for (iz=0;iz<nz;iz++){
          a[0][iz] = (3*m[0][iz] + 4*m[1][iz] + 2*m[2][iz])/9;
          a[1][iz] = (2*m[0][iz] + 3*m[1][iz] + 2*m[2][iz] + 2*m[3][iz])/9;
          a[nmx-2][iz] = (2*m[nmx-4][iz] + 2*m[nmx-3][iz] + 3*m[nmx-2][iz] + 2*m[nmx-1][iz])/9;
          a[nmx-1][iz] = (2*m[nmx-3][iz] + 4*m[nmx-2][iz] + 3*m[nmx-1][iz])/9;
        }
      }
      for (ix=2;ix<nmx-2;ix++){
        for (iz=0;iz<nz;iz++){
          a[ix][iz] = (m[ix-2][iz] + 2*m[ix-1][iz] + 3*m[ix][iz] + 2*m[ix+1][iz] + m[ix+2][iz])/9;
    	}
      }
    }
    else {
      for (ix=0;ix<nmx;ix++){
        for (iz=0;iz<nz;iz++){
    	  a[0][iz]  = (3*m[0][iz] +   2*m[1][iz] + 1*m[2][iz])/9;
    	  a[1][iz]  = (4*m[0][iz] +   3*m[1][iz] + 2*m[2][iz] + 1*m[3][iz])/9;
    	  a[2][iz]  = (2*m[0][iz] +   2*m[1][iz] + 3*m[2][iz] + 2*m[3][iz] + 1*m[4][iz])/9;
    	  a[3][iz]  = (2*m[1][iz] +   2*m[2][iz] + 3*m[3][iz] + 2*m[4][iz] + 1*m[5][iz])/9;

    	  a[nmx-1][iz]  = (3*m[nmx-1][iz] +   2*m[nmx-2][iz] + 1*m[nmx-3][iz])/9;
    	  a[nmx-2][iz]  = (4*m[nmx-1][iz] +   3*m[nmx-2][iz] + 2*m[nmx-3][iz] + 1*m[nmx-4][iz])/9;
    	  a[nmx-3][iz]  = (2*m[nmx-1][iz] +   2*m[nmx-2][iz] + 3*m[nmx-3][iz] + 2*m[nmx-4][iz] + 1*m[nmx-5][iz])/9;
    	  a[nmx-4][iz]  = (2*m[nmx-2][iz] +   2*m[nmx-3][iz] + 3*m[nmx-4][iz] + 2*m[nmx-5][iz] + 1*m[nmx-6][iz])/9;
        }
      }
      for (ix=4;ix<nmx-4;ix++){
        for (iz=0;iz<nz;iz++){
          a[ix][iz] = (m[ix-2][iz] + 2*m[ix-1][iz] + 3*m[ix][iz] + 2*m[ix+1][iz] + m[ix+2][iz])/9;
  	}
      }
    }
  }
  for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = a[ix][iz]; 
  free2complex(a);

  return;
}

void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
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
  for(iw=ia;iw<ib;iw++) in2[iw] = out1[iw]*((float) (iw-ia)/(ib-ia)); 
  for(iw=ib;iw<ic;iw++) in2[iw] = out1[iw]; 
  for(iw=ic;iw<id;iw++) in2[iw] = out1[iw]*(1 - (float) (iw-ic)/(id-ic)); 
  for(iw=id;iw<nw;iw++) in2[iw] = czero; 
  fftwf_execute(p2); /* take the FFT along the time dimension */
  for(it=0;it<nt;it++) trace[it] = out2[it]/ntfft; 
  
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_free(in1); fftwf_free(out1);
  fftwf_free(in2); fftwf_free(out2);
  return;
}

void adapt(sf_complex **f, sf_complex **m, sf_complex **d, int nw, int nmx, int Niter)
/* find lateraly smooth filter that matches m to d */
{
  int ix,iw,k;
  float gamma,gamma_old,delta,alpha,beta,misfit;
  sf_complex **r,**ss,**g,**s,**v,czero;

  r = sf_complexalloc2(nw,nmx);
  ss = sf_complexalloc2(nw,nmx);
  g = sf_complexalloc2(nw,nmx);
  s = sf_complexalloc2(nw,nmx);
  v = sf_complexalloc2(nw,nmx);

  __real__ czero = 0;
  __imag__ czero = 0;
  for (ix=0;ix<nmx;ix++){
    for (iw=0;iw<nw;iw++){
      f[ix][iw] = czero;				
      v[ix][iw] = czero;
      g[ix][iw] = czero;
    }
  }
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) r[ix][iw] = d[ix][iw];
  conv_op(g,r,m,nw,nmx,true);     /* ADJOINT */
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) s[ix][iw] = g[ix][iw];
  gamma = cgdotc(g,nw,nmx);
  gamma_old = gamma;
  for (k=0;k<Niter;k++){
    conv_op(s,ss,m,nw,nmx,false); /* FORWARD */
    delta = cgdotc(ss,nw,nmx);
    alpha = gamma/(delta + 0.00000001);
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) v[ix][iw] = v[ix][iw] +  s[ix][iw]*alpha;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) r[ix][iw] = r[ix][iw] -  ss[ix][iw]*alpha;
    misfit = cgdotc(r,nw,nmx);
    fprintf(stderr,"misfit=%f\n",misfit);
    conv_op(g,r,m,nw,nmx,true);   /* ADJOINT */
    gamma = cgdotc(g,nw,nmx);
    beta = gamma/(gamma_old + 0.00000001);
    gamma_old = gamma;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) s[ix][iw] = g[ix][iw] + s[ix][iw]*beta;
  }
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) f[ix][iw] = v[ix][iw];

  free2complex(r);
  free2complex(ss);
  free2complex(g);
  free2complex(s);
  free2complex(v);

  return;
}

void conv_op(sf_complex **filter,sf_complex **d,sf_complex **m,int nw,int nmx,bool adj)
/* convolution/correlation fwd adj pair */
{
  int ix,iw;

  if (adj){
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) filter[ix][iw] = d[ix][iw]*conjf(m[ix][iw]);
  }
  else{
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) d[ix][iw] = filter[ix][iw]*m[ix][iw];
  }
  return;
}

