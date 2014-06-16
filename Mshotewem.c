/* Shot Profile PP and PS Wave Equation Migration with angle gather imaging condition.
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
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float **dip,float fmin, float fmax,
                 int numthreads,
                 bool adj, bool inv, bool verbose);
void eextrap1f(float **dmigpp_h,float **dmigps_h,
              sf_complex **dp_g_wx, sf_complex **ds_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
              int nmx, float omx, float dmx, 
              int nsx, float osx, float dsx, 
              int nhx, float ohx, float dhx, 
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
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float **dip,float fmin,float fmax,
                 int numthreads,
                 float *misfit1, float *misfit2,
                 int Niter,int Nextern,
                 bool verbose);
float cgdot(float **x,int nt,int nm);
float cgdotc(sf_complex **x,int nt,int nm);
float cgdotc1d(sf_complex *x,int nt);
void update_weights(float **w1, float **w2, float **m1, float **m2, int nz, int nmx, int mode);
void apply_weight(float **m1, float **w1, int nz, int nmx, int mode);
void triangle_filter(float **m,int nz,int nmx,bool adj);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void adapt(sf_complex **f, sf_complex **m, sf_complex **d, int nw, int nmx, int Niter);
void conv_op(sf_complex **filter,sf_complex **d,sf_complex **m,int nw,int nmx,bool adj);
void fkfilter(float **d, float dt, int nt, float dx, int nmx, int nhx, float pa, float pb, float pc, float pd);
void fk_op(sf_complex **m,float **d,int nw,int nk,int nt,int nx,bool adj);
void window_match(float **w,float **d1,float **d2,int nt,int nLt,int nLx,int nx);
void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lt,int lx1,int lx2,int lx3,int lx4);
void triangle_filter2(float **z,int nt,int nmx,int nhx,bool adj);
void offset_to_angle(float **d_h, float **d_a,
                     int nt, float ot, float dt, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx,
                     int ix, 
                     float **vp, float **vs, float **dip, 
                     float fmin, float fmax,
                     bool adj, bool ps, bool verbose);
void ps_angle(float **d_theta0, float **d_theta,
              int nz, float oz, float dz, 
              int npx, float opx, float dpx,
              int ix, 
              float **vp, float **vs, float **dip, 
              bool adj, bool verbose);

int main(int argc, char* argv[])
{

  sf_file in1,in2,out1,out2,velp,vels,dipfile,source_wavelet,misfitfile1,misfitfile2;
  int nt,nmx,nz,nsx,nhx,npx;
  int it,ix,iz;
  float ot,omx,oz,osx,ohx,opx;
  float dt,dmx,dz,dsx,dhx,dpx;
  float **dmigpp,**dmigps,**dp,**ds,**vp,**vs,**dip,*wd,*wav;
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

  sf_init (argc,argv);
  in1 = sf_input("in1");
  in2 = sf_input("in2");
  out1 = sf_output("out1");
  out2 = sf_output("out2");
  velp = sf_input("vp");
  vels = sf_input("vs");
  dipfile = sf_input("dip");
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

  if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
  if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
  if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");

  if (adj || dottest || inv || debug){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
    if (!sf_getint("npx",&npx))   npx=201; /* length of angle axis */
    if (!sf_getfloat("opx",&opx)) opx=-2; /* origin of angle axis */
    if (!sf_getfloat("dpx",&dpx)) dpx=0.02;   /* increment of angle axis */
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
  if (adj){
    if (!sf_histint(  in1,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histfloat(in1,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(in1,"o1",&ot)) ot=0.0;
    if (!sf_histint(  in1,"n2",&nmx)) sf_error("No n2= in input"); /* this assumes that your image sampling is the same as your receiver sampling. */
    if (!sf_histfloat(in1,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(in1,"o2",&omx)) omx=0.0;
    if (!sf_histint(  in1,"n3",&nsx)) nsx=1;
    if (!sf_histfloat(in1,"d3",&dsx)) dsx=1.0;
    if (!sf_histfloat(in1,"o3",&osx)) osx=0.0;
  }
  else{
    if (!sf_histint(  in1,"n1",&nz)) sf_error("No n1= in input");
    if (!sf_histfloat(in1,"d1",&dz)) sf_error("No d1= in input");
    if (!sf_histfloat(in1,"o1",&oz)) sf_error("No o1= in input");
    if (!sf_histint(  in1,"n2",&nmx)) sf_error("No n2= in input");
    if (!sf_histfloat(in1,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(in1,"o2",&omx)) sf_error("No o2= in input");
    if (!sf_histint(  in1,"n3",&npx)) sf_error("No n3= in input");
    if (!sf_histfloat(in1,"d3",&dpx)) sf_error("No d3= in input");
    if (!sf_histfloat(in1,"o3",&opx)) sf_error("No o3= in input");
  }

  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;

  if (adj){
    sf_putstring(out1,"title","PP Migration");
    sf_putfloat(out1,"o1",oz);
    sf_putfloat(out1,"d1",dz);
    sf_putfloat(out1,"n1",nz);
    sf_putstring(out1,"label1","Depth");
    sf_putstring(out1,"unit1","m");
    sf_putfloat(out1,"o2",omx);
    sf_putfloat(out1,"d2",dmx);
    sf_putfloat(out1,"n2",nmx);
    sf_putstring(out1,"label2","X");
    sf_putstring(out1,"unit2","m");
    sf_putfloat(out1,"o3",opx);
    sf_putfloat(out1,"d3",dpx);
    sf_putfloat(out1,"n3",npx);
    sf_putstring(out1,"label3","tan\\F10 q\\F3 ");
    sf_putstring(out1,"unit3"," ");
    sf_putstring(out2,"title","PS Migration");
    sf_putfloat(out2,"o1",oz);
    sf_putfloat(out2,"d1",dz);
    sf_putfloat(out2,"n1",nz);
    sf_putstring(out2,"label1","Depth");
    sf_putstring(out2,"unit1","m");
    sf_putfloat(out2,"o2",omx);
    sf_putfloat(out2,"d2",dmx);
    sf_putfloat(out2,"n2",nmx);
    sf_putstring(out2,"label2","X");
    sf_putstring(out2,"unit2","m");
    sf_putfloat(out2,"o3",opx);
    sf_putfloat(out2,"d3",dpx);
    sf_putfloat(out2,"n3",npx);
    sf_putstring(out2,"label3","tan\\F10 q\\F3 ");
    sf_putstring(out2,"unit3"," ");
  }
  else{
    sf_putstring(out1,"title","PP data");
    sf_putfloat(out1,"o1",ot);
    sf_putfloat(out1,"d1",dt);
    sf_putfloat(out1,"n1",nt);
    sf_putstring(out1,"label1","Time");
    sf_putstring(out1,"unit1","s");
    sf_putfloat(out1,"o2",omx);
    sf_putfloat(out1,"d2",dmx);
    sf_putfloat(out1,"n2",nmx);
    sf_putstring(out1,"label2","Receiver-X");
    sf_putstring(out1,"unit2","m");
    sf_putfloat(out1,"o3",osx);
    sf_putfloat(out1,"d3",dsx);
    sf_putfloat(out1,"n3",nsx);
    sf_putstring(out1,"label3","Source-X");
    sf_putstring(out1,"unit3","m");
    sf_putstring(out2,"title","PS data");
    sf_putfloat(out2,"o1",ot);
    sf_putfloat(out2,"d1",dt);
    sf_putfloat(out2,"n1",nt);
    sf_putstring(out2,"label1","Time");
    sf_putstring(out2,"unit1","s");
    sf_putfloat(out2,"o2",omx);
    sf_putfloat(out2,"d2",dmx);
    sf_putfloat(out2,"n2",nmx);
    sf_putstring(out2,"label2","Receiver-X");
    sf_putstring(out2,"unit2","m");
    sf_putfloat(out2,"o3",osx);
    sf_putfloat(out2,"d3",dsx);
    sf_putfloat(out2,"n3",nsx);
    sf_putstring(out2,"label3","Source-X");
    sf_putstring(out2,"unit3","m");
  }
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  vs = sf_floatalloc2(nz,nmx);
  dip = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  sf_floatread(vs[0],nz*nmx,vels);
  sf_floatread(dip[0],nz*nmx,dipfile);
  dp = sf_floatalloc2(nt,nmx*nsx);
  ds = sf_floatalloc2(nt,nmx*nsx);
  dmigpp = sf_floatalloc2(nz,nmx*npx);
  dmigps = sf_floatalloc2(nz,nmx*npx);
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
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = 0.0;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = 0.0;
  }
  else{
    sf_floatread(dmigpp[0],nz*nmx*npx,in1);
    sf_floatread(dmigps[0],nz*nmx*npx,in2);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) dp[ix][it] = 0.0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ds[ix][it] = 0.0;
  }


  if (dottest){
    mseed = (unsigned long) time(NULL);
    init_genrand(mseed);
    dseed = genrand_int32();
    dp_1 = sf_floatalloc2(nt,nmx*nsx);
    dp_2 = sf_floatalloc2(nt,nmx*nsx);
    ds_1 = sf_floatalloc2(nt,nmx*nsx);
    ds_2 = sf_floatalloc2(nt,nmx*nsx);
    dmigpp_1 = sf_floatalloc2(nz,nmx*npx);
    dmigpp_2 = sf_floatalloc2(nz,nmx*npx);
    dmigps_1 = sf_floatalloc2(nz,nmx*npx);
    dmigps_2 = sf_floatalloc2(nz,nmx*npx);
    init_genrand(dseed);
    for (ix=0;ix<nmx*nsx;ix++){
      for (it=0;it<nt;it++){
        dp_1[ix][it] = 0.0;
        dp_2[ix][it] = (float) 1.0*sf_randn_one_bm();
        ds_1[ix][it] = 0.0;
        ds_2[ix][it] = (float) 1.0*sf_randn_one_bm();
      }
    }
    for (ix=0;ix<nmx*npx;ix++){
      for (iz=0;iz<nz;iz++){
        dmigpp_1[ix][iz] = (float) 1.0*sf_randn_one_bm();
        dmigpp_2[ix][iz] = 0.0;
        dmigps_1[ix][iz] = (float) 1.0*sf_randn_one_bm();
        dmigps_2[ix][iz] = 0.0;
      }
    }

    ewem_sp2d_op(dp_1,ds_1,dmigpp_1,dmigps_1,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,false,true,verbose);
    ewem_sp2d_op(dp_2,ds_2,dmigpp_2,dmigps_2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,true,true,verbose);

    tmp_sum1_p=0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) tmp_sum1_p += dp_1[ix][it]*dp_2[ix][it];
    tmp_sum2_p=0;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2_p += dmigpp_1[ix][iz]*dmigpp_2[ix][iz];
    fprintf(stderr,"DOT PRODUCT (PP): %6.5f and %6.5f\n",tmp_sum1_p,tmp_sum2_p);
    tmp_sum1_s=0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) tmp_sum1_s += ds_1[ix][it]*ds_2[ix][it];
    tmp_sum2_s=0;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) tmp_sum2_s += dmigps_1[ix][iz]*dmigps_2[ix][iz];
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
    ewem_sp2d_op(dp,ds,dmigpp,dmigps,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,adj,inv,verbose);
  }
  else{
    ls_shotewem(dp,ds,dmigpp,dmigps,wav,wd,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,misfit1,misfit2,Niter,Nextern,verbose);
    sf_floatwrite(misfit1,Niter*Nextern,misfitfile1);
    sf_floatwrite(misfit2,Niter*Nextern,misfitfile2);
  }

  if (adj || inv){
    sf_floatwrite(dmigpp[0],nz*nmx*npx,out1);
    sf_floatwrite(dmigps[0],nz*nmx*npx,out2);
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
  free2float(dip);
  free1float(wd);
  free1float(wav);
  
  exit (0);

}

void ewem_sp2d_op(float **dp,float **ds,float **dmigpp,float **dmigps,float *wav,
                 int nt, float ot, float dt, 
                 int nmx, float omx, float dmx,
                 int nsx, float osx, float dsx,
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float **dip,float fmin, float fmax,
                 int numthreads,
                 bool adj, bool inv, bool verbose)
/*< 2d shot profile PP PS wave equation depth migration operator >*/
{
  int iz,ix,isx,igx,ihx,ipx,ik,iw,it,nw,nk,padt,padx,ntfft,ismooth;
  float dw,dk;
  float **dmigpp_h,**dmigpp_h_gather,**dmigpp_a_gather,**dmigpp2,**dp2;
  float **dmigps_h,**dmigps_h_gather,**dmigps_a_gather,**dmigps2,**ds2;
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
  time_t start,finish;
  double elapsed_time;

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
  dp2 = sf_floatalloc2(nt,nmx*nsx);
  ds2 = sf_floatalloc2(nt,nmx*nsx);
  dmigpp2 = sf_floatalloc2(nz,nmx*npx);
  dmigps2 = sf_floatalloc2(nz,nmx*npx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  dmigpp_h = sf_floatalloc2(nz,nmx*nhx); 
  dmigpp_h_gather = sf_floatalloc2(nz,nhx); 
  dmigpp_a_gather = sf_floatalloc2(nz,npx); 
  for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) dmigpp_h[ix][iz] = 0.0;
  dmigps_h = sf_floatalloc2(nz,nmx*nhx); 
  dmigps_h_gather = sf_floatalloc2(nz,nhx); 
  dmigps_a_gather = sf_floatalloc2(nz,npx); 
  for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) dmigps_h[ix][iz] = 0.0;


  if (adj){
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = 0.0;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = 0.0;

    for (ix=0;ix<nmx*nsx;ix++){ 
      for (it=0;it<nt;it++){ 
        dp2[ix][it] = dp[ix][it];
        ds2[ix][it] = ds[ix][it];
      }
    }
    my_taper(dp2,nt,nmx,nsx,1,1,0,20,0,0,0);
    my_taper(ds2,nt,nmx,nsx,1,1,0,20,0,0,0);
  }
  else{
    for (ix=0;ix<nmx*npx;ix++){ 
      for (iz=0;iz<nz;iz++){ 
        dmigpp2[ix][iz] = dmigpp[ix][iz];
        dmigps2[ix][iz] = dmigps[ix][iz];
      }
    }
    if (inv){ 
      for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigpp2,nz,nmx,npx,0);
      for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigps2,nz,nmx,npx,0);
      fkfilter(dmigpp2,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
      fkfilter(dmigps2,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
    }    

    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) dp[ix][it] = 0.0;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ds[ix][it] = 0.0;
    for (ix=0;ix<nmx;ix++){
      for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) dmigpp_a_gather[ipx][iz] = dmigpp2[ipx*nmx + ix][iz];
      offset_to_angle(dmigpp_h_gather,dmigpp_a_gather,
                      nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,
                      ix,
                      vp,vs,dip,
                      fmin,fmax,
                      adj,false,verbose);
      for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) dmigpp_h[ihx*nmx + ix][iz] = dmigpp_h_gather[ihx][iz]; 
    }    
    for (ix=0;ix<nmx;ix++){
      for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) dmigps_a_gather[ipx][iz] = dmigps2[ipx*nmx + ix][iz];
      offset_to_angle(dmigps_h_gather,dmigps_a_gather,
                      nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,
                      ix,
                      vp,vs,dip,
                      fmin,fmax,
                      adj,true,verbose);
      for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) dmigps_h[ihx*nmx + ix][iz] = dmigps_h_gather[ihx][iz]; 
    }    

  }

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
  start=time(0);
  igx = (int) truncf(((float) isx*dsx + osx) - omx)/dmx;  
  /* source wavefield*/
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
  for (it=0;it<nt;it++) d_t[it] = wav[it];
  f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
  for (iw=0;iw<nw;iw++) d_s_wx[igx][iw] = d_w[iw];
  /* P and S receiver wavefields*/
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = dp2[isx*nmx + ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) dp_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dp_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dp_g_wx[ix][iw] = czero;
    }
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = ds2[isx*nmx + ix][it];
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
  #pragma omp parallel for private(iw) shared(dmigpp_h,dmigps_h,dp_g_wx,ds_g_wx,d_s_wx,progress)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax);
    if (verbose) progress_msg(progress);
    eextrap1f(dmigpp_h,dmigps_h,dp_g_wx,ds_g_wx,d_s_wx,iw,nw,ifmax,ntfft,dw,dk,nk,dz,nz, 
              nmx,omx,dmx, 
              nsx,osx,dsx, 
              nhx,ohx,dhx, 
              po_p,pd_p,po_s,pd_s,i,czero,p1,p2,adj,verbose);
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
  finish=time(0);
  elapsed_time=difftime(finish,start);
  if (verbose) fprintf(stderr," elapsed time = %6.2f seconds\n",elapsed_time);
}

  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) dmigpp_h_gather[ihx][iz] = dmigpp_h[ihx*nmx + ix][iz];
      offset_to_angle(dmigpp_h_gather,dmigpp_a_gather,
                      nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,
                      ix,
                      vp,vs,dip,
                      fmin,fmax,
                      adj,false,verbose);
      for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) dmigpp2[ipx*nmx + ix][iz] = dmigpp_a_gather[ipx][iz]; 
    }
    for (ix=0;ix<nmx;ix++){
      for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) dmigps_h_gather[ihx][iz] = dmigps_h[ihx*nmx + ix][iz];
      offset_to_angle(dmigps_h_gather,dmigps_a_gather,
                      nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,
                      ix,
                      vp,vs,dip,
                      fmin,fmax,
                      adj,true,verbose);
      for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) dmigps2[ipx*nmx + ix][iz] = dmigps_a_gather[ipx][iz]; 
    }
  }

  if (adj){
    for (ix=0;ix<nmx*npx;ix++){ 
      for (iz=0;iz<nz;iz++){ 
        dmigpp[ix][iz] = dmigpp2[ix][iz];
        dmigps[ix][iz] = dmigps2[ix][iz];
      }
    }
    if (inv){ 
      fkfilter(dmigpp,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
      fkfilter(dmigps,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
      for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigpp,nz,nmx,npx,1);
      for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigps,nz,nmx,npx,1);
    }
  }
  else{
    my_taper(dp,nt,nmx,nsx,1,1,0,20,0,0,0);
    my_taper(ds,nt,nmx,nsx,1,1,0,20,0,0,0);
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
  free2float(dmigpp_h);
  free2float(dmigpp_h_gather);
  free2float(dmigpp_a_gather);
  free2float(dmigps_h);
  free2float(dmigps_h_gather);
  free2float(dmigps_a_gather);
  free2float(dmigpp2);
  free2float(dmigps2);
  free2float(dp2);
  free2float(ds2);

  return;
} 

void eextrap1f(float **dmigpp_h,float **dmigps_h,
              sf_complex **dp_g_wx, sf_complex **ds_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
              int nmx, float omx, float dmx, 
              int nsx, float osx, float dsx, 
              int nhx, float ohx, float dhx, 
              float *po_p,float **pd_p,float *po_s,float **pd_s,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor,hx,sx,gx;
  int iz,ix,ihx,isx,igx; 
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
        for (ihx=0;ihx<nhx;ihx++){
          hx = ihx*dhx + ohx;
          sx = (ix*dmx + omx) - hx;
          gx = (ix*dmx + omx) + hx;
          isx = (int) truncf((sx - omx)/dmx);
          igx = (int) truncf((gx - omx)/dmx);
          if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
            #pragma omp atomic
            dmigpp_h[ihx*nmx + ix][iz] += factor*crealf(conjf(d_xs[isx])*dp_xg[igx]);
            #pragma omp atomic 
            dmigps_h[ihx*nmx + ix][iz] += factor*crealf(conjf(d_xs[isx])*ds_xg[igx]);
          } 
        }
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
      for (ix=0;ix<nmx;ix++){
        for (ihx=0;ihx<nhx;ihx++){
          hx = ihx*dhx + ohx;
          sx = (ix*dmx + omx) - hx;
          gx = (ix*dmx + omx) + hx;
          isx = (int) truncf((sx - omx)/dmx);
          igx = (int) truncf((gx - omx)/dmx);
          if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
            dp_xg[igx] = dp_xg[igx] + smig[isx][iz]*dmigpp_h[ihx*nmx + ix][iz];
            ds_xg[igx] = ds_xg[igx] + smig[isx][iz]*dmigps_h[ihx*nmx + ix][iz];
          }
        }
      }
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
                 int nhx, float ohx, float dhx,
                 int npx, float opx, float dpx,
                 int nz, float oz, float dz,
                 float **vp,float **vs,float **dip,float fmin,float fmax,
                 int numthreads,
                 float *misfit1, float *misfit2,
                 int Niter,int Nextern,
                 bool verbose)
/*< Least squares migration. >*/
{
  int k,k2,ix,it,iz,ismooth;
  float progress;
  float gamma1,gamma1_old,delta1,alpha1,beta1,**r1,**ss1,**g1,**s1,**v1,**v1wm;
  float gamma2,gamma2_old,delta2,alpha2,beta2,**r2,**ss2,**g2,**s2,**v2,**v2wm;
  float **wm1,**wm2;
  float **tmp;

  r1 = sf_floatalloc2(nt,nmx*nsx);
  ss1 = sf_floatalloc2(nt,nmx*nsx);
  g1 = sf_floatalloc2(nz,nmx*npx);
  s1 = sf_floatalloc2(nz,nmx*npx);
  v1 = sf_floatalloc2(nz,nmx*npx);
  v1wm = sf_floatalloc2(nz,nmx*npx);

  r2 = sf_floatalloc2(nt,nmx*nsx);
  ss2 = sf_floatalloc2(nt,nmx*nsx);
  g2 = sf_floatalloc2(nz,nmx*npx);
  s2 = sf_floatalloc2(nz,nmx*npx);
  v2 = sf_floatalloc2(nz,nmx*npx);
  v2wm = sf_floatalloc2(nz,nmx*npx);

  wm1 = sf_floatalloc2(nz,nmx*npx);
  wm2 = sf_floatalloc2(nz,nmx*npx);

  tmp = sf_floatalloc2(nz,npx);

  for (ix=0;ix<nmx*npx;ix++){
    for (iz=0;iz<nz;iz++){
      dmigpp[ix][iz] = 0.0;				
      dmigps[ix][iz] = 0.0;				
      wm1[ix][iz] = 0.0;
      wm2[ix][iz] = 0.0;
      g1[ix][iz] = 0.0;
      g2[ix][iz] = 0.0;
    }
  }
  progress = 0.0;

  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) v1[ix][iz] = 0.0;
  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) v2[ix][iz] = 0.0;
  for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r1[ix][it] = dp[ix][it]*wd[ix];
  for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r2[ix][it] = ds[ix][it]*wd[ix];
  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) s1[ix][iz] = g1[ix][iz];
  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) s2[ix][iz] = g2[ix][iz];
  gamma1 = cgdot(g1,nz,nmx*npx);
  gamma1_old = gamma1;
  gamma2 = cgdot(g2,nz,nmx*npx);
  gamma2_old = gamma2;
  for (k=0;k<Niter;k++){
    progress += 1.0/((float) Niter*Nextern);
    if (verbose) progress_msg(progress);
    ewem_sp2d_op(ss1,ss2,s1,s2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,false,true,false);
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss1[ix][it] = ss1[ix][it]*wd[ix];
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) ss2[ix][it] = ss2[ix][it]*wd[ix];
    delta1 = cgdot(ss1,nt,nmx*nsx);
    alpha1 = gamma1/(delta1 + 0.00000001);
    delta2 = cgdot(ss2,nt,nmx*nsx);
    alpha2 = gamma2/(delta2 + 0.00000001);
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) v1[ix][iz] = v1[ix][iz] +  s1[ix][iz]*alpha1;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) v2[ix][iz] = v2[ix][iz] +  s2[ix][iz]*alpha2;
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r1[ix][it] = (r1[ix][it] -  ss1[ix][it]*alpha1)*wd[ix];
    for (ix=0;ix<nmx*nsx;ix++) for (it=0;it<nt;it++) r2[ix][it] = (r2[ix][it] -  ss2[ix][it]*alpha2)*wd[ix];
    misfit1[k] = cgdot(r1,nt,nmx*nsx);
    misfit2[k] = cgdot(r2,nt,nmx*nsx);
    ewem_sp2d_op(r1,r2,g1,g2,wav,nt,ot,dt,nmx,omx,dmx,nsx,osx,dsx,nhx,ohx,dhx,npx,opx,dpx,nz,oz,dz,vp,vs,dip,fmin,fmax,numthreads,true,true,false);
    gamma1 = cgdot(g1,nz,nmx*npx);
    gamma2 = cgdot(g2,nz,nmx*npx);
    beta1 = gamma1/(gamma1_old + 0.00000001);
    beta2 = gamma2/(gamma2_old + 0.00000001);
    gamma1_old = gamma1;
    gamma2_old = gamma2;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) s1[ix][iz] = g1[ix][iz] + s1[ix][iz]*beta1;
    for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) s2[ix][iz] = g2[ix][iz] + s2[ix][iz]*beta2;
  }
  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigpp[ix][iz] = v1[ix][iz];
  for (ix=0;ix<nmx*npx;ix++) for (iz=0;iz<nz;iz++) dmigps[ix][iz] = v2[ix][iz];

  /* smooth along the offset axis with a triangle filter */
  fkfilter(dmigpp,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
  fkfilter(dmigps,dz,nz,dmx,nmx,npx,-0.5,-0.25,0.25,0.5);
  for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigpp,nz,nmx,npx,1);
  for (ismooth=0;ismooth<5;ismooth++) triangle_filter2(dmigps,nz,nmx,npx,1);

  if (verbose) fprintf(stderr,"\r                   \n");

  free2float(r1);
  free2float(ss1);
  free2float(g1);
  free2float(s1);
  free2float(v1);
  free2float(v1wm);

  free2float(r2);
  free2float(ss2);
  free2float(g2);
  free2float(s2);
  free2float(v2);
  free2float(v2wm);

  free2float(wm1);
  free2float(wm2);

  free2float(tmp);

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

float cgdotc1d(sf_complex *x,int nt)
{
  /*     Compute the inner product */
  /*     dot=(x,x) for complex x */     
  int it;
  float  cgdot; 
  sf_complex val;
  
  __real__ val = 0;
  __imag__ val = 0;
  for (it=0;it<nt;it++) val = val + conjf(x[it])*x[it];
  cgdot= crealf(val);
  return(cgdot);
}

void triangle_filter(float **m,int nz,int nmx,bool adj)
/*< 5 point triangle filter forward and adjoint operator. It acts on the x axis. The operator does nothing if the x axis has a length less than or equal to 5. >*/
{

  int iz, ix;
  float **a;
  a = sf_floatalloc2(nz,nmx);

  if (nmx>5){ 
    if (!adj){
      for (iz=0;iz<nz;iz++){
        a[0][iz] = (3*m[0][iz] + 4*m[1][iz] + 2*m[2][iz])/9;
        a[1][iz] = (2*m[0][iz] + 3*m[1][iz] + 2*m[2][iz] + 2*m[3][iz])/9;
        a[nmx-2][iz] = (2*m[nmx-4][iz] + 2*m[nmx-3][iz] + 3*m[nmx-2][iz] + 2*m[nmx-1][iz])/9;
        a[nmx-1][iz] = (2*m[nmx-3][iz] + 4*m[nmx-2][iz] + 3*m[nmx-1][iz])/9;
      }
      for (ix=2;ix<nmx-2;ix++){
        for (iz=0;iz<nz;iz++){
          a[ix][iz] = (m[ix-2][iz] + 2*m[ix-1][iz] + 3*m[ix][iz] + 2*m[ix+1][iz] + m[ix+2][iz])/9;
    	}
      }
    }
    else {
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
      for (ix=4;ix<nmx-4;ix++){
        for (iz=0;iz<nz;iz++){
          a[ix][iz] = (m[ix-2][iz] + 2*m[ix-1][iz] + 3*m[ix][iz] + 2*m[ix+1][iz] + m[ix+2][iz])/9;
  	}
      }
    }
  }
  for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = a[ix][iz]; 
  free2float(a);

  return;
}

void triangle_filter2(float **z,int nt,int nmx,int nhx,bool adj)
/*< 5 point triangle filter forward and adjoint operator. It acts on the offset axis. The operator does nothing if the offset axis has a length less than or equal to 5. >*/
{
  
  float **m;
  m = sf_floatalloc2(nt,nmx*nhx);

  int it, imx, ihx;
  if (nhx>5){ 
    if (!adj){
      for (imx=0;imx<nmx;imx++){
        for (it=0;it<nt;it++){
              m[(0)*nmx + imx][it] = (3*z[(0)*nmx + imx][it] + 4*z[(1)*nmx + imx][it] + 2*z[(2)*nmx + imx][it])/9;
              m[(1)*nmx + imx][it] = (2*z[(0)*nmx + imx][it] + 3*z[(1)*nmx + imx][it] + 2*z[(2)*nmx + imx][it] + 2*z[(3)*nmx + imx][it])/9;
              m[(nhx-2)*nmx + imx][it] = (2*z[(nhx-4)*nmx + imx][it] + 2*z[(nhx-3)*nmx + imx][it] + 3*z[(nhx-2)*nmx + imx][it] + 2*z[(nhx-1)*nmx + imx][it])/9;
              m[(nhx-1)*nmx + imx][it] = (2*z[(nhx-3)*nmx + imx][it] + 4*z[(nhx-2)*nmx + imx][it] + 3*z[(nhx-1)*nmx + imx][it])/9;
        }
      }
      for (ihx=2;ihx<nhx-2;ihx++){
        for (imx=0;imx<nmx;imx++){
          for (it=0;it<nt;it++){
            m[(ihx)*nmx + imx][it] = (z[(ihx-2)*nmx + imx][it] + 2*z[(ihx-1)*nmx + imx][it] + 3*z[(ihx)*nmx + imx][it] + 2*z[(ihx+1)*nmx + imx][it] + z[(ihx+2)*nmx + imx][it])/9;
    	  }
        }
      }
    }
    else {
      for (imx=0;imx<nmx;imx++){
        for (it=0;it<nt;it++){
    	      m[(0)*nmx + imx][it]  = (3*z[(0)*nmx + imx][it] +   2*z[(1)*nmx + imx][it] + 1*z[(2)*nmx + imx][it])/9;
    	      m[(1)*nmx + imx][it]  = (4*z[(0)*nmx + imx][it] +   3*z[(1)*nmx + imx][it] + 2*z[(2)*nmx + imx][it] + 1*z[(3)*nmx + imx][it])/9;
    	      m[(2)*nmx + imx][it]  = (2*z[(0)*nmx + imx][it] +   2*z[(1)*nmx + imx][it] + 3*z[(2)*nmx + imx][it] + 2*z[(3)*nmx + imx][it] + 1*z[(4)*nmx + imx][it])/9;
    	      m[(3)*nmx + imx][it]  = (2*z[(1)*nmx + imx][it] +   2*z[(2)*nmx + imx][it] + 3*z[(3)*nmx + imx][it] + 2*z[(4)*nmx + imx][it] + 1*z[(5)*nmx + imx][it])/9;

    	      m[(nhx-1)*nmx + imx][it]  = (3*z[(nhx-1)*nmx + imx][it] +   2*z[(nhx-2)*nmx + imx][it] + 1*z[(nhx-3)*nmx + imx][it])/9;
    	      m[(nhx-2)*nmx + imx][it]  = (4*z[(nhx-1)*nmx + imx][it] +   3*z[(nhx-2)*nmx + imx][it] + 2*z[(nhx-3)*nmx + imx][it] + 1*z[(nhx-4)*nmx + imx][it])/9;
    	      m[(nhx-3)*nmx + imx][it]  = (2*z[(nhx-1)*nmx + imx][it] +   2*z[(nhx-2)*nmx + imx][it] + 3*z[(nhx-3)*nmx + imx][it] + 2*z[(nhx-4)*nmx + imx][it] + 1*z[(nhx-5)*nmx + imx][it])/9;
    	      m[(nhx-4)*nmx + imx][it]  = (2*z[(nhx-2)*nmx + imx][it] +   2*z[(nhx-3)*nmx + imx][it] + 3*z[(nhx-4)*nmx + imx][it] + 2*z[(nhx-5)*nmx + imx][it] + 1*z[(nhx-6)*nmx + imx][it])/9;
        }
      }
      for (ihx=4;ihx<nhx-4;ihx++){
        for (imx=0;imx<nmx;imx++){
          for (it=0;it<nt;it++){
            m[ihx*nmx + imx][it] = (  1*z[(ihx-2)*nmx + imx][it] 
                                    + 2*z[(ihx-1)*nmx + imx][it] 
                                    + 3*z[(ihx)*nmx   + imx][it] 
                                    + 2*z[(ihx+1)*nmx + imx][it] 
                                    + 1*z[(ihx+2)*nmx + imx][it])/9;
  	  }
        }
      }
    } 
  for (ihx=0;ihx<nhx;ihx++) for (imx=0;imx<nmx;imx++) for (it=0;it<nt;it++) z[ihx*nmx + imx][it] = m[ihx*nmx + imx][it];
  }
  free2float(m);

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


void fkfilter(float **d, float dt, int nt, float dx, int nmx, int nhx, float pa, float pb, float pc, float pd)
{
  int iw,nw,ntfft,nk,ik,padt,padx,ihx,ix,it;
  float k,w,dk,dw,p;
  sf_complex **m;
  float **d_1;
  sf_complex czero;
  __real__ czero = 0;
  __imag__ czero = 0;
  padt = 2;
  padx = 2;
  ntfft = padt*nt;
  nw=ntfft/2+1;
  nk = padx*nhx;
  dk = (float) 1/nk/dx;
  dw = (float) 1/ntfft/dt;
  d_1 = sf_floatalloc2(nt,nhx);
  m = sf_complexalloc2(nw,nk);

for (ix=0;ix<nmx;ix++){
  for (ihx=0;ihx<nhx;ihx++) for (it=0;it<nt;it++) d_1[ihx][it] = d[ihx*nmx + ix][it];
  fk_op(m,d_1,nw,nk,nt,nhx,1);
  for (iw=1;iw<nw;iw++){
    w = dw*iw;
    for (ik=0;ik<nk;ik++){
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      p = k/w;
      if (p<pa)                m[ik][iw] = czero; 
      else if (p>=pa && p<pb)  m[ik][iw] = m[ik][iw]*((p-pa)/(pb-pa)); 
      else if (p>=pb && p<=pc) m[ik][iw] = m[ik][iw]; 
      else if (p>pc && p<=pd)  m[ik][iw] = m[ik][iw]*(1-(p-pc)/(pd-pc)); 
      else                     m[ik][iw] = czero;
    }
  }
  fk_op(m,d_1,nw,nk,nt,nhx,0);
  for (ihx=0;ihx<nhx;ihx++) for (it=0;it<nt;it++) d[ihx*nmx + ix][it] = d_1[ihx][it];
}
  free2complex(m);
  free2float(d_1);
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
    for (ik=0;ik<nk;ik++) m[ik][iw] = in2a[ik]/sqrtf((float) ntfft);
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
    for(it=0;it<nt;it++) d[ix][it] = out1b[it]/sqrtf((float) ntfft); 
  }
  fftwf_destroy_plan(p1b);
  fftwf_free(in1b); fftwf_free(out1b);
}
  free2complex(cpfft);

  return;

}

void update_weights(float **w1, float **w2, float **m1, float **m2, int nz, int nmx, int mode)
/* create weights to improve the similarity of m1 and m2 */
{
  int ix,iz,iw,padz,nzfft,nw;
  sf_complex **M1,**M2,**W1,**W2,*d_w,czero;
  float *d_z;

  if (mode==1){
    window_match(w2,m1,m2,nz,nmx,10,10); /* m1*m2/m1*m1 => multiply against m1 */
    window_match(w1,m2,m1,nz,nmx,10,10); /* m2*m1/m2*m2 => multiply against m2 */
  }
  else if (mode==2){ /* estimate convolutional filters */
    padz = 2;
    nzfft = padz*nz;
    nw=nzfft/2+1;
    M1 = sf_complexalloc2(nw,nmx);
    M2 = sf_complexalloc2(nw,nmx);
    W1 = sf_complexalloc2(nw,nmx);
    W2 = sf_complexalloc2(nw,nmx);
    __real__ czero = 0;
    __imag__ czero = 0;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) W1[ix][iw] = czero;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) W2[ix][iw] = czero;
    d_z = sf_floatalloc(nz);
    d_w = sf_complexalloc(nw);
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = m1[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) M1[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
      for (iz=0;iz<nz;iz++) d_z[iz] = m2[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) M2[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
    } 
   
    for (ix=0;ix<nmx;ix++){
     for (iw=0;iw<nw;iw++){ 
       W1[ix][iw] = conjf(M2[ix][iw])*M1[ix][iw]/(conjf(M2[ix][iw])*M2[ix][iw] + 0.01);
       W2[ix][iw] = conjf(M1[ix][iw])*M2[ix][iw]/(conjf(M1[ix][iw])*M1[ix][iw] + 0.01);
     }
    }

    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++) d_w[iw] = W1[ix][iw];
      f_op(d_w,d_z,nw,nz,0);
      for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz]/sqrtf((float) nzfft);
      for (iw=0;iw<nw;iw++) d_w[iw] = W2[ix][iw];
      f_op(d_w,d_z,nw,nz,0);
      for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz]/sqrtf((float) nzfft);
    }

    for (ix=0;ix<50;ix++){ /* iterate the appliction of a triangle filter 50 times to smooth the filters laterally */
      triangle_filter(w1,nz,nmx,1);
      triangle_filter(w2,nz,nmx,1);
    }
    
    /* bandpass filter */
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = w1[ix][iz];
      bpfilter(d_z,0.004,nz,5,12,80,90);
      for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz];
    }
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = w2[ix][iz];
      bpfilter(d_z,0.004,nz,5,12,80,90);
      for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz];
    }
    
    free2complex(M1);
    free2complex(M2);
    free2complex(W1);
    free2complex(W2);
    free1float(d_z);
    free1complex(d_w);
  }
  else if (mode==3){ /* estimate convolutional filters in a least squares sense */
    padz = 2;
    nzfft = padz*nz;
    nw=nzfft/2+1;
    M1 = sf_complexalloc2(nw,nmx);
    M2 = sf_complexalloc2(nw,nmx);
    W1 = sf_complexalloc2(nw,nmx);
    W2 = sf_complexalloc2(nw,nmx);
    __real__ czero = 0;
    __imag__ czero = 0;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) W1[ix][iw] = czero;
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) W2[ix][iw] = czero;
    d_z = sf_floatalloc(nz);
    d_w = sf_complexalloc(nw);
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = m1[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) M1[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
      for (iz=0;iz<nz;iz++) d_z[iz] = m2[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) M2[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
    }
    adapt(W2,M1,M2,nw,nmx,100);
    adapt(W1,M2,M1,nw,nmx,100);
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++) d_w[iw] = W1[ix][iw];
      f_op(d_w,d_z,nw,nz,0);
      for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz]/sqrtf((float) nzfft);
      for (iw=0;iw<nw;iw++) d_w[iw] = W2[ix][iw];
      f_op(d_w,d_z,nw,nz,0);
      for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz]/sqrtf((float) nzfft);
    }

    /* bandpass filter */
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = w1[ix][iz];
      bpfilter(d_z,0.004,nz,5,12,80,90);
      for (iz=0;iz<nz;iz++) w1[ix][iz] = d_z[iz];
    }
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = w2[ix][iz];
      bpfilter(d_z,0.004,nz,5,12,80,90);
      for (iz=0;iz<nz;iz++) w2[ix][iz] = d_z[iz];
    }

    free2complex(M1);
    free2complex(M2);
    free2complex(W1);
    free2complex(W2);
    free1float(d_z);
    free1complex(d_w);
  }

  return;
}

void apply_weight(float **m1, float **w1, int nz, int nmx, int mode)
/* apply weights. multiplication if mode=1 and convolution if mode=2. */
{
  int ix,iz,iw,padz,nzfft,nw;
  sf_complex **M1,**W1,*d_w;
  float *d_z;
  
  if (mode==1){
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) m1[ix][iz] = m1[ix][iz]*w1[ix][iz];
  }
  else if (mode>1){
    padz = 2;
    nzfft = padz*nz;
    nw=nzfft/2+1;
    M1 = sf_complexalloc2(nw,nmx);
    W1 = sf_complexalloc2(nw,nmx);
    d_z = sf_floatalloc(nz);
    d_w = sf_complexalloc(nw);
    for (ix=0;ix<nmx;ix++){
      for (iz=0;iz<nz;iz++) d_z[iz] = m1[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) M1[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
      for (iz=0;iz<nz;iz++) d_z[iz] = w1[ix][iz];
      f_op(d_w,d_z,nw,nz,1);
      for (iw=0;iw<nw;iw++) W1[ix][iw] = d_w[iw]/sqrtf((float) nzfft);
    }
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) M1[ix][iw] = M1[ix][iw]*W1[ix][iw];
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++) d_w[iw] = M1[ix][iw];
      f_op(d_w,d_z,nw,nz,0);
      for (iz=0;iz<nz;iz++) m1[ix][iz] = d_z[iz]/sqrtf((float) nzfft);
    }
    free2complex(M1);
    free2complex(W1);
    free1float(d_z);
    free1complex(d_w);
  }

  return;
}


void adapt(sf_complex **f, sf_complex **m, sf_complex **d, int nw, int nmx, int Niter)
/* find nmx 1-D filters that best match m to d */
{
  int ix,iw,k;
  float gamma,gamma_old,delta,alpha,beta;
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
    for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) r[ix][iw] = r[ix][iw] - ss[ix][iw]*alpha;
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

void window_match(float **w,float **d1,float **d2,int nt,int nx,int nLt,int nLx)
{
  int it,ix,iLt,iLx;
  float a,b;
  for (ix=0;ix<nx;ix++){
  for (it=0;it<nt;it++){
    a = 0.0;
    b = 0.0;
    for (iLx=-nLx/2;iLx<=nLx/2;iLx++){
    for (iLt=-nLt/2;iLt<=nLt/2;iLt++){
      if (it+iLt > 0 && it+iLt < nt && ix+iLx > 0 && ix+iLx < nx){ 
        a += d1[ix+iLx][it+iLt]*d2[ix+iLx][it+iLt];
        b += d1[ix+iLx][it+iLt]*d1[ix+iLx][it+iLt];
      }
    }
    }
    w[ix][it] = a/(b+0.001);
  }
  }
  
  return;
}

void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lt,int lx1,int lx2,int lx3,int lx4)
{
  int it,ix,ix1,ix2,ix3,ix4;
  float tt,tx1,tx2,tx3,tx4;

  tx1=1;tx2=1;tx3=1;tx4=1;
  for (ix1=0;ix1<nx1;ix1++){
    if (ix1>=0   && ix1<lx1) tx1 = cosf((1-(float) ix1/lx1)*PI/2);
    else if (ix1>nx1-lx1 && ix1<nx1) tx1 = cosf(((float) (ix1-nx1+lx1)/lx1)*PI/2);
    else tx1 = 1;
  for (ix2=0;ix2<nx2;ix2++){
    if (ix2>=0   && ix2<lx2) tx2 = cosf((1-(float) ix2/lx2)*PI/2);
    else if (ix2>nx2-lx2 && ix2<nx2) tx2 = cosf(((float) (ix2-nx2+lx2)/lx2)*PI/2);
    else tx2 = 1;
  for (ix3=0;ix3<nx3;ix3++){
    if (ix3>=0   && ix3<lx3) tx3 = cosf((1-(float) ix3/lx3)*PI/2);
    else if (ix3>nx3-lx3 && ix3<nx3) tx3 = cosf(((float) (ix3-nx3+lx3)/lx3)*PI/2);
    else tx3 = 1;
  for (ix4=0;ix4<nx4;ix4++){
    if (ix4>=0   && ix4<lx4) tx4 = cosf((1-(float) ix4/lx4)*PI/2);
    else if (ix4>nx4-lx4 && ix4<nx4) tx4 = cosf(((float) (ix4-nx4+lx4)/lx4)*PI/2);
    else tx4 = 1;
    ix = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1;
    for(it=0;it<nt;it++){
      if (it>=0   && it<lt) tt = cosf((1-(float) it/lt)*PI/2);
      else if (it>nt-lt && it<nt) tt = cosf(((float) (it-nt+lt)/lt)*PI/2);
      else tt = 1;
      d[ix][it] = tt*tx1*tx2*tx3*tx4*d[ix][it];
    }
  }
  }
  }
  }
  return;
}

void offset_to_angle(float **d_h, float **d_a,
                     int nz, float oz, float dz, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx,
                     int ix, 
                     float **vp, float **vs, float **dip, 
                     float fmin, float fmax,
                     bool adj, bool ps, bool verbose)
/*< Convert from offset to angle (adj=true) or from angle to offset (adj=false) angle is expressed as angle = tan(theta) >*/
{

  int iz,ihx,ipx,ik,iw,nw,nk,padz,padx,nzfft,ifmin,ifmax;
  float dw,dk,px;
  sf_complex czero;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_wh;
  sf_complex **d_wa;
  fftwf_complex *a,*b;
  sf_complex L;
  int *n;
  fftwf_plan p1,p2;
  float w,k;
  float **d_a0;

  __real__ czero = 0;
  __imag__ czero = 0;
  padz = 4;
  padx = 8;
  nzfft = padz*nz;
  nw=nzfft/2+1;

  if(fmax*dz*nzfft+1<nw) ifmax = trunc(fmax*dz*nzfft)+1;
  else ifmax = nw;
  if(fmin*dz*nzfft+1<ifmax) ifmin = trunc(fmin*dz*nzfft);
  else ifmin = 0;

  nk = padx*nhx;
  dk = 2*PI/((float) nk)/dhx;
  dw = 2*PI/((float) nzfft)/dz;
  d_wh = sf_complexalloc2(nw,nhx);
  d_wa = sf_complexalloc2(nw,npx);
  d_t = sf_floatalloc(nz);
  d_w = sf_complexalloc(nw);
  for (iz=0;iz<nz;iz++)  d_t[iz] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  

  n = sf_intalloc(1); 
  n[0] = nk;

if (adj){
  a  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  p1 = fftwf_plan_dft(1, n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
  for (iw=0;iw<nw;iw++) for (ipx=0;ipx<npx;ipx++) d_wa[ipx][iw] = czero;
  /* transform the time axis to the frequency domain */
  for (ihx=0;ihx<nhx;ihx++){
    for (iz=0;iz<nz;iz++) d_t[iz] = d_h[ihx][iz];
    f_op(d_w,d_t,nw,nz,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_wh[ihx][iw] = d_w[iw]/sqrtf((float) nzfft);
  }

  /* transform the offset axis to the frequency domain */
  for (iw=ifmin;iw<ifmax;iw++){
    w = iw*dw;
    for (ihx=0;ihx<nhx;ihx++) a[ihx] = d_wh[ihx][iw];
    for (ihx=nhx;ihx<nk;ihx++) a[ihx] = czero;
    fftwf_execute_dft(p1,a,a); 
    /* compute Ray Parameters */
    for (ipx=0;ipx<npx;ipx++){
      px = ipx*dpx + opx;
      k = -px*w;
      if (k>0){ 
        ik = truncf(k/dk);
      }
      else{ 
        ik = truncf((dk*nk + k)/dk);
      }
      __real__ L = cos(-k*ohx);
      __imag__ L = sin(-k*ohx);
      if (ik < nk && ik >= 0){
        d_wa[ipx][iw] += L*a[ik]/sqrtf((float) nk);
      }
    }
  }
      
  /* transform the frequency axis to the depth domain */
  for (ipx=0;ipx<npx;ipx++){
    for (iw=0;iw<nw;iw++) d_w[iw] = d_wa[ipx][iw];
    f_op(d_w,d_t,nw,nz,0); /* d_w to d_z */
    for (iz=0;iz<nz;iz++) d_a[ipx][iz] = d_t[iz]/sqrtf((float) nzfft);
  }
  fftwf_destroy_plan(p1);
  fftwf_free(a);

/*
  if (ps){
    d_a0 = sf_floatalloc2(nz,npx);
    for (ipx=0;ipx<npx;ipx++){
      for (iz=0;iz<nz;iz++){ 
        d_a0[ipx][iz] = d_a[ipx][iz];
        d_a[ipx][iz] = 0.0;
      }
    }
    ps_angle(d_a0,d_a,nz,oz,dz,npx,opx,dpx,ix,vp,vs,dip,adj,verbose);
    free2float(d_a0);
  }
*/

}
else{
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  p2 = fftwf_plan_dft(1, n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (iw=0;iw<nw;iw++) for (ihx=0;ihx<nhx;ihx++) d_wh[ihx][iw] = czero;

/*
  if (ps){
    d_a0 = sf_floatalloc2(nz,npx);
    for (ipx=0;ipx<npx;ipx++){
      for (iz=0;iz<nz;iz++){ 
        d_a0[ipx][iz] = 0.0;
      }
    }
    ps_angle(d_a0,d_a,nz,oz,dz,npx,opx,dpx,ix,vp,vs,dip,adj,verbose);
    for (ipx=0;ipx<npx;ipx++){
      for (iz=0;iz<nz;iz++){ 
        d_a[ipx][iz] = d_a0[ipx][iz];
      }
    }
    free2float(d_a0);
  }
*/

  /* transform the time axis to the frequency domain */
  for (ipx=0;ipx<npx;ipx++){
    for (iz=0;iz<nz;iz++) d_t[iz] = d_a[ipx][iz];
    f_op(d_w,d_t,nw,nz,1); /* d_t to d_w */
    for (iw=0;iw<nw;iw++) d_wa[ipx][iw] = d_w[iw]/sqrtf((float) nzfft);
  }

  /* transform the offset axis to the frequency domain */
  for (iw=ifmin;iw<ifmax;iw++){
    w = iw*dw;
    for (ihx=0;ihx<nk;ihx++) b[ihx] = czero;
    /* compute wavenumbers from ray parameters */
    for (ipx=0;ipx<npx;ipx++){
      px = ipx*dpx + opx;
      k = -px*w;
      if (k>0){ 
        ik = truncf(k/dk);
      }
      else{ 
        ik = truncf((dk*nk + k)/dk);
      }
      __real__ L = cos(k*ohx);
      __imag__ L = sin(k*ohx);
      if (ik < nk && ik >= 0){
        b[ik] += L*d_wa[ipx][iw];
      }
    }
    fftwf_execute_dft(p2,b,b); 
    for (ihx=0;ihx<nhx;ihx++) d_wh[ihx][iw] = b[ihx]/sqrtf((float) nk);
  }
      
  /* transform the frequency axis to the depth domain */
  for (ihx=0;ihx<nhx;ihx++){
    for (iw=0;iw<nw;iw++) d_w[iw] = d_wh[ihx][iw];
    f_op(d_w,d_t,nw,nz,0); /* d_w to d_z */
    for (iz=0;iz<nz;iz++) d_h[ihx][iz] = d_t[iz]/sqrtf((float) nzfft);
  }
  fftwf_destroy_plan(p2);
  fftwf_free(b);
}

  free2complex(d_wh);
  free2complex(d_wa);
  free1float(d_t);
  free1complex(d_w);
  return;
}

void ps_angle(float **d_theta0, float **d_theta,
              int nz, float oz, float dz, 
              int npx, float opx, float dpx,
              int ix, 
              float **vp, float **vs, float **dip, 
              bool adj, bool verbose)
{
  int iz,ip0,ip;
  float gamma,p,p0;

  for (iz=0;iz<nz;iz++){
    gamma = vp[ix][iz]/vs[ix][iz];
    for (ip0=0;ip0<npx;ip0++){
      p0 = dpx*ip0 + opx; 
      p = (4*gamma*p0 + dip[ix][iz]*(gamma*gamma - 1)*(p0*p0 + 1))/( p0*p0*(gamma-1)*(gamma-1) + (gamma+1)*(gamma+1) ); 
      ip = (int) truncf((p - opx)/dpx);
      if (ip>=0 && ip<npx){
        if (adj) d_theta[ip][iz] += d_theta0[ip0][iz];
        else d_theta0[ip0][iz] += d_theta[ip][iz];
      }
    }
  }

  return;
}

