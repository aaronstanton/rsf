/* Shot Profile Wave Equation Migration with angle gather imaging condition. Uses MPI over shots and OMP over frequencies.*/
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
#include <mpi.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

void wem1shot(float **d, float **m,float *wav,
              int nt, float ot, float dt, 
              int nmx, float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
              int nhx, float ohx, float dhx,
              int nz, float oz, float dz, float gz, float sz,
              float **vel, float fmin, float fmax,
              bool adj, bool verbose);

void extrap1f(float **dmig_h,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,float omx, float dmx,
              int nhx,float ohx, float dhx,
              float *po,float **pd,
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
void offset_to_angle(float **d_h, float **d_a,
                     int nt, float ot, float dt, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx, 
                     float fmin, float fmax,
                     bool adj, bool verbose);
void progress_msg(float progress);
void free1int(int *p);
void free1float(float *p);
void free2float(float **p);
void free1complex(sf_complex *p);
void free2complex(sf_complex **p);

int main(int argc, char* argv[])
{

  sf_file in,intmp,out,outtmp,velp,source_wavelet;
  int nt,nmx,nz,nsx,nhx,npx;
  int it,iz,ix,isx,ihx,ipx;
  float ot,omx,oz,osx,ohx,opx;
  float dt,dmx,dz,dsx,dhx,dpx;
  float gz,sz;
  float **m1shot,**d1shot,**vp,*wav,**m_h,**m,**m_h_gather,**m_a_gather;
  bool adj;
  bool verbose;
  float fmin,fmax;
  off_t iseek;
  int rank,num_procs;
  char tmpname[256];

  MPI_Status mpi_stat;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  sf_init(argc,argv);
  in = sf_input("infile");
  out = sf_output("outfile");
  velp = sf_input("vp");
  source_wavelet = sf_input("wav");
  //if (verbose) fprintf(stderr,"rank=%d\n",rank);
  //if (verbose) fprintf(stderr,"num_procs=%d\n",num_procs);
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (adj){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
    if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
    if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
    if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");
    if (!sf_getint("npx",&npx))   npx=201; /* length of angle axis */
    if (!sf_getfloat("opx",&opx)) opx=-2; /* origin of angle axis */
    if (!sf_getfloat("dpx",&dpx)) dpx=0.02;   /* increment of angle axis */
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
    if (!sf_getint("nsx",&nsx)) sf_error("nsx must be specified");
    if (!sf_getfloat("osx",&osx)) sf_error("osx must be specified");
    if (!sf_getfloat("dsx",&dsx)) sf_error("dsx must be specified");
    if (!sf_getint("nhx",&nhx)) sf_error("nhx must be specified");
    if (!sf_getfloat("ohx",&ohx)) sf_error("ohx must be specified");
    if (!sf_getfloat("dhx",&dhx)) sf_error("dhx must be specified");
  }
  /* read input file parameters */
  if (adj){
    if (!sf_histint(  in,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&ot)) ot=0.0;
    if (!sf_histint(  in,"n2",&nmx)) sf_error("No n2= in input"); /* this assumes that your image sampling is the same as your receiver sampling. */
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

  if (!sf_getfloat("gz",&gz)) gz = 0; /* depth of the receivers */
  if (!sf_getfloat("sz",&sz)) sz = 0; /* depth of the sources */

  if (fmax > 0.5/dt) fmax = 0.5/dt;
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  sf_fileclose(source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  sf_fileclose(velp);
  d1shot = sf_floatalloc2(nt,nmx);
  m1shot = sf_floatalloc2(nz,nmx*nhx);
  if (adj){
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and migrating shot %d...\n",isx);
      iseek = (off_t)(isx*nmx*nt)*sizeof(float);
      sf_seek(in,iseek,SEEK_SET);    
      sf_floatread(d1shot[0],nt*nmx,in);
      wem1shot(d1shot,m1shot,wav,
               nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nhx,ohx,dhx,nz,oz,dz,gz,sz,
               vp,fmin,fmax,adj,verbose);
      sprintf(tmpname, "tmpdmig_%d.rsf",isx);
      outtmp = sf_output(tmpname);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname);
      sf_putfloat(outtmp,"o1",oz);
      sf_putfloat(outtmp,"d1",dz);
      sf_putfloat(outtmp,"n1",nz);
      sf_putstring(outtmp,"label1","Depth");
      sf_putstring(outtmp,"unit1","m");
      sf_putfloat(outtmp,"o2",omx);
      sf_putfloat(outtmp,"d2",dmx);
      sf_putfloat(outtmp,"n2",nmx);
      sf_putstring(outtmp,"label2","X");
      sf_putstring(outtmp,"unit2","m");
      sf_putfloat(outtmp,"o3",ohx);
      sf_putfloat(outtmp,"d3",dhx);
      sf_putfloat(outtmp,"n3",nhx);
      sf_putstring(outtmp,"label3","Offset");
      sf_putstring(outtmp,"unit3"," ");
      sf_putfloat(outtmp,"o4",isx*dsx + osx);
      sf_putfloat(outtmp,"d4",dsx);
      sf_putfloat(outtmp,"n4",1);
      sf_putstring(outtmp,"label4","Source-x");
      sf_putstring(outtmp,"unit4","m");
      sf_putstring(outtmp,"title","Migrated Data");
      sf_floatwrite(m1shot[0],nz*nmx*nhx,outtmp);
      sf_fileclose(outtmp);
    }
  }
  else{
    m          = sf_floatalloc2(nz,nmx*npx);
    m_h_gather = sf_floatalloc2(nz,nhx);
    m_a_gather = sf_floatalloc2(nz,npx);
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and de-migrating shot %d...\n",isx);
      if (isx== rank) sf_floatread(m[0],nz*nmx*npx,in);
      for (ix=0;ix<nmx;ix++){
        for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m_a_gather[ipx][iz] = m[ipx*nmx + ix][iz];
        if (nhx>1){ 
          offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
        }
        else{ 
          for (iz=0;iz<nz;iz++) m_h_gather[0][iz] = m_a_gather[0][iz];
        } 
        for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) m1shot[ihx*nmx + ix][iz] = m_h_gather[ihx][iz]; 
      }
      wem1shot(d1shot,m1shot,wav,
               nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nhx,ohx,dhx,nz,oz,dz,gz,sz,
               vp,fmin,fmax,adj,verbose);
      sprintf(tmpname, "tmpd_%d.rsf",isx);
      outtmp = sf_output(tmpname);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname);
      sf_putfloat(outtmp,"o1",ot);
      sf_putfloat(outtmp,"d1",dt);
      sf_putfloat(outtmp,"n1",nt);
      sf_putstring(outtmp,"label1","Time");
      sf_putstring(outtmp,"unit1","s");
      sf_putfloat(outtmp,"o2",omx);
      sf_putfloat(outtmp,"d2",dmx);
      sf_putfloat(outtmp,"n2",nmx);
      sf_putstring(outtmp,"label2","Receiver-X");
      sf_putstring(outtmp,"unit2","m");
      sf_putfloat(outtmp,"o3",isx*dsx + osx);
      sf_putfloat(outtmp,"d3",dsx);
      sf_putfloat(outtmp,"n3",1);
      sf_putstring(outtmp,"label3","Source-X");
      sf_putstring(outtmp,"unit3","m");
      sf_putstring(outtmp,"title","Data");
      sf_floatwrite(d1shot[0],nt*nmx,outtmp);
      sf_fileclose(outtmp);
    }
    free2float(m_h_gather);
    free2float(m_a_gather);
    free2float(m);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (adj && rank==0){
    m          = sf_floatalloc2(nz,nmx*npx);
    m_h        = sf_floatalloc2(nz,nmx*nhx);
    m_h_gather = sf_floatalloc2(nz,nhx);
    m_a_gather = sf_floatalloc2(nz,npx);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname, "tmpdmig_%d.rsf",isx);
      intmp = sf_input(tmpname);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname); 
      sf_floatread(m1shot[0],nz*nmx*nhx,intmp);
      for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] += m1shot[ix][iz];
    }
    for (ix=0;ix<nmx;ix++){
      for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) m_h_gather[ihx][iz] = m_h[ihx*nmx + ix][iz];
      if (nhx>1){ 
        offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
      }
      else{ 
        for (iz=0;iz<nz;iz++) m_a_gather[0][iz] = m_h_gather[0][iz];
      } 
      for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m[ipx*nmx + ix][iz] = m_a_gather[ipx][iz]; 
    }
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
    sf_putfloat(out,"o3",opx);
    sf_putfloat(out,"d3",dpx);
    sf_putfloat(out,"n3",npx);
    sf_putstring(out,"label3","tan\\F10 q\\F3 ");
    sf_putstring(out,"unit3"," ");
    sf_putstring(out,"title","Migrated data");
    sf_floatwrite(m[0],nz*nmx*npx,out);
    free2float(m_h);
    free2float(m_h_gather);
    free2float(m_a_gather);
    free2float(m);
    sf_fileclose(intmp);
  }  
  else if (!adj && rank==0){
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
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname, "tmpd_%d.rsf",isx);
      intmp = sf_input(tmpname);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname); 
      sf_floatread(d1shot[0],nt*nmx,intmp);
      sf_floatwrite(d1shot[0],nt*nmx,out);
    }
    sf_fileclose(intmp);
  }  

  sf_fileclose(in);
  sf_fileclose(out);
  free2float(m1shot);
  free2float(d1shot);
  free2float(vp);
  free1float(wav);
  MPI_Finalize ();

  exit (0);
}

void wem1shot(float **d, float **m,float *wav,
              int nt, float ot, float dt, 
              int nmx, float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
              int nhx, float ohx, float dhx,
              int nz, float oz, float dz, float gz, float sz,
              float **vel, float fmin, float fmax,
              bool adj, bool verbose)
/*< wave equation depth migration operator >*/
{
  int iz,ix,igx,ik,iw,it,nw,nk,padt,padx,ntfft,numthreads;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **d_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po,**pd,progress;
  if (adj){
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
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
  /* set up fftw plans and pass them to the OMP region of the code */
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
  /**********************************************************************/
  igx = (int) truncf(((float) isx*dsx + osx) - omx)/dmx; /*position to inject source*/
  /* source wavefield*/
  for (ix=0;ix<nmx;ix++) for (iw=0;iw<nw;iw++) d_s_wx[ix][iw] = czero;
  for (it=0;it<nt;it++) d_t[it] = wav[it];
  f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
  for (iw=0;iw<nw;iw++) d_s_wx[igx][iw] = d_w[iw];
  /* receiver wavefield*/
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d_t[it] = d[ix][it];
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
  //numthreads = omp_get_num_threads();
  //if (verbose) fprintf(stderr,"using %d threads.",numthreads);
  #pragma omp parallel for private(iw) shared(m,d_g_wx,d_s_wx)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax - ifmin);
   // if (verbose) progress_msg(progress);
    extrap1f(m,d_g_wx,d_s_wx,iw,nw,ifmax,ntfft,dw,dk,nk,nz,oz,dz,gz,sz,nmx,omx,dmx,nhx,ohx,dhx,po,pd,i,czero,p1,p2,adj,verbose);
  }
  if (!adj){
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = d_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) d[ix][it] = d_t[it];
    }
  }

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

  return;
} 

void extrap1f(float **dmig_h,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,float omx, float dmx,
              int nhx,float ohx, float dhx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor,hx,sx,gx,z;
  int iz,ix,ihx,isx,igx; 
  sf_complex *d_xg,*d_xs,**smig;
  
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
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,verbose);
      } 
      if (z >= gz){
        ssop(d_xg,w,dk,nk,nmx,dz,iz,po,pd,i,czero,p1,p2,true,verbose);
        for (ix=0;ix<nmx;ix++){
          for (ihx=0;ihx<nhx;ihx++){
            hx = ihx*dhx + ohx;
            sx = (ix*dmx + omx) - hx;
            gx = (ix*dmx + omx) + hx;
            isx = (int) truncf((sx - omx)/dmx);
            igx = (int) truncf((gx - omx)/dmx);
            if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
              #pragma omp atomic
              dmig_h[ihx*nmx + ix][iz] += factor*crealf(d_xs[isx]*conjf(d_xg[igx]));
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
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,verbose); 
        for (ix=0;ix<nmx;ix++) smig[ix][iz] = d_xs[ix];;
      }
      else{
        for (ix=0;ix<nmx;ix++) smig[ix][iz] = czero;
      }
    }
    for (ix=0;ix<nmx;ix++) d_xg[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ /* extrapolate receiver wavefield */
      z = oz + dz*iz;
      if (z >= gz){
        for (ix=0;ix<nmx;ix++){ 
          for (ihx=0;ihx<nhx;ihx++){
            hx = ihx*dhx + ohx;
            sx = (ix*dmx + omx) - hx;
            gx = (ix*dmx + omx) + hx;
            isx = (int) truncf((sx - omx)/dmx);
            igx = (int) truncf((gx - omx)/dmx);
            if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
              d_xg[igx] = d_xg[igx] + smig[isx][iz]*dmig_h[ihx*nmx + ix][iz];
            }
          }
        }
        ssop(d_xg,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,false,verbose);
      }
    }
    for (ix=0;ix<nmx;ix++){
      d_g_wx[ix][iw] = d_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(d_xg);
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

void offset_to_angle(float **d_h, float **d_a,
                     int nz, float oz, float dz, 
                     int nhx, float ohx, float dhx, 
                     int npx, float opx, float dpx,
                     float fmin, float fmax,
                     bool adj, bool verbose)
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

  __real__ czero = 0;
  __imag__ czero = 0;
  padz = 4;
  padx = 4;
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

}
else{
  b  = fftwf_malloc(sizeof(fftwf_complex) * nk);
  p2 = fftwf_plan_dft(1, n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
  for (iw=0;iw<nw;iw++) for (ihx=0;ihx<nhx;ihx++) d_wh[ihx][iw] = czero;

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

void progress_msg(float progress)
{ 
  fprintf(stderr,"[%6.2f%% complete]\n",progress*100);
  return;
}

void free1int(int *p)
/*< free1int >*/
{
	free(p);
}

void free1float(float *p)
/*< free1float >*/
{
	free(p);
}

void free2float(float **p)
/*< free2float >*/
{
	free(*p);
	free(p);
}

void free1complex(sf_complex *p)
/*< free1complex >*/
{
	free(p);
}

void free2complex(sf_complex **p)
/*< free2complex >*/
{
	free(*p);
	free(p);
}

