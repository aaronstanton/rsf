/* Shot Profile Wave Equation Migration with angle gather imaging condition using poynting vectors. Uses MPI over shots and OMP over frequencies.*/
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

void wem1shot(float **d, float **m, float **ang1shot,float *wav,
              int nt, float ot, float dt, 
              int nmx, int nmx2,float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
              int nz, float oz, float dz, float gz, float sz,
              float **vel, float fmin, float fmax,
              bool adj, bool calc_ang, bool verbose);

void extrap1f(float **m,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              float **u_sx, float **u_sz, float **u_gx, float **u_gz,
              int iw, int ang_iw_max, int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,int nmx2,float omx, float dmx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose);
void ssop(sf_complex *d_x,
          sf_complex *p_x, sf_complex *p_z,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *po,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, 
          bool calc_ang,
          bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
void progress_msg(float progress);
void free1int(int *p);
void free1float(float *p);
void free2float(float **p);
void free1complex(sf_complex *p);
void free2complex(sf_complex **p);
float signf(float a);
int compare (const void * a, const void * b);
void compute_angles(float **usx,float **usz,float **ugx,float **ugz,float **m,float **W,int nx,float dx,int nz,float dz,int niter,int verbose);
float cgdot(float **x,int nz,int nx);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void smooth_2d(float **in, float **out, int nx, float dx, int nz, float dz, float xa, float xb, float xc, float xd, float za, float zb, float zc, float zd, bool adj);
void cinterp1d(sf_complex *y1,sf_complex *y2,int nx1,int factor,int npoint,bool adj);

int main(int argc, char* argv[])
{

  sf_file in,intmp,intmp_ang,out,outtmp,outtmp_ang,velp,source_wavelet;
  int nt,nmx,nmx2,nz,nsx,npx;
  int it,iz,ix,isx,ipx;
  float ot,omx,oz,osx,opx;
  float dt,dmx,dmx2,dz,dsx,dpx;
  float gz,sz;
  float **m1shot,**d1shot,**ang1shot,**vp,*wav,**m;
  bool adj;
  bool verbose;
  float fmin,fmax;
  off_t iseek;
  int rank,num_procs;
  char tmpname[256];
  char tmpname_ang[256];
  float alpha,px,px_floor;
  bool calc_ang;
  bool aa;
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
  if (!sf_getbool("calc_ang",&calc_ang)) calc_ang = true; /* flag for computing angles and storing to temporary files for each shot for use in forward operator */
  if (adj){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
    if (!sf_getint("npx",&npx)) sf_error("npx must be specified");
    if (!sf_getfloat("opx",&opx)) sf_error("opx must be specified");
    if (!sf_getfloat("dpx",&dpx)) sf_error("dpx must be specified");
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
    if (!sf_getint("nsx",&nsx)) sf_error("nsx must be specified");
    if (!sf_getfloat("osx",&osx)) sf_error("osx must be specified");
    if (!sf_getfloat("dsx",&dsx)) sf_error("dsx must be specified");
    if (!sf_getint("npx",&npx)) sf_error("npx must be specified");
    if (!sf_getfloat("opx",&opx)) sf_error("opx must be specified");
    if (!sf_getfloat("dpx",&dpx)) sf_error("dpx must be specified");
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
    if (!sf_histint(  in,"n2",&nmx2)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&dmx2)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"o2",&omx)) sf_error("No o2= in input");
    if (!sf_histint(  in,"n3",&npx)) sf_error("No n3= in input");
    if (!sf_histfloat(in,"d3",&dpx)) sf_error("No d3= in input");
    if (!sf_histfloat(in,"o3",&opx)) sf_error("No o3= in input");
  }
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */

  if (!sf_getfloat("gz",&gz)) gz = 0; /* depth of the receivers */
  if (!sf_getfloat("sz",&sz)) sz = 0; /* depth of the sources */
  if (!sf_getbool("aa",&aa)) aa = false; /* anti-aliasing. if y then wavefields are interpolated by a factor of 2 prior to imaging condition. */
  if (adj){
    if (aa){ 
      nmx2 = 2*nmx-1;
      dmx2 = dmx/2;
    }
    else{
      nmx2 = nmx;
      dmx2 = dmx;
    }
  }
  else{
    if (aa){ 
      nmx = (nmx2 + 1)/2;
      dmx = dmx2*2;
    }
    else{
      nmx = nmx2;
      dmx = dmx2;
    }
  }
  if (fmax > 0.5/dt) fmax = 0.5/dt;
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  sf_fileclose(source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  sf_fileclose(velp);
  d1shot = sf_floatalloc2(nt,nmx);
  m1shot = sf_floatalloc2(nz,nmx2);
  ang1shot = sf_floatalloc2(nz,nmx2);
  if (adj){
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and migrating shot %d...\n",isx);
      iseek = (off_t)(isx*nmx*nt)*sizeof(float);
      sf_seek(in,iseek,SEEK_SET);    
      sf_floatread(d1shot[0],nt*nmx,in);
      wem1shot(d1shot,m1shot,ang1shot,wav,
               nt,ot,dt,nmx,nmx2,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
               vp,fmin,fmax,adj,calc_ang,verbose);
      sprintf(tmpname, "tmpdmig_%d.rsf",isx);
      outtmp = sf_output(tmpname);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname);
      sf_putfloat(outtmp,"o1",oz);
      sf_putfloat(outtmp,"d1",dz);
      sf_putfloat(outtmp,"n1",nz);
      sf_putstring(outtmp,"label1","Depth");
      sf_putstring(outtmp,"unit1","m");
      sf_putfloat(outtmp,"o2",omx);
      sf_putfloat(outtmp,"d2",dmx2);
      sf_putfloat(outtmp,"n2",nmx2);
      sf_putstring(outtmp,"label2","X");
      sf_putstring(outtmp,"unit2","m");
      sf_putfloat(outtmp,"o3",isx*dsx + osx);
      sf_putfloat(outtmp,"d3",dsx);
      sf_putfloat(outtmp,"n3",1);
      sf_putstring(outtmp,"label3","Source-x");
      sf_putstring(outtmp,"unit3","m");
      sf_putstring(outtmp,"title","Migrated Shot");
      sf_floatwrite(m1shot[0],nz*nmx2,outtmp);
      sf_fileclose(outtmp);
      if (calc_ang){
        sprintf(tmpname_ang, "tmpang_%d.rsf",isx);
        outtmp_ang = sf_output(tmpname_ang);
        if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname_ang);
        sf_putfloat(outtmp_ang,"o1",oz);
        sf_putfloat(outtmp_ang,"d1",dz);
        sf_putfloat(outtmp_ang,"n1",nz);
        sf_putstring(outtmp_ang,"label1","Depth");
        sf_putstring(outtmp_ang,"unit1","m");
        sf_putfloat(outtmp_ang,"o2",omx);
        sf_putfloat(outtmp_ang,"d2",dmx2);
        sf_putfloat(outtmp_ang,"n2",nmx2);
        sf_putstring(outtmp_ang,"label2","X");
        sf_putstring(outtmp_ang,"unit2","m");
        sf_putfloat(outtmp_ang,"o3",isx*dsx + osx);
        sf_putfloat(outtmp_ang,"d3",dsx);
        sf_putfloat(outtmp_ang,"n3",1);
        sf_putstring(outtmp_ang,"label3","Source-x");
        sf_putstring(outtmp_ang,"unit3","m");
        sf_putstring(outtmp_ang,"title","Angle");
        sf_floatwrite(ang1shot[0],nz*nmx2,outtmp_ang);
        sf_fileclose(outtmp_ang);
      }
    }
  }
  else{
    m = sf_floatalloc2(nz,nmx2*npx);
    for (isx=rank;isx<nsx;isx+=num_procs){
      for (ix=0;ix<nmx2;ix++) for (iz=0;iz<nz;iz++) m1shot[ix][iz] = 0.0;
      if (verbose) fprintf(stderr,"reading and de-migrating shot %d...\n",isx);
      if (isx== rank) sf_floatread(m[0],nz*nmx2*npx,in);
      sprintf(tmpname_ang, "tmpang_%d.rsf",isx);
      intmp_ang = sf_input(tmpname_ang);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname_ang); 
      sf_floatread(ang1shot[0],nz*nmx2,intmp_ang);
      for (ix=0;ix<nmx2;ix++){
        for (iz=0;iz<nz;iz++){
          if (npx>1){
            px = ang1shot[ix][iz]; 
            ipx = (int) truncf((px - opx)/dpx);
            px_floor = truncf((px - opx)/dpx)*dpx + opx;
            if (ipx >= 0 && ipx+1 < npx){
	          alpha = (px-px_floor)/dpx;
	          m1shot[ix][iz]  = (1-alpha)*m[ipx*nmx2 + ix][iz] + alpha*m[(ipx+1)*nmx2 + ix][iz];
	        }
	      }
          else{
	        m1shot[ix][iz] = m[ix][iz];
	      }
        }
      }

      wem1shot(d1shot,m1shot,ang1shot,wav,
               nt,ot,dt,nmx,nmx2,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
               vp,fmin,fmax,adj,calc_ang,verbose);

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
    free2float(m);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (adj && rank==0){
    m = sf_floatalloc2(nz,nmx2*npx);
    for (ix=0;ix<nmx2;ix++) for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m[ipx*nmx2 + ix][iz] = 0.0;
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname, "tmpdmig_%d.rsf",isx);
      intmp = sf_input(tmpname);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname); 
      sf_floatread(m1shot[0],nz*nmx2,intmp);
      sprintf(tmpname_ang, "tmpang_%d.rsf",isx);
      intmp_ang = sf_input(tmpname_ang);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname_ang); 
      sf_floatread(ang1shot[0],nz*nmx2,intmp_ang);
      for (ix=0;ix<nmx2;ix++){ 
        for (iz=0;iz<nz;iz++){
          if (npx>1){
            px = ang1shot[ix][iz]; 
            ipx = (int) truncf((px - opx)/dpx);
            px_floor = truncf((px - opx)/dpx)*dpx + opx;
            if (ipx >= 0 && ipx+1 < npx){
	          alpha = (px-px_floor)/dpx;
	          m[ipx*nmx2 + ix][iz]     += (1-alpha)*m1shot[ix][iz];
	          m[(ipx+1)*nmx2 + ix][iz] +=     alpha*m1shot[ix][iz];
	        }
	      }
          else{
	        m[ix][iz] += m1shot[ix][iz];
	      }
        }
      }
      sf_fileclose(intmp);
      sf_fileclose(intmp_ang);
    }
    
    sf_putfloat(out,"o1",oz);
    sf_putfloat(out,"d1",dz);
    sf_putfloat(out,"n1",nz);
    sf_putstring(out,"label1","Depth");
    sf_putstring(out,"unit1","m");
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d2",dmx2);
    sf_putfloat(out,"n2",nmx2);
    sf_putstring(out,"label2","X");
    sf_putstring(out,"unit2","m");
    sf_putfloat(out,"o3",opx);
    sf_putfloat(out,"d3",dpx);
    sf_putfloat(out,"n3",npx);
    sf_putstring(out,"label3","Angle");
    sf_putstring(out,"unit3"," ");
    sf_putstring(out,"title","Migrated data");
    sf_floatwrite(m[0],nz*nmx2*npx,out);
    free2float(m);
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
      sf_fileclose(intmp);
    }
  }  

  sf_fileclose(in);
  sf_fileclose(out);
  free2float(m1shot);
  free2float(d1shot);
  free2float(ang1shot);
  free2float(vp);
  free1float(wav);
  MPI_Finalize ();

  exit (0);
}

void wem1shot(float **d, float **m, float **ang1shot,float *wav,
              int nt, float ot, float dt, 
              int nmx, int nmx2, float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
              int nz, float oz, float dz, float gz, float sz,
              float **vel, float fmin, float fmax,
              bool adj, bool calc_ang, bool verbose)
/*< wave equation depth migration operator >*/
{
  int iz,ix,igx,ik,iw,it,nw,nk,ntfft,numthreads;
  float dw,dk,padt,padx;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  float **u_sx,**u_sz,**u_gx,**u_gz;
  sf_complex *d_w;
  sf_complex **d_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po,**pd,progress;
  float norm_s,norm_g;
  float val1,val2,val,max_m,denom;
  float *m_amp,**W;
  float *ang,cross_prod,dot_prod,amp,costheta;
  int nxw,nzw,ixw,izw,index1,index2;
  int ifilter,nfilter;
  
  if (adj){
    for (ix=0;ix<nmx2;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
    for (ix=0;ix<nmx2;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] *= sqrtf(nmx*nz);
  }
  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 2;
  padx = 2;
  ntfft = (int) 2*truncf(padt*((float) nt)/2);
  nw = (int) truncf(ntfft/2)+1;
  nk = padx*nmx;
  dk = 2*PI/((float) nk)/dmx;
  dw = 2*PI/((float) ntfft)/dt;
  if(fmax*dt*ntfft+1<nw) ifmax = trunc(fmax*dt*ntfft)+1;
  else ifmax = nw;
  if(fmin*dt*ntfft+1<ifmax) ifmin = trunc(fmin*dt*ntfft);
  else ifmin = 0;
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
    for (ix=0;ix<nmx;ix++) po[iz] += vel[ix][iz];
    po[iz] /= (float) nmx;
    po[iz]  = 1/po[iz];
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
  
  u_sx = sf_floatalloc2(nz,nmx2);
  u_sz = sf_floatalloc2(nz,nmx2);
  u_gx = sf_floatalloc2(nz,nmx2);
  u_gz = sf_floatalloc2(nz,nmx2);

  for (ix=0;ix<nmx2;ix++){ 
    for (iz=0;iz<nz;iz++){
      u_sx[ix][iz] = 0.0;
      u_sz[ix][iz] = 0.0;
      u_gx[ix][iz] = 0.0;
      u_gz[ix][iz] = 0.0;
    }
  }
  
  progress = 0.0;
  
  //numthreads = omp_get_num_threads();
  //if (verbose) fprintf(stderr,"using %d threads.",numthreads);
  #pragma omp parallel for private(iw) shared(m,d_g_wx,d_s_wx,u_sx,u_sz,u_gx,u_gz)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax - ifmin);
    if (verbose) progress_msg(progress);
    extrap1f(m,d_g_wx,d_s_wx,u_sx,u_sz,u_gx,u_gz,iw,(int) truncf(ifmax/2),nw,ifmax,ntfft,dw,dk,nk,nz,oz,dz,gz,sz,nmx,nmx2,omx,dmx,po,pd,i,czero,p1,p2,adj,verbose);
  }
  
  if (adj && calc_ang){
    /* calculate incidence angle from propagation vectors */  
    for (ix=0;ix<nmx2;ix++){ 
      for (iz=0;iz<nz;iz++){
        // set the polarities of the u's
        u_sx[ix][iz] *=-1;
        u_sz[ix][iz] *= 1;
        u_gx[ix][iz] *= 1;
        u_gz[ix][iz] *= 1;
        norm_s = sqrtf(powf(u_sx[ix][iz],2) + powf(u_sz[ix][iz],2));
        norm_g = sqrtf(powf(u_gx[ix][iz],2) + powf(u_gz[ix][iz],2));
        dot_prod = u_sx[ix][iz]*u_gx[ix][iz] + u_sz[ix][iz]*u_gz[ix][iz];
        cross_prod = u_sx[ix][iz]*u_gz[ix][iz] - u_sz[ix][iz]*u_gx[ix][iz];
        if (fabsf(dot_prod/(norm_s*norm_g)) > 0.02 && fabsf(dot_prod/(norm_s*norm_g)) <= 1.0){
          ang1shot[ix][iz] = signf(cross_prod)*acosf(dot_prod/(norm_s*norm_g))*90/PI;
        }
        else{
          ang1shot[ix][iz] = 90.;
        }
      }
    }

    /* median filtering of outliers angles */
    nxw=3;
    nzw=5;
    nfilter=2; // number of times to repeat median filter
    ang = sf_floatalloc(nxw*nzw); 
    for (ifilter=0;ifilter<nfilter;ifilter++){
    for (ix=0;ix<nmx2;ix++){ 
      for (iz=0;iz<nz;iz++){
        for (ixw=0;ixw<nxw;ixw++){ for (izw=0;izw<nzw;izw++){
          index1 = ix - (int) truncf(nxw/2) + ixw;
          index2 = iz - (int) truncf(nzw/2) + izw;        
          if (index1>=0 && index1<nmx2 && index2>=0 && index2<nz){
            ang[ixw*nzw + izw] = ang1shot[index1][index2];
          }
          else{
            ang[ixw*nzw + izw] = 0.0;
          }
        }}
        qsort (ang,nxw*nzw, sizeof(*ang), compare);
        if (ang1shot[ix][iz] > 2*ang[(int) truncf(nxw*nzw)/2] || ang1shot[ix][iz] < 0.5*ang[(int) truncf(nxw*nzw)/2]){
          ang1shot[ix][iz] = ang[(int) truncf(nxw*nzw)/2];
        }
      }
    }
    }
    free1float(ang);


/*
    W = sf_floatalloc2(nz,nmx);
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) W[ix][iz] = 1.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) ang1shot[ix][iz] = 0.0;
    compute_angles(u_sx,u_sz,u_gx,u_gz,ang1shot,W,nmx,dmx,nz,dz,10,true);
    free2float(W);
*/

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
  
  if (adj){
   for (ix=0;ix<nmx2;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] *= sqrtf(nmx*nz);
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
  free2float(u_sx);
  free2float(u_sz);
  free2float(u_gx);
  free2float(u_gz);

  return;
} 

void extrap1f(float **m,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              float **u_sx, float **u_sz, float **u_gx, float **u_gz,
              int iw, int ang_iw_max, int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,int nmx2,float omx, float dmx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor,z;
  int iz,ix; 
  sf_complex *d_xg,*d_xs,*p_sx,*p_sz,*p_gx,*p_gz,**smig;
  sf_complex *d_xg2,*d_xs2,*p_sx2,*p_sz2,*p_gx2,*p_gz2;
  sf_complex *d_xg3;
  
  d_xg = sf_complexalloc(nmx);
  d_xs = sf_complexalloc(nmx);
  p_sx = sf_complexalloc(nmx);
  p_sz = sf_complexalloc(nmx);
  p_gx = sf_complexalloc(nmx);
  p_gz = sf_complexalloc(nmx);

  d_xg2 = sf_complexalloc(nmx2);
  d_xs2 = sf_complexalloc(nmx2);
  p_sx2 = sf_complexalloc(nmx2);
  p_sz2 = sf_complexalloc(nmx2);
  p_gx2 = sf_complexalloc(nmx2);
  p_gz2 = sf_complexalloc(nmx2);

  d_xg3 = sf_complexalloc(nmx);


  for (ix=0;ix<nmx;ix++) d_xs[ix] = d_xg[ix] = czero;
  for (ix=0;ix<nmx2;ix++) d_xs2[ix] = d_xg2[ix] = czero;

  if (iw==0) factor = 1;
  else factor = 2;

  w = iw*dw;
  if (adj){
    for (ix=0;ix<nmx;ix++){ 
      d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
      d_xg[ix] = d_g_wx[ix][iw]/sqrtf((float) ntfft);
    }
    for (iz=0;iz<nz;iz++){ // extrapolate source and receiver wavefields
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,true,verbose);
        if (nmx2>nmx) cinterp1d(d_xs,d_xs2,nmx,2,11,true);
        else for (ix=0;ix<nmx;ix++) d_xs2[ix] = d_xs[ix]; 
      } 
      if (z >= gz){
        ssop(d_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po,pd,i,czero,p1,p2,true,true,verbose);
        if (nmx2>nmx){
          cinterp1d(d_xg,d_xg2,nmx,2,11,true);
          cinterp1d(p_sx,p_sx2,nmx,2,11,true);
          cinterp1d(p_sz,p_sz2,nmx,2,11,true);
          cinterp1d(p_gx,p_gx2,nmx,2,11,true);
          cinterp1d(p_gz,p_gz2,nmx,2,11,true);
        }
        else{ 
          for (ix=0;ix<nmx;ix++) d_xg2[ix] = d_xg[ix]; 
          for (ix=0;ix<nmx;ix++) p_sx2[ix] = p_sx[ix]; 
          for (ix=0;ix<nmx;ix++) p_sz2[ix] = p_sz[ix]; 
          for (ix=0;ix<nmx;ix++) p_gx2[ix] = p_gx[ix]; 
          for (ix=0;ix<nmx;ix++) p_gz2[ix] = p_gz[ix]; 
        }
        for (ix=0;ix<nmx2;ix++){
          #pragma omp atomic
          m[ix][iz] += factor*crealf(d_xs2[ix]*conjf(d_xg2[ix]));
          if (iw<ang_iw_max){
            #pragma omp atomic
            u_sx[ix][iz] += crealf(p_sx2[ix]*conjf(d_xg2[ix]));
            #pragma omp atomic
            u_sz[ix][iz] += crealf(p_sz2[ix]*conjf(d_xg2[ix]));
            #pragma omp atomic
            u_gx[ix][iz] += crealf(p_gx2[ix]*conjf(d_xs2[ix]));
            #pragma omp atomic
            u_gz[ix][iz] += crealf(p_gz2[ix]*conjf(d_xs2[ix]));
          }
        }
      }
    }
  }


  else{
    smig = sf_complexalloc2(nz,nmx2);
    for (ix=0;ix<nmx;ix++) d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
    for (iz=0;iz<nz;iz++){ // extrapolate source wavefield 
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,false,verbose);
        if (nmx2>nmx){ 
          cinterp1d(d_xs,d_xs2,nmx,2,11,true);
        }
        else{
          for (ix=0;ix<nmx;ix++) d_xs2[ix] = d_xs[ix]; 
        } 
        for (ix=0;ix<nmx2;ix++) smig[ix][iz] = d_xs2[ix];
      }
      else{
        for (ix=0;ix<nmx2;ix++) smig[ix][iz] = czero;
      }
    }
    for (ix=0;ix<nmx;ix++) d_xg[ix] = czero;
    for (ix=0;ix<nmx2;ix++) d_xg2[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ // extrapolate receiver wavefield 
      z = oz + dz*iz;
      if (z >= gz){
        for (ix=0;ix<nmx2;ix++){ 
          d_xg2[ix] = smig[ix][iz]*m[ix][iz];
        }
        if (nmx2>nmx){
          cinterp1d(d_xg3,d_xg2,nmx,2,11,false); // possible problem here... should be adding to d_xg recursively maybe
        }
        else{ 
          for (ix=0;ix<nmx;ix++) d_xg3[ix] = d_xg2[ix]; 
        }
        for (ix=0;ix<nmx;ix++) d_xg[ix] = d_xg[ix] + d_xg3[ix]; 
        ssop(d_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,false,false,verbose);
      }
    }
    for (ix=0;ix<nmx;ix++){
      d_g_wx[ix][iw] = d_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }


/*
  else{
    smig = sf_complexalloc2(nz,nmx2);
    for (ix=0;ix<nmx;ix++) d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
    for (iz=0;iz<nz;iz++){ // extrapolate source wavefield
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,false,verbose); 
        if (nmx2>nmx) cinterp1d(d_xs,d_xs2,nmx,2,11,true);
        else for (ix=0;ix<nmx;ix++) d_xs2[ix] = d_xs[ix]; 
        for (ix=0;ix<nmx2;ix++) smig[ix][iz] = d_xs2[ix];
      }
      else{
        for (ix=0;ix<nmx2;ix++) smig[ix][iz] = czero;
      }
    }
    for (iz=nz-1;iz>=0;iz--){ // extrapolate receiver wavefield
      z = oz + dz*iz;
      if (z >= gz){
        for (ix=0;ix<nmx2;ix++){ 
          d_xg2[ix] = d_xg2[ix] + factor*smig[ix][iz]*m[ix][iz];
        }
        if (nmx2>nmx){
          cinterp1d(d_xg,d_xg2,nmx,2,11,false);
        }
        else{ 
          for (ix=0;ix<nmx;ix++) d_xg[ix] = d_xg2[ix]; 
        }
        ssop(d_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,false,false,verbose);
      }
    }
    for (ix=0;ix<nmx;ix++){
      d_g_wx[ix][iw] = d_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
*/
  
  free1complex(d_xg);
  free1complex(d_xs);
  free1complex(p_sx);
  free1complex(p_sz);
  free1complex(p_gx);
  free1complex(p_gz);  
  free1complex(d_xg2);
  free1complex(d_xs2);
  free1complex(p_sx2);
  free1complex(p_sz2);
  free1complex(p_gx2);
  free1complex(p_gz2);
  free1complex(d_xg3);
  
  
  return;
}

void ssop(sf_complex *d_x,
          sf_complex *p_x, sf_complex *p_z,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *po,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, 
          bool calc_ang,
          bool verbose)
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
      __real__ L = cosf(w*pd[ix][iz]*dz);
      __imag__ L = sinf(w*pd[ix][iz]*dz); 
      a[ix] = d_x[ix]*L;
    }
  }
  for(ix=nmx;ix<nk;ix++) a[ix] = (fftwf_complex) czero;

  fftwf_execute_dft(p1,a,a); 
  for (ik=0;ik<nk;ik++){ 
    if (ik< (int) truncf(nk/2)) k = (float) dk*ik;
    else         k = -((float) dk*nk - dk*ik);
    s = (w*w)*(po[iz]*po[iz]) - (k*k);
    /*
    if (s>=0){ 
      __real__ L = cosf(sqrtf(s)*dz);
      __imag__ L = sinf(sqrtf(s)*dz);
    }
    else L = czero;
    */
    if (s>=0){ 
      __real__ L = cosf(sqrtf(s)*dz);
      __imag__ L = sinf(sqrtf(s)*dz);
    }
    else{ 
      __real__ L = expf(-sqrtf(fabsf(s))*fabsf(dz));
      __imag__ L = 0.0;
    }
    d_k[ik] = ((sf_complex) a[ik])*L/sqrtf((float) nk);        
  }
  for(ik=0; ik<nk;ik++) b[ik] = (fftwf_complex) d_k[ik];
  fftwf_execute_dft(p2,b,b);
  if (adj){
    for(ix=0; ix<nmx;ix++){ 
      __real__ L = cosf(w*pd[ix][iz]*dz);
      __imag__ L = sinf(w*pd[ix][iz]*dz);
      d_x[ix] = ((sf_complex) b[ix])*L/sqrtf((float) nk);
    }
  }
  else{
    for(ix=0; ix<nmx;ix++){ 
      d_x[ix] = ((sf_complex) b[ix])/sqrtf((float) nk);
    }
  }


  if (calc_ang){
    for(ik=0; ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      b[ik] = (fftwf_complex) d_k[ik]*k;
    }
    fftwf_execute_dft(p2,b,b);
    for(ix=0; ix<nmx;ix++){ 
      p_x[ix] = ((sf_complex) b[ix])/sqrtf((float) nk);
    }
    for(ik=0; ik<nk;ik++){ 
      if (ik<nk/2) k = dk*ik;
      else         k = -(dk*nk - dk*ik);
      s = (w*w)*(po[iz]*po[iz]) - (k*k);
      if (s>=0){ 
        b[ik] = (fftwf_complex) d_k[ik]*sqrtf(s);
      }
      else{ 
        b[ik] = (fftwf_complex) czero;
      }
      
    }
    fftwf_execute_dft(p2,b,b);
    for(ix=0; ix<nmx;ix++){ 
      p_z[ix] = ((sf_complex) b[ix])/sqrtf((float) nk);
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

float signf(float a)
/*< sign of a float >*/
{
 float b;
 if (a>0)      b = 1.0;
 else if (a<0) b =-1.0;
 else          b = 0.0;
 return b;
}

int compare (const void * a, const void * b)
{
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

void compute_angles(float **usx,float **usz,float **ugx,float **ugz,float **m,float **W,int nx,float dx,int nz,float dz,int niter,int verbose)
/*< Non-quadratic regularization with CG-LS. The inner CG routine is taken from Algorithm 2 of Scales, 1987. Make sure linear operator passes the dot product. >*/
{
  float **s,**s_tmp,**ss,**g,**g_tmp,**r,**A,alpha,beta,delta,gamma,gamma_old,misfit;
  float xa,xb,xc,xd,za,zb,zc,zd;
  int ix,iz,iter;
  g  = sf_floatalloc2(nz,nx);
  g_tmp  = sf_floatalloc2(nz,nx);
  r  = sf_floatalloc2(nz,nx);
  s  = sf_floatalloc2(nz,nx);
  s_tmp  = sf_floatalloc2(nz,nx);
  ss = sf_floatalloc2(nz,nx);
  A  = sf_floatalloc2(nz,nx);

  xa=0;
  xb=0;
  xc=0.03;
  xd=0.04;
  za=0;
  zb=0;
  zc=0.03;
  zd=0.04;

  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = 0.0;				
  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) r[ix][iz] = W[ix][iz]*(usx[ix][iz]*ugx[ix][iz] + usz[ix][iz]*ugz[ix][iz]); // dot product of us and ug
  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) A[ix][iz] = sqrtf(usx[ix][iz]*usx[ix][iz] + usz[ix][iz]*usz[ix][iz])*sqrtf(ugx[ix][iz]*ugx[ix][iz] + ugz[ix][iz]*ugz[ix][iz]);

  //adjoint
  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) g_tmp[ix][iz] = W[ix][iz]*A[ix][iz]*r[ix][iz];
  smooth_2d(g_tmp,g,nx,dx,nz,dz,xa,xb,xc,xd,za,zb,zc,zd,true);
  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) s[ix][iz] = g[ix][iz];
  gamma = cgdot(g,nz,nx);
  gamma_old = gamma;
  for (iter=1;iter<=niter;iter++){
    //forward
    smooth_2d(s,s_tmp,nx,dx,nz,dz,xa,xb,xc,xd,za,zb,zc,zd,false);
    for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) ss[ix][iz] = W[ix][iz]*A[ix][iz]*s_tmp[ix][iz];
    delta = cgdot(ss,nz,nx);
    alpha = gamma/(delta + 0.00000001);
    for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = m[ix][iz] +  s[ix][iz]*alpha;
    for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) r[ix][iz] = r[ix][iz] -  ss[ix][iz]*alpha;
    misfit = cgdot(r,nz,nx);
    fprintf(stderr,"misfit=%f\n",misfit);
    //adjoint
    for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) g_tmp[ix][iz] = W[ix][iz]*A[ix][iz]*r[ix][iz];
    smooth_2d(g_tmp,g,nx,dx,nz,dz,xa,xb,xc,xd,za,zb,zc,zd,true);
    gamma = cgdot(g,nz,nx);
    beta = gamma/(gamma_old + 0.00000001);
    gamma_old = gamma;
    for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) s[ix][iz] = g[ix][iz] + s[ix][iz]*beta;
  }

  for (ix=0;ix<nx;ix++) for (iz=0;iz<nz;iz++) g[ix][iz] = m[ix][iz];
  smooth_2d(g,m,nx,dx,nz,dz,xa,xb,xc,xd,za,zb,zc,zd,false);

  for (ix=0;ix<nx;ix++){ 
    for (iz=0;iz<nz;iz++){
      if (m[ix][iz] < -1.0 || m[ix][iz] > 1.0) m[ix][iz] = 0.0;
      else m[ix][iz] = signf(usx[ix][iz]*ugz[ix][iz] - usz[ix][iz]*ugx[ix][iz])*acosf(m[ix][iz])*90/PI;
    }
  }
  
  free2float(ss);
  free2float(g);
  free2float(g_tmp);
  free2float(s);
  free2float(s_tmp);
  free2float(r);
  free2float(A);
  return;
}

float cgdot(float **x,int nz,int nx)
/*< Compute the inner product for matrix of floats, x >*/
{
  int iz,ix;
  float cgdot;
  
  cgdot = 0;
  for (ix=0;ix<nx;ix++){  
    for (iz=0;iz<nz;iz++){ 
      cgdot = cgdot + x[ix][iz]*x[ix][iz];
    }
  }
  return(cgdot);
}

void smooth_2d(float **in, float **out, int nx, float dx, int nz, float dz, float xa, float xb, float xc, float xd, float za, float zb, float zc, float zd, bool adj)
/*< smooth in 2d by applying a bandpass filter on each dimension >*/
{
  float *trace1,*trace2;
  int ix,iz;

  trace1 = sf_floatalloc(nz);
  trace2 = sf_floatalloc(nx);
  if (adj){
    for (ix=0;ix<nx;ix++){  
      for (iz=0;iz<nz;iz++) trace1[iz] = in[ix][iz];    
      bpfilter(trace1,dz,nz,za,zb,zc,zd);
      for (iz=0;iz<nz;iz++) out[ix][iz] = trace1[iz];    
    }
    for (iz=0;iz<nz;iz++){  
      for (ix=0;ix<nx;ix++) trace2[ix] = out[ix][iz];    
      bpfilter(trace2,dx,nx,xa,xb,xc,xd);
      for (ix=0;ix<nx;ix++) out[ix][iz] = trace2[ix];    
    }
  }
  else{
    for (iz=0;iz<nz;iz++){  
      for (ix=0;ix<nx;ix++) trace2[ix] = in[ix][iz];    
      bpfilter(trace2,dx,nx,xa,xb,xc,xd);
      for (ix=0;ix<nx;ix++) out[ix][iz] = trace2[ix];    
    }
    for (ix=0;ix<nx;ix++){  
      for (iz=0;iz<nz;iz++) trace1[iz] = out[ix][iz];    
      bpfilter(trace1,dz,nz,za,zb,zc,zd);
      for (iz=0;iz<nz;iz++) out[ix][iz] = trace1[iz];    
    }
  }
  free1float(trace1);
  free1float(trace2);
  return;
}

void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
/*< bandpass filter >*/
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

void cinterp1d(sf_complex *y1,sf_complex *y2,int nx1,int factor,int npoint,bool adj)
/*< npoint sinc interpolation of complex data. input:  an array of length n, output: an array of length factor*n - 1 >*/
{
  int ix1,ix2,index;
  float a;
  
  
  if (adj){ 
    for (ix2=0; ix2<factor*nx1-1; ix2++){
      __real__ y2[ix2] = 0.0;
      __imag__ y2[ix2] = 0.0;    
    }
  }
  else{ 
    for (ix1=0; ix1<nx1; ix1++){
      __real__ y1[ix1] = 0.0;
      __imag__ y1[ix1] = 0.0;    
    }
  }
  for (ix2=0; ix2<factor*nx1-1; ix2++){
    index = (int) truncf((ix2+1)/factor);
    for (ix1=index - (int) truncf(npoint/2); ix1< index + (int) truncf(npoint/2); ix1++){
      if (ix2-ix1*factor != 0) a = sinf(PI * (float) (ix2-ix1*factor)/factor)/(PI * (float) (ix2-ix1*factor)/factor);
      else a = 1;
      if (adj){
        if (ix1>=0 && ix1<nx1){
          __real__ y2[ix2] += crealf(y1[ix1])*a;
          __imag__ y2[ix2] += cimagf(y1[ix1])*a;
        }
      }
      else{
        if (ix1>=0 && ix1<nx1){
          __real__ y1[ix1] += crealf(y2[ix2])*a;
          __imag__ y1[ix1] += cimagf(y2[ix2])*a;
        }
      }
    }
  }
  
  return;
}
