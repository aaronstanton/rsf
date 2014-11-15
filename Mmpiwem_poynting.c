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
              int nmx, float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
              int nz, float oz, float dz, float gz, float sz,
              float **vel, float fmin, float fmax,
              bool adj, bool verbose);

void extrap1f(float **m,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              float **u_sx, float **u_sz, float **u_gx, float **u_gz,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,float omx, float dmx,
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

int main(int argc, char* argv[])
{

  sf_file in,intmp,out,outtmp,outtmp_ang,velp,source_wavelet;
  int nt,nmx,nz,nsx,npx;
  int it,iz,ix,isx,ipx;
  float ot,omx,oz,osx,opx;
  float dt,dmx,dz,dsx,dpx;
  float gz,sz;
  float **m1shot,**d1shot,**ang1shot,**vp,*wav,**m;
  bool adj;
  bool verbose;
  float fmin,fmax;
  off_t iseek;
  int rank,num_procs;
  char tmpname[256];
  char tmpname_ang[256];

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
  m1shot = sf_floatalloc2(nz,nmx);
  ang1shot = sf_floatalloc2(nz,nmx);
  if (adj){
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and migrating shot %d...\n",isx);
      iseek = (off_t)(isx*nmx*nt)*sizeof(float);
      sf_seek(in,iseek,SEEK_SET);    
      sf_floatread(d1shot[0],nt*nmx,in);
      wem1shot(d1shot,m1shot,ang1shot,wav,
               nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
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
      sf_putfloat(outtmp,"o3",isx*dsx + osx);
      sf_putfloat(outtmp,"d3",dsx);
      sf_putfloat(outtmp,"n3",1);
      sf_putstring(outtmp,"label3","Source-x");
      sf_putstring(outtmp,"unit3","m");
      sf_putstring(outtmp,"title","Migrated Shot");
      sf_floatwrite(m1shot[0],nz*nmx,outtmp);
      sf_fileclose(outtmp);
      sprintf(tmpname_ang, "tmpang_%d.rsf",isx);
      outtmp_ang = sf_output(tmpname_ang);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname_ang);
      sf_putfloat(outtmp_ang,"o1",oz);
      sf_putfloat(outtmp_ang,"d1",dz);
      sf_putfloat(outtmp_ang,"n1",nz);
      sf_putstring(outtmp_ang,"label1","Depth");
      sf_putstring(outtmp_ang,"unit1","m");
      sf_putfloat(outtmp_ang,"o2",omx);
      sf_putfloat(outtmp_ang,"d2",dmx);
      sf_putfloat(outtmp_ang,"n2",nmx);
      sf_putstring(outtmp_ang,"label2","X");
      sf_putstring(outtmp_ang,"unit2","m");
      sf_putfloat(outtmp_ang,"o3",isx*dsx + osx);
      sf_putfloat(outtmp_ang,"d3",dsx);
      sf_putfloat(outtmp_ang,"n3",1);
      sf_putstring(outtmp_ang,"label3","Source-x");
      sf_putstring(outtmp_ang,"unit3","m");
      sf_putstring(outtmp_ang,"title","Angle");
      sf_floatwrite(ang1shot[0],nz*nmx,outtmp_ang);
      sf_fileclose(outtmp_ang);
    }
  }
  else{
    m = sf_floatalloc2(nz,nmx*npx);
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and de-migrating shot %d...\n",isx);
      if (isx== rank) sf_floatread(m[0],nz*nmx*npx,in);
      for (ix=0;ix<nmx;ix++){
        for (iz=0;iz<nz;iz++){
          // (get ipx from ang1shot[ix][iz];)  ipx = truncf((px - opx)/dpx);  
          ipx=0;
          m1shot[ix][iz] = m[ipx*nmx + ix][iz];
        }
      }
      wem1shot(d1shot,m1shot,ang1shot,wav,
               nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
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
    free2float(m);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (adj && rank==0){
    m = sf_floatalloc2(nz,nmx*npx);
    for (ix=0;ix<nmx;ix++) for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m[ipx*nmx + ix][iz] = 0.0;
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname, "tmpdmig_%d.rsf",isx);
      intmp = sf_input(tmpname);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname); 
      sf_floatread(m1shot[0],nz*nmx,intmp);
      for (ix=0;ix<nmx;ix++){ 
        for (iz=0;iz<nz;iz++){ 
          // (get ipx from ang1shot[ix][iz];)  ipx = truncf((px - opx)/dpx);  
          ipx=0;
          m[ipx*nmx + ix][iz] += m1shot[ix][iz]; 
        }
      }
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
    sf_putstring(out,"label3","Angle");
    sf_putstring(out,"unit3"," ");
    sf_putstring(out,"title","Migrated data");
    sf_floatwrite(m[0],nz*nmx*npx,out);
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
  free2float(ang1shot);
  free2float(vp);
  free1float(wav);
  MPI_Finalize ();

  exit (0);
}

void wem1shot(float **d, float **m, float **ang1shot,float *wav,
              int nt, float ot, float dt, 
              int nmx, float omx, float dmx,
              int isx, int nsx, float osx, float dsx,
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
  float **u_sx,**u_sz,**u_gx,**u_gz;
  sf_complex *d_w;
  sf_complex **d_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po,**pd,progress;
  float norm_s,norm_g,val1x,val2x,val1z,val2z;
  float ang,az_x,az_z,az,dip_x,dip_z,dip;
  
  if (adj){
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) m[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
  }
  __real__ czero = 0;
  __imag__ czero = 0;
  __real__ i = 0;
  __imag__ i = 1;
  padt = 1;
  padx = 1;
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
  
  u_sx = sf_floatalloc2(nz,nmx);
  u_sz = sf_floatalloc2(nz,nmx);
  u_gx = sf_floatalloc2(nz,nmx);
  u_gz = sf_floatalloc2(nz,nmx);

  for (ix=0;ix<nmx;ix++){ 
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
  #pragma omp parallel for private(iw) shared(m,d_g_wx,d_s_wx)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax - ifmin);
   // if (verbose) progress_msg(progress);
    extrap1f(m,d_g_wx,d_s_wx,u_sx,u_sz,u_gx,u_gz,iw,nw,ifmax,ntfft,dw,dk,nk,nz,oz,dz,gz,sz,nmx,omx,dmx,po,pd,i,czero,p1,p2,adj,verbose);
  }
  
  /* divide the propagation vectors by the image, then normalize them */
  for (ix=0;ix<nmx;ix++){ 
    for (iz=0;iz<nz;iz++){
      u_sx[ix][iz] = u_sx[ix][iz]/(m[ix][iz] + 0.0001);
      u_sz[ix][iz] = u_sz[ix][iz]/(m[ix][iz] + 0.0001);
      u_gx[ix][iz] = u_gx[ix][iz]/(m[ix][iz] + 0.0001);
      u_gz[ix][iz] = u_gz[ix][iz]/(m[ix][iz] + 0.0001);
      norm_s = sqrtf(powf(u_sx[ix][iz],2) + powf(u_sz[ix][iz],2));
      norm_g = sqrtf(powf(u_gx[ix][iz],2) + powf(u_gz[ix][iz],2));
      u_sx[ix][iz] = u_sx[ix][iz]/(norm_s + 0.0001);
      u_sz[ix][iz] = u_sz[ix][iz]/(norm_s + 0.0001);
      u_gx[ix][iz] = u_gx[ix][iz]/(norm_g + 0.0001);
      u_gz[ix][iz] =-u_gz[ix][iz]/(norm_g + 0.0001);
      val1x = u_sx[ix][iz] - u_gx[ix][iz];
      val1z = u_sz[ix][iz] - u_gz[ix][iz];
      val2x = u_sx[ix][iz] + u_gx[ix][iz];
      val2z = u_sz[ix][iz] + u_gz[ix][iz];
      ang = 90 - atanf( sqrtf(powf(val1x,2) + powf(val1z,2))/(sqrtf(powf(val2x,2) + powf(val2z,2)) + 0.00001))*180/PI;
      //ang = 90 - acosf(u_sx[ix][iz]*u_gx[ix][iz] + u_sz[ix][iz]*u_gz[ix][iz])*90/PI;
      //dip_x = (u_gx[ix][iz] + u_sx[ix][iz])/2*fabsf(cosf(ang*PI/180));
      //dip_z = (u_gz[ix][iz] + u_sz[ix][iz])/2*fabsf(cosf(ang*PI/180));
      //dip = acosf(dip_z/(sqrtf(powf(dip_x,2)+powf(dip_z,2))+0.0001))*180/PI;
      dip = 2*(90 - atanf(fabsf(u_sz[ix][iz] - u_gz[ix][iz])/(fabsf(u_gx[ix][iz] - u_sx[ix][iz])+0.001))*180/PI);
      //az_x = (u_gx[ix][iz] - u_sx[ix][iz])/2*fabsf(sinf(ang*PI/180));
      //az_z = (u_gz[ix][iz] - u_sz[ix][iz])/2*fabsf(sinf(ang*PI/180));
      //az = acosf(az_z/(sqrtf(powf(az_x,2)+powf(az_z,2))+0.0001))*180/PI;
      ang1shot[ix][iz] = ang;
    }
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
  free2float(u_sx);
  free2float(u_sz);
  free2float(u_gx);
  free2float(u_gz);
  return;
} 

void extrap1f(float **m,
              sf_complex **d_g_wx, sf_complex **d_s_wx,
              float **u_sx, float **u_sz, float **u_gx, float **u_gz,
              int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
              int nz, float oz, float dz, float gz, float sz,
              int nmx,float omx, float dmx,
              float *po,float **pd,
              sf_complex i,sf_complex czero,
              fftwf_plan p1,fftwf_plan p2,
              bool adj, bool verbose)
/*< extrapolate 1 frequency >*/
{
  float w,factor,z;
  int iz,ix; 
  sf_complex *d_xg,*d_xs,*p_sx,*p_sz,*p_gx,*p_gz,**smig;
  
  d_xg = sf_complexalloc(nmx);
  d_xs = sf_complexalloc(nmx);
  p_sx = sf_complexalloc(nmx);
  p_sz = sf_complexalloc(nmx);
  p_gx = sf_complexalloc(nmx);
  p_gz = sf_complexalloc(nmx);

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
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,true,verbose);
      } 
      if (z >= gz){
        ssop(d_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po,pd,i,czero,p1,p2,true,true,verbose);
        for (ix=0;ix<nmx;ix++){
          #pragma omp atomic
          m[ix][iz] += factor*crealf(d_xs[ix]*conjf(d_xg[ix]));
          #pragma omp atomic
          u_sx[ix][iz] += crealf(p_sx[ix]*conjf(d_xg[ix]));
          #pragma omp atomic
          u_sz[ix][iz] += crealf(p_sz[ix]*conjf(d_xg[ix]));
          #pragma omp atomic
          u_gx[ix][iz] += crealf(d_xs[ix]*conjf(p_gx[ix]));
          #pragma omp atomic
          u_gz[ix][iz] += crealf(d_xs[ix]*conjf(p_gz[ix]));
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
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,true,false,verbose); 
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
          d_xg[ix] = d_xg[ix] + smig[ix][iz]*m[ix][iz];
        }
        ssop(d_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po,pd,i,czero,p1,p2,false,false,verbose);
      }
    }
    for (ix=0;ix<nmx;ix++){
      d_g_wx[ix][iw] = d_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(d_xg);
  free1complex(d_xs);
  free1complex(p_sx);
  free1complex(p_sz);
  free1complex(p_gx);
  free1complex(p_gz);

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
        b[ik] = (fftwf_complex) d_k[ik]*sqrt(s);
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

