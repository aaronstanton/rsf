/* Shot Profile Wave Equation Migration of 2C data with angle gather imaging condition. Using MPI.*/
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

void ewem1shot(float **dx_1shot, float **dz_1shot,
               float **mppx, float **mpsx, float **mppz, float **mpsz,
               float *wav,
               int nt, float ot, float dt, 
               int nmx, float omx, float dmx,
               int isx, int nsx, float osx, float dsx,
               int nhx, float ohx, float dhx,
               int nz, float oz, float dz,
               float **vp, float **vs,
               float fmin, float fmax,
               bool adj, bool verbose);
void eextrap1f(float **mppx, float **mpsx, float **mppz, float **mpsz,
               sf_complex **dpx_g_wx, sf_complex **dsx_g_wx, sf_complex **dpz_g_wx, sf_complex **dsz_g_wx, sf_complex **d_s_wx,
               int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
               int nmx,float omx, float dmx,
               int nhx,float ohx, float dhx,
               float *po,float **pd,float *po_s,float **pd_s,
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
void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile);

int main(int argc, char* argv[])
{

  sf_file fp_dx,fp_dz,fp_mppx,fp_mpsx,fp_mppz,fp_mpsz;
  sf_file fp_tmp_dx,fp_tmp_dz;
  sf_file fp_tmp_mppx,fp_tmp_mpsx,fp_tmp_mppz,fp_tmp_mpsz;
  sf_file velp,vels,source_wavelet;
  int nt,nmx,nz,nsx,nhx,npx;
  int it,iz,ix,isx,ihx,ipx;
  float ot,omx,oz,osx,ohx,opx;
  float dt,dmx,dz,dsx,dhx,dpx;
  float **mppx_1shot,**mppz_1shot,**mpsx_1shot,**mpsz_1shot;
  float **dx_1shot,**dz_1shot;
  float **vp,**vs,*wav;
  float **m,**m_h,**m_h_gather,**m_a_gather;
  float **mppx,**mpsx,**mppz,**mpsz;
  bool adj;
  bool verbose;
  float fmin,fmax;
  off_t iseek;
  int rank,num_procs;
  char tmpname1[256];
  char tmpname2[256];
  char tmpname3[256];
  char tmpname4[256];

  MPI_Status mpi_stat;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sf_init(argc,argv);
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  velp = sf_input("vp");
  vels = sf_input("vs");
  source_wavelet = sf_input("wav"); // assumed to be a P-wave source
  if (adj){
    fp_dx = sf_input("ux");
    fp_dz = sf_input("uz");
    fp_mppx = sf_output("mppx");
    fp_mpsx = sf_output("mpsx");
    fp_mppz = sf_output("mppz");
    fp_mpsz = sf_output("mpsz");
  }
  else{
    fp_dx = sf_output("dx");
    fp_dz = sf_output("dz");
    fp_mppx = sf_input("mppx");
    fp_mpsx = sf_input("mpsx");
    fp_mppz = sf_input("mppz");
    fp_mpsz = sf_input("mpsz");
  }
  //if (verbose) fprintf(stderr,"rank=%d\n",rank);
  //if (verbose) fprintf(stderr,"num_procs=%d\n",num_procs);
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
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
    if (!sf_histint(  fp_dx,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histfloat(fp_dx,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(fp_dx,"o1",&ot)) ot=0.0;
    if (!sf_histint(  fp_dx,"n2",&nmx)) sf_error("No n2= in input"); /* this assumes that your image sampling is the same as your receiver sampling. */
    if (!sf_histfloat(fp_dx,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(fp_dx,"o2",&omx)) omx=0.0;
    if (!sf_histint(  fp_dx,"n3",&nsx)) nsx=1;
    if (!sf_histfloat(fp_dx,"d3",&dsx)) dsx=1.0;
    if (!sf_histfloat(fp_dx,"o3",&osx)) osx=0.0;
  }
  else{
    if (!sf_histint(  fp_mppx,"n1",&nz)) sf_error("No n1= in input");
    if (!sf_histfloat(fp_mppx,"d1",&dz)) sf_error("No d1= in input");
    if (!sf_histfloat(fp_mppx,"o1",&oz)) sf_error("No o1= in input");
    if (!sf_histint(  fp_mppx,"n2",&nmx)) sf_error("No n2= in input");
    if (!sf_histfloat(fp_mppx,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(fp_mppx,"o2",&omx)) sf_error("No o2= in input");
    if (!sf_histint(  fp_mppx,"n3",&npx)) sf_error("No n3= in input");
    if (!sf_histfloat(fp_mppx,"d3",&dpx)) sf_error("No d3= in input");
    if (!sf_histfloat(fp_mppx,"o3",&opx)) sf_error("No o3= in input");
  }
  if (!sf_getfloat("fmin",&fmin)) fmin = 0; /* min frequency to process */
  if (!sf_getfloat("fmax",&fmax)) fmax = 0.5/dt; /* max frequency to process */
  if (fmax > 0.5/dt) fmax = 0.5/dt;
  wav = sf_floatalloc(nt);
  sf_floatread(wav,nt,source_wavelet);
  vp = sf_floatalloc2(nz,nmx);
  vs = sf_floatalloc2(nz,nmx);
  sf_floatread(vp[0],nz*nmx,velp);
  sf_floatread(vs[0],nz*nmx,vels);
  dx_1shot = sf_floatalloc2(nt,nmx);
  dz_1shot = sf_floatalloc2(nt,nmx);
  mppx_1shot = sf_floatalloc2(nz,nmx*nhx);
  mpsx_1shot = sf_floatalloc2(nz,nmx*nhx);
  mppz_1shot = sf_floatalloc2(nz,nmx*nhx);
  mpsz_1shot = sf_floatalloc2(nz,nmx*nhx);
  if (adj){
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and migrating shot %d...\n",isx);
      iseek = (off_t)(isx*nmx*nt)*sizeof(float);
      sf_seek(fp_dx,iseek,SEEK_SET);    
      sf_floatread(dx_1shot[0],nt*nmx,fp_dx);
      sf_seek(fp_dz,iseek,SEEK_SET);    
      sf_floatread(dz_1shot[0],nt*nmx,fp_dz);
      ewem1shot(dx_1shot,dz_1shot,
                mppx_1shot,mpsx_1shot,mppz_1shot,mpsz_1shot,
                wav,
                nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nhx,ohx,dhx,nz,oz,dz,
                vp,vs,
                fmin,fmax,adj,verbose);
      sprintf(tmpname1, "tmp_mppx_%d.rsf",isx);
      fp_tmp_mppx = sf_output(tmpname1);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname1);
      write5d(mppx_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              nhx, ohx, dhx,         "Offset",   "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              "mppx", fp_tmp_mppx);
      sprintf(tmpname2, "tmp_mpsx_%d.rsf",isx);
      fp_tmp_mpsx = sf_output(tmpname2);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname2);
      write5d(mpsx_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              nhx, ohx, dhx,         "Offset",   "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              "mpsx", fp_tmp_mpsx);
      sprintf(tmpname3, "tmp_mppz_%d.rsf",isx);
      fp_tmp_mppz = sf_output(tmpname3);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname3);
      write5d(mppz_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              nhx, ohx, dhx,         "Offset",   "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              "mppz", fp_tmp_mppz);
      sprintf(tmpname4, "tmp_mpsz_%d.rsf",isx);
      fp_tmp_mpsz = sf_output(tmpname4);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname4);
      write5d(mpsz_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              nhx, ohx, dhx,         "Offset",   "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              "mpsz", fp_tmp_mpsz);
    }
  }
  else{
    mppx          = sf_floatalloc2(nz,nmx*npx);
    mpsx          = sf_floatalloc2(nz,nmx*npx);
    mppz          = sf_floatalloc2(nz,nmx*npx);
    mpsz          = sf_floatalloc2(nz,nmx*npx);
    m_h_gather = sf_floatalloc2(nz,nhx);
    m_a_gather = sf_floatalloc2(nz,npx);
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and de-migrating shot %d...\n",isx);
      if (isx== rank){
        sf_floatread(mppx[0],nz*nmx*npx,fp_mppx);
        sf_floatread(mpsx[0],nz*nmx*npx,fp_mpsx);
        sf_floatread(mppz[0],nz*nmx*npx,fp_mppz);
        sf_floatread(mpsz[0],nz*nmx*npx,fp_mpsz);
        for (ix=0;ix<nmx;ix++){
          // mppx
          for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m_a_gather[ipx][iz] = mppx[ipx*nmx + ix][iz];
          if (nhx>1) offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
          else for (iz=0;iz<nz;iz++) m_h_gather[0][iz] = m_a_gather[0][iz]; 
          for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) mppx_1shot[ihx*nmx + ix][iz] = m_h_gather[ihx][iz]; 
          // mpsx
          for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m_a_gather[ipx][iz] = mpsx[ipx*nmx + ix][iz];
          if (nhx>1) offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
          else for (iz=0;iz<nz;iz++) m_h_gather[0][iz] = m_a_gather[0][iz];
          for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) mpsx_1shot[ihx*nmx + ix][iz] = m_h_gather[ihx][iz]; 
          // mppz
          for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m_a_gather[ipx][iz] = mppz[ipx*nmx + ix][iz];
          if (nhx>1) offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
          else for (iz=0;iz<nz;iz++) m_h_gather[0][iz] = m_a_gather[0][iz];
          for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) mppz_1shot[ihx*nmx + ix][iz] = m_h_gather[ihx][iz]; 
          // mpsz
          for (ipx=0;ipx<npx;ipx++) for (iz=0;iz<nz;iz++) m_a_gather[ipx][iz] = mpsz[ipx*nmx + ix][iz];
          if (nhx>1)offset_to_angle(m_h_gather,m_a_gather,nz,oz,dz,nhx,ohx,dhx,npx,opx,dpx,0,2*fmax*(dt/dz),adj,verbose);
          else for (iz=0;iz<nz;iz++) m_h_gather[0][iz] = m_a_gather[0][iz];
          for (ihx=0;ihx<nhx;ihx++) for (iz=0;iz<nz;iz++) mpsz_1shot[ihx*nmx + ix][iz] = m_h_gather[ihx][iz]; 
        }
      }
      ewem1shot(dx_1shot,dz_1shot,
                mppx_1shot,mpsx_1shot,mppz_1shot,mpsz_1shot,
                wav,
                nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nhx,ohx,dhx,nz,oz,dz,
                vp,vs,
                fmin,fmax,adj,verbose);
      sprintf(tmpname1, "tmp_dx_%d.rsf",isx);
      fp_tmp_dx = sf_output(tmpname1);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname1);
      write5d(dx_1shot,
              nt,  ot,  dt,          "Time",    "m",  
              nmx, omx, dmx,         "Receiver-X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "dx", fp_tmp_dx);
      sprintf(tmpname2, "tmp_dz_%d.rsf",isx);
      fp_tmp_dz = sf_output(tmpname2);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname2);
      write5d(dz_1shot,
              nt,  ot,  dt,          "Time",    "m",  
              nmx, omx, dmx,         "Receiver-X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "dz", fp_tmp_dz);
    }
    free2float(m_h_gather);
    free2float(m_a_gather);
    free2float(mppx);
    free2float(mpsx);
    free2float(mppz);
    free2float(mpsz);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (adj && rank==0){
    m          = sf_floatalloc2(nz,nmx*npx);
    m_h        = sf_floatalloc2(nz,nmx*nhx);
    m_h_gather = sf_floatalloc2(nz,nhx);
    m_a_gather = sf_floatalloc2(nz,npx);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname1, "tmp_mppx_%d.rsf",isx);
      fp_tmp_mppx = sf_input(tmpname1);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname1); 
      sf_floatread(mppx_1shot[0],nz*nmx*nhx,fp_tmp_mppx);
      for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] += mppx_1shot[ix][iz];
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
    write5d(m,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "tan\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mppx", fp_mppx);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname2, "tmp_mpsx_%d.rsf",isx);
      fp_tmp_mpsx = sf_input(tmpname2);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname2); 
      sf_floatread(mpsx_1shot[0],nz*nmx*nhx,fp_tmp_mpsx);
      for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] += mpsx_1shot[ix][iz];
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
    write5d(m,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "tan\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mpsx", fp_mpsx);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname3, "tmp_mppz_%d.rsf",isx);
      fp_tmp_mppz = sf_input(tmpname3);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname3); 
      sf_floatread(mppz_1shot[0],nz*nmx*nhx,fp_tmp_mppz);
      for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] += mppz_1shot[ix][iz];
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
    write5d(m,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "tan\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mppz", fp_mppz);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname4, "tmp_mpsz_%d.rsf",isx);
      fp_tmp_mpsz = sf_input(tmpname4);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname4); 
      sf_floatread(mpsz_1shot[0],nz*nmx*nhx,fp_tmp_mpsz);
      for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*nhx;ix++) m_h[ix][iz] += mpsz_1shot[ix][iz];
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
    write5d(m,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "tan\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mpsz", fp_mpsz);
    free2float(m);
    free2float(m_h);
    free2float(m_h_gather);
    free2float(m_a_gather);
  }  
  else if (!adj && rank==0){
    sf_putfloat(fp_dx,"o1",ot);
    sf_putfloat(fp_dx,"d1",dt);
    sf_putfloat(fp_dx,"n1",nt);
    sf_putstring(fp_dx,"label1","Time");
    sf_putstring(fp_dx,"unit1","s");
    sf_putfloat(fp_dx,"o2",omx);
    sf_putfloat(fp_dx,"d2",dmx);
    sf_putfloat(fp_dx,"n2",nmx);
    sf_putstring(fp_dx,"label2","Receiver-X");
    sf_putstring(fp_dx,"unit2","m");
    sf_putfloat(fp_dx,"o3",osx);
    sf_putfloat(fp_dx,"d3",dsx);
    sf_putfloat(fp_dx,"n3",nsx);
    sf_putstring(fp_dx,"label3","Source-X");
    sf_putstring(fp_dx,"unit3","m");
    sf_putstring(fp_dx,"title","Data: X");
    sf_putfloat(fp_dz,"o1",ot);
    sf_putfloat(fp_dz,"d1",dt);
    sf_putfloat(fp_dz,"n1",nt);
    sf_putstring(fp_dz,"label1","Time");
    sf_putstring(fp_dz,"unit1","s");
    sf_putfloat(fp_dz,"o2",omx);
    sf_putfloat(fp_dz,"d2",dmx);
    sf_putfloat(fp_dz,"n2",nmx);
    sf_putstring(fp_dz,"label2","Receiver-X");
    sf_putstring(fp_dz,"unit2","m");
    sf_putfloat(fp_dz,"o3",osx);
    sf_putfloat(fp_dz,"d3",dsx);
    sf_putfloat(fp_dz,"n3",nsx);
    sf_putstring(fp_dz,"label3","Source-X");
    sf_putstring(fp_dz,"unit3","m");
    sf_putstring(fp_dz,"title","Data: Z");
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname1, "tmp_dx_%d.rsf",isx);
      fp_tmp_dx = sf_input(tmpname1);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname1); 
      sf_floatread(dx_1shot[0],nt*nmx,fp_tmp_dx);
      sf_floatwrite(dz_1shot[0],nt*nmx,fp_dx);
      sprintf(tmpname2, "tmp_dz_%d.rsf",isx);
      fp_tmp_dz = sf_input(tmpname2);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname2); 
      sf_floatread(dx_1shot[0],nt*nmx,fp_tmp_dz);
      sf_floatwrite(dz_1shot[0],nt*nmx,fp_dz);
    }
  }  

  free2float(mppx_1shot);
  free2float(mpsx_1shot);
  free2float(mppz_1shot);
  free2float(mpsz_1shot);
  free2float(dx_1shot);
  free2float(dz_1shot);
  free2float(vp);
  free2float(vs);
  free1float(wav);
  MPI_Finalize ();

  exit (0);
}

void ewem1shot(float **dx_1shot, float **dz_1shot,
               float **mppx, float **mpsx, float **mppz, float **mpsz,
               float *wav,
               int nt, float ot, float dt, 
               int nmx, float omx, float dmx,
               int isx, int nsx, float osx, float dsx,
               int nhx, float ohx, float dhx,
               int nz, float oz, float dz,
               float **vp, float **vs,
               float fmin, float fmax,
               bool adj, bool verbose)
/*< scalar wave equation depth migration operator for 2C data >*/
{
  int iz,ix,igx,ik,iw,it,nw,nk,padt,padx,ntfft,numthreads;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w;
  sf_complex **dpx_g_wx,**dsx_g_wx,**dpz_g_wx,**dsz_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po_p,**pd_p,*po_s,**pd_s,progress;
  if (adj){
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) mppx[ix][iz] = 0.0;
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) mpsx[ix][iz] = 0.0;
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) mppz[ix][iz] = 0.0;
    for (ix=0;ix<nmx*nhx;ix++) for (iz=0;iz<nz;iz++) mpsz[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) dx_1shot[ix][it] = 0.0;
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) dz_1shot[ix][it] = 0.0;
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
  dpx_g_wx = sf_complexalloc2(nw,nmx);
  dsx_g_wx = sf_complexalloc2(nw,nmx);
  dpz_g_wx = sf_complexalloc2(nw,nmx);
  dsz_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);
  for (it=0;it<nt;it++)  d_t[it] = 0.0;  
  for (iw=0;iw<nw;iw++)  d_w[iw] = czero;  
  /* decompose p and s-wave slownesses into layer average, and layer perturbation */
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
      // x component 
      for (it=0;it<nt;it++) d_t[it] = dx_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) dpx_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dpx_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dpx_g_wx[ix][iw] = czero;
      for (iw=0;iw<ifmin;iw++) dsx_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dsx_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dsx_g_wx[ix][iw] = czero;
      // z component 
      for (it=0;it<nt;it++) d_t[it] = dz_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1); /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++) dpz_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dpz_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dpz_g_wx[ix][iw] = czero;
      for (iw=0;iw<ifmin;iw++) dsz_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dsz_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++) dsz_g_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++){
	    dpx_g_wx[ix][iw] = czero;
	    dsx_g_wx[ix][iw] = czero;
	    dpz_g_wx[ix][iw] = czero;
	    dsz_g_wx[ix][iw] = czero;
      }
    }
  }
  progress = 0.0;
  numthreads = omp_get_num_threads();
  //if (verbose) fprintf(stderr,"using %d threads.",numthreads);
  #pragma omp parallel for private(iw) shared(mppx,mpsx,mppz,mpsz,dpx_g_wx,dsx_g_wx,dpz_g_wx,dsz_g_wx)
  for (iw=ifmin;iw<ifmax;iw++){ 
    progress += 1.0/((float) ifmax - ifmin);
    if (verbose) progress_msg(progress);
    eextrap1f(mppx,mpsx,mppz,mpsz,dpx_g_wx,dsx_g_wx,dpz_g_wx,dsz_g_wx,d_s_wx,iw,nw,ifmax,ntfft,dw,dk,nk,dz,nz,nmx,omx,dmx,nhx,ohx,dhx,po_p,pd_p,po_s,pd_s,i,czero,p1,p2,adj,verbose);
  }
  if (!adj){
    for (ix=0;ix<nmx;ix++){
      // x component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dpx_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dx_1shot[ix][it] = d_t[it];
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dsx_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dx_1shot[ix][it] += d_t[it];
      // z component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dpz_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dz_1shot[ix][it] = d_t[it];
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dsz_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dz_1shot[ix][it] += d_t[it];
    }
  }

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  free1float(d_t);
  free1complex(d_w);
  free2complex(dpx_g_wx);
  free2complex(dsx_g_wx);
  free2complex(dpz_g_wx);
  free2complex(dsz_g_wx);
  free2complex(d_s_wx);
  free1float(po_p);
  free2float(pd_p);
  free1float(po_s);
  free2float(pd_s);
  return;
} 

void eextrap1f(float **mppx, float **mpsx, float **mppz, float **mpsz,
               sf_complex **dpx_g_wx, sf_complex **dsx_g_wx, sf_complex **dpz_g_wx, sf_complex **dsz_g_wx, sf_complex **d_s_wx,
               int iw,int nw,int ifmax,int ntfft,float dw,float dk,int nk,float dz,int nz,
               int nmx,float omx, float dmx,
               int nhx,float ohx, float dhx,
               float *po_p,float **pd_p,float *po_s,float **pd_s,
               sf_complex i,sf_complex czero,
               fftwf_plan p1,fftwf_plan p2,
               bool adj, bool verbose)
/*< scalar extrapolation of 1 frequency of 2C data >*/
{
  float w,factor,hx,sx,gx;
  int iz,ix,ihx,isx,igx; 
  sf_complex *dpx_xg,*dsx_xg,*dpz_xg,*dsz_xg,*d_xs,**smig;
  
  dpx_xg = sf_complexalloc(nmx);
  dsx_xg = sf_complexalloc(nmx);
  dpz_xg = sf_complexalloc(nmx);
  dsz_xg = sf_complexalloc(nmx);
  d_xs = sf_complexalloc(nmx);

  if (iw==0) factor = 1;
  else factor = 2;

  w = iw*dw;
  if (adj){
    for (ix=0;ix<nmx;ix++){ 
      d_xs[ix] = d_s_wx[ix][iw]/sqrtf((float) ntfft);
      dpx_xg[ix] = dpx_g_wx[ix][iw]/sqrtf((float) ntfft);
      dsx_xg[ix] = dsx_g_wx[ix][iw]/sqrtf((float) ntfft);
      dpz_xg[ix] = dpz_g_wx[ix][iw]/sqrtf((float) ntfft);
      dsz_xg[ix] = dsz_g_wx[ix][iw]/sqrtf((float) ntfft);
    }
    for (iz=0;iz<nz;iz++){ /* extrapolate source and receiver wavefields */
      ssop(d_xs,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose); 
      ssop(dpx_xg,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose);
      ssop(dsx_xg,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,true,verbose);
      ssop(dpz_xg,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,true,verbose);
      ssop(dsz_xg,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,true,verbose);
      for (ix=0;ix<nmx;ix++){
        for (ihx=0;ihx<nhx;ihx++){
          hx = ihx*dhx + ohx;
          sx = (ix*dmx + omx) - hx;
          gx = (ix*dmx + omx) + hx;
          isx = (int) truncf((sx - omx)/dmx);
          igx = (int) truncf((gx - omx)/dmx);
          if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
            #pragma omp atomic
            mppx[ihx*nmx + ix][iz] += factor*crealf(d_xs[isx]*conjf(dpx_xg[igx]));
            #pragma omp atomic
            mpsx[ihx*nmx + ix][iz] += factor*crealf(d_xs[isx]*conjf(dsx_xg[igx]));
            #pragma omp atomic
            mppz[ihx*nmx + ix][iz] += factor*crealf(d_xs[isx]*conjf(dpz_xg[igx]));
            #pragma omp atomic
            mpsz[ihx*nmx + ix][iz] += factor*crealf(d_xs[isx]*conjf(dsz_xg[igx]));
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
    for (ix=0;ix<nmx;ix++) dpx_xg[ix] = czero;
    for (ix=0;ix<nmx;ix++) dsx_xg[ix] = czero;
    for (ix=0;ix<nmx;ix++) dpz_xg[ix] = czero;
    for (ix=0;ix<nmx;ix++) dsz_xg[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ /* extrapolate receiver wavefield */
      for (ix=0;ix<nmx;ix++){ 
        for (ihx=0;ihx<nhx;ihx++){
          hx = ihx*dhx + ohx;
          sx = (ix*dmx + omx) - hx;
          gx = (ix*dmx + omx) + hx;
          isx = (int) truncf((sx - omx)/dmx);
          igx = (int) truncf((gx - omx)/dmx);
          if (isx >=0 && isx < nmx && igx >=0 && igx < nmx){
            dpx_xg[igx] = dpx_xg[igx] + smig[isx][iz]*mppx[ihx*nmx + ix][iz];
            dsx_xg[igx] = dsx_xg[igx] + smig[isx][iz]*mpsx[ihx*nmx + ix][iz];
            dpz_xg[igx] = dpz_xg[igx] + smig[isx][iz]*mppz[ihx*nmx + ix][iz];
            dsz_xg[igx] = dsz_xg[igx] + smig[isx][iz]*mpsz[ihx*nmx + ix][iz];
          }
        }
      }
      ssop(dpx_xg,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,false,verbose);
      ssop(dsx_xg,w,dk,nk,nmx,-dz,iz,po_s,pd_s,i,czero,p1,p2,false,verbose);
      ssop(dpz_xg,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,false,verbose);
      ssop(dsz_xg,w,dk,nk,nmx,-dz,iz,po_s,pd_s,i,czero,p1,p2,false,verbose);
    }
    for (ix=0;ix<nmx;ix++){
      dpx_g_wx[ix][iw] = dpx_xg[ix]/sqrtf((float) ntfft);
      dsx_g_wx[ix][iw] = dsz_xg[ix]/sqrtf((float) ntfft);
      dpz_g_wx[ix][iw] = dpx_xg[ix]/sqrtf((float) ntfft);
      dsz_g_wx[ix][iw] = dsz_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(dpx_xg);
  free1complex(dsx_xg);
  free1complex(dpz_xg);
  free1complex(dsz_xg);
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
