/* Shot Profile Vector Wave Equation Migration of 2C data with angle gather imaging condition. Uses MPI over shots and OMP over frequencies.*/
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

void vwem1shot(float **dx_1shot, float **dz_1shot,
               float **mppx, float **mppz, float **mpsx, float **mpsz, 
               float **ang1shot,
               float *wav,
               int nt, float ot, float dt, 
               int nmx, float omx, float dmx,
               int isx, int nsx, float osx, float dsx,
               int nz, float oz, float dz, float gz, float sz,
               float **vp, float **vs,
               float fmin, float fmax,
               bool adj, bool calc_ang, bool verbose);
void vextrap1f(float **mppx, float **mppz, float **mpsx, float **mpsz,
               sf_complex **dpx_g_wx, sf_complex **dpz_g_wx, sf_complex **dsx_g_wx, sf_complex **dsz_g_wx, sf_complex **d_s_wx,
               float **u_sx, float **u_sz, float **u_gx, float **u_gz, 
               int iw, int ang_iw_max,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
               int nz, float oz,  float dz, float gz, float sz,
               int nmx,float omx, float dmx,
               float *po_p,float **pd_p,float *po_s,float **pd_s,
               sf_complex i,sf_complex czero,
               fftwf_plan p1,fftwf_plan p2,
               bool adj, bool verbose);
void ssop(sf_complex *d_x,
          sf_complex *p_x, sf_complex *p_z,
          float w,float dk,int nk,int nmx,float dz,int iz,
          float *po,float **pd,
          sf_complex i,sf_complex czero,
          fftwf_plan p1,fftwf_plan p2,
          bool adj, bool calc_ang, bool verbose);
void f_op(sf_complex *m,float *d,int nw,int nt,bool adj);
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
float signf(float a);
int compare (const void * a, const void * b);

int main(int argc, char* argv[])
{

  sf_file fp_dx,fp_dz,fp_mpp,fp_mps;
  sf_file fp_tmp_dx,fp_tmp_dz;
  sf_file fp_tmp_mppx,fp_tmp_mppz,fp_tmp_mpsx,fp_tmp_mpsz,fp_tmp_ang;
  sf_file velp,vels,source_wavelet;
  int nt,nmx,nz,nsx,npx;
  int iz,ix,isx,ipx;
  float ot,omx,oz,osx,opx;
  float dt,dmx,dz,dsx,dpx;
  float gz,sz;
  float **mppx_1shot,**mppz_1shot,**mpsx_1shot,**mpsz_1shot,**ang1shot;
  float **dx_1shot,**dz_1shot;
  float **vp,**vs,*wav;
  float **mpp,**mps;
  bool adj;
  bool verbose;
  float fmin,fmax;
  off_t iseek;
  int rank,num_procs;
  char tmpname1[256];
  char tmpname2[256];
  char tmpname3[256];
  char tmpname4[256];
  char tmpname_ang[256];
  float px,px_floor,alpha;
  bool calc_ang;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  sf_init(argc,argv);
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getbool("calc_ang",&calc_ang)) calc_ang = true; /* flag for computing angles and storing to temporary files for each shot for use in forward operator */
  velp = sf_input("vp");
  vels = sf_input("vs");
  source_wavelet = sf_input("wav"); // assumed to be a P-wave source
  if (adj){
    fp_dx = sf_input("ux");
    fp_dz = sf_input("uz");
    fp_mpp = sf_output("mpp");
    fp_mps = sf_output("mps");
  }
  else{
    fp_dx = sf_output("ux");
    fp_dz = sf_output("uz");
    fp_mpp = sf_input("mpp");
    fp_mps = sf_input("mps");
  }
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/

  if (!sf_getfloat("gz",&gz)) gz = 0; /* depth of the receivers */
  if (!sf_getfloat("sz",&sz)) sz = 0; /* depth of the sources */

  if (adj){
    if (!sf_getint("nz",&nz)) sf_error("nz must be specified");
    if (!sf_getfloat("oz",&oz)) sf_error("oz must be specified");
    if (!sf_getfloat("dz",&dz)) sf_error("dz must be specified");
    if (!sf_getint("npx",&npx))   npx=91; /* length of angle axis */
    if (!sf_getfloat("opx",&opx)) opx=-90; /* origin of angle axis */
    if (!sf_getfloat("dpx",&dpx)) dpx=2;   /* increment of angle axis */
  }
  else{
    if (!sf_getint("nt",&nt)) sf_error("nt must be specified");
    if (!sf_getfloat("ot",&ot)) sf_error("ot must be specified");
    if (!sf_getfloat("dt",&dt)) sf_error("dt must be specified");
    if (!sf_getint("nsx",&nsx)) sf_error("nsx must be specified");
    if (!sf_getfloat("osx",&osx)) sf_error("osx must be specified");
    if (!sf_getfloat("dsx",&dsx)) sf_error("dsx must be specified");
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
    if (!sf_histint(  fp_mpp,"n1",&nz)) sf_error("No n1= in input");
    if (!sf_histfloat(fp_mpp,"d1",&dz)) sf_error("No d1= in input");
    if (!sf_histfloat(fp_mpp,"o1",&oz)) sf_error("No o1= in input");
    if (!sf_histint(  fp_mpp,"n2",&nmx)) sf_error("No n2= in input");
    if (!sf_histfloat(fp_mpp,"d2",&dmx)) sf_error("No d2= in input");
    if (!sf_histfloat(fp_mpp,"o2",&omx)) sf_error("No o2= in input");
    if (!sf_histint(  fp_mpp,"n3",&npx)) sf_error("No n3= in input");
    if (!sf_histfloat(fp_mpp,"d3",&dpx)) sf_error("No d3= in input");
    if (!sf_histfloat(fp_mpp,"o3",&opx)) sf_error("No o3= in input");
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
  mppx_1shot = sf_floatalloc2(nz,nmx);
  mppz_1shot = sf_floatalloc2(nz,nmx);
  mpsx_1shot = sf_floatalloc2(nz,nmx);
  mpsz_1shot = sf_floatalloc2(nz,nmx);
  ang1shot = sf_floatalloc2(nz,nmx);
  if (adj){
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and migrating shot %d... \n",isx);
      iseek = (off_t)(isx*nmx*nt)*sizeof(float);
      sf_seek(fp_dx,iseek,SEEK_SET);    
      sf_floatread(dx_1shot[0],nt*nmx,fp_dx);
      sf_seek(fp_dz,iseek,SEEK_SET);    
      sf_floatread(dz_1shot[0],nt*nmx,fp_dz);
      vwem1shot(dx_1shot,dz_1shot,
                mppx_1shot,mppz_1shot,mpsx_1shot,mpsz_1shot,
                ang1shot,
                wav,
                nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
                vp,vs,
                fmin,fmax,adj,calc_ang,verbose);
      sprintf(tmpname1, "tmp_mppx_%d.rsf",isx);
      fp_tmp_mppx = sf_output(tmpname1);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname1);
      write5d(mppx_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "mppx", fp_tmp_mppx);
      sprintf(tmpname2, "tmp_mppz_%d.rsf",isx);
      fp_tmp_mppz = sf_output(tmpname2);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname2);
      write5d(mppz_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "mppz", fp_tmp_mppz);
      sprintf(tmpname3, "tmp_mpsx_%d.rsf",isx);
      fp_tmp_mpsx = sf_output(tmpname3);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname3);
      write5d(mpsx_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "mpsx", fp_tmp_mpsx);
      sprintf(tmpname4, "tmp_mpsz_%d.rsf",isx);
      fp_tmp_mpsz = sf_output(tmpname4);
      if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname4);
      write5d(mpsz_1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "mpsz", fp_tmp_mpsz);
      if (calc_ang){        
        sprintf(tmpname_ang, "tmp_ang_%d.rsf",isx);
        fp_tmp_ang = sf_output(tmpname_ang);
        if (verbose) fprintf(stderr,"writing %s to disk.\n",tmpname_ang);
        write5d(ang1shot,
              nz,  oz,  dz,          "Depth",    "m",  
              nmx, omx, dmx,         "X",        "m",  
              1, isx*dsx + osx, dsx, "Source-X", "m",  
              1, 0, 1,   " ",        " ",
              1, 0, 1,   " ",        " ",
              "Angle of Incidence", fp_tmp_ang);
      }
    }
  }
  else{
    mpp        = sf_floatalloc2(nz,nmx*npx);
    mps        = sf_floatalloc2(nz,nmx*npx);
    for (isx=rank;isx<nsx;isx+=num_procs){
      if (verbose) fprintf(stderr,"reading and de-migrating shot %d...\n",isx);
      if (isx== rank){
        sf_floatread(mpp[0],nz*nmx*npx,fp_mpp);
        sf_floatread(mps[0],nz*nmx*npx,fp_mps);
      }
      sprintf(tmpname_ang, "tmp_ang_%d.rsf",isx);
      fp_tmp_ang = sf_input(tmpname_ang);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname_ang); 
      sf_floatread(ang1shot[0],nz*nmx,fp_tmp_ang);
      for (ix=0;ix<nmx;ix++){
        for (iz=0;iz<nz;iz++){
          if (npx>1){
            px = ang1shot[ix][iz]; 
            px_floor = truncf((px - opx)/dpx)*dpx + opx;
            ipx = (int) truncf((px - opx)/dpx);
            if (ipx >= 0 && ipx+1 < npx){
	          alpha = (px-px_floor)/dpx;
	          mppx_1shot[ix][iz]  = sinf(fabsf(px)*PI/180)*((1-alpha)*mpp[ipx*nmx + ix][iz] + alpha*mpp[(ipx+1)*nmx + ix][iz]);
	          mppz_1shot[ix][iz]  = cosf(fabsf(px)*PI/180)*((1-alpha)*mpp[ipx*nmx + ix][iz] + alpha*mpp[(ipx+1)*nmx + ix][iz]);
	          mpsx_1shot[ix][iz]  = cosf(fabsf(px)*PI/180)*((1-alpha)*mps[ipx*nmx + ix][iz] + alpha*mps[(ipx+1)*nmx + ix][iz]);
	          mpsz_1shot[ix][iz]  = sinf(fabsf(px)*PI/180)*((1-alpha)*mps[ipx*nmx + ix][iz] + alpha*mps[(ipx+1)*nmx + ix][iz]);
	        }
	      }
          else{
	        mppx_1shot[ix][iz] = mpp[ix][iz];
	        mppz_1shot[ix][iz] = mpp[ix][iz];
	        mpsx_1shot[ix][iz] = mps[ix][iz];
	        mpsz_1shot[ix][iz] = mps[ix][iz];
	      }	      
        }
      }
      vwem1shot(dx_1shot,dz_1shot,
                mppx_1shot,mppz_1shot,mpsx_1shot,mpsz_1shot,
                ang1shot,
                wav,
                nt,ot,dt,nmx,omx,dmx,isx,nsx,osx,dsx,nz,oz,dz,gz,sz,
                vp,vs,
                fmin,fmax,adj,calc_ang,verbose);
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
    free2float(mpp);
    free2float(mps);
    sf_fileclose(fp_mpp);
    sf_fileclose(fp_mps);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  sf_fileclose(velp);
  sf_fileclose(vels);
  sf_fileclose(source_wavelet);

  if (adj && rank==0){
    mpp = sf_floatalloc2(nz,nmx*npx);
    mps = sf_floatalloc2(nz,nmx*npx);
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*npx;ix++) mpp[ix][iz] = 0.0; 
    for (iz=0;iz<nz;iz++) for (ix=0;ix<nmx*npx;ix++) mps[ix][iz] = 0.0; 
    for (isx=0;isx<nsx;isx++){
      sprintf(tmpname1, "tmp_mppx_%d.rsf",isx);
      fp_tmp_mppx = sf_input(tmpname1);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname1); 
      sf_floatread(mppx_1shot[0],nz*nmx,fp_tmp_mppx);
      sprintf(tmpname2, "tmp_mppz_%d.rsf",isx);
      fp_tmp_mppz = sf_input(tmpname2);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname2); 
      sf_floatread(mppz_1shot[0],nz*nmx,fp_tmp_mppz);
      sprintf(tmpname3, "tmp_mpsx_%d.rsf",isx);
      fp_tmp_mpsx = sf_input(tmpname3);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname3); 
      sf_floatread(mpsx_1shot[0],nz*nmx,fp_tmp_mpsx);
      sprintf(tmpname4, "tmp_mpsz_%d.rsf",isx);
      fp_tmp_mpsz = sf_input(tmpname4);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname4); 
      sf_floatread(mpsz_1shot[0],nz*nmx,fp_tmp_mpsz);
      sprintf(tmpname_ang, "tmp_ang_%d.rsf",isx);
      fp_tmp_ang = sf_input(tmpname_ang);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname_ang); 
      sf_floatread(ang1shot[0],nz*nmx,fp_tmp_ang);
      for (ix=0;ix<nmx;ix++){
        for (iz=0;iz<nz;iz++){
          if (npx>1){
            px = ang1shot[ix][iz]; 
            px_floor = truncf((px - opx)/dpx)*dpx + opx;
            ipx = (int) truncf((px - opx)/dpx);
            if (ipx >= 0 && ipx+1 < npx){
	          alpha = (px-px_floor)/dpx;
	          mpp[ipx*nmx + ix][iz]     += (1-alpha)*(signf(px)*sinf(fabsf(px)*PI/180)*mppx_1shot[ix][iz] + cosf(fabsf(px)*PI/180)*mppz_1shot[ix][iz]);
	          mpp[(ipx+1)*nmx + ix][iz] +=     alpha*(signf(px)*sinf(fabsf(px)*PI/180)*mppx_1shot[ix][iz] + cosf(fabsf(px)*PI/180)*mppz_1shot[ix][iz]);
	          mps[ipx*nmx + ix][iz]     += (1-alpha)*(signf(px)*cosf(fabsf(px)*PI/180)*mpsx_1shot[ix][iz] + sinf(fabsf(px)*PI/180)*mpsz_1shot[ix][iz]);
	          mps[(ipx+1)*nmx + ix][iz] +=     alpha*(signf(px)*cosf(fabsf(px)*PI/180)*mpsx_1shot[ix][iz] + sinf(fabsf(px)*PI/180)*mpsz_1shot[ix][iz]);
	        }
	      }
          else{
	        mpp[ix][iz] += mppx_1shot[ix][iz] + mppz_1shot[ix][iz];
	        mps[ix][iz] += mpsx_1shot[ix][iz] + mpsz_1shot[ix][iz];
	      }	      
        }
      }
      sf_fileclose(fp_tmp_mppx);
      sf_fileclose(fp_tmp_mppz);
      sf_fileclose(fp_tmp_mpsx);
      sf_fileclose(fp_tmp_mpsz);
      sf_fileclose(fp_tmp_ang);
    }
    write5d(mpp,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mpp", fp_mpp);
    write5d(mps,
            nz,  oz,  dz,          "Depth",    "m",  
            nmx, omx, dmx,         "X",        "m",  
            npx, opx, dpx,         "\\F10 q\\F3 ",   " ",  
            1, 0, 1, " ", " ",  
            1, 0, 1,   " ",        " ",
            "mps", fp_mps);
    free2float(mpp);
    free2float(mps);
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
      sf_floatwrite(dx_1shot[0],nt*nmx,fp_dx);
      sprintf(tmpname2, "tmp_dz_%d.rsf",isx);
      fp_tmp_dz = sf_input(tmpname2);
      if (verbose) fprintf(stderr,"reading %s from disk.\n",tmpname2); 
      sf_floatread(dz_1shot[0],nt*nmx,fp_tmp_dz);
      sf_floatwrite(dz_1shot[0],nt*nmx,fp_dz);
      sf_fileclose(fp_tmp_dx);
      sf_fileclose(fp_tmp_dz);
    }
  }  

  free2float(mppx_1shot);
  free2float(mppz_1shot);
  free2float(mpsx_1shot);
  free2float(mpsz_1shot);
  free2float(dx_1shot);
  free2float(dz_1shot);
  free2float(vp);
  free2float(vs);
  free1float(wav);
  sf_fileclose(fp_dx);
  sf_fileclose(fp_dz);
  MPI_Finalize ();
  exit (0);
}

void vwem1shot(float **dx_1shot, float **dz_1shot,
               float **mppx, float **mppz, float **mpsx, float **mpsz, 
               float **ang1shot,
               float *wav,
               int nt, float ot, float dt, 
               int nmx, float omx, float dmx,
               int isx, int nsx, float osx, float dsx,
               int nz, float oz, float dz, float gz, float sz,
               float **vp, float **vs,
               float fmin, float fmax,
               bool adj, bool calc_ang, bool verbose)
/*< Vector depth migration operator for 2C data >*/
{
  int iz,ix,igx,ik,iw,it,nw,nk,padt,padx,ntfft;
  int igz;
  float dw,dk;
  sf_complex czero,i;
  int ifmin,ifmax;
  float *d_t;
  sf_complex *d_w,*d_x,*d_z,*d_p,*d_s;
  sf_complex **dpx_g_wx,**dpz_g_wx,**dsx_g_wx,**dsz_g_wx;
  sf_complex **d_s_wx;
  fftwf_complex *a,*b;
  int *n;
  fftwf_plan p1,p2;
  float *po_p,**pd_p,*po_s,**pd_s,progress;
  float **u_sx,**u_sz,**u_gx,**u_gz,*ang;
  float norm_s,norm_g;
  float dot_prod,cross_prod;
  int nxw,nzw,ixw,izw,index1,index2;
  int ifilter,nfilter;
  
  if (adj){
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) mppx[ix][iz] = 0.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) mppz[ix][iz] = 0.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) mpsx[ix][iz] = 0.0;
    for (ix=0;ix<nmx;ix++) for (iz=0;iz<nz;iz++) mpsz[ix][iz] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) dx_1shot[ix][it] = 0.0;
    for (ix=0;ix<nmx;ix++) for (it=0;it<nt;it++) dz_1shot[ix][it] = 0.0;
  }
  igz = (int) truncf(gz/dz);
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
  dpz_g_wx = sf_complexalloc2(nw,nmx);
  dsx_g_wx = sf_complexalloc2(nw,nmx);
  dsz_g_wx = sf_complexalloc2(nw,nmx);
  d_s_wx = sf_complexalloc2(nw,nmx);
  d_t = sf_floatalloc(nt);
  d_w = sf_complexalloc(nw);

  d_x = sf_complexalloc(nk);
  d_z = sf_complexalloc(nk);
  d_p = sf_complexalloc(nk);
  d_s = sf_complexalloc(nk);

  u_sx = sf_floatalloc2(nz,nmx);
  u_sz = sf_floatalloc2(nz,nmx);
  u_gx = sf_floatalloc2(nz,nmx);
  u_gz = sf_floatalloc2(nz,nmx);

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
      for (it=0;it<nt;it++)        d_t[it] = dx_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     dpx_g_wx[ix][iw] = dsx_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dpx_g_wx[ix][iw] = dsx_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++)    dpx_g_wx[ix][iw] = dsx_g_wx[ix][iw] = czero;
      // z component 
      for (it=0;it<nt;it++)        d_t[it] = dz_1shot[ix][it];
      f_op(d_w,d_t,nw,nt,1);       /* d_t to d_w */
      for (iw=0;iw<ifmin;iw++)     dpz_g_wx[ix][iw] = dsz_g_wx[ix][iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) dpz_g_wx[ix][iw] = dsz_g_wx[ix][iw] = d_w[iw];
      for (iw=ifmax;iw<nw;iw++)    dpz_g_wx[ix][iw] = dsz_g_wx[ix][iw] = czero;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (iw=0;iw<nw;iw++){
	    dpx_g_wx[ix][iw] = czero;
	    dpz_g_wx[ix][iw] = czero;
	    dsx_g_wx[ix][iw] = czero;
	    dsz_g_wx[ix][iw] = czero;
      }
    }
  }
  
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
  #pragma omp parallel for private(iw) shared(mppx,mppz,mpsx,mpsz,dpx_g_wx,dpz_g_wx,dsx_g_wx,dsz_g_wx,progress)
  for (iw=ifmin;iw<ifmax;iw++){ 
    //progress += 1.0/((float) ifmax - ifmin);
    //if (verbose) progress_msg(progress);
    vextrap1f(mppx,mppz,mpsx,mpsz,dpx_g_wx,dpz_g_wx,dsx_g_wx,dsz_g_wx,d_s_wx,u_sx,u_sz,u_gx,u_gz,iw,(int) truncf(ifmax/2),nw,ifmax,ntfft,dw,dk,nk,nz,oz,dz,gz,sz,nmx,omx,dmx,po_p,pd_p,po_s,pd_s,i,czero,p1,p2,adj,verbose);
  }

  if (adj && calc_ang){
    /* calculate incidence angle from propagation vectors */  
    for (ix=0;ix<nmx;ix++){ 
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
    for (ix=0;ix<nmx;ix++){ 
      for (iz=0;iz<nz;iz++){
        for (ixw=0;ixw<nxw;ixw++){ for (izw=0;izw<nzw;izw++){
          index1 = ix - (int) truncf(nxw/2) + ixw;
          index2 = iz - (int) truncf(nzw/2) + izw;        
          if (index1>=0 && index1<nmx && index2>=0 && index2<nz){
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
    
  }

  if (!adj){
    for (ix=0;ix<nmx;ix++){
      // x component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dpx_g_wx[ix][iw] + dsx_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dx_1shot[ix][it] = d_t[it];
      // z component 
      for (iw=0;iw<ifmin;iw++) d_w[iw] = czero;
      for (iw=ifmin;iw<ifmax;iw++) d_w[iw] = dpz_g_wx[ix][iw] + dsz_g_wx[ix][iw];
      for (iw=ifmax;iw<nw;iw++) d_w[iw] = czero;
      f_op(d_w,d_t,nw,nt,0); /* d_w to d_t */
      for (it=0;it<nt;it++) dz_1shot[ix][it] = d_t[it];
    }
  }

  free1int(n); 
  fftwf_free(a);
  fftwf_free(b);
  fftwf_destroy_plan(p1);fftwf_destroy_plan(p2);
  free1float(d_t);
  free1complex(d_w);
  free2complex(dpx_g_wx);
  free2complex(dpz_g_wx);
  free2complex(dsx_g_wx);
  free2complex(dsz_g_wx);
  free2complex(d_s_wx);
  free1float(po_p);
  free2float(pd_p);
  free1float(po_s);
  free2float(pd_s);
  free1complex(d_x);
  free1complex(d_z);
  free2float(u_sx);
  free2float(u_sz);
  free2float(u_gx);
  free2float(u_gz);
  return;
} 

void vextrap1f(float **mppx, float **mppz, float **mpsx, float **mpsz,
               sf_complex **dpx_g_wx, sf_complex **dpz_g_wx, sf_complex **dsx_g_wx, sf_complex **dsz_g_wx, sf_complex **d_s_wx,
               float **u_sx, float **u_sz, float **u_gx, float **u_gz, 
               int iw, int ang_iw_max,int nw,int ifmax,int ntfft,float dw,float dk,int nk,
               int nz, float oz,  float dz, float gz, float sz,
               int nmx,float omx, float dmx,
               float *po_p,float **pd_p,float *po_s,float **pd_s,
               sf_complex i,sf_complex czero,
               fftwf_plan p1,fftwf_plan p2,
               bool adj, bool verbose)
/*< vector extrapolation of 1 frequency of 2C data >*/
{
  float w,factor,z;
  int iz,ix; 
  sf_complex *dpx_xg,*dpz_xg,*dsx_xg,*dsz_xg,*d_xs,**smig,*p_sx,*p_sz,*p_gx,*p_gz;
  
  dpx_xg = sf_complexalloc(nmx);
  dpz_xg = sf_complexalloc(nmx);
  dsx_xg = sf_complexalloc(nmx);
  dsz_xg = sf_complexalloc(nmx);
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
      dpx_xg[ix] = dpx_g_wx[ix][iw]/sqrtf((float) ntfft);
      dpz_xg[ix] = dpz_g_wx[ix][iw]/sqrtf((float) ntfft);
      dsx_xg[ix] = dsx_g_wx[ix][iw]/sqrtf((float) ntfft);
      dsz_xg[ix] = dsz_g_wx[ix][iw]/sqrtf((float) ntfft);
    }
    for (iz=0;iz<nz;iz++){ /* extrapolate source and receiver wavefields */
      z = oz + dz*iz;
      if (z >= sz){
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,true,true,verbose); 
      }
      if (z >= gz){
        ssop(dpx_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,true,true,verbose);
        ssop(dpz_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po_p,pd_p,i,czero,p1,p2,true,true,verbose);
        ssop(dsx_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,true,false,verbose);
        ssop(dsz_xg,p_gx,p_gz,w,dk,nk,nmx,dz,iz,po_s,pd_s,i,czero,p1,p2,true,false,verbose);
        for (ix=0;ix<nmx;ix++){
          #pragma omp atomic
          mppx[ix][iz] += factor*crealf(d_xs[ix]*conjf(dpx_xg[ix]));
          #pragma omp atomic
          mppz[ix][iz] += factor*crealf(d_xs[ix]*conjf(dpz_xg[ix]));
          #pragma omp atomic
          mpsx[ix][iz] += factor*crealf(d_xs[ix]*conjf(dsx_xg[ix]));
          #pragma omp atomic
          mpsz[ix][iz] += factor*crealf(d_xs[ix]*conjf(dsz_xg[ix]));
          if (iw<ang_iw_max){
            #pragma omp atomic
            u_sx[ix][iz] += crealf(p_sx[ix]*conjf(dpz_xg[ix]));
            #pragma omp atomic
            u_sz[ix][iz] += crealf(p_sz[ix]*conjf(dpz_xg[ix]));
            #pragma omp atomic
            u_gx[ix][iz] += crealf(dpx_xg[ix]*conjf(d_xs[ix]));
            #pragma omp atomic
            u_gz[ix][iz] += crealf(dpz_xg[ix]*conjf(d_xs[ix]));
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
        ssop(d_xs,p_sx,p_sz,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,true,false,verbose); 
        for (ix=0;ix<nmx;ix++) smig[ix][iz] = d_xs[ix];
      }
      else{
        for (ix=0;ix<nmx;ix++) smig[ix][iz] = czero;
      }
    }
    for (ix=0;ix<nmx;ix++) dpx_xg[ix] = dpz_xg[ix] = czero;
    for (ix=0;ix<nmx;ix++) dsx_xg[ix] = dsz_xg[ix] = czero;
    for (iz=nz-1;iz>=0;iz--){ /* extrapolate receiver wavefield */
      z = oz + dz*iz;
      if (z >= gz){
        for (ix=0;ix<nmx;ix++){ 
          dpx_xg[ix] = dpx_xg[ix] + smig[ix][iz]*mppx[ix][iz];
          dpz_xg[ix] = dpz_xg[ix] + smig[ix][iz]*mppz[ix][iz];
          dsx_xg[ix] = dsx_xg[ix] + smig[ix][iz]*mpsx[ix][iz];
          dsz_xg[ix] = dsz_xg[ix] + smig[ix][iz]*mpsz[ix][iz];
        }
        ssop(dpx_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,false,false,verbose);
        ssop(dpz_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po_p,pd_p,i,czero,p1,p2,false,false,verbose);
        ssop(dsx_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po_s,pd_s,i,czero,p1,p2,false,false,verbose);
        ssop(dsz_xg,p_gx,p_gz,w,dk,nk,nmx,-dz,iz,po_s,pd_s,i,czero,p1,p2,false,false,verbose);
      }
    }
    for (ix=0;ix<nmx;ix++){
      dpx_g_wx[ix][iw] = dpx_xg[ix]/sqrtf((float) ntfft);
      dpz_g_wx[ix][iw] = dpz_xg[ix]/sqrtf((float) ntfft);
      dsx_g_wx[ix][iw] = dsx_xg[ix]/sqrtf((float) ntfft);
      dsz_g_wx[ix][iw] = dsz_xg[ix]/sqrtf((float) ntfft);
    }
    free2complex(smig);
  }
  free1complex(dpx_xg);
  free1complex(dpz_xg);
  free1complex(dsx_xg);
  free1complex(dsz_xg);
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
          bool adj, bool calc_ang, bool verbose)
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
    if (ik<nk/2) k = dk*ik;
    else         k = -(dk*nk - dk*ik);
    s = (w*w)*(po[iz]*po[iz]) - (k*k);
    if (s>=0){ 
      __real__ L = cosf(sqrtf(s)*dz);
      __imag__ L = sinf(sqrtf(s)*dz);
    }
    else L = czero;   // evanescent waves
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
        b[ik] = czero;
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
    free1float(in1a); fftwf_free(out1a);
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
    fftwf_free(in1b); free1float(out1b);
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
