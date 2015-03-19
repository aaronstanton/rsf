/* PSTM using multiple constant velocities.
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
#include "myfree.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "myfree.h"
#include <fftw3.h>

void ktop(float **d, float **m, float **vp,
          int nt, int nmx, int nhx, 
          float ot, float omx, float ohx, 
          float dt, float dmx, float dhx,
          float aperture, bool adj, bool verbose);
void kt1ofc(float **d, float **m, float **vp, 
               int nt, int nmx, int nhx,
               float ot, float omx, float ohx, 
               float dt, float dmx, float dhx,
               int ihx, float hx, 
               float aperture, bool adj, bool verbose);
void ktfwd(float **d, float *m, float **vp, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icipx,
               float hx,float aperture);
void ktadj(float *d, float **m, float **vp,
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icmpx,
               float hx,float aperture);
float spherical_divergence(float ts,float tg,float v);
float angle_taper(float ts,float tg, float v, float hx);
void rho_filt(float *m,int nt,int adj);

int main(int argc, char* argv[])
{

  sf_file in,out,vel,weight;
  int n1,n2,n3,n4;
  int nt,nmx,nhx,nv;
  int it,ix,iv,ihx;
  float o1,o2,o3,o4;
  float d1,d2,d3,d4;
  float ot,omx,ohx,ov;
  float dt,dmx,dhx,dv;
  float **d,**m,**m1,**vp,*trace,**w;
  bool adj;
  bool verbose;
  float aperture;
  float sum;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  vel = sf_input("vel");
  weight = sf_input("w");
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getfloat("aperture",&aperture)) aperture=1000;
  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d2= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;
  if (!sf_histint(  in,"n3",&n3)) sf_error("No n3= in input");
  if (!sf_histfloat(in,"d3",&d3)) sf_error("No d3= in input");
  if (!sf_histfloat(in,"o3",&o3)) o3=0.;
  
  /* get number of velocities from weight file*/
  if (!sf_histint(  weight,"n4",&n4)) sf_error("No n4= in w");
  if (!sf_histfloat(weight,"d4",&d4)) sf_error("No d4= in w");
  if (!sf_histfloat(weight,"o4",&o4)) o4=1.;

  nt=n1; nmx=n2; nhx=n3; nv=n4; 
  dt=d1; dmx=d2; dhx=d3; dv=d4; 
  ot=o1; omx=o2; ohx=o3; ov=o4;

  sf_putfloat(out,"o1",ot);
  sf_putfloat(out,"o2",omx);
  sf_putfloat(out,"o3",ohx);
  sf_putfloat(out,"d1",dt);
  sf_putfloat(out,"d2",dmx);
  sf_putfloat(out,"d3",dhx);
  sf_putfloat(out,"n1",nt);
  sf_putfloat(out,"n2",nmx);
  sf_putfloat(out,"n3",nhx);
  sf_putstring(out,"label1","Time");
  sf_putstring(out,"label2","Midpoint");
  sf_putstring(out,"label3","Offset");
  sf_putstring(out,"unit1","s");
  sf_putstring(out,"unit2","m");
  sf_putstring(out,"unit3","m");

  if (adj){
    sf_putstring(out,"title","m");
    sf_putfloat(out,"o4",ov);
    sf_putfloat(out,"d4",dv);
    sf_putfloat(out,"n4",nv);
    sf_putstring(out,"label4","Velocity Perturbation");
    sf_putstring(out,"unit4","");
  }
  else{
    sf_putstring(out,"title","d");
  }

  vp = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc(n1);
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,vel);
    for (it=0;it<nt;it++) vp[ix][it] = trace[it];
  }

  m = sf_floatalloc2(nt,nmx*nhx*nv);
  d = sf_floatalloc2(nt,nmx*nhx);
  w = sf_floatalloc2(nt,nmx*nv);

  if (adj){
    for (ix=0;ix<nmx;ix++) for (ihx=0;ihx<nhx;ihx++){
      sf_floatread(trace,n1,in);
      for (it=0;it<nt;it++) d[ihx*nmx + ix][it] = trace[it];
    }
    for (ix=0;ix<nmx*nhx*nv;ix++) for (it=0;it<nt;it++) m[ix][it] = 0.0;
  }
  else{
    for (ix=0;ix<nmx;ix++) for (ihx=0;ihx<nhx;ihx++) for (iv=0;iv<nv;iv++){
      sf_floatread(trace,n1,in);
      for (it=0;it<nt;it++) m[iv*nhx*nmx + ihx*nmx + ix][it] = trace[it];
    }
    for (ix=0;ix<nhx*nmx;ix++) for (it=0;it<nt;it++) d[ix][it] = 0.0;
  }

  for (ix=0;ix<nmx;ix++) for (iv=0;iv<nv;iv++){
    sf_floatread(trace,n1,weight);
    for (it=0;it<nt;it++) w[iv*nmx + ix][it] = trace[it];
  }
 
  m1 = sf_floatalloc2(nt,nmx*nhx);
  for (ix=0;ix<nmx*nhx;ix++) for (it=0;it<nt;it++) m1[ix][it] = 0.0;

  for (iv=0;iv<nv;iv++){
    if (!adj){
      for (ix=0;ix<nmx;ix++) for (ihx=0;ihx<nhx;ihx++) for (it=0;it<nt;it++){
        m1[ihx*nmx + ix][it] = m[iv*nhx*nmx + ihx*nmx + ix][it]*w[iv*nmx + ix][it];
      }
    } 
    ktop(d,m1,vp,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,aperture,adj,verbose);
    if (adj){
      for (ix=0;ix<nmx;ix++) for (ihx=0;ihx<nhx;ihx++) for (it=0;it<nt;it++){
        m[iv*nhx*nmx + ihx*nmx + ix][it] = m1[ihx*nmx + ix][it]*w[iv*nmx + ix][it];
      }
    } 
  }
  if (adj){
    for (ix=0; ix<nmx*nhx*nv; ix++) {
      for (it=0; it<nt; it++) trace[it] = m[ix][it];	
      sf_floatwrite(trace,nt,out);
    }
  }
  else{
    for (ix=0; ix<nmx*nhx; ix++) {
      for (it=0; it<nt; it++) trace[it] = d[ix][it];	
      sf_floatwrite(trace,nt,out);
    }
  }

  exit (0);
}

void ktop(float **d, float **m, float **vp,
          int nt, int nmx, int nhx, 
          float ot, float omx, float ohx, 
          float dt, float dmx, float dhx,
          float aperture, bool adj, bool verbose)
/*< Kirchhoff time migration operator >*/
{
  int ihx,ix,it;
  float **z; float *trace,hx;

  trace = sf_floatalloc(nt);

  #ifdef _OPENMP
  #pragma omp parallel for shared(d,m) 
  #endif
  for (ihx=0;ihx<nhx;ihx++){ /* read and demigrate or migrate the offset class */
    hx = ohx + ihx*dhx;
    kt1ofc(d,m,vp,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,ihx,hx,
               aperture,adj,verbose);
  }
    
  return;
} 

void kt1ofc(float **d, float **m, float **vp, 
               int nt, int nmx, int nhx,
               float ot, float omx, float ohx, 
               float dt, float dmx, float dhx,
               int ihx, float hx, 
               float aperture, bool adj, bool verbose)
/*< de-migrate or migrate 1 offset class >*/
{
  int ix,it;
  float **doc,**moc,*trace;
  if (adj) moc = sf_floatalloc2(nt,nmx);
  else doc = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc(nt);

  for (it=0;it<nt;it++) trace[it] = 0;
 
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) moc[ix][it] = 0;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) doc[ix][it] = 0;
    }
  } 

  if (verbose){
    if (adj) fprintf(stderr,"migrating offset class %d of %d\n",ihx+1,nhx);
    else     fprintf(stderr,"demigrating offset class %d of %d\n",ihx+1,nhx);
  }
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) moc[ix][it] = 0;
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) doc[ix][it] = 0;
    }
  } 
  for (ix=0;ix<nmx;ix++){
    if (adj){
      for (it=0;it<nt;it++) trace[it] = d[ihx*nmx + ix][it];
      ktadj(trace,moc,vp,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture);
    }
    else{
      for (it=0;it<nt;it++) trace[it] = m[ihx*nmx + ix][it];
      rho_filt(trace,nt,0);
      ktfwd(doc,trace,vp,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture);
    }
  }
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) m[ihx*nmx + ix][it] += moc[ix][it];
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d[ihx*nmx + ix][it] += doc[ix][it];
    }
  } 

  if (adj) free2float(moc);
  else free2float(doc);
  free1float(trace);
  return;
}

void ktfwd(float **d, float *m, float **vp, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icipx,
               float hx,float aperture)
/*< forward Kirchhoff time migration of 1 model trace to all data traces >*/
{
  int it,icmpx,jt;
  float dist,sx,gx,v2,dists,distg,dists2,distg2;
  float cmpx,cipx,ocipx,dcipx,t,t0,t02,ts,tg,t_floor;
  float res,res0,sphe,cos1;
  float gammainv;
  float tp0,ts0,tp02,ts02,vp2,vs2,gamma_eff;
  float gamma0,ds,dg,z,cos_s,cos_g,sin_s,sin_g,gamma02,w;

  ocipx=ocmpx; dcipx=dcmpx;

  cipx = ocipx + icipx*dcipx;
  for (icmpx=0;icmpx<ncmpx;icmpx++){
    cmpx = ocmpx + icmpx*dcmpx;
    sx = cmpx - hx/2;
    gx = cmpx + hx/2;
    dist = fabsf(cipx - cmpx);
    if (dist > aperture) continue;
    dists = fabsf(sx-cipx); 
    distg = fabsf(gx-cipx);
    dists2 = dists*dists;
    distg2 = distg*distg;
    for (it=1;it<nt;it++){
      t0 = 0.5*(ot + it*dt);
      t02 = t0*t0;
      v2=vp[icipx][it]*vp[icipx][it];
      ts = sqrtf(t02 + dists2/v2);
      tg = sqrtf(t02 + distg2/v2);
      w = spherical_divergence(ts,tg,vp[icipx][it]);
      t  = ts + tg;
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
      // if (cos1 < 0.1) continue;
      // if (fabs(t - 2*t0) > 0.1) continue;  
      t_floor = trunc(t/dt) + 1;
      jt = (int) t_floor;
      if (jt >= 0 && jt+1 < nt){
	res = (t-t_floor)/dt;
	res0 = 1.0-res;
	d[icmpx][jt]   += res0*(m[it]);
	d[icmpx][jt+1] += res*(m[it]);
      }
    }
  } 
  return;
}

void ktadj(float *d, float **m, float **vp,
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icmpx,
               float hx,float aperture)
/*< adjoint Kirchhoff time migration of 1 data trace to all model points>*/
{
  int it,icipx,ncipx,jt;
  float dist,sx,gx,v2,dists,distg,dists2,distg2;
  float cmpx,cipx,ocipx,dcipx,t,t0,t02,ts,tg,t_floor;
  float res,res0,sphe,cos1;
  float tp0,ts0,tp02,ts02,vp2,vs2,gamma_eff;
  float ds,dg,z,cos_s,cos_g,sin_s,sin_g,gamma02,w;
 
  ocipx=ocmpx; dcipx=dcmpx; ncipx=ncmpx;
   
  cmpx = ocmpx + dcmpx*icmpx; 
  sx = cmpx - hx/2;
  gx = cmpx + hx/2;
  for (icipx=0;icipx<ncipx;icipx++){
    cipx = ocipx + icipx*dcipx;
    dist = fabsf(cipx - cmpx);
    if (dist > aperture) continue;
    dists = fabsf(sx-cipx); 
    distg = fabsf(gx-cipx);
    dists2 = dists*dists;
    distg2 = distg*distg;
    for (it=1;it<nt;it++){
      t0 = 0.5*(ot + it*dt);
      t02 = t0*t0;
      v2=vp[icipx][it]*vp[icipx][it];
      ts = sqrtf(t02 + dists2/v2);
      tg = sqrtf(t02 + distg2/v2);
      w = spherical_divergence(ts,tg,vp[icipx][it]);
      t  = ts + tg;
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
     // if (cos1 < 0.1) continue;       
     // if (fabs(t - 2*t0) > 0.1) continue;  
      t_floor = trunc(t/dt) + 1;
      jt = (int) t_floor;
      if (jt >= 0 && jt+1 < nt){
	res = (t-t_floor)/dt;
	res0 = 1.0-res;
	m[icipx][it] += (res0*d[jt] + res*d[jt+1]);
      }
    }
  } 
  return;
}

float spherical_divergence(float ts,float tg,float v)
/*< spherical divergence correction for KT operator >*/
{
  float sphe;
  sphe = 1.0/sqrt(ts*tg*v*v*v);
  return sphe;
}

float angle_taper(float ts,float tg, float v, float hx)
/*< angle taper for KT operator >*/
{
  float cos1,cos2;
  cos2 = ts*ts + tg*tg - (hx/v)*(hx/v);
  cos2 = cos2/(2*ts*tg);
  cos1 = sqrt((1 + cos2)/2);
  return cos1;
}

void rho_filt(float *m,int nt,int adj)
/*< forward and adjoint rho filter for KT operator >*/
{
  float rho[31];
  float *trace;
  int it,irho,k;
  trace = sf_floatalloc(nt);

  rho[0]  =  0.0000683;rho[1]  = -0.0006419;rho[2]  =  0.0019237;rho[3]  = -0.0039358;rho[4]  =  0.0069924;
  rho[5]  = -0.0109374;rho[6]  =  0.0164332;rho[7]  = -0.0231245;rho[8]  =  0.0322571;rho[9]  = -0.0435157;
  rho[10] =  0.0593533;rho[11] = -0.0808676;rho[12] =  0.1152946;rho[13] = -0.1773558;rho[14] =  0.3434832;
  rho[15] =  0.8357222;rho[16] = -0.7415759;rho[17] =  0.1119745;rho[18] = -0.1769119;rho[19] =  0.0580648;
  rho[20] = -0.0820505;rho[21] =  0.0331987;rho[22] = -0.0421669;rho[23] =  0.0183178;rho[24] = -0.0207074;
  rho[25] =  0.0088969;rho[26] = -0.0085778;rho[27] =  0.0032669;rho[28] = -0.0023109;rho[29] =  0.0005415;
  rho[30] = -0.0000806;

  for (it=0;it<nt;it++) trace[it] = 0.0;
  for (it=16;it<nt-16;it++){
    for (irho=0;irho<31;irho++){
      if (adj){
        k = it - irho + 16;
      }
      else{
        k = it + irho - 16; 
      }
        if (k > 0 && k < nt) trace[k] = trace[k] + m[it]*rho[irho];
    }
  }
  for (it=16;it<nt-16;it++) m[it] = trace[it];  
  
  return;
}





