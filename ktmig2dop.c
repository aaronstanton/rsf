/* allocation and deallocation routines */
/*
  Copyright (C) 213 University of Alberta
  
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
#include <rsf.h>
/*^*/

#include "ktmig2dop.h"

void kt_2d_fwd(float **d, float *m, float **vp, float **vs, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icipx,
               float hx,float aperture,float gamma,bool ps)
/*< forward Kirchhoff time migration of 1 model trace to all data traces >*/
{
  int it,icmpx,jt;
  float dist,sx,gx,v2,dists,distg,dists2,distg2;
  float cmpx,cipx,ocipx,dcipx,t,t0,t02,ts,tg,t_floor;
  float res,res0,sphe,cos1;
  float gammainv;
  float tp0,ts0,tp02,ts02,vp2,vs2;
  
  ocipx=ocmpx; dcipx=dcmpx;

  gammainv = 1/gamma;

  cipx = ocipx + icipx*dcipx;
  for (icmpx=0;icmpx<ncmpx;icmpx++){
    cmpx = ocmpx + icmpx*dcmpx;
    if (!ps){ 
      sx = cmpx - hx/2;
      gx = cmpx + hx/2;
    }
    else {
      sx = cmpx - hx*(1/(1 + gammainv));
      gx = cmpx + hx*(1 - 1/(1 + gammainv));
    }
    dist = fabsf(cipx - cmpx);
    if (dist > aperture) continue;
    dists = fabsf(sx-cipx); 
    distg = fabsf(gx-cipx);
    dists2 = dists*dists;
    distg2 = distg*distg;
    for (it=1;it<nt;it++){
      t0 = 0.5*(ot + it*dt);
      if (!ps){
	t02 = t0*t0;
	v2=vp[icipx][it]*vp[icipx][it];
	ts = sqrtf(t02 + dists2/v2);
	tg = sqrtf(t02 + distg2/v2);
      }
      else{
	tp0 = (2*t0)/(1+gamma);
	ts0 = (2*t0)*gamma/(1+gamma);
	tp02 = tp0*tp0; 
	ts02 = ts0*ts0;
	vp2=vp[icipx][it]*vp[icipx][it];
	vs2=vs[icipx][it]*vs[icipx][it];
	ts = sqrtf(tp02 + dists2/vp2);
	tg = sqrtf(ts02 + distg2/vs2);
      }
      t  = ts + tg;
      sphe = spherical_divergence(ts,tg,vp[icipx][it]);
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
      if (cos1 < 0.1) continue;          
      t_floor = trunc(t/dt) + 1;
      jt = (int) t_floor;
      if (jt >= 0 && jt+1 < nt){
	res = (t-t_floor)/dt;
	res0 = 1.0-res;
	d[icmpx][jt]   += cos1*sphe*res0*(m[it]);
	d[icmpx][jt+1] += cos1*sphe*res*(m[it]);
      }
    }
  } 
  return;
}

void kt_2d_adj(float *d, float **m, float **vp, float **vs, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icmpx,
               float hx,float aperture,float gamma,bool ps)
/*< adjoint Kirchhoff time migration of 1 data trace to all model points>*/
{
  int it,icipx,ncipx,jt;
  float dist,sx,gx,v2,dists,distg,dists2,distg2;
  float cmpx,cipx,ocipx,dcipx,t,t0,t02,ts,tg,t_floor;
  float res,res0,sphe,cos1;
  float gammainv;
  float tp0,ts0,tp02,ts02,vp2,vs2;
 
  ocipx=ocmpx; dcipx=dcmpx; ncipx=ncmpx;
   
  gammainv = 1/gamma;
  cmpx = ocmpx + dcmpx*icmpx; 
  if (!ps){ 
    sx = cmpx - hx/2;
    gx = cmpx + hx/2;
  }
  else {
    sx = cmpx - hx*(1/(1 + gammainv));
    gx = cmpx + hx*(1 - 1/(1 + gammainv));
  }
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
      if (!ps){
	t02 = t0*t0;
	v2=vp[icipx][it]*vp[icipx][it];
	ts = sqrtf(t02 + dists2/v2);
	tg = sqrtf(t02 + distg2/v2);
      }
      else{
	tp0 = (2*t0)/(1+gamma);
	ts0 = (2*t0)*gamma/(1+gamma);
	tp02 = tp0*tp0; 
	ts02 = ts0*ts0;
	vp2=vp[icipx][it]*vp[icipx][it];
	vs2=vs[icipx][it]*vs[icipx][it];
	ts = sqrtf(tp02 + dists2/vp2);
	tg = sqrtf(ts02 + distg2/vs2);
      }
      t  = ts + tg;
      sphe = spherical_divergence(ts,tg,vp[icipx][it]);
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
      if (cos1 < 0.1) continue;          
      t_floor = trunc(t/dt) + 1;
      jt = (int) t_floor;
      if (jt >= 0 && jt+1 < nt){
	res = (t-t_floor)/dt;
	res0 = 1.0-res;
	m[icipx][it] += cos1*sphe*(res0*d[jt] + res*d[jt+1]);
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

