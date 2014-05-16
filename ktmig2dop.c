/* 2d Kirchhoff migration routine */
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
#ifdef _OPENMP
#include <omp.h>
#endif
#include "myfree.h"
#include <fftw3.h>
/*^*/


#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif 

#include "ktmig2dop.h"

void kt_2d_op(float **d, float **m, float **vp, float **vs, 
               int nt, int nmx, int nhx, 
               float ot, float omx, float ohx, 
               float dt, float dmx, float dhx,
               float aperture, float gamma, bool ps, int numthreads, bool adj, bool verbose)
/*< Kirchhoff time migration operator >*/
{
  int ihx,ix,it;
  float **z; float *trace,hx;

  trace = sf_floatalloc(nt);
  z = sf_floatalloc2(nt,nmx*nhx);
  
  if (adj){ 
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++) z[ix][it] = d[ix][it]; /* save data in z and put back afterward */ 
      for (it=0;it<nt;it++) trace[it] = d[ix][it]; 
      bpfilter(trace,dt,nt,0,5,30,40);
      for (it=0;it<nt;it++) d[ix][it] = trace[it]; 
    }
  }
  else{
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++) z[ix][it] = m[ix][it];
    }
    triangle_filter(m,z,nt,nmx,nhx,adj);
    fkfilter(m,dt,nt,dhx,nmx,nhx,-0.5,-0.25,0.25,0.5);
  }

omp_set_num_threads(numthreads);

#ifdef _OPENMP
#pragma omp parallel for \
    shared(d,m) 
#endif
  for (ihx=0;ihx<nhx;ihx++){ /* read and demigrate or migrate the offset class */
    hx = ohx + ihx*dhx;
    kt_2d_1ofc(d,m,vp,vs,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,ihx,hx,
               aperture,gamma,ps,adj,verbose);
  }
    
  if (adj){ 
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++) d[ix][it] = z[ix][it]; /* put data back */ 
      for (it=0;it<nt;it++) trace[it] = m[ix][it]; 
      rho_filt(trace,nt,1);
      for (it=0;it<nt;it++) m[ix][it] = trace[it]; 
    }
    fkfilter(m,dt,nt,dhx,nmx,nhx,-0.5,-0.25,0.25,0.5);
    triangle_filter(m,z,nt,nmx,nhx,adj);
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++) m[ix][it] = z[ix][it];
    }
  }
  else{
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++) trace[it] = d[ix][it]; 
      bpfilter(trace,dt,nt,0,5,30,40);
      for (it=0;it<nt;it++) d[ix][it] = trace[it]; 
    }
  }

  return;
} 

void kt_2d_1ofc(float **d, float **m, float **vp, float **vs, 
               int nt, int nmx, int nhx,
               float ot, float omx, float ohx, 
               float dt, float dmx, float dhx,
               int ihx, float hx, 
               float aperture, float gamma, bool ps, bool adj, bool verbose)
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
      kt_2d_adj(trace,moc,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,gamma,ps);
    }
    else{
      for (it=0;it<nt;it++) trace[it] = m[ihx*nmx + ix][it];
      rho_filt(trace,nt,0);
      kt_2d_fwd(doc,trace,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,gamma,ps);
    }
  }
  if (adj){
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) m[ihx*nmx + ix][it] = moc[ix][it];
    }
  }
  else{
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) d[ihx*nmx + ix][it] = doc[ix][it];
    }
  } 

  if (adj) free2float(moc);
  else free2float(doc);
  free1float(trace);
  return;
}

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
  float tp0,ts0,tp02,ts02,vp2,vs2,gamma_eff;
  float gamma0,ds,dg,z,cos_s,cos_g,sin_s,sin_g,gamma02,w;

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
        w = spherical_divergence(ts,tg,vp[icipx][it]);
      }
      else{
        gamma0 = vp[icipx][it]/vs[icipx][it];
	tp0 = (2*t0)/(1+gamma0);
	ts0 = (2*t0)*gamma0/(1+gamma0);
	tp02 = tp0*tp0; 
	ts02 = ts0*ts0;
	vp2=vp[icipx][it]*vp[icipx][it];
	vs2=vs[icipx][it]*vs[icipx][it];
	ts = sqrtf(tp02 + dists2/vp2);
	tg = sqrtf(ts02 + distg2/vs2);


        ds = vp[icipx][it]*ts;
        dg = vs[icipx][it]*tg;
        z = sqrt(ds*ds - dists2);
        cos_s = z/ds;
        cos_g = z/dg;
        sin_s = dists/ds;
        sin_g = distg/dg;
        cos1 = cos_s*cos_g - sin_s*sin_g;        
        gamma02 = gamma0*gamma0;
        w = z*(1+cos1)*(gamma02*ts+tg)*(gamma02*gamma02*ts*ts+tg*tg)/(2*vp2*vp[icipx][it]*sqrt(1+2*gamma0*cos1+gamma02)*gamma0*ts*ts*tg*tg);

        gamma_eff = (vp2/vs2)/gamma;
        sphe = spherical_divergence(ts,tg,sqrt((1+gamma_eff)/((1+gamma)*gamma_eff))*vp[icipx][it]);
      }
      t  = ts + tg;
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
      if (cos1 < 0.1) continue;
    
      if (fabs(t - 2*t0) > 0.1) continue; 
         
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
  float tp0,ts0,tp02,ts02,vp2,vs2,gamma_eff;
  float gamma0,ds,dg,z,cos_s,cos_g,sin_s,sin_g,gamma02,w;
 
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
        w = spherical_divergence(ts,tg,vp[icipx][it]);
      }
      else{
        gamma0 = vp[icipx][it]/vs[icipx][it];
        tp0 = (2*t0)/(1+gamma0);
        ts0 = (2*t0)*gamma0/(1+gamma0);
        tp02 = tp0*tp0;
        ts02 = ts0*ts0;
        vp2=vp[icipx][it]*vp[icipx][it];
        vs2=vs[icipx][it]*vs[icipx][it];
        ts = sqrtf(tp02 + dists2/vp2);
        tg = sqrtf(ts02 + distg2/vs2);

        ds = vp[icipx][it]*ts;
        dg = vs[icipx][it]*tg;
        z = sqrt(ds*ds - dists2);
        cos_s = z/ds;
        cos_g = z/dg;
        sin_s = dists/ds;
        sin_g = distg/dg;
        cos1 = cos_s*cos_g - sin_s*sin_g;
        gamma02 = gamma0*gamma0;
        w = z*(1+cos1)/(2*vp2*vp[icipx][it]*sqrt(1+2*gamma0*cos1+gamma02))*(tg/(gamma02*ts) + (gamma02*ts)/tg)*sqrt((gamma02*ts + tg)/(ts*tg));
        gamma_eff = (vp2/vs2)/gamma;
        sphe = spherical_divergence(ts,tg,sqrt((1+gamma_eff)/((1+gamma)*gamma_eff))*vp[icipx][it]);
      }
      t  = ts + tg;
      cos1 = angle_taper(ts,tg,vp[icipx][it],hx);
      if (cos1 < 0.1) continue;       
    
      if (fabs(t - 2*t0) > 0.1) continue; 
         
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

void triangle_filter(float **m,float **z,int nt,int nmx,int nhx,bool adj)
/*< 5 point triangle filter forward and adjoint operator. It acts on the offset axis. The operator does nothing if the offset axis has a length less than or equal to 5. >*/
{
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
    	      z[(0)*nmx + imx][it]  = (3*m[(0)*nmx + imx][it] +   2*m[(1)*nmx + imx][it] + 1*m[(2)*nmx + imx][it])/9;
    	      z[(1)*nmx + imx][it]  = (4*m[(0)*nmx + imx][it] +   3*m[(1)*nmx + imx][it] + 2*m[(2)*nmx + imx][it] + 1*m[(3)*nmx + imx][it])/9;
    	      z[(2)*nmx + imx][it]  = (2*m[(0)*nmx + imx][it] +   2*m[(1)*nmx + imx][it] + 3*m[(2)*nmx + imx][it] + 2*m[(3)*nmx + imx][it] + 1*m[(4)*nmx + imx][it])/9;
    	      z[(3)*nmx + imx][it]  = (2*m[(1)*nmx + imx][it] +   2*m[(2)*nmx + imx][it] + 3*m[(3)*nmx + imx][it] + 2*m[(4)*nmx + imx][it] + 1*m[(5)*nmx + imx][it])/9;

    	      z[(nhx-1)*nmx + imx][it]  = (3*m[(nhx-1)*nmx + imx][it] +   2*m[(nhx-2)*nmx + imx][it] + 1*m[(nhx-3)*nmx + imx][it])/9;
    	      z[(nhx-2)*nmx + imx][it]  = (4*m[(nhx-1)*nmx + imx][it] +   3*m[(nhx-2)*nmx + imx][it] + 2*m[(nhx-3)*nmx + imx][it] + 1*m[(nhx-4)*nmx + imx][it])/9;
    	      z[(nhx-3)*nmx + imx][it]  = (2*m[(nhx-1)*nmx + imx][it] +   2*m[(nhx-2)*nmx + imx][it] + 3*m[(nhx-3)*nmx + imx][it] + 2*m[(nhx-4)*nmx + imx][it] + 1*m[(nhx-5)*nmx + imx][it])/9;
    	      z[(nhx-4)*nmx + imx][it]  = (2*m[(nhx-2)*nmx + imx][it] +   2*m[(nhx-3)*nmx + imx][it] + 3*m[(nhx-4)*nmx + imx][it] + 2*m[(nhx-5)*nmx + imx][it] + 1*m[(nhx-6)*nmx + imx][it])/9;
        }
      }
      for (ihx=4;ihx<nhx-4;ihx++){
        for (imx=0;imx<nmx;imx++){
          for (it=0;it<nt;it++){
            z[ihx*nmx + imx][it] = (m[(ihx-2)*nmx + imx][it] 
                                    + 2*m[(ihx-1)*nmx + imx][it] 
                                    + 3*m[(ihx)*nmx + imx][it] 
                                    + 2*m[(ihx+1)*nmx + imx][it] 
                                    +   m[(ihx+2)*nmx + imx][it])/9;
  	  }
        }
      }
    }
  }
    
  return;
}

void cg_irls_kt2d(float **d,int nd,
             float **m,int nm,
             float *wd,int nwd,
	     int itmax_external,int itmax_internal,
             float **vp,float **vs,
             int nt,int nmx,int nhx,
             float ot,float omx,float ohx,
             float dt,float dmx,float dhx,
             float *misfit,
             float aperture,float psgamma,bool ps,
             int numthreads,
             int verbose)
/*< Non-quadratic regularization with CG-LS. The inner CG routine is taken from Algorithm 2 of Scales, 1987. Make sure linear operator passes the dot product. In this case (PSTM), the linear operator is a Kirchhoff demigration operator. >*/
{
  float **v,**Pv,**Ps,**s,**ss,**g,**r;
  float alpha,beta,delta,gamma,gamma_old,**P,Max_m,progress; 
  int ix,it,j,k;
  v  = sf_floatalloc2(nt,nm);
  P  = sf_floatalloc2(nt,nm);
  Pv = sf_floatalloc2(nt,nm);
  Ps = sf_floatalloc2(nt,nm);
  g  = sf_floatalloc2(nt,nm);
  r  = sf_floatalloc2(nt,nd);
  s  = sf_floatalloc2(nt,nm);
  ss = sf_floatalloc2(nt,nd);

  progress = 0.0;

  for (ix=0;ix<nm;ix++){
    for (it=0;it<nt;it++){
      m[ix][it] = 0;				
      P[ix][it] = 1;
      v[ix][it] = m[ix][it];
    }
  }

  for (ix=0;ix<nd;ix++){
    for (it=0;it<nt;it++) r[ix][it] = d[ix][it];				
  }

  for (j=1;j<=itmax_external;j++){
    for (ix=0;ix<nm;ix++){
      for (it=0;it<nt;it++){ 
        Pv[ix][it] = v[ix][it]*P[ix][it];
      }
    }

    kt_2d_op(r,Pv,vp,vs,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,
             aperture,psgamma,ps,numthreads,false,false);

    for (ix=0;ix<nd;ix++){
      for (it=0;it<nt;it++){ 
        r[ix][it] = r[ix][it]*wd[ix];
      }
    }

    for (ix=0;ix<nd;ix++){
      for (it=0;it<nt;it++){ 
        r[ix][it] = d[ix][it] - r[ix][it];
      }
    }


    for (ix=0;ix<nd;ix++){
      for (it=0;it<nt;it++){ 
        r[ix][it] = r[ix][it]*wd[ix];
      }
    }

    kt_2d_op(r,g,vp,vs,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,
             aperture,psgamma,ps,numthreads,true,false);

    for (ix=0;ix<nm;ix++){
      for (it=0;it<nt;it++){
        g[ix][it] = g[ix][it]*P[ix][it];
        s[ix][it] = g[ix][it];
      }
    }

    gamma = cgdot(g,nt,nm);
    gamma_old = gamma;

    for (k=1;k<=itmax_internal;k++){
      progress += 1.0/((float) itmax_internal*itmax_external);
      if (verbose) progress_msg(progress);
      for (ix=0;ix<nm;ix++){
        for (it=0;it<nt;it++){
          Ps[ix][it] = s[ix][it]*P[ix][it];
        }
      }

      kt_2d_op(ss,Ps,vp,vs,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,
               aperture,psgamma,ps,numthreads,false,false);

      for (ix=0;ix<nd;ix++){
        for (it=0;it<nt;it++){ 
          ss[ix][it] = ss[ix][it]*wd[ix];
        }
      }

      delta = cgdot(ss,nt,nd);
      alpha = gamma/(delta + 0.00000001);


      for (ix=0;ix<nm;ix++){
        for (it=0;it<nt;it++){
          v[ix][it] = v[ix][it] +  s[ix][it]*alpha;
        }
      }

      for (ix=0;ix<nd;ix++){
        for (it=0;it<nt;it++){
          r[ix][it] = r[ix][it] -  ss[ix][it]*alpha;
        }
      }


      for (ix=0;ix<nd;ix++){
        for (it=0;it<nt;it++){ 
          r[ix][it] = r[ix][it]*wd[ix];
        }
      }

      misfit[(j-1)*itmax_internal + (k-1)] = cgdot(r,nt,nd);

      kt_2d_op(r,g,vp,vs,nt,nmx,nhx,ot,omx,ohx,dt,dmx,dhx,
               aperture,psgamma,ps,numthreads,true,false);


      for (ix=0;ix<nm;ix++){
        for (it=0;it<nt;it++){
          g[ix][it] = g[ix][it]*P[ix][it];
        }
      }

      gamma = cgdot(g,nt,nm);
      beta = gamma/(gamma_old + 0.00000001);

      gamma_old = gamma;
      for (ix=0;ix<nm;ix++){
        for (it=0;it<nt;it++){
          s[ix][it] = g[ix][it] + s[ix][it]*beta;
        }
      }
    }

    for (ix=0;ix<nm;ix++){
      for (it=0;it<nt;it++){
        m[ix][it] = v[ix][it]*P[ix][it];
      }
    }

    Max_m = max_abs(m,nt,nm);

    for (ix=0;ix<nm;ix++){
      for (it=0;it<nt;it++){
        P[ix][it] = fabsf(m[ix][it]*(1/Max_m));
      }
    }

  }

  return;
  
}

float max_abs(float **x,int nt,int nm)
/*< Compute Mx = max absolute value of matrix of floats, x >*/
{
  int it,ix;
  float Mx;
  
  Mx = 0;
  for (ix=0;ix<nm;ix++){
    for (it=0;it<nt;it++){   
      if(Mx<fabsf(x[ix][it])) Mx=fabsf(x[ix][it]);
    }
  }
  return(Mx);
}

float cgdot(float **x,int nt,int nm)
/*< Compute the inner product for matrix of floats, x >*/
{
  int it,ix;
  float cgdot;
  
  cgdot = 0;
  for (ix=0;ix<nm;ix++){  
    for (it=0;it<nt;it++){ 
      cgdot = cgdot + x[ix][it]*x[ix][it];
    }
  }
  return(cgdot);
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

void fkfilter(float **d, float dt, int nt, float dx, int nmx, int nhx, float pa, float pb, float pc, float pd)
/*< fk filter >*/
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
/*< fk operator >*/
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

void progress_msg(float progress)
/*< progress message >*/
{ 
  fprintf(stderr,"\r[%6.2f%% complete]",progress*100);
  return;
}

