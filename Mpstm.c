/* PSTM operator.
*/
/*
  Copyright (C) 2013 University of Alberta
  
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
#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif
#include "myfree.h"

void kt_2d_fwd(float **d, float *m, float **vp, float **vs, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icipx,
               float hx,float aperture,float fwidth,float gamma,bool ps);
void kt_2d_adj(float *d, float **m, float **v, float **vs,
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icmpx,
               float hx,float aperture,float fwidth,float gamma,bool ps);
float spherical_divergence(float ts,float tg,float v);
float angle_taper(float ts,float tg, float v, float hx);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void rho_filt(float *m,int nt,int adj);

int main(int argc, char* argv[])
{

  sf_file in,out,velp,vels;
  int n1,n2;
  int nt,nmx;
  int it,ix;
  float o1,o2;
  float d1,d2;
  float ot,omx;
  float dt,dmx;
  float **d,**m,**vp,**vs,*trace,*traces;
  bool adj;
  bool ps;
  float gamma;
  float aperture;
  float hx;
  
  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  if (!sf_getbool("ps",&ps)) ps = false; /* flag for PS data */
  if (ps) vels = sf_input("vs");
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getfloat("gamma",&gamma)) gamma=2;
  if (!sf_getfloat("aperture",&aperture)) aperture=1000;
  if (!sf_getfloat("hx",&hx)) sf_error("parameter hx required.");
  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;

  nt=n1; nmx=n2;  
  dt=d1; dmx=d2;  
  ot=o1; omx=o2;

  vp = sf_floatalloc2(nt,nmx);
  if (ps) vs = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc(n1);
  traces = sf_floatalloc(n1);
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,velp);
    for (it=0;it<nt;it++) vp[ix][it] = trace[it];
    if (ps){ 
      sf_floatread(traces,n1,vels);
      for (it=0;it<nt;it++) vs[ix][it] = traces[it];
    }
  }
  if (adj){ 
    m = sf_floatalloc2(nt,nmx);
    for (ix=0;ix<n2;ix++){
      for (it=0;it<nt;it++) m[ix][it] = 0.0;
    }
  }
  else{ 
    d = sf_floatalloc2(nt,nmx);
    for (ix=0;ix<n2;ix++){
      for (it=0;it<nt;it++) d[ix][it] = 0.0;
    }
  }

  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,in);
    if (!adj) rho_filt(trace,nt,0);
    if (adj) kt_2d_adj(trace,m,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,fwidth,gamma,ps);
    else     kt_2d_fwd(d,trace,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,fwidth,gamma,ps);
  }

  if (adj){ 
    for (ix=0;ix<nmx;ix++){
      for (it=0;it<nt;it++) trace[it] = m[ix][it]; 
      rho_filt(trace,nt,1);
      for (it=0;it<nt;it++) m[ix][it] = trace[it]; 
    }
  }

  if (adj){
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"n1",nt);
    sf_putfloat(out,"n2",nmx);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","Midpoint");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"title","Reflectivity");
    for (ix=0; ix<nmx; ix++) {
     for (it=0; it<nt; it++) trace[it] = m[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"n1",nt);
    sf_putfloat(out,"n2",nmx);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","Midpoint");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"title","Synthesized data");
    for (ix=0; ix<nmx; ix++) {
     for (it=0; it<nt; it++) trace[it] = d[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }

    exit (0);
}

void kt_2d_fwd(float **d, float *m, float **vp, float **vs, 
               int nt, int ncmpx, float ot, float ocmpx, float dt, float dcmpx,
               int icipx,
               float hx,float aperture,float fwidth,float gamma,bool ps)
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
               float hx,float aperture,float fwidth,float gamma,bool ps)
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



void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
{
  int iw,nw,ntfft;
  float *in1, *out2;
  int ia,ib,ic,id;
  int it;
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
float spherical_divergence(float ts,float tg,float v)
{
  float sphe;
  sphe = 1.0/sqrt(ts*tg*v*v*v);
  return sphe;
}
float angle_taper(float ts,float tg, float v, float hx)
{
  float cos1,cos2;
  cos2 = ts*ts + tg*tg - (hx/v)*(hx/v);
  cos2 = cos2/(2*ts*tg);
  cos1 = sqrt((1 + cos2)/2);
  return cos1;
}

void rho_filt(float *m,int nt,int adj)
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

