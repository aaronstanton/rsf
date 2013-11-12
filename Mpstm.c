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

int main(int argc, char* argv[])
{

  sf_file in,out,vel;
  int n1,n2,n3;
  int nt,nmx,nhx,nsx,ngx;
  float o1,o2,o3;
  float d1,d2,d3;
  float ot,omx,ohx,osx,ogx;
  float dt,dmx,dhx,dsx,dgx;
  float **d,**m,**v,*trace;
  int ix,it,imx,ihx,isx,igx,jt;
  float dist,sx,gx,hx,v2,dists,distg,dists2,distg2;
  float mx,midx,t,t0,t02,ts,tg,t_floor,res,res0;
  float ang,angmax,angtaper;
  bool adj;
  bool ps;
  float geoms,obliq;
  float gamma,gammainv;
  float tp0,ts0,tp02,ts02,vp2,vs2;
  float antialias;
  int index[6];
  float weight[6];
  float tx;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  vel = sf_input("vel");
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getbool("ps",&ps)) ps = false; /* flag for PS data */
  if (!sf_getfloat("angmax",&angmax)) angmax=40;
  if (!sf_getfloat("antialias",&antialias)) antialias=1;
  if (!sf_getfloat("gamma",&gamma)) gamma=2;

  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;
  if (!sf_histint(  in,"n3",&n3)) sf_error("No n3= in input");
  if (!sf_histfloat(in,"d3",&d3)) sf_error("No d3= in input");
  if (!sf_histfloat(in,"o3",&o3)) o3=0.;

  if (adj){
    nt=n1; nsx=n2; ngx=n3;  
    dt=d1; dsx=d2; dgx=d3;  
    ot=o1; osx=o2; ogx=o3;  
    if (!sf_getint(  "nmx",&nmx)) sf_error("Parameter nmx required");
    if (!sf_getfloat("omx",&omx)) sf_error("Parameter omx required");
    if (!sf_getfloat("dmx",&dmx)) sf_error("Parameter dmx required");
    if (!sf_getint(  "nhx",&nhx)) sf_error("Parameter nhx required");
    if (!sf_getfloat("ohx",&ohx)) sf_error("Parameter ohx required");
    if (!sf_getfloat("dhx",&dhx)) sf_error("Parameter dhx required");
  }
  else{
    nt=n1; nmx=n2; nhx=n3;  
    dt=d1; dmx=d2; dhx=d3;  
    ot=o1; omx=o2; ohx=o3;  
    if (!sf_getint(  "nsx",&nsx)) sf_error("Parameter nsx required");
    if (!sf_getfloat("osx",&osx)) sf_error("Parameter osx required");
    if (!sf_getfloat("dsx",&dsx)) sf_error("Parameter dsx required");
    if (!sf_getint(  "ngx",&ngx)) sf_error("Parameter ngx required");
    if (!sf_getfloat("ogx",&ogx)) sf_error("Parameter ogx required");
    if (!sf_getfloat("dgx",&dgx)) sf_error("Parameter dgx required");
  }
  d  = sf_floatalloc2(nt,nsx*ngx);
  m = sf_floatalloc2(nt,nmx*nhx);
  v = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc(n1);
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,vel);
    for (it=0;it<nt;it++) v[ix][it] = trace[it];
  }
  for (ix=0;ix<n2*n3;ix++){
    sf_floatread(trace,n1,in);
    if (adj){ 
      for (it=0;it<nt;it++) d[ix][it] = trace[it];
    }
    else{ 
      for (it=0;it<nt;it++) m[ix][it] = trace[it];
    }
  }


  kirchhoff_2d_time_op(float **d, float ***m, float **v, int nsx, int ngx, int nt, int nmx)


  if (adj){
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
    sf_putstring(out,"label2","Distance");
    sf_putstring(out,"label3","Offset");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"unit3","m");
    sf_putstring(out,"title","Reflectivity");
    for (ix=0; ix<nmx*nhx; ix++) {
     for (it=0; it<nt; it++) trace[it] = m[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"o2",osx);
    sf_putfloat(out,"o3",ogx);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"d2",dsx);
    sf_putfloat(out,"d3",dgx);
    sf_putfloat(out,"n1",nt);
    sf_putfloat(out,"n2",nsx);
    sf_putfloat(out,"n3",ngx);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","Source-X");
    sf_putstring(out,"label3","Receiver-x");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"unit3","m");
    sf_putstring(out,"title","Synthesized data");
    for (ix=0; ix<nsx*ngx; ix++) {
     for (it=0; it<nt; it++) trace[it] = d[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }

    exit (0);
}

void kirchhoff_2d_time_op(float **d, float ***m, float **v, int nsx, int ngx, int nt, int nmx, int nhx, bool adj)
{

  if (adj){
    for (ix=0;ix<nmx*nhx;ix++){
      for (it=0;it<nt;it++){ 
        m[ix][it] = 0;
      }
    }
  }
  else{ 
    for (ix=0;ix<nsx*ngx;ix++){
      for (it=0;it<nt;it++){ 
        d[ix][it] = 0;
      }
    }
  }

  gammainv = 1/gamma;

  for (isx=0;isx<nsx;isx++){
    for (igx=0;igx<ngx;igx++){
      sx = osx + isx*dsx;
      gx = ogx + igx*dgx;
      hx = gx - sx;
      ihx = (int) trunc(fabsf((hx - ohx)/dhx));
      if (!ps) midx = (sx + gx)/2;
      else midx = sx + hx/(1 + gammainv);
      for (imx=0;imx<nmx;imx++){
        mx = omx + imx*dmx;
        dist = mx - midx;
        dists = fabsf(sx-mx); 
        distg = fabsf(gx-mx);
        dists2 = dists*dists;
        distg2 = distg*distg;
        for (it=0;it<nt;it++){
          t0 = 0.5*(ot + it*dt);
          if (!ps){
            t02 = t0*t0;
            v2=v[imx][it]*v[imx][it];
            ts = sqrtf(t02 + dists2/v2);
            tg = sqrtf(t02 + distg2/v2);
	    tx = dists/(v2*(ts+dt))+ distg/(v2*(tg+dt));
          }
          else{
            tp0 = (2*t0)/(1+gamma);
            ts0 = (2*t0)*gamma/(1+gamma);
            tp02 = tp0*tp0; 
            ts02 = ts0*ts0;
            vp2=v[imx][it]*v[imx][it];
            vs2=v[imx][it]*v[imx][it]/4; /* Fix this part of the code */
            ts = sqrtf(tp02 + dists2/vp2);
            tg = sqrtf(ts02 + distg2/vs2);
	    tx = dists/(vp2*(ts+dt))+ distg/(vs2*(tg+dt));
          }
          t  = ts + tg;

          geoms=sqrtf(1/(t*v[imx][it] + 0.001));
          obliq=sqrt(.5*(1 + (t0*t0/(4*ts*tg + 0.001)) 
                - (1/(ts*tg + 0.001))*sqrt(ts*ts - t0*t0/4)*sqrt(tg*tg - t0*t0/4)));
          ang=180.0*fabsf(acos(t0/t+0.001))/PI;  
	  if(ang<=angmax) angtaper=1.0;
	  if(ang>angmax) angtaper=cos((ang-angmax)*PI/20);

          t_floor = trunc(t/dt);
          jt = (int) t_floor;
          if (jt-2 > 0 && jt+3 < nt && ihx < nhx){
	    res = (t-t_floor)/dt;
	    res0 = 1.0-res;
            if (adj){ /* data space --> model space */
              m[ihx*nmx + imx][it] += geoms*obliq*(res0*d[igx*nsx+isx][jt] + res*d[igx*nsx+isx][jt+1]);
            }
            else{ /* model space --> data space */
              d[igx*nsx+isx][jt]   += geoms*obliq*res0*m[ihx*nmx+imx][it];
              d[igx*nsx+isx][jt+1] += geoms*obliq*res*m[ihx*nmx+imx][it];
            }
          }
        }
      }
    }
  }



  return;
}



