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
#include "myfree.h"
#include "ktmig2dop.h"

int main(int argc, char* argv[])
{

  sf_file in,out,velp,vels;
  int n1,n2,n3;
  int nt,nmx,nhx;
  int it,ix,ihx;
  float o1,o2,o3;
  float d1,d2,d3;
  float ot,omx,ohx;
  float dt,dmx,dhx;
  float **d,**m,**vp,**vs,*trace,*traces;
  bool adj;
  bool ps;
  bool verbose;
  float gamma;
  float aperture;
  float hx;
  
  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  velp = sf_input("vp");
  if (!sf_getbool("ps",&ps)) ps = false; /* flag for PS data */
  if (!sf_getbool("verbose",&verbose)) verbose = false; /* verbosity flag*/
  if (ps) vels = sf_input("vs");
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getfloat("gamma",&gamma)) gamma=2;
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

  nt=n1; nmx=n2; nhx=n3;  
  dt=d1; dmx=d2; dhx=d3;  
  ot=o1; omx=o2; ohx=o3;

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
    sf_putstring(out,"title","Reflectivity");
  }
  else{
    sf_putstring(out,"title","Synthesized data");
  }

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
  }
  else{ 
    d = sf_floatalloc2(nt,nmx);
  }

  for (ihx=0;ihx<nhx;ihx++){
    hx = ohx + ihx*dhx;
    if (verbose) fprintf(stderr,"processing offset class %d of %d (hx=%6.0fm)\n",ihx+1,nhx,hx);
    if (adj){ 
      for (ix=0;ix<nmx;ix++){
	for (it=0;it<nt;it++) m[ix][it] = 0.0;
      }
    }
    else{ 
      for (ix=0;ix<nmx;ix++){
	for (it=0;it<nt;it++) d[ix][it] = 0.0;
      }
    }

    for (ix=0;ix<nmx;ix++){ /* read and demigrate or migrate the offset class */
      sf_floatread(trace,n1,in);
      if (!adj) rho_filt(trace,nt,0);
      if (adj) kt_2d_adj(trace,m,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,gamma,ps);
      else     kt_2d_fwd(d,trace,vp,vs,nt,nmx,ot,omx,dt,dmx,ix,hx,aperture,gamma,ps);
    }
    
    if (adj){ 
      for (ix=0;ix<nmx;ix++){
	for (it=0;it<nt;it++) trace[it] = m[ix][it]; 
	rho_filt(trace,nt,1);
	for (it=0;it<nt;it++) m[ix][it] = trace[it]; 
      }
    }
    
    if (adj){
      for (ix=0; ix<nmx; ix++) {
	for (it=0; it<nt; it++) trace[it] = m[ix][it];	
	sf_floatwrite(trace,nt,out);
      }
    }
    else{
      for (ix=0; ix<nmx; ix++) {
	for (it=0; it<nt; it++) trace[it] = d[ix][it];	
	sf_floatwrite(trace,nt,out);
      }
    }
  }
  
  exit (0);
}


