/* Bin 2D receiver-source data to midpoint-offset.
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

#include <rsf.h>
#include "myfree.h"

int main(int argc, char* argv[])
{
    int it,igx,isx,imx,ihx,ix;
    int nt,ngx,nsx,nmx,nhx;
    float ot,ogx,osx,omx,ohx;
    float dt,dgx,dsx,dmx,dhx;
    float gx,sx,mx,hx; 
    float **d_gxsx,**d_mxhx; 
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");

    if (!sf_histint(in,"n1",&nt)) sf_error("No n1= in input");
    if (!sf_histint(in,"n2",&ngx)) sf_error("No n2= in input");
    if (!sf_histint(in,"n3",&nsx)) sf_error("No n3= in input");
    if (!sf_histfloat(in,"o1",&ot)) sf_error("No o1= in input");
    if (!sf_histfloat(in,"o2",&ogx)) sf_error("No o2= in input");
    if (!sf_histfloat(in,"o3",&osx)) sf_error("No o3= in input");
    if (!sf_histfloat(in,"d1",&dt)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"d2",&dgx)) sf_error("No d2= in input");
    if (!sf_histfloat(in,"d3",&dsx)) sf_error("No d3= in input");

    if (!sf_getint("nmx",&nmx)) sf_error("Need nmx=");
    if (!sf_getfloat("omx",&omx)) sf_error("Need omx=");
    if (!sf_getfloat("dmx",&dmx)) sf_error("Need dmx=");
    if (!sf_getint("nhx",&nhx)) sf_error("Need nhx=");
    if (!sf_getfloat("ohx",&ohx)) sf_error("Need ohx=");
    if (!sf_getfloat("dhx",&dhx)) sf_error("Need dhx=");

    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"n2",nmx);
    sf_putstring(out,"label3","Midpoint");
    sf_putstring(out,"unit3","m");
    sf_putfloat(out,"o3",ohx);
    sf_putfloat(out,"d3",dhx);
    sf_putfloat(out,"n3",nhx);
    sf_putstring(out,"label3","Offset");
    sf_putstring(out,"unit3","m");

    d_gxsx = sf_floatalloc2(nt,ngx*nsx);
    d_mxhx = sf_floatalloc2(nt,nmx*nhx);
    sf_floatread(d_gxsx[0],nt*ngx*nsx,in);
    for (ix=0;ix<nmx*nhx;ix++) for (it=0;it<nt;it++) d_mxhx[ix][it] = 0.0;
    for (igx=0;igx<ngx;igx++){
      for (isx=0;isx<nsx;isx++){
        gx = ogx + dgx*igx;
        sx = osx + dsx*isx;
        mx = (gx + sx)/2;
        hx = (gx - sx)/2;
        imx = (mx - omx)/dmx;
        ihx = (hx - ohx)/dhx;
        if (imx > 0 && imx < nmx && ihx > 0 && ihx < nhx){
          for (it=0;it<nt;it++) d_mxhx[ihx*nmx + imx][it] = d_gxsx[isx*ngx + igx][it];
        }
      }
    }
    sf_floatwrite(d_mxhx[0],nt*nmx*nhx,out);
    exit (0);
}

