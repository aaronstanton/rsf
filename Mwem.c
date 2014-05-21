/* Wave Equation Migration.
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

int main(int argc, char* argv[])
{
  wem_op(d,
         nt,ot,dt,
         nmx,omx,dmx,
         nhx,ohx,dhx,
         m,
         nt,ot,dt,
         nmx,omx,dmx,
         adj);
  exit (0);
}

void wem_op(float **d,
         int nt,float ot,float dt,
         int nmx,float omx,float dmx,
         int nhx,float ohx,float dhx,
         float **m,
         int nz,float oz,float dz,
         int nmx,float omx,float dmx,
         int nhx,float ohx,float dhx,
         bool adj);
/*< wave equation depth migration operator >*/
{

  wem_set_up_vel();
  for (iw=iw_min;iw<iw_max;iw++){
    wem_define_src();
    for (iz=0;iz<nz;iz++){
      wem_extrap_src();
      wem_extrap_rec();
    }
  }

  return;
}

void wem_set_up_vel(float **c,float **po,float **pd,int nz,int nmx)
/*< decompose slowness into layer average, and layer purturbation >*/
{
  int iz,ix;

  for (iz=0;iz<nz;iz++){
    po[iz] = 0.0;
    for (ix=0;ix<nmx;ix++) po[iz] += 1.0/c[ix][iz];
    po[iz] /= (float) nmx;
    for (ix=0;ix<nmx;ix++){ 
      pd[ix][iz] = 1.0/c[ix][iz] - po[iz];
    }
  }
  return;
}

void wem_define_src()
/*< load sources into array >*/
{
  /* find index of min offset to place the source */
  imin_hx = 0;
  for (ihx=0;ihx<nhx;ihx++){
      if (dhx*ihx + ohx < dhx*imin_hx + ohx) imin_hx = ihx;
  }
  for (imx=0;imx<nmx;imx++){
    d_s_wx[imin_hx*nmx + imx][iw] = wavelet[iw];
  }
  return;
}


