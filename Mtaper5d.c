/* Cosine tapering of one or more axes.
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

#ifndef PI
#define PI (3.141592653589793)
#endif

void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lt,int lx1,int lx2,int lx3,int lx4);
int main(int argc, char* argv[])
{
    int n1,n2,n3,n4,n5,lt,lx1,lx2,lx3,lx4;
    float **d;
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");
    if (!sf_getint("lt",&lt))   lt  = 0; /* length of taper for axis 1 */
    if (!sf_getint("lx1",&lx1)) lx1 = 5; /* length of taper for axis 2 */
    if (!sf_getint("lx2",&lx2)) lx2 = 0; /* length of taper for axis 3 */
    if (!sf_getint("lx3",&lx3)) lx3 = 0; /* length of taper for axis 4 */
    if (!sf_getint("lx4",&lx4)) lx4 = 0; /* length of taper for axis 5 */
    /* read input file parameters */
    if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histint(in,"n3",&n3))   n3=1;
    if (!sf_histint(in,"n4",&n4))   n4=1;
    if (!sf_histint(in,"n5",&n5))   n5=1;
    d = sf_floatalloc2(n1,n2*n3*n4*n5);
    sf_floatread(d[0],n1*n2*n3*n4*n5,in);
    my_taper(d,n1,n2,n3,n4,n5,lt,lx1,lx2,lx3,lx4);
    sf_floatwrite(d[0],n1*n2*n3*n4*n5,out);
    free2float(d);
    exit (0);
}

void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lt,int lx1,int lx2,int lx3,int lx4)
{
  int it,ix,ix1,ix2,ix3,ix4;
  float tt,tx1,tx2,tx3,tx4;

  tx1=1;tx2=1;tx3=1;tx4=1;
  for (ix1=0;ix1<nx1;ix1++){
    if (ix1>=0   && ix1<lx1) tx1 = 1 - cos(((float) (ix1)/lx1)*PI/2);
    if (ix1>=lx1 && ix1<=nx1-lx1) tx1 = 1;
    if (ix1>nx1-lx1 && ix1<nx1) tx1 = cos(((float) (ix1-nx1+lx1)/lx1)*PI/2);
  for (ix2=0;ix2<nx2;ix2++){
    if (ix2>=0   && ix2<lx2) tx2 = 1 - cos(((float) (ix2)/lx2)*PI/2);
    if (ix2>=lx2 && ix2<=nx2-lx2) tx2 = 1;
    if (ix2>nx2-lx2 && ix2<nx2) tx2 = cos(((float) (ix2-nx2+lx2)/lx2)*PI/2);
  for (ix3=0;ix3<nx3;ix3++){
    if (ix3>=0   && ix3<lx3) tx3 = 1 - cos(((float) (ix3)/lx3)*PI/2);
    if (ix3>=lx3 && ix3<=nx3-lx3) tx3 = 1;
    if (ix3>nx3-lx3 && ix3<nx3) tx3 = cos(((float) (ix3-nx3+lx3)/lx3)*PI/2);
  for (ix4=0;ix4<nx4;ix4++){
    if (ix4>=0   && ix4<lx4) tx4 = 1 - cos(((float) (ix4)/lx4)*PI/2);
    if (ix4>=lx4 && ix4<=nx4-lx4) tx4 = 1;
    if (ix4>nx4-lx4 && ix4<nx4) tx4 = cos(((float) (ix4-nx4+lx4)/lx4)*PI/2);
    ix = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1;
    for(it=0;it<nt;it++){
      if (it>=0   && it<lt) tt = 1 - cos(((float) (it)/lt)*PI/2);
      if (it>=lt && it<=nt-lt) tt = 1;
      if (it>nt-lt && it<nt) tt = cos(((float) (it-nt+lt)/lt)*PI/2);
      d[ix][it] = tt*tx1*tx2*tx3*tx4*d[ix][it];
    }
  }
  }
  }
  }
  return;
}

