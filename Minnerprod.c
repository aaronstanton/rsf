/* Compute the inner product of two datasets with the same dimension
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
#include <rsf.h>
#include "myfree.h"
int main(int argc, char* argv[])
{ 
    int n1,n2,n3,n4,n5,i1,i2,i3,i4,i5,ix;
    float *trace1,*trace2,sum;
    sf_file in1, in2;
    sf_init (argc,argv);
 
    in1 = sf_input("in1");
    in2 = sf_input("in2");

    /* read input file parameters */
    if (!sf_histint(in1,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histint(in1,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histint(in1,"n3",&n3))   n3=1;
    if (!sf_histint(in1,"n4",&n4))   n4=1;
    if (!sf_histint(in1,"n5",&n5))   n5=1;

    trace1 = sf_floatalloc (n1);
    trace2 = sf_floatalloc (n1);
    sum = 0.0;
    for (ix=0; ix<n2*n3*n4*n5; ix++) {
      sf_floatread(trace1,n1,in1);
      sf_floatread(trace2,n1,in2);
      for (i1=0; i1<n1; i1++){
        sum += trace1[i1]*trace2[i1];
      }
    }
 
    fprintf(stderr,"%6.2f\n",sum);
    free1float(trace1);
    free1float(trace2);
    exit (0);
}

