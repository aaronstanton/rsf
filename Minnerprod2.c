/* Compute the inner product of two 2-component datasets with the same dimension
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
    int n1,n2,n3,n4,n5,i1,ix;
    float *trace1a,*trace1b,*trace2a,*trace2b,sum;
    sf_file in1a,in1b,in2a,in2b;
    sf_init (argc,argv);
    in1a = sf_input("in1a");
    in1b = sf_input("in1b");
    in2a = sf_input("in2a");
    in2b = sf_input("in2b");
    /* read input file parameters */
    if (!sf_histint(in1a,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histint(in1a,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histint(in1a,"n3",&n3)) n3=1;
    if (!sf_histint(in1a,"n4",&n4)) n4=1;
    if (!sf_histint(in1a,"n5",&n5)) n5=1;
    trace1a = sf_floatalloc (n1);
    trace1b = sf_floatalloc (n1);
    trace2a = sf_floatalloc (n1);
    trace2b = sf_floatalloc (n1);
    sum = 0.0;
    for (ix=0; ix<n2*n3*n4*n5; ix++) {
      sf_floatread(trace1a,n1,in1a);
      sf_floatread(trace1b,n1,in1b);
      sf_floatread(trace2a,n1,in2a);
      sf_floatread(trace2b,n1,in2b);
      for (i1=0; i1<n1; i1++){
        sum += trace1a[i1]*trace2a[i1] + trace1b[i1]*trace2b[i1];
      }
    }
    fprintf(stderr,"%6.2f\n",sum);
    free1float(trace1a);
    free1float(trace1b);
    free1float(trace2a);
    free1float(trace2b);
    exit (0);
}

