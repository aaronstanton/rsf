/* static preserving sparse basis persuit denoising 
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

#include <rsf.h>
#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif

int main(int argc, char* argv[])
{
    int mode; 
    int ix,seed;
    int n1,n2,n3,n4,n5;
    float *trace;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float tshift,range; 
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");

    if (!sf_getint("seed",&seed)) seed = time(NULL);
    /* random seed */
    init_genrand((unsigned long) seed);


    if (!sf_getint("mode",&mode)) mode=1; /* mode of static shift */
    if (!sf_getfloat("range",&range)) range=0.005; /* 5ms static shift */


    /* read input file parameters */
    if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&o1)) o1=0.;
    if (!sf_histint(in,"n2",&n2)) sf_error("No n2= in input");
    if (!sf_histfloat(in,"d2",&d2)) d2=1;
    if (!sf_histfloat(in,"o2",&o2)) o2=0.;
    if (!sf_histint(in,"n3",&n3))   n3=1;
    if (!sf_histfloat(in,"d3",&d3)) d3=1;
    if (!sf_histfloat(in,"o3",&o3)) o3=0.;
    if (!sf_histint(in,"n4",&n4))   n4=1;
    if (!sf_histfloat(in,"d4",&d4)) d4=1;
    if (!sf_histfloat(in,"o4",&o4)) o4=0.;
    if (!sf_histint(in,"n5",&n5))   n5=1;
    if (!sf_histfloat(in,"d5",&d5)) d5=1;
    if (!sf_histfloat(in,"o5",&o5)) o5=0.;

    sf_putfloat(out,"o1",o1);
    sf_putfloat(out,"o2",o2);
    sf_putfloat(out,"o3",o3);
    sf_putfloat(out,"o4",o4);
    sf_putfloat(out,"o5",o5);
    sf_putfloat(out,"d1",d1);
    sf_putfloat(out,"d2",d2);
    sf_putfloat(out,"d3",d3);
    sf_putfloat(out,"d4",d4);
    sf_putfloat(out,"d5",d5);
    sf_putfloat(out,"n1",n1);
    sf_putfloat(out,"n2",n2);
    sf_putfloat(out,"n3",n3);
    sf_putfloat(out,"n4",n4);
    sf_putfloat(out,"n5",n5);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","ix1");
    sf_putstring(out,"label3","ix2");
    sf_putstring(out,"label4","ix3");
    sf_putstring(out,"label5","ix4");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","index");
    sf_putstring(out,"unit3","index");
    sf_putstring(out,"unit4","index");
    sf_putstring(out,"unit5","index"); 

    trace = sf_floatalloc (n1);

    for (ix=0; ix<n2*n3*n4*n5; ix++) {	
      sf_floatread(trace,n1,in);
      tshift = range*sf_randn_one_bm();
      time_shift(trace,d1,n1,tshift);
      sf_floatwrite(trace,n1,out);
    }

  for (iter=1;iter<Niter;iter++){
    linear_radon_op(fwd);
    myxcorr();
    get_gradient();
    soft_threshold();
    update_model();
    linear_radon_op(adj);
  }
  




    exit (0);
}

void get_gradient()
{
  return;
}

void soft_threshold()
{
  return;
}

void myxcorr()
{
  return;
}

void linear_radon_op()
{
  return;
}

void fk_op()
{
  return;
}


