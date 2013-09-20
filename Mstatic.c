/* add random static shifts to seismic data.
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

#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>

#include <stdio.h>

#include <time.h>

#include <rsf.h>
#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif

void time_shift(float *d,float dt,int nt,float tshift);

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

    exit (0);
}
void time_shift(float *d,float dt,int nt,float tshift)
/* apply a time shift to one trace */
{
  float *in1, *out2, omega;
  sf_complex *out1, *in2, shift_op, *D;
  fftwf_plan p1, p2;
  int padfactor,ntfft,nw,it,iw,ifmin,ifmax;

  padfactor = 2;
  ntfft = padfactor*nt;
  nw = (int) ntfft/2 + 1; 
  D = sf_complexalloc(ntfft);
  ifmin = 0;
  ifmax = nw;
  in1 = sf_floatalloc(ntfft);
  out1 = sf_complexalloc(nw);
  p1 = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  for(it=0;it<nt;it++)     in1[it] = d[it];
  for(it=nt;it<ntfft;it++) in1[it] = 0;
  fftwf_execute(p1);
  for(iw=0;iw<nw;iw++) D[iw] = out1[iw]; 
  for (iw=ifmin;iw<ifmax;iw++){
    omega = (float) 2*PI*iw/ntfft/dt;
    __real__ shift_op = cos(omega*tshift);
    __imag__ shift_op =-sin(omega*tshift);
    D[iw] = D[iw]*shift_op;
  } 
  in2 = sf_complexalloc(ntfft);
  out2 = sf_floatalloc(ntfft);
  p2 = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for(iw=0;iw<nw;iw++){
    in2[iw] = D[iw];
  }
  fftwf_execute(p2);
  for(it=0;it<nt;it++){
    d[it] = out2[it]; 
  }
  for (it=0; it<nt; it++) d[it]=d[it]/ntfft;
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);

  return;
}
