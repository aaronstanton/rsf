/* Generate synthetic data containing hyperbolic events.*/
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

#include <rsf.h>

#include "mysegy.h"

#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif

void my_event(float *d,float dt,int nt,float sx,float sy,float gx,float gy,
              int nevent,float *amp,float *f0,float *t0,float *vx,float *vy);
void my_ricker(float *w, float f0,float dt);

int main(int argc, char* argv[])
{ 
    int ix,ih;
    int nkeys;
    int ievent, nevent, n1, n2, n3, n4, n5, i2, i3, i4, i5, itrace[SF_MAXKEYS];
    const char *label, *headname;
    float sx, sy, gx, gy, o1, o2, o3, o4, o5, d1, d2, d3, d4, d5; 
    float *trace, *amp=NULL, *f0=NULL, *t0=NULL, *vx=NULL, *vy=NULL;
    sf_file din, dout, hdr;

    sf_init (argc,argv);

    if (!sf_stdin()) { /* no input file in stdin */
	din = NULL;
    } else {
	din = sf_input("in");
    }

    dout = sf_output("out");
    
    if (NULL == din) {
	sf_setformat(dout,"native_float");
    } else if (SF_FLOAT != sf_gettype(din)) {
	sf_error("Need float input");
    }
    
    if (NULL != (label = sf_getstring("title")))
	sf_putstring(dout,"title",label);
    /* title for plots */

    /******/
    if (!sf_getint("nevent",&nevent)) nevent=1;
    if (!sf_getint("n1",&n1)) n1=100;
    if (!sf_getint("n2",&n2)) n2=4;
    if (!sf_getint("n3",&n3)) n3=4;
    if (!sf_getint("n4",&n4)) n4=10;
    if (!sf_getint("n5",&n5)) n5=10;
    if (!sf_getfloat("o1",&o1)) o1=0;
    if (!sf_getfloat("o2",&o2)) o2=100;
    if (!sf_getfloat("o3",&o3)) o3=100;
    if (!sf_getfloat("o4",&o4)) o4=50;
    if (!sf_getfloat("o5",&o5)) o5=50;
    if (!sf_getfloat("d1",&d1)) d1=0.004;
    if (!sf_getfloat("d2",&d2)) d2=10;
    if (!sf_getfloat("d3",&d3)) d3=10;
    if (!sf_getfloat("d4",&d4)) d4=20;
    if (!sf_getfloat("d5",&d5)) d5=5;

    sf_putfloat(dout,"o1",o1);
    sf_putfloat(dout,"o2",o2);
    sf_putfloat(dout,"o3",o3);
    sf_putfloat(dout,"o4",o4);
    sf_putfloat(dout,"o5",o5);
    sf_putfloat(dout,"d1",d1);
    sf_putfloat(dout,"d2",d2);
    sf_putfloat(dout,"d3",d3);
    sf_putfloat(dout,"d4",d4);
    sf_putfloat(dout,"d5",d5);
    sf_putfloat(dout,"n1",n1);
    sf_putfloat(dout,"n2",n2);
    sf_putfloat(dout,"n3",n3);
    sf_putfloat(dout,"n4",n4);
    sf_putfloat(dout,"n5",n5);
    sf_putstring(dout,"label1","Time");
    sf_putstring(dout,"label2","Source-x");
    sf_putstring(dout,"label3","Source-y");
    sf_putstring(dout,"label4","Receiver-x");
    sf_putstring(dout,"label5","Receiver-y");
    sf_putstring(dout,"unit1","s");
    sf_putstring(dout,"unit2","m");
    sf_putstring(dout,"unit3","m");
    sf_putstring(dout,"unit4","m");
    sf_putstring(dout,"unit5","m"); 

    /* write header */
    nkeys = SF_NKEYS; /* adding one new header */
    /* example for adding a non-standard header: nkeys = SF_NKEYS+1;*/
    /* initialize standard headers */
    segy_init(nkeys,NULL); 
    /* initialize a non-standard header */
    /*nonstandard_header_init("hello",2,nkeys-1); */
    hdr = sf_output("tfile");
    sf_putint(hdr,"n1",nkeys);
    sf_putint(hdr,"n2",n2*n3*n4*n5);
    sf_setformat(hdr,"native_int");
    segy2hist(hdr,nkeys);

    if (NULL == (headname = sf_getstring("tfile"))) headname = "tfile";
    /* output trace header file name in data rsf file*/
    if (NULL != dout) sf_putstring(dout,"head",headname);


    amp = sf_floatalloc(nevent);
    if (!sf_getfloats("amp",amp,nevent)) {
      /* amplitude of events */
      for (ievent=0; ievent < nevent; ievent++) {
        amp[ievent]=1;
      }
    }
    f0 = sf_floatalloc(nevent);
    if (!sf_getfloats("f0",f0,nevent)) {
      /* peak frequency of events in Hz */
      for (ievent=0; ievent < nevent; ievent++) {
        f0[ievent]=20;
      }
    }
    t0 = sf_floatalloc(nevent);
    if (!sf_getfloats("t0",t0,nevent)) {
      /* zero offset time of events in seconds */
      for (ievent=0; ievent < nevent; ievent++) {
        t0[ievent]=0.1;
      }
    }
    vx = sf_floatalloc(nevent);
    if (!sf_getfloats("vx",vx,nevent)) {
      /* velocity of events in the x direction */
      for (ievent=0; ievent < nevent; ievent++) {
        vx[ievent]=1500;
      }
    }
    vy = sf_floatalloc(nevent);
    if (!sf_getfloats("vy",vy,nevent)) {
      /* velocity of events in the y direction */
      for (ievent=0; ievent < nevent; ievent++) {
        vy[ievent]=1500;
      }
    }


    fprintf(stderr,"n1=%d,n2=%d,n3=%d,n4=%d,n5=%d \n",n1,n2,n3,n4,n5);

    for (ih=0;ih<nkeys;ih++) itrace[ih] = 0;  

    trace = sf_floatalloc (n1);
    ix = 0;
    for (i5=0;i5<n5;i5++){ 
      for (i4=0;i4<n4;i4++){ 
        for (i3=0;i3<n3;i3++){ 
          for (i2=0;i2<n2;i2++){ 
            sx = o2 + i2*d2;
            sy = o3 + i3*d3;
            gx = o4 + i4*d4;
            gy = o5 + i5*d5;
            my_event(trace,d1,n1,sx,sy,gx,gy,nevent,amp,f0,t0,vx,vy);
	    sf_floatwrite(trace,n1,dout);
            itrace[21] = sx;
            itrace[22] = sy;
            itrace[23] = gx;
            itrace[24] = gy;
            /* example for adding a non-standard header: itrace[nkeys-1] = 1; */
	    sf_intwrite(itrace, nkeys, hdr);
            ix++;
	  }
        }
      }
    }
    exit (0);
}

void my_event(float *d,float dt,int nt,float sx,float sy,float gx,float gy,
              int nevent,float *amp,float *f0,float *t0,float *vx,float *vy)
/* create one trace with nevents on it */
{
  sf_complex shift_op;
  sf_complex czero;
  int padfactor;
  int ntfft;
  int nw;
  int it,iw;
  int ifmin,ifmax;
  sf_complex *D;
  float omega;
  float* w;
  sf_complex* W;
  float delay;
  float hx, hy, tshift;
  float* in1;
  sf_complex* out1;
  fftwf_plan p1;
  sf_complex* in2;
  float* out2;
  fftwf_plan p2;
  int N;
  int ievent;
  
  hx = gx - sx;
  hy = gy - sy;

  __real__ czero=0;
  __imag__ czero=0;
  
  padfactor = 2;
  ntfft = padfactor*nt;
  nw = (int) ntfft/2 + 1; 
  D = sf_complexalloc(ntfft);

  w = sf_floatalloc(ntfft);
  
  for (iw=0;iw<nw;iw++){ 
    D[iw] =  czero;
  }

  ifmin = 0;
  ifmax = nw;

  N = ntfft;
  W = sf_complexalloc(nw);
  in1 = sf_floatalloc(N);
  out1 = sf_complexalloc(nw);
  p1 = fftwf_plan_dft_r2c_1d(N, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);

  for (ievent=0;ievent<nevent;ievent++){
    /* calculate ricker wavelet for each event */ 
    delay = dt*(trunc((2.2/f0[ievent]/dt))+1)/2;
    for (it=0;it<ntfft;it++) w[it] = 0;
    my_ricker(w,f0[ievent],dt);
    for(it=0;it<ntfft;it++) in1[it] = w[it];
    fftwf_execute(p1);
    for(iw=0;iw<nw;iw++) W[iw] = out1[iw]; 

    for (iw=ifmin;iw<ifmax;iw++){
      omega = (float) 2*PI*iw/ntfft/dt;
      /* normal moveout */
      tshift = sqrt(t0[ievent]*t0[ievent] + hx*hx/(vx[ievent]*vx[ievent]) + hy*hy/(vy[ievent]*vy[ievent]) ) - delay;
      __real__ shift_op = cos(omega*tshift);
      __imag__ shift_op =-sin(omega*tshift);
      D[iw] = D[iw] + (W[iw]*shift_op)*amp[ievent];
    } 
  }
  /**********************************************************************************************/
  N = ntfft;
  in2 = sf_complexalloc(N);
  out2 = sf_floatalloc(N);
  p2 = fftwf_plan_dft_c2r_1d(N, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for(iw=0;iw<nw;iw++){
    in2[iw] = D[iw];
  }
  fftwf_execute(p2);
  for(it=0;it<nt;it++){
    d[it] = out2[it]; 
  }
  /**********************************************************************************************/
  for (it=0; it<nt; it++) d[it]=d[it]/ntfft;

  fftwf_destroy_plan(p1);

  return;
}
void my_ricker(float *w, float f0,float dt)
{
  int iw, nw, nc;
  float alpha, beta;  
  nw = (int) 2*trunc((float) (2.2/f0/dt)/2) + 1;
  nc = (int) trunc((float) nw/2);
 
  for (iw=0;iw<nw-2;iw++){
    alpha = (nc-iw+1)*f0*dt*PI;
  	beta = alpha*alpha;
    w[iw] = (1-beta*2)*exp(-beta);
  }

  return;
}


