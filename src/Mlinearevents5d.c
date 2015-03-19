/* Generate synthetic 5d data containing linear events.*/
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

#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif

void my_event(float *d,float dt,int nt,float x1,float x2,float x3,float x4,
              int nevent,float *amp,float *f0,float *t0,
              float *v1,float *v2,float *v3,float *v4);
void my_ricker(float *w, float f0,float dt);

int main(int argc, char* argv[])
{ 
    int ix,ih;
    int nkeys;
    int ievent, nevent, n1, n2, n3, n4, n5, i2, i3, i4, i5;
    float x1, x2, x3, x4, o1, o2, o3, o4, o5, d1, d2, d3, d4, d5; 
    float *trace, *amp=NULL, *f0=NULL, *t0=NULL, *v1=NULL, *v2=NULL, *v3=NULL, *v4=NULL;
    sf_file din, dout;

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
    
    /******/
    if (!sf_getint("nevent",&nevent)) nevent=1;
    if (!sf_getint("n1",&n1)) n1=100;
    if (!sf_getint("n2",&n2)) n2=10;
    if (!sf_getint("n3",&n3)) n3=10;
    if (!sf_getint("n4",&n4)) n4=10;
    if (!sf_getint("n5",&n5)) n5=10;
    if (!sf_getfloat("o1",&o1)) o1=0;
    if (!sf_getfloat("o2",&o2)) o2=0;
    if (!sf_getfloat("o3",&o3)) o3=0;
    if (!sf_getfloat("o4",&o4)) o4=0;
    if (!sf_getfloat("o5",&o5)) o5=0;
    if (!sf_getfloat("d1",&d1)) d1=0.004;
    if (!sf_getfloat("d2",&d2)) d2=1;
    if (!sf_getfloat("d3",&d3)) d3=1;
    if (!sf_getfloat("d4",&d4)) d4=1;
    if (!sf_getfloat("d5",&d5)) d5=1;

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
    sf_putstring(dout,"label2","ix1");
    sf_putstring(dout,"label3","ix2");
    sf_putstring(dout,"label4","ix3");
    sf_putstring(dout,"label5","ix4");
    sf_putstring(dout,"unit1","s");
    sf_putstring(dout,"unit2","index");
    sf_putstring(dout,"unit3","index");
    sf_putstring(dout,"unit4","index");
    sf_putstring(dout,"unit5","index"); 

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
    v1 = sf_floatalloc(nevent);
    if (!sf_getfloats("v1",v1,nevent)) {
      /* velocity of events in direction 1 */
      for (ievent=0; ievent < nevent; ievent++) {
        v1[ievent]=1500;
      }
    }
    v2 = sf_floatalloc(nevent);
    if (!sf_getfloats("v2",v2,nevent)) {
      /* velocity of events in direction 2 */
      for (ievent=0; ievent < nevent; ievent++) {
        v2[ievent]=1500;
      }
    }
    v3 = sf_floatalloc(nevent);
    if (!sf_getfloats("v3",v3,nevent)) {
      /* velocity of events in direction 3 */
      for (ievent=0; ievent < nevent; ievent++) {
        v3[ievent]=1500;
      }
    }
    v4 = sf_floatalloc(nevent);
    if (!sf_getfloats("v4",v4,nevent)) {
      /* velocity of events in direction 4 */
      for (ievent=0; ievent < nevent; ievent++) {
        v4[ievent]=1500;
      }
    }


    fprintf(stderr,"n1=%d,n2=%d,n3=%d,n4=%d,n5=%d \n",n1,n2,n3,n4,n5);

    trace = sf_floatalloc (n1);
    ix = 0;
    for (i5=0;i5<n5;i5++){ 
      for (i4=0;i4<n4;i4++){ 
        for (i3=0;i3<n3;i3++){ 
          for (i2=0;i2<n2;i2++){ 
            x1 = o2 + i2*10;
            x2 = o3 + i3*10;
            x3 = o4 + i4*10;
            x4 = o5 + i5*10;
            my_event(trace,d1,n1,x1,x2,x3,x4,nevent,amp,f0,t0,v1,v2,v3,v4);
	    sf_floatwrite(trace,n1,dout);
            ix++;
	  }
        }
      }
    }
    exit (0);
}

void my_event(float *d,float dt,int nt,float x1,float x2,float x3,float x4,
              int nevent,float *amp,float *f0,float *t0,
              float *v1,float *v2,float *v3,float *v4)
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
  float tshift;
  float* in1;
  sf_complex* out1;
  fftwf_plan p1;
  sf_complex* in2;
  float* out2;
  fftwf_plan p2;
  int N;
  int ievent;

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
      /* linear moveout in 4 spatial dimensions */
      tshift = t0[ievent] + x1/v1[ievent] + x2/v2[ievent] + x3/v3[ievent] + x4/v4[ievent] - delay;
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


