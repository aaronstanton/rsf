/* Radial filtering of axis 1 and chosen spatial axis. Default spatial axis is 2.
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
#include <fftw3.h>
#include "myfree.h"

#ifndef PI
#define PI (3.141592653589793)
#endif

void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           float fa,float fb,float fc,float fd,
                           int axis);
void radial_filter(float **d,float ot, float dt, int nt,float ox, float dx, int nx,float fa,float fb,float fc,float fd);
void radial_op(float **d,float **m,int nt,int nx,int np,float op,float dp,bool adj);
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);
void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile);
int main(int argc, char* argv[])
{
    int n1,n2,n3,n4,n5,i1,ix,axis;
    float **d;
    float *trace;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float fa,fb,fc,fd;
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");
    if (!sf_getfloat("fa",&fa)) fa = 0; /* minimum frequency (amplitude set to 0) */
    if (!sf_getfloat("fb",&fb)) fb = 3; /* minimum frequency taper (amplitude set to 1) */
    if (!sf_getfloat("fc",&fc)) fc = 50; /* maximum frequency taper (amplitude set to 1) */
    if (!sf_getfloat("fd",&fd)) fd = 60; /* maximum frequency (amplitude set to 0) */
    if (!sf_getint("axis",&axis)) axis = 2; /* spatial axis to perform radial filtering (2,3,4, or 5) */
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
    d = sf_floatalloc2(n1,n2*n3*n4*n5);
    trace = sf_floatalloc(n1);
    for (ix=0;ix<n2*n3*n4*n5;ix++) {
      sf_floatread(trace,n1,in);
      for (i1=0;i1<n1;i1++) d[ix][i1] = trace[i1];
    }
    radial_filter_gathers(d,
                          o1,d1,n1,
                          o2,d2,n2,
                          o3,d3,n3,
                          o4,d4,n4,
                          o5,d5,n5,
                          fa,fb,fc,fd,
                          axis);
    
    for (ix=0;ix<n2*n3*n4*n5;ix++) {
      for (i1=0;i1<n1;i1++) trace[i1] = d[ix][i1];
      sf_floatwrite(trace,n1,out);
    }
    free2float(d);
    exit (0);
}

void radial_filter_gathers(float **d,
                           float o1,float d1,int n1,
                           float o2,float d2,int n2,
                           float o3,float d3,int n3,
                           float o4,float d4,int n4,
                           float o5,float d5,int n5,
                           float fa,float fb,float fc,float fd,
                           int axis)
{
  float **d_gather;
  int i1,i2,i3,i4,i5;
  // process gathers
  if (axis==2){
    d_gather = sf_floatalloc2(n1,n2);
    for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d_gather[i2][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1];
      radial_filter(d_gather,o1,d1,n1,o2,d2,n2,fa,fb,fc,fd); 
      for (i2=0;i2<n2;i2++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i2][i1]; 
    }}}
  }
  else if (axis==3){
    d_gather = sf_floatalloc2(n1,n3);
    for (i2=0;i2<n2;i2++){ for (i4=0;i4<n4;i4++){ for (i5=0;i5<n5;i5++){
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d_gather[i3][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o3,d3,n3,fa,fb,fc,fd); 
      for (i3=0;i3<n3;i3++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i3][i1]; 
    }}}
  }
  else if (axis==4){
    d_gather = sf_floatalloc2(n1,n4);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i5=0;i5<n5;i5++){
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d_gather[i4][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o4,d4,n4,fa,fb,fc,fd); 
      for (i4=0;i4<n4;i4++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i4][i1]; 
    }}}
  }
  else if (axis==5){
    d_gather = sf_floatalloc2(n1,n5);
    for (i2=0;i2<n2;i2++){ for (i3=0;i3<n3;i3++){ for (i4=0;i4<n4;i4++){
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d_gather[i5][i1] = d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1]; 
      radial_filter(d_gather,o1,d1,n1,o5,d5,n5,fa,fb,fc,fd); 
      for (i5=0;i5<n5;i5++) for (i1=0;i1<n1;i1++) d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1] = d_gather[i5][i1]; 
    }}}
  }
                           
  free2float(d_gather);
  return;
}


void radial_filter(float **d,float ot, float dt, int nt,float ox, float dx, int nx,float fa,float fb,float fc,float fd)
{
  int np,ip,it;
  float op,dp;
  float *trace,**m;
//  sf_file outtmp;
//  char tmpname[256];
  
//  sprintf(tmpname, "tmp_radial_transform.rsf");
//  outtmp = sf_output(tmpname);

  np=1801;
  op=-90;
  dp=0.1;
  m = sf_floatalloc2(nt,np);
  trace = sf_floatalloc(nt);  
  radial_op(d,m,nt,nx,np,op,dp,true); 
  
//  write5d(m,
//          nt,ot,dt,"Time","s",  
//          np,op,dp,"Angle","Degrees",
//          1,0,1," "," ",
//          1,0,1," "," ",
//          1,0,1," "," ",
//          "Radial Domain",outtmp);
//  sf_fileclose(outtmp);
  
  for (ip=0;ip<np;ip++){  
    for (it=0;it<nt;it++) trace[it] = m[ip][it];    
    bpfilter(trace,dt,nt,fa,fb,fc,fd);
    for (it=0;it<nt;it++) m[ip][it] = trace[it];    
  }
  radial_op(d,m,nt,nx,np,op,dp,false); 

  free1float(trace);
  free2float(m);
  return;
}

void radial_op(float **d,float **m,int nt,int nx,int np,float op,float dp,bool adj)
{
  int ip,it,ix;
  float ox,dx,ot,dt,p,p_floor,x,t,alpha,beta;
  
  ox=-1;
  dx=2/(float) nx;
  ot=0;
  dt=1/(float) nt;
  
  if (adj){
    for (it=0;it<nt;it++) for (ip=0;ip<np;ip++) m[ip][it] = 0.0; 
  }
  else{
    for (it=0;it<nt;it++) for (ix=0;ix<nx;ix++) d[ix][it] = 0.0; 
  }
  
  for (it=0;it<nt;it++){ 
    for (ix=0;ix<nx;ix++){
      x = ix*dx + ox;
      t = it*dt + ot;
      p = (180/PI)*atanf(x/t);
      //fprintf(stderr,"p=%f\n",p);
      ip = (int) truncf((p - op)/dp);
      p_floor = truncf((p - op)/dp)*dp + op;
      alpha = (p-p_floor)/dp;
      beta = 1-alpha;
      if (ip >= 0 && ip+1 < np){
        if (adj){
          m[ip][it]   +=  beta*d[ix][it];
          m[ip+1][it] += alpha*d[ix][it];
        }
        else{
          d[ix][it] += (1/(beta*beta + alpha*alpha))*(beta*m[ip][it] + alpha*m[ip+1][it]);
	    }
	  }
    }
  }
  return;
}

void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
/*< bandpass filter >*/
{
  int iw,nw,ntfft,ia,ib,ic,id,it;
  float *in1, *out2;
  sf_complex *in2,*out1;
  sf_complex czero;
  fftwf_plan p1;
  fftwf_plan p2;

  __real__ czero = 0;
  __imag__ czero = 0;
  ntfft = 4*nt;
  nw=ntfft/2+1;
  if(a>0) ia = trunc(a*dt*ntfft);
  else ia = 0;
  if(b>0) ib = trunc(b*dt*ntfft);
  else ib = 1;
  if(c*dt*ntfft<nw) ic = trunc(c*dt*ntfft);
  else ic = nw-1;
  if(d*dt*ntfft<nw) id = trunc(d*dt*ntfft);
  else id = nw;

  out1 = sf_complexalloc(nw);
  in1  = sf_floatalloc(ntfft);
  p1   = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  out2 = sf_floatalloc(ntfft);
  in2  = sf_complexalloc(ntfft);
  p2   = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);

  for (it=0; it<nt; it++) in1[it]=trace[it];
  for (it=nt; it< ntfft;it++) in1[it] = 0.0;
  fftwf_execute(p1);
  for(iw=0;iw<ia;iw++)  in2[iw] = czero; 
  for(iw=ia;iw<ib;iw++) in2[iw] = out1[iw]*((float) (iw-ia)/(ib-ia))/sqrtf((float) ntfft); 
  for(iw=ib;iw<ic;iw++) in2[iw] = out1[iw]/sqrtf((float) ntfft); 
  for(iw=ic;iw<id;iw++) in2[iw] = out1[iw]*(1 - (float) (iw-ic)/(id-ic))/sqrtf((float) ntfft); 
  for(iw=id;iw<nw;iw++) in2[iw] = czero; 
  fftwf_execute(p2); /* take the FFT along the time dimension */
  for(it=0;it<nt;it++) trace[it] = out2[it]/sqrtf((float) ntfft); 
  
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_free(in1); fftwf_free(out1);
  fftwf_free(in2); fftwf_free(out2);
  return;
}

void write5d(float **data,
             int n1, float o1, float d1, const char *label1, const char *unit1,  
             int n2, float o2, float d2, const char *label2, const char *unit2,  
             int n3, float o3, float d3, const char *label3, const char *unit3,  
             int n4, float o4, float d4, const char *label4, const char *unit4,  
             int n5, float o5, float d5, const char *label5, const char *unit5,
             const char *title, sf_file outfile)
/*< write a 5d array of floats to disk >*/
{
  sf_putfloat(outfile,"o1",o1);
  sf_putfloat(outfile,"d1",d1);
  sf_putint(outfile,"n1",n1);
  sf_putstring(outfile,"label1",label1);
  sf_putstring(outfile,"unit1",unit1);
  sf_putfloat(outfile,"o2",o2);
  sf_putfloat(outfile,"d2",d2);
  sf_putint(outfile,"n2",n2);
  sf_putstring(outfile,"label2",label2); 
  sf_putstring(outfile,"unit2",unit2);
  sf_putfloat(outfile,"o3",o3);
  sf_putfloat(outfile,"d3",d3);
  sf_putint(outfile,"n3",n3);
  sf_putstring(outfile,"label3",label3);
  sf_putstring(outfile,"unit3",unit3);
  sf_putfloat(outfile,"o4",o4);
  sf_putfloat(outfile,"d4",d4);
  sf_putint(outfile,"n4",n4);
  sf_putstring(outfile,"label4",label4);
  sf_putstring(outfile,"unit4",unit4);
  sf_putfloat(outfile,"o5",o5);
  sf_putfloat(outfile,"d5",d5);
  sf_putint(outfile,"n5",n5);
  sf_putstring(outfile,"label5",label5);
  sf_putstring(outfile,"unit5",unit5);
  sf_putstring(outfile,"title",title);
  sf_floatwrite(data[0],n1*n2*n3*n4*n5,outfile);
  sf_fileclose(outfile);
  return;
}
