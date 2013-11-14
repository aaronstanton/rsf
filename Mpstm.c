/* PSTM operator.
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

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif 

#ifndef PI
#define PI (3.141592653589793)
#endif

#include <rsf.h>
#include <fftw3.h>
#ifndef PI
#define PI (3.141592653589793)
#endif
#include "myfree.h"

void kirchhoff_2d_time_op(float **d, float **m, float **v, 
                            int nt,   int nmx,   int nhx,   int nsx,   int ngx,
                          float ot, float omx, float ohx, float osx, float ogx,
                          float dt, float dmx, float dhx, float dsx, float dgx,
                          float angmax,
                          float gamma,
                          bool adj, bool ps);

void reduce_rank_2d(float **m,int nt,int nmx,int nhx,float dt, float flo, float fhi, int rank, int verbose);
void rr2d(sf_complex *freqslice,int nx1, int nx2, int rank);
void unfold(sf_complex *in, sf_complex **out,int *n,int a);
void fold(sf_complex **in, sf_complex *out,int *n,int a);
void csvd(sf_complex **A,sf_complex **U, float *S,sf_complex **VT,int M,int N);
void mult_svd(sf_complex **A, sf_complex **U, float *S, sf_complex **VT, int M, int N, int rank);
void cgesdd_(const char *jobz,const int *M,const int *N,sf_complex *Avec,const int *lda,float *Svec,sf_complex *Uvec,const int *ldu,sf_complex *VTvec,const int *ldvt,sf_complex *work,const int *lwork,float *rwork,int *iwork,int *info);	

void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d);

int main(int argc, char* argv[])
{

  sf_file in,out,vel;
  int n1,n2,n3;
  int nt,nmx,nhx,nsx,ngx;
  int it,ix;
  float o1,o2,o3;
  float d1,d2,d3;
  float ot,omx,ohx,osx,ogx;
  float dt,dmx,dhx,dsx,dgx;
  float **d,**m,**v,*trace;
  float angmax;
  bool adj;
  bool ps;
  float gamma;
  float rank;

  sf_init (argc,argv);
  in = sf_input("in");
  out = sf_output("out");
  vel = sf_input("vel");
  if (!sf_getbool("adj",&adj)) adj = true; /* flag for adjoint */
  if (!sf_getbool("ps",&ps)) ps = false; /* flag for PS data */
  if (!sf_getfloat("angmax",&angmax)) angmax=40;
  if (!sf_getfloat("gamma",&gamma)) gamma=2;
  if (!sf_getfloat("rank",&rank)) rank=5;
  
  /* read input file parameters */
  if (!sf_histint(  in,"n1",&n1)) sf_error("No n1= in input");
  if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o1",&o1)) o1=0.;
  if (!sf_histint(  in,"n2",&n2)) sf_error("No n2= in input");
  if (!sf_histfloat(in,"d2",&d2)) sf_error("No d1= in input");
  if (!sf_histfloat(in,"o2",&o2)) o2=0.;
  if (!sf_histint(  in,"n3",&n3)) sf_error("No n3= in input");
  if (!sf_histfloat(in,"d3",&d3)) sf_error("No d3= in input");
  if (!sf_histfloat(in,"o3",&o3)) o3=0.;

  if (adj){
    nt=n1; nsx=n2; ngx=n3;  
    dt=d1; dsx=d2; dgx=d3;  
    ot=o1; osx=o2; ogx=o3;  
    if (!sf_getint(  "nmx",&nmx)) sf_error("Parameter nmx required");
    if (!sf_getfloat("omx",&omx)) sf_error("Parameter omx required");
    if (!sf_getfloat("dmx",&dmx)) sf_error("Parameter dmx required");
    if (!sf_getint(  "nhx",&nhx)) sf_error("Parameter nhx required");
    if (!sf_getfloat("ohx",&ohx)) sf_error("Parameter ohx required");
    if (!sf_getfloat("dhx",&dhx)) sf_error("Parameter dhx required");
  }
  else{
    nt=n1; nmx=n2; nhx=n3;  
    dt=d1; dmx=d2; dhx=d3;  
    ot=o1; omx=o2; ohx=o3;  
    if (!sf_getint(  "nsx",&nsx)) sf_error("Parameter nsx required");
    if (!sf_getfloat("osx",&osx)) sf_error("Parameter osx required");
    if (!sf_getfloat("dsx",&dsx)) sf_error("Parameter dsx required");
    if (!sf_getint(  "ngx",&ngx)) sf_error("Parameter ngx required");
    if (!sf_getfloat("ogx",&ogx)) sf_error("Parameter ogx required");
    if (!sf_getfloat("dgx",&dgx)) sf_error("Parameter dgx required");
  }
  d  = sf_floatalloc2(nt,nsx*ngx);
  m = sf_floatalloc2(nt,nmx*nhx);
  v = sf_floatalloc2(nt,nmx);
  trace = sf_floatalloc(n1);
  for (ix=0;ix<nmx;ix++){
    sf_floatread(trace,n1,vel);
    for (it=0;it<nt;it++) v[ix][it] = trace[it];
  }
  for (ix=0;ix<n2*n3;ix++){
    sf_floatread(trace,n1,in);
    if (adj){ 
      for (it=0;it<nt;it++) d[ix][it] = trace[it];
    }
    else{ 
      for (it=0;it<nt;it++) m[ix][it] = trace[it];
    }
  }

  kirchhoff_2d_time_op(d, m, v, 
                       nt, nmx, nhx, nsx, ngx,
                       ot, omx, ohx, osx, ogx,
                       dt, dmx, dhx, dsx, dgx,
                       angmax,
                       gamma,
                       adj, ps);

  if (adj){
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"o2",omx);
    sf_putfloat(out,"o3",ohx);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"d2",dmx);
    sf_putfloat(out,"d3",dhx);
    sf_putfloat(out,"n1",nt);
    sf_putfloat(out,"n2",nmx);
    sf_putfloat(out,"n3",nhx);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","Distance");
    sf_putstring(out,"label3","Offset");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"unit3","m");
    sf_putstring(out,"title","Reflectivity");
    for (ix=0; ix<nmx*nhx; ix++) {
     for (it=0; it<nt; it++) trace[it] = m[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }
  else{
    sf_putfloat(out,"o1",ot);
    sf_putfloat(out,"o2",osx);
    sf_putfloat(out,"o3",ogx);
    sf_putfloat(out,"d1",dt);
    sf_putfloat(out,"d2",dsx);
    sf_putfloat(out,"d3",dgx);
    sf_putfloat(out,"n1",nt);
    sf_putfloat(out,"n2",nsx);
    sf_putfloat(out,"n3",ngx);
    sf_putstring(out,"label1","Time");
    sf_putstring(out,"label2","Source-X");
    sf_putstring(out,"label3","Receiver-x");
    sf_putstring(out,"unit1","s");
    sf_putstring(out,"unit2","m");
    sf_putstring(out,"unit3","m");
    sf_putstring(out,"title","Synthesized data");
    for (ix=0; ix<nsx*ngx; ix++) {
     for (it=0; it<nt; it++) trace[it] = d[ix][it];	
     sf_floatwrite(trace,nt,out);
    }
  }

    exit (0);
}

void kirchhoff_2d_time_op(float **d, float **m, float **v, 
                            int nt,   int nmx,   int nhx,   int nsx,   int ngx,
                          float ot, float omx, float ohx, float osx, float ogx,
                          float dt, float dmx, float dhx, float dsx, float dgx,
                          float angmax,
                          float gamma,
                          bool adj, bool ps)
{
  int ix,it,imx,ihx,isx,igx,jt;
  float dist,sx,gx,hx,v2,dists,distg,dists2,distg2;
  float mx,midx,t,t0,t02,ts,tg,t_floor;
  float res,res0,ang,angtaper;
  float geoms,obliq,gammainv;
  float tp0,ts0,tp02,ts02,vp2,vs2;
  float fnyq;
  float pmin, p;
  float *trace;
  int ic,nc;
  float fwidth;
  float fhi;
  float **mlo, **dlo;
  float wlo,whi,ref;
  int fplo,fphi;
  
  fwidth=20;
  fnyq= 1.0/(2*dt);
  nc=trunc(fnyq/fwidth)+1;
  fprintf(stderr,"nc= %d\n",nc);
  
  trace = sf_floatalloc(nt);
  fnyq = 1.0/(2*dt);
    
  gammainv = 1/gamma;
  if (adj){
    dlo = sf_floatalloc2(nt,nc);
    for (isx=0;isx<nsx;isx++){
      for (igx=0;igx<ngx;igx++){
	for (ic=0;ic<nc;ic++){
	  for (it=0;it<nt;it++) trace[it] = d[igx*nsx+isx][it];
	  fhi = fnyq*ic/nc;
	  bpfilter(trace,dt,nt,0.01,0.1,fhi,fhi + 8);
	  for (it=0;it<nt;it++) dlo[ic][it] = trace[it];
	}  
        sx = osx + isx*dsx;
        gx = ogx + igx*dgx;
        hx = gx - sx;
        ihx = (int) trunc(fabsf((hx - ohx)/dhx));
        if (!ps) midx = (sx + gx)/2;
        else midx = sx + hx/(1 + gammainv);
        for (imx=0;imx<nmx;imx++){
          mx = omx + imx*dmx;
          dist = mx - midx;
          dists = fabsf(sx-mx); 
          distg = fabsf(gx-mx);
          dists2 = dists*dists;
          distg2 = distg*distg;
          for (it=0;it<nt;it++){
            t0 = 0.5*(ot + it*dt);
            if (!ps){
              t02 = t0*t0;
              v2=v[imx][it]*v[imx][it];
              ts = sqrtf(t02 + dists2/v2);
              tg = sqrtf(t02 + distg2/v2);
            }
            else{
              tp0 = (2*t0)/(1+gamma);
              ts0 = (2*t0)*gamma/(1+gamma);
              tp02 = tp0*tp0; 
              ts02 = ts0*ts0;
              vp2=v[imx][it]*v[imx][it];
              vs2=v[imx][it]*v[imx][it]/4; /* Fix this part of the code */
              ts = sqrtf(tp02 + dists2/vp2);
              tg = sqrtf(ts02 + distg2/vs2);
            }
            t  = ts + tg;

            geoms=sqrtf(1/(t*v[imx][it] + 0.001));
            obliq=sqrtf(.5*(1 + ((2*t0)*(2*t0)/(4*ts*tg + 0.001)) 
                  - (1/(ts*tg + 0.001))*sqrt(ts*ts - (2*t0)*(2*t0)/4)*sqrt(tg*tg - (2*t0)*(2*t0)/4)));
            ang=180.0*fabsf(acos((2*t0)/t+0.001))/PI;  
	    if(ang<=angmax) angtaper=1.0;
	    if(ang>angmax) angtaper=cos((ang-angmax)*PI/20);
            pmin= 1/(2*dmx*fnyq);
	    p= dists/(v2*ts+0.001) + distg/(v2*tg+0.001);
	    p=sqrtf(p*p);
	    if(p==0) fplo=nc;
	    else fplo=trunc(nc*pmin/p);
	    ref=fmodf(nc*pmin,p);
	    wlo=1-ref;
	    fphi=++fplo;
	    whi=ref;
	    /*fprintf(stderr,"pmin=%f p=%f\n",pmin,p);*/  
            t_floor = trunc(t/dt);
            jt = (int) t_floor;
            if (jt-2 > 0 && jt+3 < nt && ihx < nhx){
	      res = (t-t_floor)/dt;
	      res0 = 1.0-res;
              if (fplo>=nc-1){ 
                m[ihx*nmx + imx][it] += geoms*obliq*angtaper*(res0*dlo[nc-1][jt] + res*dlo[nc-1][jt+1]);
              }
              else {
                m[ihx*nmx + imx][it] += geoms*obliq*angtaper*(res0*(wlo*dlo[fplo][jt]+whi*dlo[fphi][jt]) + res*(wlo*dlo[fplo][jt+1]+whi*dlo[fphi][jt+1]));
              }
            }
          }
        } 
      }
    }
  }
  else {
    mlo = sf_floatalloc2(nt,nc);
    for (imx=0;imx<nmx;imx++){
      fprintf(stderr,"imx=%d nmx=%d\n",imx,nmx);
      mx = omx + imx*dmx;
      for (isx=0;isx<nsx;isx++){
	for (igx=0;igx<ngx;igx++){
	  sx = osx + isx*dsx;
          gx = ogx + igx*dgx;
          hx = gx - sx;
          ihx = (int) trunc(fabsf((hx - ohx)/dhx));
	  if (ihx<nhx){
            for (ic=0;ic<nc;ic++){
	      for (it=0;it<nt;it++) trace[it] = m[ihx*nmx+imx][it];
	      fhi = fnyq*ic/nc;
	      bpfilter(trace,dt,nt,0.01,0.1,fhi,fhi + 8);
	      for (it=0;it<nt;it++) mlo[ic][it] = trace[it];
            }
	  }
	  else{
            for (ic=0;ic<nc;ic++){
	      for (it=0;it<nt;it++) mlo[ic][it] = 0;
	    }
	  }  
          if (!ps) midx = (sx + gx)/2;
          else midx = sx + hx/(1 + gammainv);
          dist = mx - midx;
          dists = fabsf(sx-mx); 
          distg = fabsf(gx-mx);
          dists2 = dists*dists;
          distg2 = distg*distg;
          for (it=0;it<nt;it++){
            t0 = 0.5*(ot + it*dt);
            if (!ps){
              t02 = t0*t0;
              v2=v[imx][it]*v[imx][it];
              ts = sqrtf(t02 + dists2/v2);
              tg = sqrtf(t02 + distg2/v2);
            }
            else{
              tp0 = (2*t0)/(1+gamma);
              ts0 = (2*t0)*gamma/(1+gamma);
              tp02 = tp0*tp0; 
              ts02 = ts0*ts0;
              vp2=v[imx][it]*v[imx][it];
              vs2=v[imx][it]*v[imx][it]/4; /* Fix this part of the code */
              ts = sqrtf(tp02 + dists2/vp2);
              tg = sqrtf(ts02 + distg2/vs2);
            }
            t  = ts + tg;

            geoms=sqrtf(1/(t*v[imx][it] + 0.001));
            obliq=sqrtf(.5*(1 + ((2*t0)*(2*t0)/(4*ts*tg + 0.001)) 
                  - (1/(ts*tg + 0.001))*sqrt(ts*ts - (2*t0)*(2*t0)/4)*sqrt(tg*tg - (2*t0)*(2*t0)/4)));
            ang=180.0*fabsf(acos((2*t0)/t+0.001))/PI;  
	    if(ang<=angmax) angtaper=1.0;
	    if(ang>angmax) angtaper=cos((ang-angmax)*PI/20);

            pmin= 1/(2*dmx*fnyq);
	    p= dists/(v2*ts+0.001) + distg/(v2*tg+0.001);
	    p=sqrtf(p*p);
	    if(p==0) fplo=nc;
	    else fplo=trunc(nc*pmin/p);
	    ref=fmodf(nc*pmin,p);
	    wlo=1-ref;
	    fphi=++fplo;
	    whi=ref;
	    /*fprintf(stderr,"pmin=%f p=%f\n",pmin,p);*/  
            t_floor = trunc(t/dt);
            jt = (int) t_floor;
            if (jt-2 > 0 && jt+3 < nt && ihx < nhx){
	      res = (t-t_floor)/dt;
	      res0 = 1.0-res;
              if (fplo>=nc-1){ 
                d[igx*nsx+isx][jt]   += geoms*obliq*angtaper*res0*(wlo*mlo[nc-1][it]+whi*mlo[nc-1][it]);
                d[igx*nsx+isx][jt+1] += geoms*obliq*angtaper*res*(wlo*mlo[nc-1][it]+whi*mlo[nc-1][it]);
              }
              else { 
                d[igx*nsx+isx][jt]   += geoms*obliq*angtaper*res0*(wlo*mlo[fplo][it]+whi*mlo[fphi][it]);
                d[igx*nsx+isx][jt+1] += geoms*obliq*angtaper*res*(wlo*mlo[fplo][it]+whi*mlo[fphi][it]);
              }
            }
          }
        } 
      }
    }
  }
    
  return;
}

void reduce_rank_2d(float **m,int nt,int nmx,int nhx,float dt, float flo, float fhi, int rank, int verbose)
{
  int nx = nmx*nhx;
  int it, ix, iw;
  sf_complex czero;
  int ntfft,nw;
  float **pfft; 
  sf_complex **cpfft;
  sf_complex *out;
  fftwf_plan p1;
  int iflo;
  int ifhi;
  float *out2;
  fftwf_plan p2;
  sf_complex *in2;
  sf_complex *freqslice;
  float* in;

  __real__ czero = 0;
  __imag__ czero = 0;
  ntfft = 4*nt;
  nw=ntfft/2+1;

  pfft  = sf_floatalloc2(ntfft,nx);
  cpfft = sf_complexalloc2(nw,nx);

  for (ix=0;ix<nx;ix++){
    for (it=0; it<nt; it++) pfft[ix][it]=m[ix][it];
    for (it=nt; it< ntfft;it++) pfft[ix][it] = 0.0;
  }
  out = sf_complexalloc(nw);
  in = sf_floatalloc(ntfft);
  p1 = fftwf_plan_dft_r2c_1d(ntfft, in, (fftwf_complex*)out, FFTW_ESTIMATE);

  for (ix=0;ix<nx;ix++){
    for(it=0;it<ntfft;it++){
      in[it] = pfft[ix][it];
    }
    fftwf_execute(p1); 
    for(iw=0;iw<nw;iw++){
      cpfft[ix][iw] = out[iw]; 
    }
  }
  fftwf_destroy_plan(p1);
  fftwf_free(in); fftwf_free(out);

  freqslice= sf_complexalloc(nx);

  if(flo>0){ 
    iflo = trunc(flo*dt*ntfft);
  }
  else{
    iflo = 0;
  }
  if(fhi*dt*ntfft<nw){ 
    ifhi = trunc(fhi*dt*ntfft);
  }
  else{
    ifhi = 0;
  }

  /* process frequency slices */
  for (iw=iflo;iw<ifhi;iw++){
    if (verbose) fprintf(stderr,"\r                                         ");
    if (verbose) fprintf(stderr,"\rfrequency slice %d of %d",iw-iflo+1,ifhi-iflo);
	  
    for (ix=0;ix<nx;ix++){
      freqslice[ix] = cpfft[ix][iw];
    }

    rr2d(freqslice,nmx,nhx,rank);
    
    for (ix=0;ix<nx;ix++){
      cpfft[ix][iw] = freqslice[ix];
    }

  }

  /* zero all other frequencies */
  for (ix=0;ix<nx;ix++){
    for (iw=ifhi;iw<nw;iw++){
      cpfft[ix][iw] = czero;
    }
  }

  out2 = sf_floatalloc(ntfft);
  in2 = sf_complexalloc(ntfft);
  p2 = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);
  for (ix=0;ix<nx;ix++){
    for(iw=0;iw<nw;iw++){
      in2[iw] = cpfft[ix][iw];
    }
    fftwf_execute(p2);
    for(it=0;it<nt;it++){
      pfft[ix][it] = out2[it]; 
    }
  }
  if (verbose) fprintf(stderr,"\n");

  fftwf_destroy_plan(p2);
  fftwf_free(in2); fftwf_free(out2);
  for (ix=0;ix<nx;ix++) for (it=0; it<nt; it++) m[ix][it]=pfft[ix][it]/ntfft;

  return;
}

void rr2d(sf_complex *freqslice,int nx1, int nx2, int rank)
{  

  int M, N;
  int *n;
  sf_complex **uf;  
  sf_complex **U;
  sf_complex **VT;
  float *S;

  n = sf_intalloc(4);
  n[0] = nx1;
  n[1] = nx2;
  n[2] = 1;
  n[3] = 1;
  uf = sf_complexalloc2(nx2,nx1);  
  M = nx1;
  N = nx2;
  U = sf_complexalloc2(M,M);
  S = sf_floatalloc(M);
  VT = sf_complexalloc2(N,M);

  if (nx1 > rank){    
    unfold(freqslice,uf,n,1);
    M = nx1;
    N = nx2;
    csvd(uf,U,S,VT,M,N);
    mult_svd(uf,U,S,VT,M,N,rank);
    fold(uf,freqslice,n,1);
  }

  free2complex(uf);
  free2complex(U);
  free1float(S);
  free2complex(VT);

  return;
}

void unfold(sf_complex *in, sf_complex **out,int *n,int a)
{   
  /* 
  unfold a long-vector representing a 4D tensor into a matrix.	
  in  = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4])
  out = matrix that is the unfolding of "in" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;

  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];
  
  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1][ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


 
  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix2][ix1*nx3*nx4+ix3*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    


  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
            out[ix3][ix1*nx2*nx4+ix2*nx4+ix4]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix4][ix1*nx2*nx3+ix2*nx3+ix3]=in[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  return;
}

void fold(sf_complex **in, sf_complex *out,int *n,int a)
{   
  /* 
  fold a matrix back to a long-vector representing a 4D tensor.	
  in  = the unfolding of "out" with dimensions (if a=1): (n[1],n[2]*n[3]*n[4])
  out = long vector representing a tensor with dimensions (n[1],n[2],n[3],n[4]) 
  */
  int nx1,nx2,nx3,nx4;
  int ix1,ix2,ix3,ix4;
  
  nx1=n[0];
  nx2=n[1];
  nx3=n[2];
  nx4=n[3];

  if (a==1){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix1][ix2*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==2){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix2][ix1*nx3*nx4+ix3*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==3){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix3][ix1*nx2*nx4+ix2*nx4+ix4];
	  }
	}
      }
    }
  }    

  if (a==4){
    for (ix1=0;ix1<nx1;ix1++){
      for (ix2=0;ix2<nx2;ix2++){
	for (ix3=0;ix3<nx3;ix3++){
	  for (ix4=0;ix4<nx4;ix4++){
	    out[ix1*nx2*nx3*nx4+ix2*nx3*nx4+ix3*nx4+ix4]=in[ix4][ix1*nx2*nx3+ix2*nx3+ix3];
	  }
	}
      }
    }
  }    

  return;
}

void csvd(sf_complex **A,sf_complex **U,float *S,sf_complex **VT,int M,int N)
{

  sf_complex czero;
  sf_complex* Avec;
  int i,j,n;
  float* Svec;  
  char jobz;
  int lda;
  int ldu;
  int ldvt;
  int lwork; 
  int info;
  sf_complex *work;
  int lrwork;
  float *rwork;  
  float *iwork;  
  sf_complex* Uvec; 
  sf_complex* VTvec; 
  
  __real__ czero = 0;
  __imag__ czero = 0;

  Avec = sf_complexalloc(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      Avec[n] = A[j][i];
      n++;
    }
  }
  Uvec = sf_complexalloc(M*M); 
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      Uvec[n] = czero;
      n++;
    }
  }
  VTvec = sf_complexalloc(M*N); 
  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VTvec[n] = czero;
      n++;
    }
  }
  Svec = sf_floatalloc(M);  
  
  jobz = 'S';
  lda   = M;
  ldu   = M;
  ldvt  = M;
  lwork = 4*(5*M + N); 
  work = sf_complexalloc(lwork); 

  /* make float array rwork with dimension lrwork (where lrwork >= min(M,N)*max(5*min(M,N)+7,2*max(M,N)+2*min(M,N)+1)) */
  lrwork = 4*(5*M*M+ 7*M);
  rwork = sf_floatalloc(lrwork);  

  /* make float array iwork with dimension 8*M (where A is MxN) */
  iwork = sf_floatalloc(4*8*M);  
  cgesdd_(&jobz, &M, &N, (sf_complex*)Avec, &lda, Svec, (sf_complex*)Uvec, &ldu, (sf_complex*)VTvec, &ldvt, (sf_complex*)work, &lwork, (float*)rwork, (int*)iwork, &info);  
  if (info != 0)  fprintf(stderr,"Error in cgesdd: info = %d\n",info);

  n = 0;
  for (i=0;i<N;i++){
    for (j=0;j<M;j++){
      VT[j][i] = VTvec[n];
      n++;
    }
  }
  n = 0;
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      U[j][i] = Uvec[n];
      n++;
    }
  }

  for (i=0;i<M;i++){
    S[i] = Svec[i];
  }
  

  free1complex(Avec);
  free1complex(VTvec);
  free1complex(Uvec);
  free1float(Svec);
  free1complex(work);
  free1float(rwork);
  free1float(iwork);

  return;
}

void mult_svd(sf_complex **A, sf_complex **U,float *S,sf_complex **VT,int M,int N,int rank)
{   
  int i,j,k;
  sf_complex czero;
  sf_complex **SVT;
  sf_complex sum;

  SVT = sf_complexalloc2(N,M);
  __real__ czero = 0;
  __imag__ czero = 0;
  sum = czero;
  
  for (i=0;i<rank;i++){
    for (j=0;j<N;j++){
      SVT[i][j] = czero;
      if (i < rank) SVT[i][j] = VT[i][j]*S[i];
    }
  }
  for (i=0;i<M;i++){
    for (j=0;j<N;j++){
      sum = czero;
      for (k=0;k<rank;k++){
      sum = sum + SVT[k][j]*U[i][k];
      }
      A[i][j] = sum;
    }
  }
 
  free2complex(SVT);

  return;
}
void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
{
  int iw,nw,ntfft;
  float *in1, *out2;
  int ia,ib,ic,id;
  int it;
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
  for(iw=ia;iw<ib;iw++) in2[iw] = out1[iw]*((float) (iw-ia)/(ib-ia)); 
  for(iw=ib;iw<ic;iw++) in2[iw] = out1[iw]; 
  for(iw=ic;iw<id;iw++) in2[iw] = out1[iw]*(1 - (float) (iw-ic)/(id-ic)); 
  for(iw=id;iw<nw;iw++) in2[iw] = czero; 
  fftwf_execute(p2); /* take the FFT along the time dimension */
  for(it=0;it<nt;it++) trace[it] = out2[it]/ntfft; 
  
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_free(in1); fftwf_free(out1);
  fftwf_free(in2); fftwf_free(out2);
  return;
}


