/* Zero pad 5d data to be regular.
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

#include <rsf.h>

#include "mysegy.h"

int main(int argc, char* argv[])
{
    int mode; 
    int ik,nk_in,nk_out,ix,it;
    int n1,i1,n2,i2,i3,i4;
    int nx1,nx2,nx3,nx4;
    char *hdr_in_name;
    char *hdr_out_name;
    int *ix1_in, *ix2_in, *ix3_in, *ix4_in;
    int *ix1_out, *ix2_out, *ix3_out, *ix4_out;
    float *trace,**data_in,**data_out;
    float d1,o1; 
    int *hdr_in_array, *hdr_out_array, **hdr_in_all, **hdr_out_all;
    int min_ix1,min_ix2,min_ix3,min_ix4,max_ix1,max_ix2,max_ix3,max_ix4;
    /*int min_ix1_out,min_ix2_out,min_ix3_out,min_ix4_out,max_ix1_out,max_ix2_out,max_ix3_out,max_ix4_out;*/
    sf_file in,out,hdr_in,hdr_out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");

    if (!sf_getint("mode",&mode)) mode=1; /* mode of geometry computation */

    hdr_in_name = sf_getstring("headin");
    if (NULL == hdr_in_name) sf_error("Need headin=");
    hdr_in = sf_input(hdr_in_name);

    if (SF_INT != sf_gettype(hdr_in)) sf_error("Need int headin file");
    if (!sf_histint(hdr_in,"n1",&nk_in)) sf_error("No n1= in headin");
    n2 = sf_leftsize(hdr_in,1);

    /* write header */
    nk_out = SF_NKEYS + 16; /* adding new headers */
    /* initialize standard headers */
    segy_init(nk_out,NULL); 
    /* initialize a non-standard header */
    nonstandard_header_init("mx",2,SF_NKEYS+1);
    nonstandard_header_init("my",2,SF_NKEYS+2);
    nonstandard_header_init("hx",2,SF_NKEYS+3);
    nonstandard_header_init("hy",2,SF_NKEYS+4);
    nonstandard_header_init("h", 2,SF_NKEYS+5);
    nonstandard_header_init("az",2,SF_NKEYS+6);
    nonstandard_header_init("isx",2,SF_NKEYS+7);
    nonstandard_header_init("isy",2,SF_NKEYS+8);
    nonstandard_header_init("igx",2,SF_NKEYS+9);
    nonstandard_header_init("igy",2,SF_NKEYS+10);
    nonstandard_header_init("imx",2,SF_NKEYS+11);
    nonstandard_header_init("imy",2,SF_NKEYS+12);
    nonstandard_header_init("ihx",2,SF_NKEYS+13);
    nonstandard_header_init("ihy",2,SF_NKEYS+14);
    nonstandard_header_init("ih", 2,SF_NKEYS+15);
    nonstandard_header_init("iaz",2,SF_NKEYS+16);

    hdr_in_array = sf_intalloc(nk_in);
    hdr_out_array = sf_intalloc(nk_out);

    ix1_in = sf_intalloc(n2);
    ix2_in = sf_intalloc(n2);
    ix3_in = sf_intalloc(n2);
    ix4_in = sf_intalloc(n2);
    hdr_in_all = sf_intalloc2(n2,nk_in);

    /* read input file parameters */
    if (!sf_histint(in,"n1",&n1)) sf_error("No n1= in input");
    if (!sf_histfloat(in,"d1",&d1)) sf_error("No d1= in input");
    if (!sf_histfloat(in,"o1",&o1)) o1=0.;

    trace = sf_floatalloc (n1);
    data_in  = sf_floatalloc2(n2,n1);

    min_ix1=999999;min_ix2=999999;min_ix3=999999;min_ix4=999999;
    max_ix1=-999999;max_ix2=-999999;max_ix3=-999999;max_ix4=-999999;
    for (i2=0; i2<n2; i2++) {	
      sf_intread (hdr_in_array,nk_in,hdr_in);
      for (ik=0;ik<nk_in;ik++) hdr_in_all[ik][i2]=hdr_in_array[ik];
      if (mode==1){
	ix1_in[i2] = hdr_in_array[segykey("isx")];
	ix2_in[i2] = hdr_in_array[segykey("isy")];
	ix3_in[i2] = hdr_in_array[segykey("igx")];
	ix4_in[i2] = hdr_in_array[segykey("igy")];
      }
      else if (mode==2){
	ix1_in[i2] = hdr_in_array[segykey("imx")];
	ix2_in[i2] = hdr_in_array[segykey("imy")];
	ix3_in[i2] = hdr_in_array[segykey("ihx")];
	ix4_in[i2] = hdr_in_array[segykey("ihy")];
      }
      else if (mode==3){
	ix1_in[i2] = hdr_in_array[segykey("imx")];
	ix2_in[i2] = hdr_in_array[segykey("imy")];
	ix3_in[i2] = hdr_in_array[segykey("ih")];
        ix4_in[i2] = hdr_in_array[segykey("iaz")];
      }
      if (ix1_in[i2] < min_ix1) min_ix1 = ix1_in[i2];
      if (ix2_in[i2] < min_ix2) min_ix2 = ix2_in[i2];
      if (ix3_in[i2] < min_ix3) min_ix3 = ix3_in[i2];
      if (ix4_in[i2] < min_ix4) min_ix4 = ix4_in[i2];
      if (ix1_in[i2] > max_ix1) max_ix1 = ix1_in[i2];
      if (ix2_in[i2] > max_ix2) max_ix2 = ix2_in[i2];
      if (ix3_in[i2] > max_ix3) max_ix3 = ix3_in[i2];
      if (ix4_in[i2] > max_ix4) max_ix4 = ix4_in[i2];
      sf_floatread(trace,n1,in);
      for (i1=0;i1<n1;i1++) data_in[i1][i2] = trace[i1];
    }
    sf_fileclose (hdr_in);

    fprintf(stderr,"min_ix1=%d, min_ix2=%d, min_ix3=%d, min_ix4=%d\n",min_ix1,min_ix2,min_ix3,min_ix4);
    fprintf(stderr,"max_ix1=%d, max_ix2=%d, max_ix3=%d, max_ix4=%d\n",max_ix1,max_ix2,max_ix3,max_ix4);


    /* let user overwrite */
    sf_getint ("min_ix1",&min_ix1); /* ix1 minimum (calculated if not specified) */
    sf_getint ("max_ix1",&max_ix1); /* ix1 maximum (calculated if not specified) */
    sf_getint ("min_ix2",&min_ix2); /* ix2 minimum (calculated if not specified) */
    sf_getint ("max_ix2",&max_ix2); /* ix2 maximum (calculated if not specified) */
    sf_getint ("min_ix3",&min_ix3); /* ix3 minimum (calculated if not specified) */
    sf_getint ("max_ix3",&max_ix3); /* ix3 maximum (calculated if not specified) */
    sf_getint ("min_ix4",&min_ix4); /* ix4 minimum (calculated if not specified) */
    sf_getint ("max_ix4",&max_ix4); /* ix4 maximum (calculated if not specified) */

    nx1 = max_ix1 - min_ix1 + 1;    
    nx2 = max_ix2 - min_ix2 + 1;    
    nx3 = max_ix3 - min_ix3 + 1;    
    nx4 = max_ix4 - min_ix4 + 1;    

    for (i2=0; i2<n2; i2++){
      ix1_in[i2] = ix1_in[i2]-min_ix1;
      ix2_in[i2] = ix2_in[i2]-min_ix2;
      ix3_in[i2] = ix3_in[i2]-min_ix3;
      ix4_in[i2] = ix4_in[i2]-min_ix4;
    }

    fprintf(stderr,"n1=%d n2=%d nx1=%d,nx2=%d,nx3=%d,nx4=%d\n",n1,n2,nx1,nx2,nx3,nx4);
    
    data_out = sf_floatalloc2(nx1*nx2*nx3*nx4,n1);
    
    hdr_out_name = sf_getstring("headout");
    if (NULL == hdr_out_name) sf_error("Need headout=");
    hdr_out = sf_output(hdr_out_name);
    sf_putint(hdr_out,"n1",nk_out);
    sf_putint(hdr_out,"n2",nx1*nx2*nx3*nx4);
    sf_putint(hdr_out,"n3",1);
    sf_putint(hdr_out,"n4",1);
    sf_putint(hdr_out,"n5",1);
    sf_putfloat(hdr_out,"d1",1);
    sf_putfloat(hdr_out,"d2",1);
    sf_putfloat(hdr_out,"d3",1);
    sf_putfloat(hdr_out,"d4",1);
    sf_putfloat(hdr_out,"d5",1);
    sf_putfloat(hdr_out,"o1",1);
    sf_putfloat(hdr_out,"o2",0);
    sf_putfloat(hdr_out,"o3",0);
    sf_putfloat(hdr_out,"o4",0);
    sf_putfloat(hdr_out,"o5",0);
    sf_putstring(hdr_out,"label1","");
    sf_putstring(hdr_out,"label2","");
    sf_putstring(hdr_out,"label3","");
    sf_putstring(hdr_out,"label4","");
    sf_putstring(hdr_out,"label5","");
    sf_putstring(hdr_out,"unit1","");
    sf_putstring(hdr_out,"unit2","");
    sf_putstring(hdr_out,"unit3","");
    sf_putstring(hdr_out,"unit4","");
    sf_putstring(hdr_out,"unit5",""); 

    sf_setformat(hdr_out,"native_int");
    segy2hist(hdr_out,nk_out);

    fprintf(stderr,"nk_in=%d, nk_out=%d, n2=%d\n",nk_in,nk_out,n2);


    ix1_out = sf_intalloc(nx1*nx2*nx3*nx4);
    ix2_out = sf_intalloc(nx1*nx2*nx3*nx4);
    ix3_out = sf_intalloc(nx1*nx2*nx3*nx4);
    ix4_out = sf_intalloc(nx1*nx2*nx3*nx4);
    hdr_out_all = sf_intalloc2(nx1*nx2*nx3*nx4,nk_out);

    ix=0;
    for (i1=0;i1<nx1;i1++){
      for (i2=0;i2<nx2;i2++){
        for (i3=0;i3<nx3;i3++){
          for (i4=0;i4<nx4;i4++){
            ix1_out[ix] = i1;
            ix2_out[ix] = i2;
            ix3_out[ix] = i3;
            ix4_out[ix] = i4;
            for (ik=0; ik<nk_out; ik++) hdr_out_all[ik][ix] = 0;
            for (it=0; it<n1; it++) data_out[it][ix] = 0;
	    ix++;
          }
        }
      }
    }

    for (i2=0; i2<n2; i2++){
      /* ix1/ix2/ix3/ix4 sort */
      /*ix = ix1_in[i2]*nx2*nx3*nx4 + ix2_in[i2]*nx3*nx4 + ix3_in[i2]*nx4 + ix4_in[i2];*/
      /* ix4/ix3/ix2/ix1 sort */
      ix = ix4_in[i2]*nx3*nx2*nx1 + ix3_in[i2]*nx2*nx1 + ix2_in[i2]*nx1 + ix1_in[i2];
      /* fprintf(stderr,"ix1_in[%d]=%d, ix2_in[%d]=%d, ix3_in[%d]=%d, ix4_in[%d]=%d\n",i2,ix1_in[i2],i2,ix2_in[i2],i2,ix3_in[i2],i2,ix4_in[i2]); */
      for (ik=0; ik<nk_out; ik++) hdr_out_all[ik][ix] = hdr_in_all[ik][i2];
      for (i1=0; i1<n1; i1++) data_out[i1][ix] = data_in[i1][i2];
    }

    sf_putfloat(out,"o1",0);
    sf_putfloat(out,"o2",min_ix1);
    sf_putfloat(out,"o3",min_ix2);
    sf_putfloat(out,"o4",min_ix3);
    sf_putfloat(out,"o5",min_ix4);
    sf_putfloat(out,"d1",d1);
    sf_putfloat(out,"d2",1);
    sf_putfloat(out,"d3",1);
    sf_putfloat(out,"d4",1);
    sf_putfloat(out,"d5",1);
    sf_putfloat(out,"n1",n1);
    sf_putfloat(out,"n2",nx1);
    sf_putfloat(out,"n3",nx2);
    sf_putfloat(out,"n4",nx3);
    sf_putfloat(out,"n5",nx4);
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

/*
    min_ix1_out=999999;min_ix2_out=999999;min_ix3_out=999999;min_ix4_out=999999;
    max_ix1_out=-999999;max_ix2_out=-999999;max_ix3_out=-999999;max_ix4_out=-999999;
    for (i2=0; i2<nx1*nx2*nx3*nx4; i2++) {	
      if (ix1_out[i2] < min_ix1_out) min_ix1_out = ix1_out[i2];
      if (ix2_out[i2] < min_ix2_out) min_ix2_out = ix2_out[i2];
      if (ix3_out[i2] < min_ix3_out) min_ix3_out = ix3_out[i2];
      if (ix4_out[i2] < min_ix4_out) min_ix4_out = ix4_out[i2];
      if (ix1_out[i2] > max_ix1_out) max_ix1_out = ix1_out[i2];
      if (ix2_out[i2] > max_ix2_out) max_ix2_out = ix2_out[i2];
      if (ix3_out[i2] > max_ix3_out) max_ix3_out = ix3_out[i2];
      if (ix4_out[i2] > max_ix4_out) max_ix4_out = ix4_out[i2];
    }
    fprintf(stderr,"min_ix1_out=%d, min_ix2_out=%d, min_ix3_out=%d, min_ix4_out=%d\n",min_ix1_out,min_ix2_out,min_ix3_out,min_ix4_out);
    fprintf(stderr,"max_ix1_out=%d, max_ix2_out=%d, max_ix3_out=%d, max_ix4_out=%d\n",max_ix1_out,max_ix2_out,max_ix3_out,max_ix4_out);

    fprintf(stderr,"min_ix1=%d, min_ix2=%d, min_ix3=%d, min_ix4=%d\n",min_ix1,min_ix2,min_ix3,min_ix4);
    fprintf(stderr,"max_ix1=%d, max_ix2=%d, max_ix3=%d, max_ix4=%d\n",max_ix1,max_ix2,max_ix3,max_ix4);
*/

    for (i2=0; i2<nx1*nx2*nx3*nx4; i2++) {	
      /* pass on all input headers */
      for (ik=0;ik<nk_out;ik++) hdr_out_array[ik]=0;
      for (ik=0;ik<nk_out;ik++) hdr_out_array[ik]=hdr_out_all[ik][i2];
      if (mode==1){
        hdr_out_array[segykey("isx")] = (int) ix1_out[i2] + min_ix1;
        hdr_out_array[segykey("isy")] = (int) ix2_out[i2] + min_ix2;
        hdr_out_array[segykey("igx")] = (int) ix3_out[i2] + min_ix3;
        hdr_out_array[segykey("igy")] = (int) ix4_out[i2] + min_ix4;
      }
      else if (mode==2){
        hdr_out_array[segykey("imx")] = (int) ix1_out[i2] + min_ix1;
        hdr_out_array[segykey("imy")] = (int) ix2_out[i2] + min_ix2;
        hdr_out_array[segykey("ihx")] = (int) ix3_out[i2] + min_ix3;
        hdr_out_array[segykey("ihy")] = (int) ix4_out[i2] + min_ix4;
      }
      else if (mode==3){
        hdr_out_array[segykey("imx")] = (int) ix1_out[i2] + min_ix1;
        hdr_out_array[segykey("imy")] = (int) ix2_out[i2] + min_ix2;
        hdr_out_array[segykey("ih")]  = (int) ix3_out[i2] + min_ix3;
        hdr_out_array[segykey("iaz")] = (int) ix4_out[i2] + min_ix4;
      }
      sf_intwrite(hdr_out_array, nk_out, hdr_out);
      for (i1=0;i1<n1;i1++) trace[i1] = data_out[i1][i2];
      sf_floatwrite(trace,n1,out);
    }
    sf_fileclose (hdr_out);

    exit (0);
}

