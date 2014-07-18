/* Merge overlapping datasets by stacking
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <unistd.h>

#include <rsf.h>
#include "myfree.h"

#ifndef PI
#define PI (3.141592653589793)
#endif

void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lti,int ltf,int lx1i,int lx1f,int lx2i,int lx2f,int lx3i,int lx3f,int lx4i,int lx4f);

int main(int argc, char* argv[])
{
  int i,i1,i2,i3,i4,i5,nin,iseek;
  int   N1,N2,N3,N4,N5;
  float O1,O2,O3,O4,O5;
  float D1,D2,D3,D4,D5;
  int   T1,T2,T3,T4,T5;
  int   n1,n2,n3,n4,n5;
  float o1,o2,o3,o4,o5;
  float d1,d2,d3,d4,d5;
  int   t1i,t2i,t3i,t4i,t5i;
  int   t1f,t2f,t3f,t4f,t5f;
  float *a,**d;
  const char **filename;
  char *label1,*label2,*label3,*label4,*label5;
  char *unit1,*unit2,*unit3,*unit4,*unit5;
  bool verbose;
  sf_file infile,outfile;
  sf_init(argc,argv);

  filename = (const char**) sf_alloc ((size_t) argc,sizeof(char*));
  if (!sf_stdin()) { /* no input file in stdin */
    nin=0;
  } else {
    filename[0] = "in";
    nin=1;
  }
  for (i=1; i< argc; i++) { /* collect inputs */
    if (NULL != strchr(argv[i],'=')) continue; /* not a file */
    filename[nin] = argv[i];
    nin++;
  }
  if (0==nin) sf_error ("no input");

  a = sf_floatalloc(1);  
  a[0] = 0.0;

  if (!sf_getbool("verbose",&verbose)) verbose=false;
  sf_getint("n1",&N1);
  sf_getint("n2",&N2);
  sf_getint("n3",&N3);
  sf_getint("n4",&N4);
  sf_getint("n5",&N5);
  sf_getfloat("o1",&O1);
  sf_getfloat("o2",&O2);
  sf_getfloat("o3",&O3);
  sf_getfloat("o4",&O4);
  sf_getfloat("o5",&O5);
  sf_getfloat("d1",&D1);
  sf_getfloat("d2",&D2);
  sf_getfloat("d3",&D3);
  sf_getfloat("d4",&D4);
  sf_getfloat("d5",&D5);
  sf_getint("t1",&T1);
  sf_getint("t2",&T2);
  sf_getint("t3",&T3);
  sf_getint("t4",&T4);
  sf_getint("t5",&T5);
  label1 = sf_getstring("label1");
  label2 = sf_getstring("label2");
  label3 = sf_getstring("label3");
  label4 = sf_getstring("label4");
  label5 = sf_getstring("label5");
  unit1 = sf_getstring("unit1");
  unit2 = sf_getstring("unit2");
  unit3 = sf_getstring("unit3");
  unit4 = sf_getstring("unit4");
  unit5 = sf_getstring("unit5");

  /* calling sf_input before sf_output to meet criteria of sf_output */
  infile = sf_input(filename[0]);
  outfile = sf_output("outfile");

  sf_putfloat(outfile,"o1",O1);
  sf_putfloat(outfile,"o2",O2);
  sf_putfloat(outfile,"o3",O3);
  sf_putfloat(outfile,"o4",O4);
  sf_putfloat(outfile,"o5",O5);
  sf_putfloat(outfile,"d1",D1);
  sf_putfloat(outfile,"d2",D2);
  sf_putfloat(outfile,"d3",D3);
  sf_putfloat(outfile,"d4",D4);
  sf_putfloat(outfile,"d5",D5);
  sf_putfloat(outfile,"n1",N1);
  sf_putfloat(outfile,"n2",N2);
  sf_putfloat(outfile,"n3",N3);
  sf_putfloat(outfile,"n4",N4);
  sf_putfloat(outfile,"n5",N5);
  sf_putstring(outfile,"label1",label1);
  sf_putstring(outfile,"label2",label2);
  sf_putstring(outfile,"label3",label3);
  sf_putstring(outfile,"label4",label4);
  sf_putstring(outfile,"label5",label5);
  sf_putstring(outfile,"unit1",unit1);
  sf_putstring(outfile,"unit2",unit2);
  sf_putstring(outfile,"unit3",unit3);
  sf_putstring(outfile,"unit4",unit4);
  sf_putstring(outfile,"unit5",unit5); 

  /* initialize output file with zeros */  
  for (i=0;i<N1*N2*N3*N4*N5;i++){
    sf_floatwrite(a,1,outfile); 
  }

  /* read and add each input patch to the final output */
  for (i=0;i<nin;i++){
    fprintf(stderr,"filename[%d]=%s\n",i,filename[i]);
    infile = sf_input(filename[i]);
    sf_histint(infile,"n1",&n1);
    sf_histint(infile,"n2",&n2);
    sf_histint(infile,"n3",&n3);
    sf_histint(infile,"n4",&n4);
    sf_histint(infile,"n5",&n5);
    sf_histfloat(infile,"o1",&o1);
    sf_histfloat(infile,"o2",&o2);
    sf_histfloat(infile,"o3",&o3);
    sf_histfloat(infile,"o4",&o4);
    sf_histfloat(infile,"o5",&o5);
    sf_histfloat(infile,"d1",&d1);
    sf_histfloat(infile,"d2",&d2);
    sf_histfloat(infile,"d3",&d3);
    sf_histfloat(infile,"d4",&d4);
    sf_histfloat(infile,"d5",&d5);
    d = sf_floatalloc2(n1,n2*n3*n4*n5);
    sf_floatread(d[0],n1*n2*n3*n4*n5,infile);
    if (o1 > O1) t1i = T1;
    else         t1i = 0;
    if (o1+d1*n1 < O1+D1*N1) t1f = T1;
    else                     t1f = 0;
    if (o2 > O2) t2i = T2;
    else         t2i = 0;
    if (o2+d2*n2 < O2+D2*N2) t2f = T2;
    else                     t2f = 0;
    if (o3 > O3) t3i = T3;
    else         t3i = 0;
    if (o3+d3*n3 < O3+D3*N3) t3f = T3;
    else                     t3f = 0;
    if (o4 > O4) t4i = T4;
    else         t4i = 0;
    if (o4+d4*n4 < O4+D4*N4) t4f = T4;
    else                     t4f = 0;
    if (o5 > O5) t5i = T5;
    else         t5i = 0;
    if (o5+d5*n5 < O5+D5*N5) t5f = T5;
    else                     t5f = 0;
    my_taper(d,n1,n2,n3,n4,n5,t1i,t1f,t2i,t2f,t3i,t3f,t4i,t4f,t5i,t5f);
    o1 = o1/d1;
    o2 = o2/d2;
    o3 = o3/d3;
    o4 = o4/d4;
    o5 = o5/d5;
    for (i5=0;i5<n5;i5++){
      for (i4=0;i4<n4;i4++){
        for (i3=0;i3<n3;i3++){
          for (i2=0;i2<n2;i2++){
            for (i1=0;i1<n1;i1++){
              iseek = (off_t)((float) (o5+i5)*N4*N3*N2*N1 + (o4+i4)*N3*N2*N1 + (o3+i3)*N2*N1 + (o2+i2)*N1 + (o1+i1))*sizeof(float);
              if (verbose) fprintf(stderr,"N1=%d o1=%f o2=%f i2=%d\n",N1,o1,o2,i2);
              if (verbose) fprintf(stderr,"%f\n",(float) ((o5+i5)*N4*N3*N2*N1 + (o4+i4)*N3*N2*N1 + (o3+i3)*N2*N1 + (o2+i2)*N1 + (o1+i1)));
              sf_seek(outfile,iseek,SEEK_SET);             
              sf_floatread(a,1,outfile);            
              sf_seek(outfile,iseek,SEEK_SET);
              a[0] += d[i5*n4*n3*n2 + i4*n3*n2 + i3*n2 + i2][i1];
              sf_floatwrite(a,1,outfile);
            }
          }
        }
      }
      free2float(d);
    }
    sf_fileclose(infile);
  }  

  exit(0);
}

void my_taper(float **d,int nt,int nx1,int nx2,int nx3,int nx4,int lti,int ltf,int lx1i,int lx1f,int lx2i,int lx2f,int lx3i,int lx3f,int lx4i,int lx4f)
{
  int it,ix,ix1,ix2,ix3,ix4;
  float tt,tx1,tx2,tx3,tx4;

  tx1=1;tx2=1;tx3=1;tx4=1;
  for (ix1=0;ix1<nx1;ix1++){
    if (ix1>=0   && ix1<lx1i) tx1 = 1 - cos(((float) (ix1)/lx1i)*PI/2);
    if (ix1>=lx1i && ix1<=nx1-lx1f) tx1 = 1;
    if (ix1>nx1-lx1f && ix1<nx1) tx1 = cos(((float) (ix1-nx1+lx1f)/lx1f)*PI/2);
  for (ix2=0;ix2<nx2;ix2++){
    if (ix2>=0   && ix2<lx2i) tx2 = 1 - cos(((float) (ix2)/lx2i)*PI/2);
    if (ix2>=lx2i && ix2<=nx2-lx2f) tx2 = 1;
    if (ix2>nx2-lx2f && ix2<nx2) tx2 = cos(((float) (ix2-nx2+lx2f)/lx2f)*PI/2);
  for (ix3=0;ix3<nx3;ix3++){
    if (ix3>=0   && ix3<lx3i) tx3 = 1 - cos(((float) (ix3)/lx3i)*PI/2);
    if (ix3>=lx3i && ix3<=nx3-lx3f) tx3 = 1;
    if (ix3>nx3-lx3f && ix3<nx3) tx3 = cos(((float) (ix3-nx3+lx3f)/lx3f)*PI/2);
  for (ix4=0;ix4<nx4;ix4++){
    if (ix4>=0   && ix4<lx4i) tx4 = 1 - cos(((float) (ix4)/lx4i)*PI/2);
    if (ix4>=lx4i && ix4<=nx4-lx4f) tx4 = 1;
    if (ix4>nx4-lx4f && ix4<nx4) tx4 = cos(((float) (ix4-nx4+lx4f)/lx4f)*PI/2);
    ix = ix4*nx3*nx2*nx1 + ix3*nx2*nx1 + ix2*nx1 + ix1;
    for(it=0;it<nt;it++){
      if (it>=0   && it<lti) tt = 1 - cos(((float) (it)/lti)*PI/2);
      if (it>=lti && it<=nt-ltf) tt = 1;
      if (it>nt-ltf && it<nt) tt = cos(((float) (it-nt+ltf)/ltf)*PI/2);
      d[ix][it] = tt*tx1*tx2*tx3*tx4*d[ix][it];
    }
  }
  }
  }
  }
  return;
}

