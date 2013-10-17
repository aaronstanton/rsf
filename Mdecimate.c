/* decimate seismic data.
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

int main(int argc, char* argv[])
{
    int mode,inc2,inc3,inc4,inc5; 
    int seed;
    int n1,i1,n2,i2,n3,i3,n4,i4,n5,i5;
    float *trace;
    float d1,o1,d2,o2,d3,o3,d4,o4,d5,o5;
    float r,perc; 
    float a2,a3,a4,a5;
    sf_file in,out;
    sf_init (argc,argv);
    in = sf_input("in");
    out = sf_output("out");

    if (!sf_getint("seed",&seed)) seed = time(NULL);
    /* random seed */
    init_genrand((unsigned long) seed);


    if (!sf_getint("mode",&mode)) mode=1; /* mode of decimation 1=random (specify perc), 2=regular inc2 to inc5 specify increment between non-zero traces in each dimension, 3=regular inc2 to inc5 specify increment between zero traces in each dimension. */
    if (!sf_getint("inc2",&inc2)) inc2=2;    /* if mode=2 then the increment of live traces for dimension 2 (ones in between are zeroed)*/
    if (!sf_getint("inc3",&inc3)) inc3=2;    /* if mode=2 then the increment of live traces for dimension 3 (ones in between are zeroed)*/
    if (!sf_getint("inc4",&inc4)) inc4=2;    /* if mode=2 then the increment of live traces for dimension 4 (ones in between are zeroed)*/
    if (!sf_getint("inc5",&inc5)) inc5=2;    /* if mode=2 then the increment of live traces for dimension 5 (ones in between are zeroed)*/
    if (!sf_getfloat("perc",&perc)) perc=40; /* percentage of traces decimated */

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


    if (n2==1) inc2=1; 
    if (n3==1) inc3=1; 
    if (n4==1) inc4=1; 
    if (n5==1) inc5=1; 

    trace = sf_floatalloc (n1);
    for (i2=0; i2<n2; i2++) {	
    for (i3=0; i3<n3; i3++) {
    for (i4=0; i4<n4; i4++) {
    for (i5=0; i5<n5; i5++) {
      sf_floatread(trace,n1,in);
      if (mode==1){ /* randomly set traces to zero */
        r = genrand_real1();
        if (r < perc/100){
          for (i1=0;i1<n1;i1++) trace[i1] = 0;
        }
      }
      else if (mode==2){ /* regularly set traces to zero, user chooses increment between non-zero traces */
        a2=a3=a4=a5=1;
        if ((float) i2/inc2 - round(i2/inc2) > 0.0001) a2=0;
        if ((float) i3/inc3 - round(i3/inc3) > 0.0001) a3=0;
        if ((float) i4/inc4 - round(i4/inc4) > 0.0001) a4=0;
        if ((float) i5/inc5 - round(i5/inc5) > 0.0001) a5=0;
        for (i1=0;i1<n1;i1++) trace[i1] = trace[i1]*a2*a3*a4*a5;
      }
      else if (mode==3){ /* regularly set traces to zero, user chooses increment between zero traces */
        a2=a3=a4=a5=1;
        if ((float) i2/inc2 - round(i2/inc2) > 0.0001 || n2==1) a2=0;
        if ((float) i3/inc3 - round(i3/inc3) > 0.0001 || n3==1) a3=0;
        if ((float) i4/inc4 - round(i4/inc4) > 0.0001 || n4==1) a4=0;
        if ((float) i5/inc5 - round(i5/inc5) > 0.0001 || n5==1) a5=0;
        for (i1=0;i1<n1;i1++) trace[i1] = trace[i1]*(1-a2)*(1-a3)*(1-a4)*(1-a5);
      }
      sf_floatwrite(trace,n1,out);
    }
    }
    }
    }

    exit (0);
}

