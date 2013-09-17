/* Set geometry headers (cmpx, cmpy, offsetx, offsety, offset, azimuth) and bin the data. Also can use binned headers to re-compute shot and receiver coordinates. The program acts solely upon the header file. 
mode=1: sx/sy/gx/gy     -> isx/isy/igx/igy
mode=2: sx/sy/gx/gy     -> imx/imy/ihx/ihy
mode=3: sx/sy/gx/gy     -> imx/imy/ih/iaz
mode=4: isx/isy/igx/igy -> sx/sy/gx/gy
mode=5: imx/imy/ihx/ihy -> sx/sy/gx/gy
mode=6: imx/imy/ih/iaz  -> sx/sy/gx/gy
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

#ifndef PI
#define PI (3.141592653589793)
#endif

int main(int argc, char* argv[])
{
    int mode; 
    int nk;
    int n1, n2, i2, itrace[SF_MAXKEYS];
    const char *label, *headname;
    float sx, sy, gx, gy; 
    sf_file hdr;
    sf_init (argc,argv);

    if (!sf_getint("mode",&mode) mode=1; /* mode of geometry computation */
    if (!sf_getfloat("osx",&osx)) osx=0;
    if (!sf_getfloat("osy",&osy)) osy=0;
    if (!sf_getfloat("dsx",&dsx)) dsx=1;
    if (!sf_getfloat("dsy",&dsy)) dsy=1;
    if (!sf_getfloat("omx",&omx)) omx=0;
    if (!sf_getfloat("omy",&omy)) omy=0;
    if (!sf_getfloat("dmx",&dmx)) dmx=1;
    if (!sf_getfloat("dmy",&dmy)) dmy=1;
    if (!sf_getfloat("dhx",&dhx)) dhx=1;
    if (!sf_getfloat("dhy",&dhy)) dhy=1;
    if (!sf_getfloat("dh",&dh))   dh=1;
    if (!sf_getfloat("daz",&daz)) daz=1;

    header = sf_getstring("head");
    /* header file */
    if (NULL == header) { 
	header = sf_histstring(in,"head");
	if (NULL == header) sf_error("Need head=");
    }

    head = sf_input(header);

    if (SF_INT != sf_gettype(head)) sf_error("Need int header");
    if (!sf_histint(head,"n1",&nk)) sf_error("No n1= in head");
    n2 = sf_leftsize(head,1);

    segy_init(nk,head);

    hdr = sf_intalloc(nk);
    sx = sf_floatalloc(n2);
    sy = sf_floatalloc(n2);
    gx = sf_floatalloc(n2);
    gy = sf_floatalloc(n2);

    for (i2=0; i2<n2; i2++) {	
	sf_intread (hdr,nk,head);
	sx[i2] = (float) hdr[segykey("sx")];
	sy[i2] = (float) hdr[segykey("sy")];
	gx[i2] = (float) hdr[segykey("gx")];
	gy[i2] = (float) hdr[segykey("gy")];
    }
    sf_fileclose (head);



    if      (mode==1) geom_mode1(sx,sy,gx,gy,
	  		         isx,isy,igx,igy,
			         osx,osy,ogx,ogy,
			         dsx,dsy,dgx,dgy,n2);
    else if (mode==2) geom_mode2(mx,my,hx,hy,
	  		         imx,imy,ihx,ihy,
			         omx,omy,ohx,ohy,
			         dmx,dmy,dhx,dhy,n2);
    










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
    for (i2=0;i2<n2;i2++){ 
      for (i3=0;i3<n3;i3++){ 
        for (i4=0;i4<n4;i4++){ 
          for (i5=0;i5<n5;i5++){ 
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


