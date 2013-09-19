/* Set geometry headers (cmpx, cmpy, offsetx, offsety, offset, azimuth) and bin the data. Also can use binned headers to re-compute shot and receiver coordinates. The program acts solely upon the header file. 
mode=1: sx/sy/gx/gy     -> isx/isy/igx/igy
mode=2: sx/sy/gx/gy     -> imx/imy/ihx/ihy
mode=3: sx/sy/gx/gy     -> imx/imy/ih/iaz
mode=4: isx/isy/igx/igy -> sx/sy/gx/gy
mode=5: imx/imy/ihx/ihy -> sx/sy/gx/gy
mode=6: imx/imy/ih/iaz  -> sx/sy/gx/gy

TO BE FIXED: problem with mode=1 & 4 and mode=3 & 6 : azimuth calculation problematic. mode=2 & 5 appears to work well.
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

void geom_bin_header(float *x1,float *x2,float *x3,float *x4,
	              int *ix1,int *ix2,int *ix3,int *ix4,
		      float ox1,float ox2,float ox3,float ox4,
		      float dx1,float dx2,float dx3,float dx4,float ang,
		      int fwd,int haz_flag);
void geom_calc_header(float *sx,float *sy,float *gx,float *gy,
		       float *mx,float *my,float *hx,float *hy,
	               float *h,float *az,float gamma);

void geom_calc_sxsygxgy_from_mxmyhxhy(float *mx,float *my,float *hx,float *hy,
				      float *sx,float *sy,float *gx,float *gy,
				      float gamma);

void geom_calc_sxsygxgy_from_mxmyhaz(float *mx,float *my,float *h,float *az,
				      float *sx,float *sy,float *gx,float *gy,
				      float gamma);

int main(int argc, char* argv[])
{
    int mode; 
    int ik,nk_in,nk_out;
    int n2, i2;
    char *hdr_in_name;
    char *hdr_out_name;
    float *sx, *sy, *gx, *gy, *mx, *my, *hx, *hy, *h, *az; 
    int *isx, *isy, *igx, *igy, *imx, *imy, *ihx, *ihy, *ih, *iaz; 
    float osx, osy, ogx, ogy, omx, omy, ohx, ohy, oh, oaz; 
    float dsx, dsy, dgx, dgy, dmx, dmy, dhx, dhy, dh, daz;
    float gamma,ang; 
    int *hdr_in_array, *hdr_out_array, **hdr_in_all;
    sf_file hdr_in,hdr_out;
    sf_init (argc,argv);

    if (!sf_getint("mode",&mode)) mode=1; /* mode of geometry computation */
    if (!sf_getfloat("osx",&osx)) osx=0;
    if (!sf_getfloat("osy",&osy)) osy=0;
    if (!sf_getfloat("dsx",&dsx)) dsx=1;
    if (!sf_getfloat("dsy",&dsy)) dsy=1;
    if (!sf_getfloat("ogx",&ogx)) ogx=0;
    if (!sf_getfloat("ogy",&ogy)) ogy=0;
    if (!sf_getfloat("dgx",&dgx)) dgx=1;
    if (!sf_getfloat("dgy",&dgy)) dgy=1;
    if (!sf_getfloat("omx",&omx)) omx=0;
    if (!sf_getfloat("omy",&omy)) omy=0;
    if (!sf_getfloat("dmx",&dmx)) dmx=1;
    if (!sf_getfloat("dmy",&dmy)) dmy=1;
    if (!sf_getfloat("ohx",&ohx)) ohx=0;
    if (!sf_getfloat("ohy",&ohy)) ohy=0;
    if (!sf_getfloat("dhx",&dhx)) dhx=1;
    if (!sf_getfloat("dhy",&dhy)) dhy=1;
    if (!sf_getfloat("oh",&oh))   oh=0;
    if (!sf_getfloat("oaz",&oaz)) oaz=0;
    if (!sf_getfloat("dh",&dh))   dh=1;
    if (!sf_getfloat("daz",&daz)) daz=1;
    if (!sf_getfloat("gamma",&gamma)) gamma=1; /* vp/vs ratio, can be used for converted wave binning */
    if (!sf_getfloat("ang",&ang)) ang=90; /* azimuth of constant inline direction (commonly defined as the direction that receiver lines run) measured in degrees CC from East */

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
    hdr_out_name = sf_getstring("headout");
    if (NULL == hdr_out_name) sf_error("Need headout=");
    hdr_out = sf_output(hdr_out_name);
    sf_putint(hdr_out,"n1",nk_out);
    sf_putint(hdr_out,"n2",n2);
    sf_setformat(hdr_out,"native_int");
    segy2hist(hdr_out,nk_out);

    hdr_in_array = sf_intalloc(nk_in);
    hdr_out_array = sf_intalloc(nk_out);

    sx = sf_floatalloc(1);
    sy = sf_floatalloc(1);
    gx = sf_floatalloc(1);
    gy = sf_floatalloc(1);
    mx = sf_floatalloc(1);
    my = sf_floatalloc(1);
    hx = sf_floatalloc(1);
    hy = sf_floatalloc(1);
    h  = sf_floatalloc(1);
    az = sf_floatalloc(1);
    isx = sf_intalloc(1);
    isy = sf_intalloc(1);
    igx = sf_intalloc(1);
    igy = sf_intalloc(1);
    imx = sf_intalloc(1);
    imy = sf_intalloc(1);
    ihx = sf_intalloc(1);
    ihy = sf_intalloc(1);
    ih  = sf_intalloc(1);
    iaz = sf_intalloc(1);
    hdr_in_all = sf_intalloc2(n2,nk_in);

    sx[0]=0;sy[0]=0;gx[0]=0;gy[0]=0;
    mx[0]=0;my[0]=0;hx[0]=0;hy[0]=0;h[0]=0;az[0]=0;
    isx[0]=0;isy[0]=0;igx[0]=0;igy[0]=0;
    imx[0]=0;imy[0]=0;ihx[0]=0;ihy[0]=0;ih[0]=0;iaz[0]=0;

    for (i2=0; i2<n2; i2++) {	
	sf_intread (hdr_in_array,nk_in,hdr_in);
	for (ik=0;ik<nk_in;ik++) hdr_in_all[ik][i2]=hdr_in_array[ik];
	if (mode<=3){
	  sx[0] = (float) hdr_in_array[segykey("sx")];
	  sy[0] = (float) hdr_in_array[segykey("sy")];
	  gx[0] = (float) hdr_in_array[segykey("gx")];
	  gy[0] = (float) hdr_in_array[segykey("gy")];
	}
	else if (mode==4){
	  isx[0] = hdr_in_array[segykey("isx")];
	  isy[0] = hdr_in_array[segykey("isy")];
	  igx[0] = hdr_in_array[segykey("igx")];
	  igy[0] = hdr_in_array[segykey("igy")];
	}
	else if (mode==5){
	  imx[0] = hdr_in_array[segykey("imx")];
	  imy[0] = hdr_in_array[segykey("imy")];
	  ihx[0] = hdr_in_array[segykey("ihx")];
	  ihy[0] = hdr_in_array[segykey("ihy")];
	}
	else if (mode==6){
	  imx[0] = hdr_in_array[segykey("imx")];
	  imy[0] = hdr_in_array[segykey("imy")];
	  ih[0]  = hdr_in_array[segykey("ih")];
	  iaz[0] = hdr_in_array[segykey("iaz")];
	}

    if (mode==1){      /* binning sx/sy/gx/gy */
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
      geom_bin_header(sx,sy,gx,gy,
	              isx,isy,igx,igy,
		      osx,osy,ogx,ogy,
		      dsx,dsy,dgx,dgy,ang,1,0);
    }
    else if (mode==2){ /* binning mx/my/hx/hy */
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
      geom_bin_header(mx,my,hx,hy,
	              imx,imy,ihx,ihy,
		      omx,omy,ohx,ohy,
		      dmx,dmy,dhx,dhy,ang,1,0);
    }
    else if (mode==3){ /* binning mx/my/h/az */
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
      geom_bin_header(mx,my,h,az,
	              imx,imy,ih,iaz,
		      omx,omy,oh,oaz,
		      dmx,dmy,dh,daz,ang,1,1);
    }
    else if (mode==4){ /* use binned isx/isy/igx/igy to get headers */
      geom_bin_header(sx,sy,gx,gy,
	              isx,isy,igx,igy,
		      osx,osy,ogx,ogy,
		      dsx,dsy,dgx,dgy,ang,0,0);
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
    }
    else if (mode==5){ /* use binned imx/imy/ihx/ihy to get headers */
      geom_bin_header(mx,my,hx,hy,
	              imx,imy,ihx,ihy,
		      omx,omy,ohx,ohy,
		      dmx,dmy,dhx,dhy,ang,0,0);
      geom_calc_sxsygxgy_from_mxmyhxhy(mx,my,hx,hy,sx,sy,gx,gy,gamma);
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
    }
    else if (mode==6){ /* use binned imx/imy/ih/iaz to get headers */
      geom_bin_header(mx,my,h,az,
	              imx,imy,ih,iaz,
		      omx,omy,oh,oaz,
		      dmx,dmy,dh,daz,ang,0,1);
      geom_calc_sxsygxgy_from_mxmyhaz(mx,my,h,az,sx,sy,gx,gy,gamma);
      geom_calc_header(sx,sy,gx,gy,
		       mx,my,hx,hy,
	               h,az,gamma);
    } 
    /* zero all headers that were not present on input */ 
    for (ik=0;ik<nk_out;ik++) hdr_out_array[ik] = 0;
    /* pass on all input headers */
    for (ik=0;ik<nk_in;ik++) hdr_out_array[ik]=hdr_in_all[ik][i2];
    /* update the new non-standard headers*/ 
    hdr_out_array[segykey("sx")] = (int) sx[0];
    hdr_out_array[segykey("sy")] = (int) sy[0];
    hdr_out_array[segykey("gx")] = (int) gx[0];
    hdr_out_array[segykey("gy")] = (int) gy[0];
    hdr_out_array[segykey("mx")] = (int) mx[0];
    hdr_out_array[segykey("my")] = (int) my[0];
    hdr_out_array[segykey("hx")] = (int) hx[0];
    hdr_out_array[segykey("hy")] = (int) hy[0];
    hdr_out_array[segykey("h")]  = (int) h[0];
    hdr_out_array[segykey("az")] = (int) az[0];
    if (mode==1 || mode==4){
      hdr_out_array[segykey("isx")] = (int) isx[0];
      hdr_out_array[segykey("isy")] = (int) isy[0];
      hdr_out_array[segykey("igx")] = (int) igx[0];
      hdr_out_array[segykey("igy")] = (int) igy[0];
    }
    if (mode==2 || mode==5){
      hdr_out_array[segykey("imx")] = (int) imx[0];
      hdr_out_array[segykey("imy")] = (int) imy[0];
      hdr_out_array[segykey("ihx")] = (int) ihx[0];
      hdr_out_array[segykey("ihy")] = (int) ihy[0];
    }
    if (mode==3 || mode==6){
      hdr_out_array[segykey("imx")] = (int) imx[0];
      hdr_out_array[segykey("imy")] = (int) imy[0];
      hdr_out_array[segykey("ih")]  = (int) ih[0];
      hdr_out_array[segykey("iaz")] = (int) iaz[0];
    }
    sf_intwrite(hdr_out_array, nk_out, hdr_out);
    }
    sf_fileclose (hdr_in);
    sf_fileclose (hdr_out);

    exit (0);
}
void geom_bin_header(float *x1,float *x2,float *x3,float *x4,
	              int *ix1,int *ix2,int *ix3,int *ix4,
		      float ox1,float ox2,float ox3,float ox4,
		      float dx1,float dx2,float dx3,float dx4,float ang,
		      int fwd,int haz_flag)
{
  float deg2rad,ang2,x1_rot,x2_rot,x3_rot,x4_rot;
  deg2rad = PI/180;

  if (fwd){
    if (ang > 90) ang2=-deg2rad*(ang-90);
    else ang2=deg2rad*(90-ang);
    x1_rot = (x1[0]-ox1)*cos(ang2) - (x2[0]-ox2)*sin(ang2) + ox1;
    x2_rot = (x1[0]-ox1)*sin(ang2) + (x2[0]-ox2)*cos(ang2) + ox2;
    x3_rot = (x3[0]-ox3)*cos(ang2) - (x4[0]-ox4)*sin(ang2) + ox3;
    x4_rot = (x3[0]-ox3)*sin(ang2) + (x4[0]-ox4)*cos(ang2) + ox4;
    ix1[0] = (int) round((x1_rot-ox1)/dx1);
    ix2[0] = (int) round((x2_rot-ox2)/dx2);
    if (haz_flag==0){
      ix3[0] = (int) round((x3_rot-ox3)/dx3);
      ix4[0] = (int) round((x4_rot-ox4)/dx4);
    }
    else{
      ix3[0] = (int) round((x3[0]-ox3)/dx3);
      ix4[0] = (int) round((x4[0]-ox4)/dx4);
    }
  }
  else{
    if (ang > 90) ang2=-deg2rad*(ang-90);
    else ang2=deg2rad*(90-ang);
    x1_rot = (float) ix1[0]*dx1 + ox1;
    x2_rot = (float) ix2[0]*dx2 + ox2;
    x3_rot = (float) ix3[0]*dx3 + ox3;
    x4_rot = (float) ix4[0]*dx4 + ox4;
    x1[0] =  (x1_rot-ox1)*cos(ang2) + (x2_rot-ox2)*sin(ang2) + ox1;
    x2[0] = -(x1_rot-ox1)*sin(ang2) + (x2_rot-ox2)*cos(ang2) + ox2;
    if (haz_flag==0){
      x3[0] =  (x3_rot-ox3)*cos(ang2) + (x4_rot-ox4)*sin(ang2) + ox3;
      x4[0] = -(x3_rot-ox3)*sin(ang2) + (x4_rot-ox4)*cos(ang2) + ox4;
    }
    else{
      x3[0] = x3_rot;
      x4[0] = x4_rot;
    }
  }

  return;
}

void geom_calc_header(float *sx,float *sy,float *gx,float *gy,
		       float *mx,float *my,float *hx,float *hy,
	               float *h,float *az,float gamma)
{
  float gammainv, rad2deg;
  rad2deg = 180/PI;
  gammainv = 1/gamma;

  hx[0] = gx[0] - sx[0];
  hy[0] = gy[0] - sy[0];
  h[0]  = sqrt(hx[0]*hx[0] + hy[0]*hy[0]);
  /* azimuth measured from source to receiver
     CC from East and ranges from 0 to 359.999 degrees*/
  az[0] = rad2deg*atan2((gy[0]-sy[0]),(gx[0]-sx[0]));
  if (az[0] < 0.) az[0] += 360.0;
  mx[0] = sx[0] + hx[0]/(1 + gammainv);
  my[0] = sy[0] + hy[0]/(1 + gammainv);

  return;
}

void geom_calc_sxsygxgy_from_mxmyhxhy(float *mx,float *my,float *hx,float *hy,
				      float *sx,float *sy,float *gx,float *gy,
				      float gamma)
{
  float gammainv;
  gammainv = 1/gamma;
  sx[0] = mx[0] - hx[0]/(1 + gammainv);
  sy[0] = my[0] - hy[0]/(1 + gammainv);
  gx[0] = mx[0] + hx[0]*(1-(1/(1 + gammainv)));
  gy[0] = my[0] + hy[0]*(1-(1/(1 + gammainv)));

  return;
}

void geom_calc_sxsygxgy_from_mxmyhaz(float *mx,float *my,float *h,float *az,
				      float *sx,float *sy,float *gx,float *gy,
				      float gamma)
{
  float gammainv, deg2rad, hx, hy;
  deg2rad = PI/180;
  gammainv = 1/gamma;

  if (az[0] <= 90){
    hx = h[0]*cos(deg2rad*az[0]);
    hy = h[0]*sin(deg2rad*az[0]);
  }
  else if (az[0] > 90 && az[0] <= 180) {
    hx =-h[0]*cos(PI-(deg2rad*az[0]));
    hy = h[0]*sin(PI-(deg2rad*az[0]));
  }
  else if (az[0] > 180 && az[0] <= 270) {
    hx =-h[0]*cos((deg2rad*az[0])-PI);
    hy =-h[0]*sin((deg2rad*az[0])-PI);
  }
  else {
    hx = h[0]*cos(2*PI-(deg2rad*az[0]));
    hy =-h[0]*sin(2*PI-(deg2rad*az[0]));
  }
  sx[0] = mx[0] - hx/(1 + gammainv);
  sy[0] = my[0] - hy/(1 + gammainv);
  gx[0] = mx[0] + hx*(1-(1/(1 + gammainv)));
  gy[0] = my[0] + hy*(1-(1/(1 + gammainv)));

  return;
}

