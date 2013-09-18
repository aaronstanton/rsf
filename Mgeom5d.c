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
    int ik,nk_in,nk_out;
    int n2, i2;
    char *hdr_in_name;
    char *hdr_out_name;
    float *sx, *sy, *gx, *gy, *mx, *my, *hx, *hy, *h, *az; 
    int *isx, *isy, *igx, *igy, *imx, *imy, *ihx, *ihy, *ih, *iaz; 
    float osx, osy, ogx, ogy, omx, omy, ohx, ohy, oh, oaz; 
    float dsx, dsy, dgx, dgy, dmx, dmy, dhx, dhy, dh, daz; 
    int *hdr_in_array, *hdr_out_array;
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

    for (i2=0; i2<n2; i2++) {	
	sf_intread (hdr_in_array,nk_in,hdr_in);
	if (mode<=3){
	  sx = (float) hdr_in_array[segykey("sx")];
	  sy = (float) hdr_in_array[segykey("sy")];
	  gx = (float) hdr_in_array[segykey("gx")];
	  gy = (float) hdr_in_array[segykey("gy")];
	}
	else if (mode==4){
	  isx = hdr_in_array[segykey("isx")];
	  isy = hdr_in_array[segykey("isy")];
	  igx = hdr_in_array[segykey("igx")];
	  igy = hdr_in_array[segykey("igy")];
	}
	else if (mode==5){
	  imx = hdr_in_array[segykey("imx")];
	  imy = hdr_in_array[segykey("imy")];
	  ihx = hdr_in_array[segykey("ihx")];
	  ihy = hdr_in_array[segykey("ihy")];
	}
	else if (mode==6){
	  imx = hdr_in_array[segykey("imx")];
	  imy = hdr_in_array[segykey("imy")];
	  ih  = hdr_in_array[segykey("ih")];
	  iaz = hdr_in_array[segykey("iaz")];
	}


    if (mode==1){      /* binning sx/sy/gx/gy */
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
      geom_bin_header(sx,sy,gx,gy,
	               isx,isy,igx,igy,
		       osx,osy,ogx,ogy,
		       dsx,dsy,dgx,dgy,1);
    }
    else if (mode==2){ /* binning mx/my/hx/hy */
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
      geom_bin_header(mx,my,hx,hy,
	               imx,imy,ihx,ihy,
		       omx,omy,ohx,ohy,
		       dmx,dmy,dhx,dhy,1);
    }
    else if (mode==3){ /* binning mx/my/h/az */
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
      geom_bin_header(mx,my,h,az,
	               imx,imy,ih,iaz,
		       omx,omy,oh,oaz,
		       dmx,dmy,dh,daz,1);
    }
    else if (mode==4){ /* use binned isx/isy/igx/igy to get headers */
      geom_bin_header(sx,sy,gx,gy,
	               isx,isy,igx,igy,
		       osx,osy,ogx,ogy,
		       dsx,dsy,dgx,dgy,0);
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
    }
    else if (mode==5){ /* use binned imx/imy/ihx/ihy to get headers */
      geom_bin_header(mx,my,hx,hy,
	               imx,imy,ihx,ihy,
		       omx,omy,ohx,ohy,
		       dmx,dmy,dhx,dhy,0);
      geom_calc_sxsygxgy_from_mxmyhxhy(mx,my,hx,hy,sx,sy,gx,gy,n2);
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
    }
    else if (mode==6){ /* use binned imx/imy/ih/iaz to get headers */
      geom_bin_header(mx,my,h,az,
	               imx,imy,ih,iaz,
		       omx,omy,oh,oaz,
		       dmx,dmy,dh,daz,0);
      geom_calc_sxsygxgy_from_mxmyhaz(mx,my,h,az,sx,sy,gx,gy,n2);
      geom_calc_header(sx,sy,gx,gy,
		        mx,my,hx,hy,
	                h,az);
    }
 
    /* zero all headers that were not present on input */ 
    for (ik=0;ik<nk_out;ik++) hdr_out_array[ik] = 0;

      /* pass on all input headers */
      /*for (ik=0;ik<nk_in;ik++) itrace[ik] = hdrin[i2][ik];*/
      /* update the new non-standard headers*/  
      if (mode<=3){ 
        hdr_out_array[segykey("sx")] = (int) sx;
        hdr_out_array[segykey("sy")] = (int) sx;
        hdr_out_array[segykey("gx")] = (int) gx;
        hdr_out_array[segykey("gy")] = (int) gy;
        hdr_out_array[segykey("mx")] = (int) mx;
        hdr_out_array[segykey("my")] = (int) my;
        hdr_out_array[segykey("hx")] = (int) hx;
        hdr_out_array[segykey("hy")] = (int) hy;
        hdr_out_array[segykey("h")]  = (int) h;
        hdr_out_array[segykey("az")] = (int) az;
      }
      if (mode==1){
        hdr_out_array[segykey("isx")] = (int) isx;
        hdr_out_array[segykey("isy")] = (int) isx;
        hdr_out_array[segykey("igx")] = (int) igx;
        hdr_out_array[segykey("igy")] = (int) igy;
      }
      if (mode==2){
        hdr_out_array[segykey("imx")] = (int) imx;
        hdr_out_array[segykey("imy")] = (int) imy;
        hdr_out_array[segykey("ihx")] = (int) ihx;
        hdr_out_array[segykey("ihy")] = (int) ihy;
      }
      if (mode==3){
        hdr_out_array[segykey("imx")] = (int) imx;
        hdr_out_array[segykey("imy")] = (int) imy;
        hdr_out_array[segykey("ih")]  = (int) ih;
        hdr_out_array[segykey("iaz")] = (int) iaz;
      }
      sf_intwrite(hdr_out_array, nk_out, hdr_out);
    }
    sf_fileclose (hdr_in);
    sf_fileclose (hdr_out);

    exit (0);
}
void geom_bin_headers(float x1,float x2,float x3,float x4,
	              int ix1,int ix2,int ix3,int ixy,
		      float ox1,float ox2,float ox3,float ox4,
		      float dx1,float dx2,float dx3,float dx4,float ang,
		      int fwd,int haz_flag)
{
  float rad2deg,deg2rad,ang2,x1_rot,x2_rot,x3_rot,x4_rot;
  rad2deg = 180/PI;
  deg2rad = PI/180;

  if (fwd){
    if (ang > 90) ang2=-deg2rad*(ang-90);
    else ang2=deg2rad*(90-ang);
    x1_rot = (x1-ox1)*cos(ang2) - (x2-ox2)*sin(ang2) + ox1;
    x2_rot = (x1-ox1)*sin(ang2) + (x2-ox2)*cos(ang2) + ox2;
    x3_rot = (x3-ox3)*cos(ang2) - (x4-ox4)*sin(ang2) + ox3;
    x4_rot = (x3-ox3)*sin(ang2) + (x4-ox4)*cos(ang2) + ox4;
    ix1 = (int) round((x1_rot-ox1)/dx1);
    ix2 = (int) round((x2_rot-ox2)/dx2);
    if (haz_flag==0){
      ix3 = (int) round((x3_rot-ox3)/dx3);
      ix4 = (int) round((x4_rot-ox4)/dx4);
    }
    else{
      ix3 = (int) round((x3-ox3)/dx3);
      ix4 = (int) round((x4-ox4)/dx4);
    }
  }
  else{
    x1_rot = (float) ix1*dx1 + ox1;
    x2_rot = (float) ix2*dx2 + ox2;
    x3_rot = (float) ix3*dx3 + ox3;
    x4_rot = (float) ix4*dx4 + ox4;
    x1 =  (x1_rot-ox1)*cos(ang2) + (x2_rot-ox2)*sin(ang2) + ox1;
    x2 = -(x1_rot-ox1)*sin(ang2) + (x2_rot-ox2)*cos(ang2) + ox2;
    if (haz_flag==0){
      x3 =  (x3_rot-ox3)*cos(ang2) + (x4_rot-ox4)*sin(ang2) + ox3;
      x4 = -(x3_rot-ox3)*sin(ang2) + (x4_rot-ox4)*cos(ang2) + ox4;
    }
    else{
      x3 = x3_rot;
      x4 = x4_rot;
    }
  }

  return;
}
void geom_calc_sxsygxgy_from_mxmyhxhy(float mx,float my,float hx,float hy,
				      float sx,float sy,float gx,float gy)
{
  return;
}
void geom_calc_headers(float sx,float sy,float gx,float gy,
		       float mx,float my,float hx,float hy,
	               float h,float az,float gamma)
{
  float gammainv;
  gammainv = 1/gamma;	 		

  hx = gx - sx;
  hy = gy - sy;
  h  = sqrt(hx*hx + hy*hy);
  /* azimuth measured from source to receiver
     CC from East and ranges from 0 to 359.999 degrees*/
  az = rad2deg*atan2((gy-sy),(gx-sx));
  if (az < 0.) az += 360.0;
  mx = sx + hx/(1 + gammainv);
  my = sy + hy/(1 + gammainv);

  return;
}


