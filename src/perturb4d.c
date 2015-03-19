/* allocation and deallocation routines */
/*
  Copyright (C) 213 University of Alberta
  
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
#include <math.h>
#include <time.h>

#include <rsf.h>
/*^*/

#include "perturb4d.h"

void perturb4d(sf_complex *d,int n1,int n2,int n3,int n4,float var1,float var2,float var3,float var4)
/*< perturb the position of data in a 4d array >*/
{
  int seed,i,i1,i2,i3,i4,ip,ip1,ip2,ip3,ip4;
  float std1,std2,std3,std4;
  seed = time(NULL);
  /* random seed */
  init_genrand((unsigned long) seed);
  std1 = sqrtf(var1);
  std2 = sqrtf(var2);
  std3 = sqrtf(var3);
  std4 = sqrtf(var4);
  i = 0;
  for(i1=0;i1<n1;i1++){
    for(i2=0;i2<n2;i2++){
      for(i3=0;i3<n3;i3++){
        for(i4=0;i4<n4;i4++){
          ip1 = (int) i1 + roundf(std1*genrand_real1());
          if (ip1 < 0) ip1 = 0; else if (ip1 >= n1) ip1 = n1-1;
          ip2 = (int) i2 + roundf(std2*genrand_real1());
          if (ip2 < 0) ip2 = 0; else if (ip2 >= n2) ip2 = n2-1;
          ip3 = (int) i3 + roundf(std3*genrand_real1());
          if (ip3 < 0) ip3 = 0; else if (ip3 >= n3) ip3 = n3-1;
          ip4 = (int) i4 + roundf(std4*genrand_real1());
          if (ip4 < 0) ip4 = 0; else if (ip4 >= n4) ip4 = n4-1;
          ip = ip1*n2*n3*n4 + ip2*n3*n4 + ip3*n4 + ip4; 
          d[i] = d[ip]; 
          i++;
        }
      }
    }
  }
  return;
}


