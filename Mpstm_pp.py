#!/usr/bin/env python
''' PSTM operator for PP data  

REQUIRES the PYTHON API, NUMPY, SCIPY, and MATH
'''

# Import RSF API
try:
    import rsf.api as rsf
    import sys, numpy, scipy, math
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY, MATH '''
    sys.exit(1)

# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
adj = par.bool  ("adj", True) # True=>data to model, False=>model to data
# Declare input and outputs
fin = rsf.Input()    # no argument means stdin
fout = rsf.Output()  # no argument means stdout
 
# Get dimensions of input header or output header
n1 = fin.int('n1')
n2 = fin.int('n2')
n3 = fin.int('n3')
d1 = fin.float('d1')
d2 = fin.float('d2')
d3 = fin.float('d3')
o1 = fin.float('o1')
o2 = fin.float('o2')
o3 = fin.float('o3')

if (adj==True):
  nt  = n1
  nsx = n2
  ngx = n3
  ot  = o1
  osx = o2
  ogx = o3
  dt  = d1
  dsx = d2
  dgx = d3
  nhx = par.int("nhx", 10) 
  dhx = par.float("dhx", 1) 
  ohx = par.float("ohx", 0) 
  nmx = par.int("nmx", 10) 
  dmx = par.float("dmx", 1) 
  omx = par.float("omx", 0) 
  data = numpy.zeros((ngx,nsx,nt),'f') # Note, we reverse array dims
  model = numpy.zeros((nhx,nmx,nt),'f') # Note, we reverse array dims
  # Read input data
  fin.read(data)
  fin.close()
else:
  nt  = n1
  nmx = n2
  nhx = n3
  ot  = o1
  omx = o2
  ohx = o3
  dt  = d1
  dmx = d2
  dhx = d3
  ngx = par.int("ngx", 10) 
  dgx = par.float("dgx", 1) 
  ogx = par.float("ogx", 0) 
  nsx = par.int("nsx", 10) 
  dsx = par.float("dsx", 1) 
  osx = par.float("osx", 0) 
  data = numpy.zeros((ngx,nsx,nt),'f') # Note, we reverse array dims
  model = numpy.zeros((nhx,nmx,nt),'f') # Note, we reverse array dims
  # Read input data
  fin.read(model)
  fin.close()

sx = numpy.zeros((nsx),'f') 
for isx in range(0, nsx): sx[isx] = osx + isx*dsx;
gx = numpy.zeros((ngx),'f')
for igx in range(0, ngx): gx[igx] = ogx + igx*dgx;
mx = numpy.zeros((nmx),'f') 
for imx in range(0, nmx): mx[imx] = omx + imx*dmx;
hx = numpy.zeros((nhx),'f')
for ihx in range(0, nhx): hx[ihx] = ohx + ihx*dhx;

v = numpy.zeros((nmx,nt),'f') # Note, we reverse array dims
for it in range(0, nt):
  for imx in range(0, nmx):
    v[imx,it] = 1500
# Read velocity model
# fvel.read(v)
# fvel.close()

for isx in range(0, nsx):
  for igx in range(0, ngx):
    print >> sys.stderr, "igx=" + str(igx)
    hx = gx[igx] - sx[isx]
    ih = int((hx - ohx)/dhx)
    for imx in range(0, nmx):  
      for it in range(0, nt):
        t0 = ot + it*dt
        t = math.sqrt(t0**2 + (hx/v[imx,it])**2)
        t_floor = math.floor(t/dt)
        jt = int(t_floor)
        if (jt+1 < nt and ih < nhx):
	  res = (t-t_floor)/dt;
	  res0 = 1.0-res;
          if (adj==True): # data space --> model space
            model[ihx,imx,it] = model[ihx,imx,it] + res0*data[igx,isx,jt] + res*data[igx,isx,jt+1]
          else: # model space --> data space
            data[igx,isx,jt]   = data[igx,isx,jt] + res0*model[ihx,imx,it]
            data[igx,isx,jt+1] = data[igx,isx,jt+1] + res*model[ihx,imx,it]

if (adj==True): # data space --> model space
  # Setup output headers
  fout.put('n1',nt)
  fout.put('n2',nmx)
  fout.put('n3',nhx)
  fout.put('o1',ot)
  fout.put('o2',omx)
  fout.put('o3',ohx)
  fout.put('d1',dt)
  fout.put('d2',dmx)
  fout.put('d3',dhx)
  # Write output data
  fout.write(model)
else: # model space --> data space
  # Setup output headers
  fout.put('n1',nt)
  fout.put('n2',nsx)
  fout.put('n3',ngx)
  fout.put('o1',ot)
  fout.put('o2',osx)
  fout.put('o3',ogx)
  fout.put('d1',dt)
  fout.put('d2',dsx)
  fout.put('d3',dgx)
  # Write output data
  fout.write(data)

# Clean up files
fout.close()
fin.close()
