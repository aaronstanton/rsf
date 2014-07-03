#!/usr/bin/env python
''' Test using python as a wrapper for a C code  

REQUIRES the PYTHON API, NUMPY, and SCIPY
'''

# Import RSF API
try:
    import rsf.api as rsf
    import os.path, sys, numpy, scipy
    from subprocess import call
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY '''
    sys.exit(1)

# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
d   = par.string("d", "d.rsf")
m   = par.string("m", "m.rsf")
v   = par.string("v", "v.rsf")
wav = par.string("wav", "wav.rsf")
np  = par.int("np",1)
npersocket = par.int("npersocket",1)
npersocket = par.int("npersocket",1)
nz  = par.int("nz",1)
dz  = par.float("dz",1)
oz  = par.float("oz",1)
nt  = par.int("nt",1)
dt  = par.float("dt",1)
ot  = par.float("ot",1)
nhx  = par.int("nhx",1)
dhx  = par.float("dhx",1)
ohx  = par.float("ohx",1)
npx  = par.int("npx",1)
dpx  = par.float("dpx",1)
opx  = par.float("opx",1)
nsx  = par.int("nsx",1)
dsx  = par.float("dsx",1)
osx  = par.float("osx",1)
fmin  = par.float("fmin",1)
fmax  = par.float("fmax",1)

forward = "/usr/lib64/openmpi/bin/mpiexec -np %d --npersocket %d \
~/rsf/bin/sfmpishotwem \
adj=n infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,npersocket,m,d,v,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

print >> sys.stderr, forward

os.system(forward)

print >> sys.stderr, "Done."

