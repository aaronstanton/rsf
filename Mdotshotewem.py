#!/usr/bin/env python
''' dot product test for sfshotewem

REQUIRES the PYTHON API
'''

# Import RSF API
try:
    import rsf.api as rsf
    import os.path, sys
    from subprocess import call
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY '''
    sys.exit(1)

# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
d1     = par.string("ux", "d1.rsf")
d2     = par.string("uz", "d2.rsf")
m1     = par.string("mpp", "m1.rsf")
m2     = par.string("mps", "m2.rsf")
vp     = par.string("vp", "vp.rsf")
vs     = par.string("vs", "vs.rsf")
wav    = par.string("wav", "wav.rsf")
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

d1_fwd = "tmp_d1_fwd.rsf"
d2_fwd = "tmp_d2_fwd.rsf"
m1_adj = "tmp_m1_adj.rsf"
m2_adj = "tmp_m2_adj.rsf"

forward = "mpiexec -np 1 \
~/rsf/bin/sfshotewem adj=n \
ux=%s uz=%s \
mpp=%s mps=%s \
vp=%s vs=%s wav=%s \
verbose=n \
nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (d1_fwd,d2_fwd,m1,m2,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

adjoint = "mpiexec -np 1 \
~/rsf/bin/sfshotewem adj=y \
ux=%s uz=%s \
mpp=%s mps=%s \
vp=%s vs=%s wav=%s \
verbose=n \
nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (d1,d2,m1_adj,m2_adj,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

dot1 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(d1,d2,d1_fwd,d2_fwd)
dot2 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(m1,m2,m1_adj,m2_adj)

print >> sys.stderr, "Forward..."
print >> sys.stderr, forward
os.system(forward)
print >> sys.stderr, "Adjoint..."
print >> sys.stderr, adjoint
os.system(adjoint)
print >> sys.stderr, "Inner product 1:"
os.system(dot1)
print >> sys.stderr, "Inner product 2:"
os.system(dot2)

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_d1_fwd.rsf tmp_d2_fwd.rsf tmp_m1_adj.rsf tmp_m2_adj.rsf"
os.system(cleanup)

print >> sys.stderr, "Done."

