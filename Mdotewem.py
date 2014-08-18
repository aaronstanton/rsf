#!/usr/bin/env python
''' dot product test for sfmpiewem

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
sz  = par.float("sz",0)
gz  = par.float("gz",0)
reg  = par.bool("reg",False)
pa  = par.float("pa",-100)
pb  = par.float("pb",-50)
pc  = par.float("pc",50)
pd  = par.float("pd",100)

d1_fwd = "tmp_dot_d1_fwd.rsf"
d2_fwd = "tmp_dot_d2_fwd.rsf"
m1_adj = "tmp_dot_m1_adj.rsf"
m2_adj = "tmp_dot_m2_adj.rsf"
if (reg):
    tmp_m1 = "tmp_dot_tmp_m1.rsf"
    tmp_m1_adj = "tmp_dot_tmp_m1_adj.rsf"
    tmp_m2 = "tmp_dot_tmp_m2.rsf"
    tmp_m2_adj = "tmp_dot_tmp_m2_adj.rsf"
else:
    tmp_m1 = m1
    tmp_m1_adj = m1_adj
    tmp_m2 = m2
    tmp_m2_adj = m2_adj

forward1a = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (m1,tmp_m1,pa,pb,pc,pd) 
forward1b = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (m2,tmp_m2,pa,pb,pc,pd) 
forward2 = "mpiexec -np 1 \
~/rsf/bin/sfmpiewem adj=n \
ux=%s uz=%s \
mpp=%s mps=%s \
vp=%s vs=%s wav=%s \
verbose=n \
nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (d1_fwd,d2_fwd,tmp_m1,tmp_m2,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)

adjoint1 = "mpiexec -np 1 \
~/rsf/bin/sfmpiewem adj=y \
ux=%s uz=%s \
mpp=%s mps=%s \
vp=%s vs=%s wav=%s \
verbose=n \
nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (d1,d2,tmp_m1_adj,tmp_m2_adj,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)
adjoint2a = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_m1_adj,m1_adj,pa,pb,pc,pd)
adjoint2b = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_m2_adj,m2_adj,pa,pb,pc,pd)

dot1 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(d1,d2,d1_fwd,d2_fwd)
dot2 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(m1,m2,m1_adj,m2_adj)

print >> sys.stderr, "Forward..."
if (reg):
    os.system(forward1a)
    os.system(forward1b)
os.system(forward2)
print >> sys.stderr, "Adjoint..."
os.system(adjoint1)
if (reg):
    os.system(adjoint2a)
    os.system(adjoint2b)
print >> sys.stderr, "Inner product 1:"
os.system(dot1)
print >> sys.stderr, "Inner product 2:"
os.system(dot2)

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_dot_*.rsf"
os.system(cleanup)

print >> sys.stderr, "Done."

