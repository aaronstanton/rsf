#!/usr/bin/env python
''' dot product test for sfshotwem  

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

d_fwd = "tmp_d_fwd.rsf"
m_adj = "tmp_m_adj.rsf"

forward = "/usr/lib64/openmpi/bin/mpiexec -np %d --npersocket %d \
~/rsf/bin/sfmpishotwem \
adj=n infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,npersocket,m,d_fwd,v,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

adjoint = "/usr/lib64/openmpi/bin/mpiexec -np %d --npersocket %d \
~/rsf/bin/sfmpishotwem \
adj=y infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,npersocket,d,m_adj,v,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

dot1 = "~/rsf/bin/sfinnerprod in1=%s in2=%s" %(d,d_fwd)
dot2 = "~/rsf/bin/sfinnerprod in1=%s in2=%s" %(m,m_adj)

print >> sys.stderr, "Forward..."
os.system(forward)
print >> sys.stderr, "Adjoint..."
os.system(adjoint)
print >> sys.stderr, "Inner product 1:"
os.system(dot1)
print >> sys.stderr, "Inner product 2:"
os.system(dot2)

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_d_fwd.rsf tmp_m_adj.rsf tmpdmig_*.rsf tmpd_*.rsf"
os.system(cleanup)

print >> sys.stderr, "Done."

