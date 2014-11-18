#!/usr/bin/env python
''' dot product test for sfmpiwem  

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

# get environmental variables (such as MPIRUN) from config.py
execfile(os.environ['HOME'] +"/rsf/src/config.py")

# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
d   = par.string("d", "d.rsf")
m   = par.string("m", "m.rsf")
v   = par.string("v", "v.rsf")
wav = par.string("wav", "wav.rsf")
np  = par.int("np",1)
npersocket = par.int("npersocket",1)
sz  = par.float("sz",0)
gz  = par.float("gz",0)
nz  = par.int("nz",1)
dz  = par.float("dz",1)
oz  = par.float("oz",1)
nt  = par.int("nt",1)
dt  = par.float("dt",1)
ot  = par.float("ot",1)
npx  = par.int("npx",1)
dpx  = par.float("dpx",1)
opx  = par.float("opx",1)
nsx  = par.int("nsx",1)
dsx  = par.float("dsx",1)
osx  = par.float("osx",1)
fmin  = par.float("fmin",1)
fmax  = par.float("fmax",1)
reg  = par.bool("reg",False)
pa  = par.float("pa",-100)
pb  = par.float("pb",-50)
pc  = par.float("pc",50)
pd  = par.float("pd",100)

d_fwd = "tmp_dot_d_fwd.rsf"
m_adj = "tmp_dot_m_adj.rsf"
if (reg):
    tmp_m = "tmp_dot_tmp_m.rsf"
    tmp_m_adj = "tmp_dot_tmp_m_adj.rsf"
else:
    tmp_m = m
    tmp_m_adj = m_adj

forward1 = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (m,tmp_m,pa,pb,pc,pd) 
forward2 = "%s -np 1 \
~/rsf/bin/sfmpiwem_poynting \
adj=n infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (MPIRUN,tmp_m,d_fwd,v,wav,nz,dz,oz,nt,dt,ot,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)

adjoint1 = "%s -np 1 \
~/rsf/bin/sfmpiwem_poynting \
adj=y infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (MPIRUN,d,tmp_m_adj,v,wav,nz,dz,oz,nt,dt,ot,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)
adjoint2 = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_m_adj,m_adj,pa,pb,pc,pd) 

dot1 = "~/rsf/bin/sfinnerprod in1=%s in2=%s" %(d,d_fwd)
dot2 = "~/rsf/bin/sfinnerprod in1=%s in2=%s" %(m,m_adj)

print >> sys.stderr, "Adjoint..."
os.system(adjoint1)
if (reg):
    os.system(adjoint2)

print >> sys.stderr, "Forward..."
if (reg):
    os.system(forward1)
os.system(forward2)

print >> sys.stderr, "Inner product 1:"
os.system(dot1)
print >> sys.stderr, "Inner product 2:"
os.system(dot2)

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_dot_*.rsf"
os.system(cleanup)

print >> sys.stderr, "Done."

