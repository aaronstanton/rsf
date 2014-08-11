#!/usr/bin/env python
''' dot product test for sfwavsep

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
d1     = par.string("d1", "d1.rsf")
d2     = par.string("d2", "d2.rsf")
m1     = par.string("m1", "m1.rsf")
m2     = par.string("m2", "m2.rsf")
vp     = par.string("vp", "vp.rsf")
vs     = par.string("vs", "vs.rsf")
mode   = par.int("mode", 1)
d1_fwd = "tmp_d1_fwd.rsf"
d2_fwd = "tmp_d2_fwd.rsf"
m1_adj = "tmp_m1_adj.rsf"
m2_adj = "tmp_m2_adj.rsf"

forward = "~/rsf/bin/sfwavsep mode=%d adj=n in1=%s in2=%s out1=%s out2=%s vp=%s vs=%s verbose=n" % (mode,m1,m2,d1_fwd,d2_fwd,vp,vs)
adjoint = "~/rsf/bin/sfwavsep mode=%d adj=y in1=%s in2=%s out1=%s out2=%s vp=%s vs=%s verbose=n" % (mode,d1,d2,m1_adj,m2_adj,vp,vs)

dot1 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(d1,d2,d1_fwd,d2_fwd)
dot2 = "~/rsf/bin/sfinnerprod2 in1a=%s in1b=%s in2a=%s in2b=%s" %(m1,m2,m1_adj,m2_adj)

print >> sys.stderr, "Forward..."
os.system(forward)
print >> sys.stderr, "Adjoint..."
os.system(adjoint)
print >> sys.stderr, "Inner product 1:"
os.system(dot1)
print >> sys.stderr, "Inner product 2:"
os.system(dot2)

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_d1_fwd.rsf tmp_d2_fwd.rsf tmp_m1_adj.rsf tmp_m2_adj.rsf"
os.system(cleanup)

print >> sys.stderr, "Done."

