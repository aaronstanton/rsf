#!/usr/bin/env python
''' least squares prestack Kirchhoff time migration. Inversion follows algorithm 2 of Scales, 1987.
References:   
Scales, John A. "Tomographic inversion via the conjugate gradient method." Geophysics 52.2 (1987): 179-185.
REQUIRES the PYTHON API and NUMPY and SCIPY
'''

# Import RSF API
try:
    import rsf.api as rsf
    import os, sys, numpy, scipy
    from subprocess import call
    from textwrap import dedent
    import math
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY '''
    sys.exit(1)

# get environmental variables (such as MPIRUN) from config.py
execfile(os.environ['HOME'] +"/rsf/src/config.py")

def innerprod(filename):
    inp  = rsf.Input(filename)
    assert 'float' == inp.type
    n1 = inp.int('n1')
    n2 = inp.size(1)
    assert n1
#    print >> sys.stderr, "n1=" , n1
#    print >> sys.stderr, "n2=" , n2
    trace = numpy.zeros(n1,'f')
    sum = 0.0
    for i2 in xrange(n2): # loop over traces
        inp.read(trace)
        a = numpy.linalg.norm(trace,2)
        sum += a*a
    inp.close()
    return sum;

def sampling_calculate(filenamein,filenameout):
    inp = rsf.Input(filenamein)
    output = rsf.Output(filenameout)
    assert 'float' == inp.type
    n1 = inp.int('n1')
    n2 = inp.size(1)
    assert n1
    assert n2
    trace = numpy.zeros(n1,'f')
    wd = numpy.zeros(n2,'f')
    output.put('n1',n2)
    output.put('d1',1)
    output.put('o1',0)
    for i2 in xrange(n2):
        inp.read(trace)
        sum = 0.0
        for i1 in xrange(n1):
            sum += trace[i1]*trace[i1]
        if (sum > 0):
            wd[i2] = 1.0
        else:
            wd[i2] = 0.0
    output.write(wd)
    inp.close()
    output.close()

def sampling_apply(filenamein_d,filenamein_wd):
    inp1 = rsf.Input(filenamein_d)
    inp2 = rsf.Input(filenamein_wd)
    filenametmp = "tmp_cg_samp.rsf"
    output =  rsf.Output(filenametmp)
    assert 'float' == inp1.type
    assert 'float' == inp2.type
    n1 = inp1.int('n1')
    n2 = inp1.int('n2')
    n3 = inp1.int('n3')
    assert n1
    assert n2
    assert n3
    d1 = inp1.float('d1')
    d2 = inp1.float('d2')
    d3 = inp1.float('d3')
    o1 = inp1.float('o1')
    o2 = inp1.float('o2')
    o3 = inp1.float('o3')
    trace = numpy.zeros(n1,'f')
    wd = numpy.zeros(n2*n3,'f')
    output.put('n1',n1)
    output.put('n2',n2)
    output.put('n3',n3)
    output.put('d1',d1)
    output.put('d2',d2)
    output.put('d3',d3)
    output.put('o1',o1)
    output.put('o2',o2)
    output.put('o3',o3)
    inp2.read(wd)
    for i2 in xrange(n2*n3):
        inp1.read(trace)
        for i1 in xrange(n1):
            trace[i1] = trace[i1]*wd[i2] 
        output.write(trace)
    inp1.close()
    inp2.close()
    output.close()
    cmd = "~/rsf/bin/sfcp %s %s" % (filenametmp,filenamein_d)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sfrm %s" % (filenametmp)
    call(cmd,shell=True)

def cgupdate(filenamein1,filenamein2,a,b):
    inp1 = rsf.Input(filenamein1)
    inp2 = rsf.Input(filenamein2)
    filenametmp = "tmp_cg_update.rsf"
    cmd = "~/rsf/bin/sfmath x=%s y=%s output=\'%f*x + %f*y\' > %s" % (filenamein1,filenamein2,a,b,filenametmp)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sfcp %s %s" % (filenametmp,filenamein1)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sfrm %s" % (filenametmp)
    call(cmd,shell=True)
   
# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
d   = par.string("d", "d.rsf")
m   = par.string("m", "m.rsf")
vp   = par.string("vp", "vp.rsf")
misfit_name = par.string("misfit", "misfit.rsf")
niter  = par.int("niter",1) 
reg  = par.bool("reg",False)
aperture  = par.float("aperture",3000)
pa  = par.float("pa",-100)
pb  = par.float("pb",-50)
pc  = par.float("pc",50)
pd  = par.float("pd",100)
misfit_file = rsf.Output(misfit_name)
misfit = numpy.zeros(niter)

r = "tmp_cg_r.rsf"
g = "tmp_cg_g.rsf"
if (reg):
    tmp_g = "tmp_cg_tmp_g.rsf"
else:
    tmp_g = g
s = "tmp_cg_s.rsf"
if (reg):
    tmp_s = "tmp_cg_tmp_s.rsf"
else:
    tmp_s = s
ss = "tmp_cg_ss.rsf"
wd = "tmp_cg_wd.rsf"                                              

forward1 = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (s,tmp_s,pa,pb,pc,pd) 
forward2 = "~/rsf/bin/sfpstm adj=n < %s > %s vp=%s ps=n aperture=%f verbose=n" % (tmp_s,ss,vp,aperture)
adjoint1 = "~/rsf/bin/sfpstm adj=y < %s > %s vp=%s ps=n aperture=%f verbose=n" % (r,tmp_g,vp,aperture)
adjoint2 = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_g,g,pa,pb,pc,pd) 

# set up arrays for CG
sampling_calculate(d,wd)
cmd = "~/rsf/bin/sfcp %s %s" % (d,r)
call(cmd,shell=True)

call(adjoint1,shell=True)
if (reg):
    call(adjoint2,shell=True)
cmd = "~/rsf/bin/sfcp %s %s" % (g,s)
call(cmd,shell=True)
cmd = "~/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g,m)
call(cmd,shell=True)
gamma = innerprod(g)
if (reg):
    call(forward1,shell=True)
call(forward2,shell=True)
sampling_apply(ss,wd)

for iter in range(1,niter+1):
    delta = innerprod(ss)
    alpha = gamma/delta
    cgupdate(m,s,1,alpha)    # m = m + alpha*s
    cgupdate(r,ss,1,-alpha)  # r = r - alpha*ss     
    misfit[iter-1] = innerprod(r)
    print >> sys.stderr, "misfit=", misfit[iter-1] 
    call(adjoint1,shell=True)
    if (reg):
        call(adjoint2,shell=True)
    gamma_old = gamma
    gamma = innerprod(g)
    beta = gamma/gamma_old
    cgupdate(s,g,beta,1)      # s = beta*s + g
    if (reg):
        call(forward1,shell=True)
    call(forward2,shell=True)
    sampling_apply(ss,wd)

if (reg):
    # Apply regularization to m 
    cmd = "~/rsf/bin/sfcp %s %s" % (m,g)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (g,m,pa,pb,pc,pd) 
    call(cmd,shell=True)

misfit_file.put('n1',niter)
misfit_file.put('o1',1)
misfit_file.put('d1',1)
misfit_file.put('label1','Iteration')
misfit_file.put('unit1','')
misfit_file.put('title','Misfit')
misfit_file.write(misfit)
misfit_file.close()

# Clean up temporary files
cleanup = "~/rsf/bin/sfrm tmp_cg_*.rsf"
call(cleanup,shell=True)

print >> sys.stderr, "Done."

