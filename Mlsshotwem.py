#!/usr/bin/env python
''' least squares shot profile wave equation migration  

REQUIRES the PYTHON API and NUMPY and SCIPY
'''

# Import RSF API
try:
    import rsf.api as rsf
    import sys, numpy, scipy
    from subprocess import call
    from textwrap import dedent
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY '''
    sys.exit(1)

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
    cmd = "/home/kstanton/rsf/bin/sfcp %s %s" % (filenametmp,filenamein_d)
    call(cmd,shell=True)
    cmd = "/home/kstanton/rsf/bin/sfrm %s" % (filenametmp)
    call(cmd,shell=True)

def cgupdate(filenamein1,filenamein2,a,b):
    inp1 = rsf.Input(filenamein1)
    inp2 = rsf.Input(filenamein2)
    filenametmp = "tmp_cg_update.rsf"
    cmd = "/home/kstanton/rsf/bin/sfmath x=%s y=%s output=\'%f*x + %f*y\' > %s" % (filenamein1,filenamein2,a,b,filenametmp)
    call(cmd,shell=True)
    cmd = "/home/kstanton/rsf/bin/sfcp %s %s" % (filenametmp,filenamein1)
    call(cmd,shell=True)
    cmd = "/home/kstanton/rsf/bin/sfrm %s" % (filenametmp)
    call(cmd,shell=True)
   
# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
d   = par.string("d", "d.rsf")
m   = par.string("m", "m.rsf")
vp   = par.string("vp", "vp.rsf")
wav = par.string("wav", "wav.rsf")
np  = par.int("np",1)
niter  = par.int("niter",1) 
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

r = "tmp_cg_r.rsf"
g = "tmp_cg_g.rsf"
s = "tmp_cg_s.rsf"
ss = "tmp_cg_ss.rsf"
wd = "tmp_cg_wd.rsf"                                              

forward = "/usr/lib64/openmpi/bin/mpiexec -np %d --npersocket %d \
/home/kstanton/rsf/bin/sfshotwem \
adj=n infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,npersocket,s,ss,vp,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

adjoint = "/usr/lib64/openmpi/bin/mpiexec -np %d --npersocket %d \
/home/kstanton/rsf/bin/sfshotwem \
adj=y infile=%s outfile=%s vp=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,npersocket,r,g,vp,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

# set up arrays for CG
sampling_calculate(d,wd)
cmd = "/home/kstanton/rsf/bin/sfcp %s %s" % (d,r)
call(cmd,shell=True)

call(adjoint,shell=True)
cmd = "/home/kstanton/rsf/bin/sfcp %s %s" % (g,s)
call(cmd,shell=True)
cmd = "/home/kstanton/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g,m)
call(cmd,shell=True)
gamma = innerprod(g)
gamma_old = gamma

print >> sys.stderr, "gamma=", gamma

for iter in range(1,niter+1):
    call(forward,shell=True)
    sampling_apply(ss,wd)
    delta = innerprod(ss)
    print >> sys.stderr, "delta=", delta
    alpha = gamma/(delta + 0.0000001)
    print >> sys.stderr, "alpha=", alpha
    cgupdate(m,s,1,alpha)   # m = m + alpha*s
    cgupdate(r,ss,1,-alpha) # r = r - alpha*ss     
    sampling_apply(r,wd)
    misfit = innerprod(r)
    print >> sys.stderr, "misfit=", misfit 
    call(adjoint,shell=True)
    gamma = innerprod(g)
    print >> sys.stderr, "gamma=", gamma
    beta = gamma/(gamma_old + 0.0000001)
    print >> sys.stderr, "beta=", beta
    gamma_old = gamma
    cgupdate(s,g,beta,1)      # s = beta*s + g

# Clean up temporary files
cleanup = "/home/kstanton/rsf/bin/sfrm tmp_cg_*.rsf"
call(cleanup,shell=True)

print >> sys.stderr, "Done."

