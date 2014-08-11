#!/usr/bin/env python
''' least squares shot profile wave equation migration of isotropic 2C data. Inversion follows algorithm 2 of Scales, 1987.
References:   
Scales, John A. "Tomographic inversion via the conjugate gradient method." Geophysics 52.2 (1987): 179-185.
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
ux   = par.string("ux", "ux.rsf")
uz   = par.string("uz", "uz.rsf")
mpp   = par.string("mpp", "mpp.rsf")
mps   = par.string("mps", "mps.rsf")
vp   = par.string("vp", "vp.rsf")
vs   = par.string("vs", "vs.rsf")
wav = par.string("wav", "wav.rsf")
misfit_name = par.string("misfit", "misfit.rsf")
np  = par.int("np",1)
npersocket = par.int("npersocket",1)
niter  = par.int("niter",1) 
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

misfit_file = rsf.Output(misfit_name)
misfit = numpy.zeros(niter)

r1 = "tmp_cg_r1.rsf"
r2 = "tmp_cg_r2.rsf"
g1 = "tmp_cg_g1.rsf"
g2 = "tmp_cg_g2.rsf"
s1 = "tmp_cg_s1.rsf"
s2 = "tmp_cg_s2.rsf"
ss1 = "tmp_cg_ss1.rsf"
ss2 = "tmp_cg_ss2.rsf"
wd = "tmp_cg_wd.rsf"                                              

forward = "/usr/lib64/openmpi/bin/mpiexec -np %d \
/home/kstanton/rsf/bin/sfshotwem \
adj=n ux=%s uz=%s mpp=%s mps=%s vp=%s vs=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,ss1,ss2,s1,s2,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

adjoint = "/usr/lib64/openmpi/bin/mpiexec -np %d \
/home/kstanton/rsf/bin/sfshotwem \
adj=y ux=%s uz=%s mpp=%s mps=%s vp=%s vs=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f" % (np,r1,r2,g1,g2,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax)

# set up arrays for CG
sampling_calculate(ux,wd)
cmd1 = "/home/kstanton/rsf/bin/sfcp %s %s" % (ux,r1)
cmd2 = "/home/kstanton/rsf/bin/sfcp %s %s" % (uz,r2)
call(cmd1,shell=True)
call(cmd2,shell=True)

call(adjoint,shell=True)
cmd1 = "/home/kstanton/rsf/bin/sfcp %s %s" % (g1,s1)
cmd2 = "/home/kstanton/rsf/bin/sfcp %s %s" % (g2,s2)
call(cmd1,shell=True)
call(cmd2,shell=True)
cmd1 = "/home/kstanton/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g1,mpp)
cmd2 = "/home/kstanton/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g2,mps)
call(cmd1,shell=True)
call(cmd2,shell=True)
gamma = innerprod(g1) + innerprod(g2)
call(forward,shell=True)
sampling_apply(ss1,wd)
sampling_apply(ss2,wd)

for iter in range(1,niter+1):
    delta = innerprod(ss1) + innerprod(ss2)
    alpha = gamma/delta
    cgupdate(mpp,s1,1,alpha)    # mpp = mpp + alpha*s1
    cgupdate(mps,s2,1,alpha)    # mps = mps + alpha*s2
    cgupdate(r1,ss1,1,-alpha)   # r1 = r1 - alpha*ss1     
    cgupdate(r2,ss2,1,-alpha)   # r2 = r2 - alpha*ss2     
    misfit[iter-1] = innerprod(r1) + innerprod(r2)
    print >> sys.stderr, "misfit=", misfit 
    call(adjoint,shell=True)
    gamma_old = gamma
    gamma = innerprod(g1) + innerprod(g2)
    beta = gamma/gamma_old
    cgupdate(s1,g1,beta,1)      # s1 = beta*s1 + g1
    cgupdate(s2,g2,beta,1)      # s2 = beta*s2 + g2
    call(forward,shell=True)
    sampling_apply(ss1,wd)
    sampling_apply(ss2,wd)

misfit_file.put('n1',niter)
misfit_file.put('o1',1)
misfit_file.put('d1',1)
misfit_file.put('label1','Iteration')
misfit_file.put('unit1','')
misfit_file.put('title','Misfit')
misfit_file.write(misfit)
misfit_file.close()

# Clean up temporary files
cleanup = "/home/kstanton/rsf/bin/sfrm tmp_cg_*.rsf"
call(cleanup,shell=True)

print >> sys.stderr, "Done."

