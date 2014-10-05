#!/usr/bin/env python
''' least squares shot profile wave equation migration of isotropic 2C data that has already been separated into P and S waves. Inversion follows algorithm 2 of Scales, 1987.
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
    inp1.close()
    inp2.close()
   
# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
ux   = par.string("ds", "ds.rsf")
uz   = par.string("dp", "dp.rsf")
mpp   = par.string("mpp", "mpp.rsf")
mps   = par.string("mps", "mps.rsf")
vp   = par.string("vp", "vp.rsf")
vs   = par.string("vs", "vs.rsf")
wav = par.string("wav", "wav.rsf")
misfit_name = par.string("misfit", "misfit.rsf")
np  = par.int("np",1)
npersocket = par.int("npersocket",1)
numthreads = par.int("numthreads",1)
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
sz  = par.float("sz",0)
gz  = par.float("gz",0)
reg  = par.bool("reg",False)
pa  = par.float("pa",-100)
pb  = par.float("pb",-50)
pc  = par.float("pc",50)
pd  = par.float("pd",100)

misfit_file = rsf.Output(misfit_name)
misfit = numpy.zeros(niter)

r_p = "tmp_cg_r_p.rsf"
r_s = "tmp_cg_r_s.rsf"
g_p = "tmp_cg_g_p.rsf"
g_s = "tmp_cg_g_s.rsf"
if (reg):
    tmp_g_p = "tmp_cg_tmp_g_p.rsf"
    tmp_g_s = "tmp_cg_tmp_g_s.rsf"
else:
    tmp_g_p = g_p
    tmp_g_s = g_s
s_p = "tmp_cg_s_p.rsf"
s_s = "tmp_cg_s_s.rsf"
if (reg):
    tmp_s_p = "tmp_cg_tmp_s_p.rsf"
    tmp_s_s = "tmp_cg_tmp_s_s.rsf"
else:
    tmp_s_p = s_p
    tmp_s_s = s_s
ss_p = "tmp_cg_ss_p.rsf"
ss_s = "tmp_cg_ss_s.rsf"
wd = "tmp_cg_wd.rsf"                                              

forward1a = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (s_s,tmp_s_s,pa,pb,pc,pd) 
forward1b = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (s_s,tmp_s_s,pa,pb,pc,pd) 
forward2 = "%s -np %d \
~/rsf/bin/sfmpiewem_sep \
adj=n dp=%s ds=%s mpp=%s mps=%s vp=%s vs=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (MPIRUN,np,ss_p,ss_s,tmp_s_p,tmp_s_s,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)

adjoint1 = "%s -np %d \
~/rsf/bin/sfmpiewem_sep \
adj=y dp=%s ds=%s mpp=%s mps=%s vp=%s vs=%s wav=%s verbose=n nz=%d dz=%f oz=%f \
nt=%d dt=%f ot=%f \
nhx=%d dhx=%f ohx=%f \
npx=%d dpx=%f opx=%f \
nsx=%d dsx=%f osx=%f \
fmin=%f fmax=%f \
sz=%f gz=%f" % (MPIRUN,np,r_p,r_s,tmp_g_p,tmp_g_s,vp,vs,wav,nz,dz,oz,nt,dt,ot,nhx,dhx,ohx,npx,dpx,opx,nsx,dsx,osx,fmin,fmax,sz,gz)
adjoint2a = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_g_p,g_p,pa,pb,pc,pd) 
adjoint2b = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (tmp_g_s,g_s,pa,pb,pc,pd) 

# set up arrays for CG
sampling_calculate(ux,wd)
cmd1 = "~/rsf/bin/sfcp %s %s" % (ux,r_s)
cmd2 = "~/rsf/bin/sfcp %s %s" % (uz,r_p)
call(cmd1,shell=True)
call(cmd2,shell=True)

call(adjoint1,shell=True)
if (reg):
    call(adjoint2a,shell=True)
    call(adjoint2b,shell=True)
cmd1 = "~/rsf/bin/sfcp %s %s" % (g_p,s_p)
cmd2 = "~/rsf/bin/sfcp %s %s" % (g_s,s_s)
call(cmd1,shell=True)
call(cmd2,shell=True)
cmd1 = "~/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g_p,mpp)
cmd2 = "~/rsf/bin/sfmath g=%s output=\'g*0\' > %s" % (g_s,mps)
call(cmd1,shell=True)
call(cmd2,shell=True)
gamma1 = innerprod(g_p)
gamma2 = innerprod(g_s)
if (reg):
    call(forward1a,shell=True)
    call(forward1b,shell=True)
call(forward2,shell=True)
sampling_apply(ss_p,wd)
sampling_apply(ss_s,wd)

for iter in range(1,niter+1):
    delta1 = innerprod(ss_p)
    delta2 = innerprod(ss_s)
    alpha1 = gamma1/delta1
    alpha2 = gamma2/delta2
    cgupdate(mpp,s_p,1,alpha1)    # mpp = mpp + alpha*s1
    cgupdate(mps,s_s,1,alpha2)    # mps = mps + alpha*s2
    cgupdate(r_p,ss_p,1,-alpha1)   # r1 = r1 - alpha*ss1     
    cgupdate(r_s,ss_s,1,-alpha2)   # r2 = r2 - alpha*ss2     
    misfit[iter-1] = innerprod(r_p) + innerprod(r_s)
    print >> sys.stderr, "misfit=", misfit[iter-1] 
    call(adjoint1,shell=True)
    if (reg):
        call(adjoint2a,shell=True)
        call(adjoint2b,shell=True)
    gamma1_old = gamma1
    gamma2_old = gamma2
    gamma1 = innerprod(g_p)
    gamma2 = innerprod(g_s)
    beta1 = gamma1/gamma1_old
    beta2 = gamma2/gamma2_old
    cgupdate(s_p,g_p,beta1,1)      # s1 = beta*s1 + g1
    cgupdate(s_s,g_s,beta2,1)      # s2 = beta*s2 + g2
    if (reg):
        call(forward1a,shell=True)
        call(forward1b,shell=True)
    call(forward2,shell=True)
    sampling_apply(ss_p,wd)
    sampling_apply(ss_s,wd)

if (reg):
    # Apply regularization to mpp and mps 
    cmd = "~/rsf/bin/sfcp %s %s" % (mpp,g_p)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (g_p,mpp,pa,pb,pc,pd) 
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sfcp %s %s" % (mps,g_s)
    call(cmd,shell=True)
    cmd = "~/rsf/bin/sffkfilter axis=3 < %s > %s pa=%f pb=%f pc=%f pd=%f" % (g_s,mps,pa,pb,pc,pd)
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

