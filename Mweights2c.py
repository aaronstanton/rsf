#!/usr/bin/env python
''' Creat weighting function for 2 component images (mpp and mps). Bounded by (0,1). Low values are assigned to regions of the image that are dissimilar, and values near to 1 are assigned to regions of the images that are similar.
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

def weights_calculate(filenamein1,filenamein2,p,filenameout):
    tmp_wm1 = "tmp_wm1.rsf"
    tmp_wm2 = "tmp_wm2.rsf"
    inp1 = rsf.Input(filenamein1)
    inp2 = rsf.Input(filenamein2)
    output1 = rsf.Output(tmp_wm1)
    assert 'float' == inp1.type
    n1 = inp1.int('n1')
    n2 = inp1.int('n2')
    n3 = inp1.int('n3')
    o1 = inp1.float('o1')
    o2 = inp1.float('o2')
    o3 = inp1.float('o3')
    d1 = inp1.float('d1')
    d2 = inp1.float('d2')
    d3 = inp1.float('d3')
    assert n1
    assert n2
    assert n3
    trace1 = numpy.zeros(n1,'f')
    trace2 = numpy.zeros(n1,'f')
    wm = numpy.zeros(n1,'f')
    output1.put('n1',n1)
    output1.put('d1',d1)
    output1.put('o1',o1)
    output1.put('n2',n2)
    output1.put('d2',d2)
    output1.put('o2',o2)
    output1.put('n3',n3)
    output1.put('d3',d3)
    output1.put('o3',o3)
    
    # open input files and calculate weights
    inp1 = rsf.Input(filenamein1)
    inp2 = rsf.Input(filenamein2)
    for i2 in xrange(n2*n3):
        inp1.read(trace1)
        inp2.read(trace2)
        a1 = numpy.percentile(numpy.fabs(trace1), p, axis=None)        
        a2 = numpy.percentile(numpy.fabs(trace2), p, axis=None)        
        for i1 in xrange(n1):
            wm[i1] = 1 - 1/(1 + math.pow((trace1[i1]*trace2[i1])/(a1*a2),2))
    	output1.write(wm)
    inp1.close()
    inp2.close()
    output1.close()
    # Smooth the weights using a triangle filter
    cmd = "~/rsf/bin/sfsmooth < %s > %s rect1=%d rect2=%d" % (tmp_wm1,tmp_wm2,20,5)
    call(cmd,shell=True)

    # open weight file and apply some fuzzy logic to stabilize
    output2 = rsf.Output(tmp_wm1)
    output2.put('n1',n1)
    output2.put('d1',d1)
    output2.put('o1',o1)
    output2.put('n2',n2)
    output2.put('d2',d2)
    output2.put('o2',o2)
    output2.put('n3',n3)
    output2.put('d3',d3)
    output2.put('o3',o3)
    inp1 = rsf.Input(tmp_wm2)
    for i2 in xrange(n2*n3):
        inp1.read(wm)
        max = numpy.amax(wm)
        for i1 in xrange(n1):
            if (wm[i1] >= 0.8*max):
                wm[i1] = 1
            else:
                wm[i1] = math.atan(wm[i1])
        output2.write(wm)
    inp1.close()
    output2.close()


    # Smooth the weights using a triangle filter
    cmd = "~/rsf/bin/sfsmooth < %s > %s rect1=%d rect2=%d" % (tmp_wm1,tmp_wm2,5,5)
    call(cmd,shell=True)

    # Stack and spray the weights along the angle axis
    cmd = "~/rsf/bin/sfstack < %s axis=3 | ~/rsf/bin/sfspray axis=3 n=%d o=%f d=%f > %s" % (tmp_wm2,n3,o3,d3,filenameout)
    call(cmd,shell=True)
    # Clean up temporary files
    cleanup = "~/rsf/bin/sfrm tmp_wm*.rsf"
    call(cleanup,shell=True)
  
# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
mpp = par.string("mpp", "mpp.rsf")  # PP image input file
mps = par.string("mps", "mps.rsf")  # PS image input file
wm  = par.string("w", "w.rsf")      # file to output weights
p  = par.float("p",80) # percentile used to calculate quantile of amplitude for scale factor applied to each sample prior to computing weight.
weights_calculate(mpp,mps,p,wm)


print >> sys.stderr, "Done."


