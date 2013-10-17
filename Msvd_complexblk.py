#!/usr/bin/env python
''' Perform SVD on a block of complex-valued matrices using SCIPY.  

REQUIRES the PYTHON API, NUMPY AND SCIPY
'''

# Import RSF API
try:
    import rsf.api as rsf
    import sys, numpy, scipy
except Exception, e:
    print \
'''ERROR: NEED PYTHON API, NUMPY, SCIPY '''
    sys.exit(1)

# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
vectors = par.bool  ("vectors", False     ) # Output singular vectors?
left    = par.string("left"   ,"left.rsf") # File to store left singular vectors
right   = par.string("right"  ,"right.rsf") # File to store right singular vectors
# Declare input and outputs
fin = rsf.Input()    # no argument means stdin
fout = rsf.Output()  # no argument means stdout
# Declare optional inputs/outputs
if vectors:
    lout = rsf.Output(left)  # left singular vectors
    rout = rsf.Output(right) # right singular vectors

# Get dimensions of input header or output header
n1 = fin.int('n1')
n2 = fin.int('n2')
n3 = fin.int('n3')

data = numpy.zeros((n3,n2,n1),'complex64') # Note, we reverse array dims

# Read our input data
fin.read(data)
svals = numpy.zeros((n3,n2,n1),'complex64') 
lvec = numpy.zeros((n3,n2,n1),'complex64') 
rvec = numpy.zeros((n3,n2,n1),'complex64') 

for i1 in range(0, n1):
  # Perform our SVD
  u,l,v = numpy.linalg.svd(numpy.squeeze(data[:,:,i1]))
  svals[:,:,i1] = numpy.diag(l)
  lvec[:,:,i1] = u
  rvec[:,:,i1] = v
  
print svals.shape

# Setup output headers
fout.put('n1',n1)
fout.put('n2',n2)
fout.put('n3',n3)
fout.put('o1',1)
fout.put('o2',1)
fout.put('o3',1)
fout.put('d1',1)
fout.put('d2',1)
fout.put('d3',1)

# Write output data
fout.write(svals)

if vectors:
    lout.put('n1',n1)
    lout.put('n2',n2)
    lout.put('n3',n3)
    rout.put('n1',n1)
    rout.put('n2',n2)
    rout.put('n3',n3)
    lout.put('o1',1)
    lout.put('o2',1)
    lout.put('o3',1)
    rout.put('o1',1)
    rout.put('o2',1)
    rout.put('o3',1)
    lout.put('d1',1)
    lout.put('d2',1)
    lout.put('d3',1)
    rout.put('d1',1)
    rout.put('d2',1)
    rout.put('d3',1)
    lout.write(lvec)
    rout.write(rvec)
    lout.close()
    rout.close()

# Clean up files
fout.close()
fin.close()
