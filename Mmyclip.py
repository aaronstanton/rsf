#!/usr/bin/env python
 
import numpy, sys, pylab
import rsf.api as rsf
 
par = rsf.Par()
input  = rsf.Input()
output = rsf.Output()
assert 'float' == input.type
 
n1 = input.int('n1')
n2 = input.int('n2')
assert n1
 
clip = par.float("clip")
assert clip

data = numpy.zeros((n2,n1),'f')
input.read(data)
pylab.imshow(data)
pylab.savefig('test.png')
 
trace = numpy.zeros(n1,'f')
 
for i2 in xrange(n2): # loop over traces
    input.read(trace)
    trace = numpy.clip(trace,-clip,clip)
    output.write(trace)


