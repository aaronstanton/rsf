#!/usr/bin/env python
 
import numpy, sys, pylab
import rsf.api as rsf
 
par  = rsf.Par()
fin  = rsf.Input()
assert 'float' == fin.type
 
n1 = fin.int('n1')
n2 = fin.int('n2')
assert n1
assert n2

print >> sys.stderr, "n1=" + str(n1)
print >> sys.stderr, "n2=" + str(n2)

imageout = par.string("imageout"   ,"test.pdf") # File for output image

data = numpy.zeros((n2,n1),'f')
fin.read(data)
numpy.clip(data + 0.0001*numpy.random.randn(n2,n1),-0.001,0.001,out=data)
pylab.imshow(data.T)
pylab.set_cmap('gray')
pylab.axis('tight')
pylab.xlabel('distance (m)')
pylab.ylabel('time (s)')
pylab.title('test of the python api')
pylab.savefig(imageout,dpi=300,format='pdf')
