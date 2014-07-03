#!/usr/bin/env python
''' test running a subprocess call within python  

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
   
# Initialize RSF command line parser    
par = rsf.Par()
# Read command line variables
verbose  = par.bool("verbose",True)

call(dedent("""\
    #!/bin/bash
    echo Hello world
    """), shell=True)

