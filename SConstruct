import os, sys, re, string
sys.path.append('../../framework')
import bldutil

progs = ''' 
syn5d
'''

try:  # distributed version
    Import('env root pkgdir bindir')
    env = env.Clone()
except: # local version
    env = bldutil.Debug()
    root = None
    SConscript('../../system/seismic/SConstruct')

src = Glob('[a-z]*.c')

env.Prepend(CPPPATH=['../../include'],
            LIBPATH=['../../lib'],
            LIBS=[env.get('DYNLIB','')+'rsf',
                  env.get('DYNLIB','')+'rsfsegy'])

for source in src:
    inc = env.RSF_Include(source,prefix='')
    obj = env.StaticObject(source)
    env.Depends(obj,inc)

mains = Split(progs)
for prog in mains:
    sources = ['M' + prog]
    bldutil.depends(env,sources,'M'+prog)
    prog = env.Program(prog,map(lambda x: x + '.c',sources))
    if root:
        env.Install(bindir,prog)

######################################################################
# SELF-DOCUMENTATION
######################################################################
if root:
    user = os.path.basename(os.getcwd())
    main = 'sf%s.py' % user
    
    docs = map(lambda prog: env.Doc(prog,'M' + prog),mains)
    env.Depends(docs,'#/framework/rsf/doc.py')	

    doc = env.RSF_Docmerge(main,docs)
    env.Install(pkgdir,doc)

