import os, sys, re, string
sys.path.append('../../framework')
import bldutil

progs = '''
syn5d
linearevents5d
geom5d
pad5d
reg5d
taper5d
merge5d
decimate
static
pstm
velpstm
zowem
velwem
spbpdn
wavsep
fkfilter
convert_to_angle
velconvert
gxsxtomxhx
innerprod
innerprod2
'''

mpi_progs = '''
mpiwem
mpiwem_poynting
mpiewem
mpiewem_poynting
mpiewem_sep
'''

pyprogs='''
imagesc
svd_complex
svd_complexblk
dotwem
dotwem_poynting
dotewem
dotewem_poynting
lswem
lswem_poynting
lsewem
lsewem_sep
dotwavsep
weights2c
'''

pymods='''
'''

try: # distributed version
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

mpicc = env.get('MPICC')
for source in src:
    inc = env.RSF_Include(source,prefix='')
    obj = env.StaticObject(source)
    env.Depends(obj,inc)

fftw = env.get('FFTW')
if fftw:
    progs += ''
elif root:
    place += ''
    for prog in Split('zomiso'):
        prog = env.RSF_Place('sf'+prog,None,var='FFTW',package='fftw')
        env.Install(bindir,prog)

mains = Split(progs)
for prog in mains:
    sources = ['M' + prog]
    bldutil.depends(env,sources,'M'+prog)
    env.StaticObject('M'+prog+'.c')
    prog = env.Program(prog,map(lambda x: x + '.o',sources))
    if root:
        env.Install(bindir,prog)

mpi_mains = Split(mpi_progs)
if mpicc and fftw:
    for prog in mpi_mains:
        sources = ['M' + prog]
        bldutil.depends(env,sources,'M'+prog)
        env.StaticObject('M'+prog+'.c',CC=mpicc)
    	prog = env.Program(prog,map(lambda x: x + '.o',sources),CC=mpicc)
        if root:
            env.Install(bindir,prog)
else:
    place += mpi_progs
    for prog in mpi_mains:
	prog = env.RSF_Place('sf'+prog,None,var='MPICC',package='mpi')
        if root:
            env.Install(bindir,prog)

######################################################################
# PYTHON METAPROGRAMS (python API not needed)
######################################################################

if root: # no compilation, just rename
    pymains = Split(pyprogs)
    exe = env.get('PROGSUFFIX','')
    for prog in pymains:
        env.InstallAs(os.path.join(bindir,'sf'+prog+exe),'M'+prog+'.py')
    for mod in Split(pymods):
        env.Install(pkgdir,mod+'.py')

######################################################################
# SELF-DOCUMENTATION
######################################################################
if root:
    user = os.path.basename(os.getcwd())
    main = 'sf%s.py' % user
    
    docs = map(lambda prog: env.Doc(prog,'M' + prog),mains+mpi_mains) + \
           map(lambda prog: env.Doc(prog,'M'+prog+'.py',lang='python'),pymains)
    env.Depends(docs,'#/framework/rsf/doc.py')	

    doc = env.RSF_Docmerge(main,docs)
    env.Install(pkgdir,doc)
