import sys,os,os.path,re
from subprocess import Popen, PIPE, check_call
import datetime as dt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optparse import OptionParser

#---------------------------------------------------------
# Main
#
def main():
    parser = OptionParser()
    parser.add_option("-s", "--size", dest="size",
                      help="Problem size")
    parser.add_option("-r", "--samples", dest="samples", type="int",
                      help="Number of samples to take in each run")

    parser.add_option("-o", "--only", dest="only", type="string", action="append",
                      help="Run only the specified task")

    parser.add_option("--ncrit", dest="ncrit", type="int", help="Ncrit")

    (opts, args) = parser.parse_args()

    samples = opts.samples or 1

    if opts.size is None or re.match(r'^m(edium)?$', opts.size, re.I):
        size = {'mt_strong'      : 100000,
                'flatmpi_strong' : 100000,
                'hybrid_strong'  : 100000,
                'hybrid_weak'    : 100000,
                'p2p'            : 1000}
    elif re.match(r'^l(arge)?$', opts.size, re.I):
        size = {'mt_strong'      : 1000000,
                'flatmpi_strong' : 1000000,
                'hybrid_strong'  : 1000000,
                'hybrid_weak'    : 1000000,
                'p2p'            : 10000}
    elif re.match(r'^s(mall)?$', opts.size, re.I):
        size = {'mt_strong'      : 10000,
                'flatmpi_strong' : 10000,
                'hybrid_strong'  : 10000,
                'hybrid_weak'    : 10000,
                'p2p'            : 100}
        

    ncrit = opts.ncrit or 64
    samples = opts.samples or 5

    mt_strong(size['mt_strong'], ncrit, samples)
    flatmpi_strong(size['flatmpi_strong'], ncrit, samples)
    hybrid_strong(size['hybrid_strong'], ncrit, samples)
    hybrid_weak(size['hybrid_weak'], ncrit, samples)
    p2p(size['p2p'], samples)

    
#---------------------------------------------------------

class Mean(object):
    def __init__(self):
        self._vals = []

    def append(self, val):
        val = float(val)
        self._vals.append(val)

    def __iadd__(self, val):
        self.append(val)
        return self

    def mean(self):
        return sum(self._vals) / len(self._vals)

    def max(self):
        return max(self._vals)


    def min(self):
        return min(self._vals)

def formatWithCommads(value):
    s = str(value)
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ",".join(reversed(groups))

Now = dt.datetime.now()
TakenAt = Now.strftime("on %Y/%m/%d %H:%M:%S")

if "PBS_JOBID" in os.environ:
    RUN_ID = "%s-%s" % (Now.strftime("%Y%m%d-%H%M%S"), os.environ["PBS_JOBID"])
else:
    RUN_ID = Now.strftime("%Y%m%d-%H%M%S")
    
OUTPUT_DIR = os.getcwd()
DIR=os.path.dirname(__file__)
env = os.environ

#---------------------------------------------------------
# Get MPI version number
#---------------------------------------------------------

def get_mpi_version():
    mpicc = Popen("%s/run.sh which mpicc" % DIR, shell=True, stdout=PIPE).communicate()[0].strip()

    print mpicc

    m = re.search('/openmpi/([^/]+)/.*/bin/mpicc', mpicc)
    if m:
        return ("Open MPI", m.group(1))

    m = re.search(r'mvapich2/([^/]+)/.*/bin/mpicc', mpicc)
    if m:
        return ("MVAPICH", m.group(1))

    m = re.search(r'mpich2/([^/]+)/.*/bin/mpicc', mpicc)
    if m:
        return ("MPICH", m.group(1))

    raise Exception("Unknown MPI type or version: '%s'" % mpicc)

MPI_Type, MPI_Version = get_mpi_version()

print "MPI Type = '%s', Version = '%s'" % (MPI_Type, MPI_Version)

#---------------------------------------------------------
# Check the working set is clean
#---------------------------------------------------------
os.chdir(DIR)

try:
    check_call(["git", "diff", "--quiet"])
    check_call(["git", "diff", "--cached", "--quiet"])
except:
    sys.stderr.write("\nError: the working directory is not clean.\n\n")
    check_call(["git", "status", "-uno"])
    exit(1)

# Record the current commit
p = Popen("git show --oneline | head -n 1", shell=True, stdout=PIPE)
commit = p.communicate()[0].strip()

check_call("git checkout master", shell=True)

print "---------------------------------------------------"
check_call("git show --oneline | head -n 1", shell=True)
check_call("date", shell=True)
check_call("hostname", shell=True)
print "---------------------------------------------------"

os.chdir(OUTPUT_DIR)

#-------------------------------------------------------------------------------
#
# Measure P2P speed
#
#-------------------------------------------------------------------------------

def p2p(NB, NumSamples):
    Ncrit = NB + 1

    branches = ['prof0', 'prof1', 'prof2', 'prof3', 'master']
    labels = ['prof0', 'prof1', 'prof2', 'prof3', 'master@%s' % commit.split()[0]]
    means = [Mean() for b in branches]

    for i in range(0, len(branches)):
        br = branches[i]
    
        print "Switching to branch '%s'" % br
        check_call("git checkout %s" % br, shell=True, cwd=DIR)

        # build parallel_tapas
        p = Popen("%s/build.sh" % DIR, shell=True, stdout=PIPE, env=env)
        bin = p.communicate()[0].strip()

        for j in range(NumSamples):
            env['MYTH_WORKER_NUM'] = "1"
            cmd = ["mpiexec", "-n", "1", bin, "--numBodies", str(NB), "--ncrit", str(Ncrit)]
            p = Popen(["%s/run.sh" % DIR] + cmd, stdout=PIPE, env=env)
            out = p.communicate()[0]

            print "----------------------------------------------"
            print "# ", " ".join(cmd)
            print out
            print "----------------------------------------------"

            try:
                upw = re.search(r'Upward pass\s+:\s+([0-9.]+) s', out).group(1)
                trav = re.search(r'Traverse\s+:\s+([0-9.]+) s', out).group(1)
                dwn = re.search(r'Downward pass\s+:\s+([0-9.]+) s', out).group(1)
            except:
                print "Error: regex search failed."
                print out
                exit(1)

            means[i].append(trav)

    means = [m.mean() for m in means]

    print "P2P"
    print "N=", NB
    print "Ncrit=", Ncrit
    print "NumSamples=", NumSamples
    print "Taken at ", TakenAt
    print commit
    print labels
    print means

    fig, ax = plt.subplots()
    index = np.arange(len(means))
    bar_width = 0.35

    plt.style.use('ggplot')
    colors = [v['color'] for v in list(plt.rcParams['axes.prop_cycle'])]

    bars = plt.bar(index + bar_width/2 , means, bar_width,
                   color=colors[0],
                   label='Traversal')

    NBs = formatWithCommads(NB)
    plt.xlabel('Versions')
    plt.ylabel('Traversal Time [s]')
    plt.title("Tapas P2P time (Ncrit > NB) \n NB=%s, Ncrit=%d, mean of %d samples\n %s\ncommit:\"%s\"" % (NBs, Ncrit, NumSamples, TakenAt, commit),
              fontsize=10)
    plt.xticks(index + bar_width, labels)
    plt.legend()

    plt.tight_layout()

    save_fname = '%s-p2p.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    plt.savefig(save_path)
    plt.clf()

#-------------------------------------------------------------------------------
#
# Single process, multithread strong scaling
#
#-------------------------------------------------------------------------------

def mt_strong(NB, Ncrit, NumSamples):
    NumThreads = [1,2,3,4,5,6,7,8,9,10,11,12]

    check_call("git checkout master", shell=True, cwd=DIR)
    p = Popen("%s/build.sh" % DIR, shell=True, stdout=PIPE, env=env)
    bin = p.communicate()[0].strip()

    upw  = [Mean() for nt in NumThreads]
    trav = [Mean() for nt in NumThreads]
    dwn  = [Mean() for nt in NumThreads]
    tree = [Mean() for nt in NumThreads]
    total = [Mean() for nt in NumThreads]
    
    for i, nt in enumerate(NumThreads):
        print "Running parallel_tapas with MYTH_WORKER_NUM=%d" % nt

        for j in range(NumSamples):
            print "MYTH_WORKER_NUM=%d, run #%d" % (nt, j)
            env['MYTH_WORKER_NUM'] = str(nt)
            print "MYTH_WORKER_NUM = ", env['MYTH_WORKER_NUM']
            print "bin = ", bin
            cmd = ["mpiexec", "-n", "1", bin, "--numBodies", str(NB), "--ncrit", str(Ncrit)]
            p = Popen(["%s/run.sh" % DIR,] + cmd, stdout=PIPE, env=env)
            out = p.communicate()[0]
            
            print "----------------------------------------------"
            print '# ', " ".join(cmd)
            print out
            print "----------------------------------------------"

            try:
                upw[i]   += re.search(r'Upward pass\s+:\s+([0-9.]+) s', out).group(1)
                trav[i]  += re.search(r'Traverse\s+:\s+([0-9.]+) s', out).group(1)
                dwn[i]   += re.search(r'Downward pass\s+:\s+([0-9.]+) s', out).group(1)
                total[i] += re.search(r'Total FMM\s+:\s+([0-9.]+) s', out).group(1)
            except:
                print "Error: regex search failed."
                print out
                raise

            try:
                with open("tree_construction.csv") as f:
                    lines = f.readlines()
                    tree[i] += re.split(r'\s+', lines[1].strip())[1]
            except:
                print "Error in reading tree_construction.csv"
                raise

    index = np.arange(12)
    upw   = np.array([m.mean() for m in upw])
    trav  = np.array([m.mean() for m in trav])
    dwn   = np.array([m.mean() for m in dwn])
    tree  = np.array([m.mean() for m in tree])
    total = np.array([m.mean() for m in total])

    other = total - upw - trav - dwn - tree

    print "Single node, multithreading, strong scaling"
    print "NB = "
    print NB
    print "Ncrit = "
    print Ncrit
    print "NumThreads = "
    print NumThreads
    print "Upward = "
    print upw
    print "Traversal = "
    print trav
    print "Downward = "
    print dwn
    print "Tree = "
    print tree
    print "Total = "
    print total
    print "Other = "
    print other

    width = 0.35
    
    plt.style.use('ggplot')
    #plt.style.use('seaborn-whitegrid')

    NBs = formatWithCommads(NB)

    #with plt.style.context('fivethirtyeight'):
    colors = [v['color'] for v in list(plt.rcParams['axes.prop_cycle'])]
    p0 = plt.bar(index, tree,  width, color=colors[0]);
    p1 = plt.bar(index, upw,   width, bottom=tree,color=colors[1])
    p2 = plt.bar(index, trav,  width, bottom=tree+upw, color=colors[2])
    p3 = plt.bar(index, dwn,   width, bottom=tree+upw+trav, color=colors[3])
    p4 = plt.bar(index, other, width, bottom=tree+upw+trav+dwn, color=colors[4])
    plt.xlabel('MYTH_WORKER_NUM')
    plt.ylabel('Runtime [s]')
    plt.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc="upper right")
    plt.xticks(index + width/2., NumThreads)
    plt.title("Tapas runtime strong scaling / Single node / multithreaded\nNB=%s, Ncrit=%d, mean of %d samples\n %s\ncommit:\"%s\"" % (NBs, Ncrit, NumSamples, TakenAt, commit),
              fontsize=10)
    
    save_fname = '%s-mt-strong.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    plt.savefig(save_path)
    plt.clf()

#-------------------------------------------------------------------------------
#
# Single node, multiple process, single thread strong strong scaling
#
#-------------------------------------------------------------------------------

def flatmpi_strong(NB, Ncrit, NumSamples):
    MaxNumProc = 12
    NumProcs = range(1, MaxNumProc + 1)

    check_call("git checkout master", shell=True, cwd=DIR)
    p = Popen("%s/build.sh" % DIR, shell=True, stdout=PIPE, env=env)
    bin = p.communicate()[0].strip()

    upw  = [Mean() for nt in NumProcs]
    trav = [Mean() for nt in NumProcs]
    dwn  = [Mean() for nt in NumProcs]
    tree = [Mean() for nt in NumProcs]
    total = [Mean() for nt in NumProcs]

    for i, nproc in enumerate(NumProcs):
        print "Running parallel_tapas with NP=%d" % nproc

        for j in range(NumSamples):
            print "NP=%d, run #%d" % (nproc, j)
            env['MYTH_WORKER_NUM'] = "1"
            print "MYTH_WORKER_NUM = ", env['MYTH_WORKER_NUM']
            print "NP = %d" % nproc
            print "bin = ", bin
            cmd = ["mpiexec", "-np", str(nproc), bin, "--numBodies", str(NB), "--ncrit", str(Ncrit)]
            p = Popen(["%s/run.sh" % DIR] + cmd, stdout=PIPE, env=env)
            out = p.communicate()[0]

            print "----------------------------------------------"
            print '# ', " ".join(cmd)
            print out
            print "----------------------------------------------"

            try:
                upw[i]   += re.search(r'Upward pass\s+:\s+([0-9.]+) s', out).group(1)
                trav[i]  += re.search(r'Traverse\s+:\s+([0-9.]+) s', out).group(1)
                dwn[i]   += re.search(r'Downward pass\s+:\s+([0-9.]+) s', out).group(1)
                total[i] += re.search(r'Total FMM\s+:\s+([0-9.]+) s', out).group(1)
            except:
                print "Error: regex search failed."
                print "outout = >>>"
                print out
                print "<<<"
                raise

            try:
                with open("tree_construction.csv") as f:
                    lines = f.readlines()
                    tree[i] += re.split(r'\s+', lines[1].strip())[1]
            except:
                print "Error in reading tree_construction.csv"
                raise


    index = np.arange(MaxNumProc)
    upw   = np.array([m.mean() for m in upw])
    trav  = np.array([m.mean() for m in trav])
    dwn   = np.array([m.mean() for m in dwn])
    tree  = np.array([m.mean() for m in tree])
    total = np.array([m.mean() for m in total])

    other = total - upw - trav - dwn - tree

    print "Single node, multithreading, strong scaling"
    print "NB = "
    print NB
    print "Ncrit = "
    print Ncrit
    print "NumProcs = "
    print NumProcs
    print "Upward = "
    print upw
    print "Traversal = "
    print trav
    print "Downward = "
    print dwn
    print "Tree = "
    print tree
    print "Total = "
    print total
    print "Other = "
    print other

    width = 0.35

    plt.style.use('ggplot')
    #plt.style.use('seaborn-whitegrid')

    #with plt.style.context('fivethirtyeight'):
    colors = [v['color'] for v in list(plt.rcParams['axes.prop_cycle'])]
    NBs = formatWithCommads(NB)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    p0 = ax.bar(index, tree,  width, color=colors[0]);
    p1 = ax.bar(index, upw,   width, bottom=tree,color=colors[1])
    p2 = ax.bar(index, trav,  width, bottom=tree+upw, color=colors[2])
    p3 = ax.bar(index, dwn,   width, bottom=tree+upw+trav, color=colors[3])
    p4 = ax.bar(index, other, width, bottom=tree+upw+trav+dwn, color=colors[4])
    ax.set_xlabel('#Procs')
    ax.set_ylabel('Runtime [s]')
    #ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc="upper right",)
    ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    #ax.set_xticks(index, [str(n) for n in NumProcs])
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    ax.set_title("Tapas runtime strong scaling / Single node / Flat MPI\nNB=%s, Ncrit=%d, mean of %d samples\n %s\ncommit:\"%s\"" % (NBs, Ncrit, NumSamples, TakenAt, commit),
              fontsize=10)
    
    save_fname = '%s-flatmpi-strong.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()

    print os.path.abspath("tree_construction.csv")
    print "---------------------"
    with open("tree_construction.csv") as f:
        print f.read()
    print "---------------------"

    # Read the tree_construction.csv file generated by the last run (with the max NP)
    with open("tree_construction.csv") as f:
        data = [re.split(r'\s+', ln.strip()) for ln in f.readlines()[1:]]

        all  = np.array( [float(data[i][1]) for i in range(MaxNumProc)] )
        smpl = np.array( [float(data[i][2]) for i in range(MaxNumProc)] )
        exch = np.array( [float(data[i][3]) for i in range(MaxNumProc)] )
        glc  = np.array( [float(data[i][4]) for i in range(MaxNumProc)] )
        ggl  = np.array( [float(data[i][5]) for i in range(MaxNumProc)] )
        othr = all - smpl - exch - glc - ggl

        print "sample = "
        print smpl
        print "exchange = "
        print exch
        print "grow_local = "
        print glc
        print "grow_global = "
        print ggl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    bottom = np.array([0.0] * MaxNumProc)
    p0 = ax.bar(index, smpl, width, bottom=bottom, color=colors[0]); bottom += smpl
    p1 = ax.bar(index, exch, width, bottom=bottom, color=colors[1]); bottom += exch
    p2 = ax.bar(index, glc,  width, bottom=bottom, color=colors[2]); bottom += glc
    p3 = ax.bar(index, ggl,  width, bottom=bottom, color=colors[3]); bottom += ggl
    p4 = ax.bar(index, othr,  width, bottom=bottom, color=colors[4]); bottom += othr
    
    ax.set_xlabel('MPI rank')
    ax.set_ylabel('Runtime [s]')
    ax.legend([p0, p1, p2, p3, p4], ["Sample", "Exchange", "GrowLocal", "GrowGlobal", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    NBs = formatWithCommads(NB)
    label_str = [
        "Tapas tree const. breakdown #Proc=%d / Single node / Flat MPI" % MaxNumProc,
        "NB=%s, Ncrit=%d" % (NBs, Ncrit),
        "%s" % TakenAt,
        "commit:\"%s\"" % commit
    ]
    ax.set_title("\n".join(label_str), fontsize=10)
    
    save_fname = '%s-flatmpi-strong-tree.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()


#-------------------------------------------------------------------------------
#
# Multiple nodes, Hybrid strong scaling
#
#-------------------------------------------------------------------------------

def hybrid_strong(NB, Ncrit, NumSamples):
    if not "PBS_NODEFILE" in os.environ:
        print "ERROR: PBS_NODEFILE is not defined"
        return

    NODEFILE = os.environ["PBS_NODEFILE"]

    print "-----------------------------------------------------------"
    print "Multiple nodes, hybrid strong scaling"
    print "NB = ", NB
    print "Ncrit = ", Ncrit
    print "NumSamples = ", NumSamples
    print "NODEFILE = %s" % NODEFILE
    print "-----------------------------------------------------------"
    
    with open(NODEFILE) as f:
        MaxNumProc = len( [ln for ln in f.readlines() if len(ln.strip()) > 0] )
        
    NumProcs = range(1, MaxNumProc + 1)
    
    check_call("git checkout master", shell=True, cwd=DIR)
    p = Popen("%s/build.sh" % DIR, shell=True, stdout=PIPE, env=env)
    bin = p.communicate()[0].strip()

    upw  = [Mean() for nt in NumProcs]
    trav = [Mean() for nt in NumProcs]
    dwn  = [Mean() for nt in NumProcs]
    tree = [Mean() for nt in NumProcs]
    total = [Mean() for nt in NumProcs]

    for i, nproc in enumerate(NumProcs):
        print "Running parallel_tapas with NP=%d" % nproc

        for j in range(NumSamples):
            print "NP=%d, run #%d" % (nproc, j)
            env['MYTH_WORKER_NUM'] = "12"
            print "MYTH_WORKER_NUM = ", env['MYTH_WORKER_NUM']
            print "NP = %d" % nproc
            print "bin = ", bin
            # Open MPI specific
            cmd = ["mpiexec", "-np", str(nproc), "-hostfile", NODEFILE, "--map-by", "node", bin, "--numBodies", str(NB), "--ncrit", str(Ncrit)]
            p = Popen(["%s/run.sh" % DIR] + cmd, stdout=PIPE, env=env)
            out = p.communicate()[0]

            print "----------------------------------------------"
            print '# ', " ".join(cmd)
            print out
            print "----------------------------------------------"

            try:
                upw[i]   += re.search(r'Upward pass\s+:\s+([0-9.]+) s', out).group(1)
                trav[i]  += re.search(r'Traverse\s+:\s+([0-9.]+) s', out).group(1)
                dwn[i]   += re.search(r'Downward pass\s+:\s+([0-9.]+) s', out).group(1)
                total[i] += re.search(r'Total FMM\s+:\s+([0-9.]+) s', out).group(1)
            except:
                print "Error: regex search failed."
                print "outout = >>>"
                print out
                print "<<<"
                raise

        try:
            with open("tree_construction.csv") as f:
                lines = f.readlines()
                tree[i] += re.split(r'\s+', lines[1].strip())[1]
        except:
            print "Error in reading tree_construction.csv"
            raise


    index = np.arange(MaxNumProc)
    upw   = np.array([m.mean() for m in upw])
    trav  = np.array([m.mean() for m in trav])
    dwn   = np.array([m.mean() for m in dwn])
    tree  = np.array([m.mean() for m in tree])
    total = np.array([m.mean() for m in total])

    other = total - upw - trav - dwn - tree

    print "Hybrid strong scaling"
    print "NB = "
    print NB
    print "Ncrit = "
    print Ncrit
    print "NumProcs = "
    print NumProcs
    print "Upward = "
    print upw
    print "Traversal = "
    print trav
    print "Downward = "
    print dwn
    print "Tree = "
    print tree
    print "Total = "
    print total
    print "Other = "
    print other

    width = 0.35

    plt.style.use('ggplot')
    #plt.style.use('seaborn-whitegrid')

    #with plt.style.context('fivethirtyeight'):
    colors = [v['color'] for v in list(plt.rcParams['axes.prop_cycle'])]
    NBs = formatWithCommads(NB)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    p0 = ax.bar(index, tree,  width, color=colors[0]);
    p1 = ax.bar(index, upw,   width, bottom=tree,color=colors[1])
    p2 = ax.bar(index, trav,  width, bottom=tree+upw, color=colors[2])
    p3 = ax.bar(index, dwn,   width, bottom=tree+upw+trav, color=colors[3])
    p4 = ax.bar(index, other, width, bottom=tree+upw+trav+dwn, color=colors[4])
    ax.set_xlabel('#Procs (=#Nodes)')
    ax.set_ylabel('Runtime [s]')
    #ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc="upper right",)
    ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    #ax.set_xticks(index, [str(n) for n in NumProcs])
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    ax.set_title("Tapas runtime strong scaling / Hybrid MPI+MT\nNB=%s, Ncrit=%d, mean of %d samples\n %s\ncommit:\"%s\"" % (NBs, Ncrit, NumSamples, TakenAt, commit),
              fontsize=10)
    
    save_fname = '%s-hybrid-strong.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()

    print os.path.abspath("tree_construction.csv")
    print "---------------------"
    with open("tree_construction.csv") as f:
        print f.read()
    print "---------------------"

    # Read the tree_construction.csv file generated by the last run (with the max NP)
    with open("tree_construction.csv") as f:
        data = [re.split(r'\s+', ln.strip()) for ln in f.readlines()[1:]]

        all  = np.array( [float(data[i][1]) for i in range(MaxNumProc)] )
        smpl = np.array( [float(data[i][2]) for i in range(MaxNumProc)] )
        exch = np.array( [float(data[i][3]) for i in range(MaxNumProc)] )
        glc  = np.array( [float(data[i][4]) for i in range(MaxNumProc)] )
        ggl  = np.array( [float(data[i][5]) for i in range(MaxNumProc)] )
        othr = all - smpl - exch - glc - ggl

        print "sample = "
        print smpl
        print "exchange = "
        print exch
        print "grow_local = "
        print glc
        print "grow_global = "
        print ggl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    bottom = np.array([0.0] * MaxNumProc)
    p0 = ax.bar(index, smpl, width, bottom=bottom, color=colors[0]); bottom += smpl
    p1 = ax.bar(index, exch, width, bottom=bottom, color=colors[1]); bottom += exch
    p2 = ax.bar(index, glc,  width, bottom=bottom, color=colors[2]); bottom += glc
    p3 = ax.bar(index, ggl,  width, bottom=bottom, color=colors[3]); bottom += ggl
    p4 = ax.bar(index, othr,  width, bottom=bottom, color=colors[4]); bottom += othr
    
    ax.set_xlabel('MPI rank')
    ax.set_ylabel('Runtime [s]')
    ax.legend([p0, p1, p2, p3, p4], ["Sample", "Exchange", "GrowLocal", "GrowGlobal", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    NBs = formatWithCommads(NB)
    label_str = [
        "Tapas tree const. breakdown #Proc=%d / Hybrid Strong" % MaxNumProc,
        "NB=%s Ncrit=%d #Nodes=%d" % (NBs, Ncrit, MaxNumProc),
        "%s" % TakenAt,
        "commit:\"%s\"" % commit
    ]
    ax.set_title("\n".join(label_str), fontsize=10)
    
    save_fname = '%s-hybrid-strong-tree.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()

#-------------------------------------------------------------------------------
#
# Multiple nodes, Hybrid weak scaling
#
#-------------------------------------------------------------------------------

def hybrid_weak(NBpp, Ncrit, NumSamples):
    if not "PBS_NODEFILE" in os.environ:
        print "ERROR: PBS_NODEFILE is not defined"
        return

    NODEFILE = os.environ["PBS_NODEFILE"]
    with open(NODEFILE) as f:
        MaxNumProc = len( [ln for ln in f.readlines() if len(ln.strip()) > 0] )

    print "-----------------------------------------------------------"
    print "Multiple nodes, hybrid weak scaling"
    print "Nbpp = ", NBpp
    print "Ncrit = ", Ncrit
    print "NumSamples = ", NumSamples
    print "NODEFILE = %s" % NODEFILE
    print "-----------------------------------------------------------"
    
    NumProcs = range(1, MaxNumProc + 1)
    
    check_call("git checkout master", shell=True, cwd=DIR)
    p = Popen("%s/build.sh" % DIR, shell=True, stdout=PIPE, env=env)
    bin = p.communicate()[0].strip()

    upw  = [Mean() for nt in NumProcs]
    trav = [Mean() for nt in NumProcs]
    dwn  = [Mean() for nt in NumProcs]
    tree = [Mean() for nt in NumProcs]
    total = [Mean() for nt in NumProcs]

    for i, nproc in enumerate(NumProcs):
        NB = NBpp * nproc
        print "Running parallel_tapas with NP=%d" % nproc

        for j in range(NumSamples):
            print "NP=%d, run #%d" % (nproc, j)
            env['MYTH_WORKER_NUM'] = "12"
            print "MYTH_WORKER_NUM = ", env['MYTH_WORKER_NUM']
            print "NP = %d" % nproc
            print "NB = %d" % NB
            print "bin = ", bin
            # Open MPI specific
            cmd = ["mpiexec", "-np", str(nproc), "-hostfile", NODEFILE, "--map-by", "node", bin, "--numBodies", str(NB), "--ncrit", str(Ncrit)]
            p = Popen(["%s/run.sh" % DIR] + cmd, stdout=PIPE, env=env)
            out = p.communicate()[0]

            print "----------------------------------------------"
            print '# ', " ".join(cmd)
            print out
            print "----------------------------------------------"

            
            try:
                upw[i]   += re.search(r'Upward pass\s+:\s+([0-9.]+) s', out).group(1)
                trav[i]  += re.search(r'Traverse\s+:\s+([0-9.]+) s', out).group(1)
                dwn[i]   += re.search(r'Downward pass\s+:\s+([0-9.]+) s', out).group(1)
                total[i] += re.search(r'Total FMM\s+:\s+([0-9.]+) s', out).group(1)
            except:
                print "Error: regex search failed."
                print "outout = >>>"
                print out
                print "<<<"
                raise

            try:
                with open("tree_construction.csv") as f:
                    lines = f.readlines()
                    tree[i] += re.split(r'\s+', lines[1].strip())[1]
            except:
                print "Error in reading tree_construction.csv"
                raise


    index = np.arange(MaxNumProc)
    upw   = np.array([m.mean() for m in upw])
    trav  = np.array([m.mean() for m in trav])
    dwn   = np.array([m.mean() for m in dwn])
    tree  = np.array([m.mean() for m in tree])
    total = np.array([m.mean() for m in total])

    other = total - upw - trav - dwn - tree

    print "Hybrid strong scaling"
    print "NB = "
    print NB
    print "Ncrit = "
    print Ncrit
    print "NumProcs = "
    print NumProcs
    print "Upward = "
    print upw
    print "Traversal = "
    print trav
    print "Downward = "
    print dwn
    print "Tree = "
    print tree
    print "Total = "
    print total
    print "Other = "
    print other

    width = 0.35

    plt.style.use('ggplot')
    #plt.style.use('seaborn-whitegrid')

    #with plt.style.context('fivethirtyeight'):
    colors = [v['color'] for v in list(plt.rcParams['axes.prop_cycle'])]
    NBs = formatWithCommads(NB)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    
    p0 = ax.bar(index, tree,  width, color=colors[0]);
    p1 = ax.bar(index, upw,   width, bottom=tree,color=colors[1])
    p2 = ax.bar(index, trav,  width, bottom=tree+upw, color=colors[2])
    p3 = ax.bar(index, dwn,   width, bottom=tree+upw+trav, color=colors[3])
    p4 = ax.bar(index, other, width, bottom=tree+upw+trav+dwn, color=colors[4])
    ax.set_xlabel('#Procs (=#Nodes)')
    ax.set_ylabel('Runtime [s]')
    #ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc="upper right",)
    ax.legend([p0, p1, p2, p3, p4], ["Tree", "Upward", "Traverse", "Downward", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    #ax.set_xticks(index, [str(n) for n in NumProcs])
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    ax.set_title("Tapas runtime weak scaling / Hybrid MPI+MT\nNB=%s, Ncrit=%d, mean of %d samples\n %s\ncommit:\"%s\"" % (NBs, Ncrit, NumSamples, TakenAt, commit),
              fontsize=10)
    
    save_fname = '%s-hybrid-weak.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()

    print os.path.abspath("tree_construction.csv")
    print "---------------------"
    with open("tree_construction.csv") as f:
        print f.read()
    print "---------------------"

    # Read the tree_construction.csv file generated by the last run (with the max NP)
    with open("tree_construction.csv") as f:
        data = [re.split(r'\s+', ln.strip()) for ln in f.readlines()[1:]]

        all  = np.array( [float(data[i][1]) for i in range(MaxNumProc)] )
        smpl = np.array( [float(data[i][2]) for i in range(MaxNumProc)] )
        exch = np.array( [float(data[i][3]) for i in range(MaxNumProc)] )
        glc  = np.array( [float(data[i][4]) for i in range(MaxNumProc)] )
        ggl  = np.array( [float(data[i][5]) for i in range(MaxNumProc)] )
        othr = all - smpl - exch - glc - ggl

        print "sample = "
        print smpl
        print "exchange = "
        print exch
        print "grow_local = "
        print glc
        print "grow_global = "
        print ggl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    bottom = np.array([0.0] * MaxNumProc)
    p0 = ax.bar(index, smpl, width, bottom=bottom, color=colors[0]); bottom += smpl
    p1 = ax.bar(index, exch, width, bottom=bottom, color=colors[1]); bottom += exch
    p2 = ax.bar(index, glc,  width, bottom=bottom, color=colors[2]); bottom += glc
    p3 = ax.bar(index, ggl,  width, bottom=bottom, color=colors[3]); bottom += ggl
    p4 = ax.bar(index, othr,  width, bottom=bottom, color=colors[4]); bottom += othr
    
    ax.set_xlabel('MPI rank')
    ax.set_ylabel('Runtime [s]')
    ax.legend([p0, p1, p2, p3, p4], ["Sample", "Exchange", "GrowLocal", "GrowGlobal", "Other"], loc=2, bbox_to_anchor=(1.05,1), borderaxespad=0.)
    ax.set_xticks(index + width/2.)
    ax.set_xticklabels([str(n) for n in NumProcs])
    NBs = formatWithCommads(NB)
    label_str = [
        "Tapas tree const. breakdown #Proc=%d / Hybrid Weak Scaling" % MaxNumProc,
        "NB=%s Ncrit=%d #Nodes=%d" % (NBs, Ncrit, MaxNumProc),
        "%s" % TakenAt,
        "commit:\"%s\"" % commit
    ]
    ax.set_title("\n".join(label_str), fontsize=10)
    
    save_fname = '%s-hybrid-weak-tree.pdf' % RUN_ID
    save_path = os.path.join(OUTPUT_DIR, save_fname)
    fig.savefig(save_path)
    fig.clf()

if __name__ == "__main__":
    main()
