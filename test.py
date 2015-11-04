# -*- coding: utf-8 -*-

import os,sys,re
import unittest
import errno
import tempfile
import shutil
import glob
import datetime
import traceback
from subprocess import Popen, PIPE, check_call, CalledProcessError, STDOUT

SourceRoot = os.path.dirname(os.path.abspath(__file__))
BuildRoot = tempfile.mkdtemp()
LogFile = None
Scale = 'small' # test scale ('small', 'medium', 'large')

assert Scale in ['small', 'medium', 'large']

def mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError as exc:
        if exc.erro == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def is_exe(path):
    return os.path.isfile(path) and os.access(path, os.X_OK)

if 'SCALE' in os.environ:
    s = os.environ['SCALE']
    if re.match(r'^s(m(a(l(l)?)?)?)?$', s, re.I): # small
        Scale = 'small'
    elif re.match(r'^m(e(d(i(u(m)?)?)?)?)?$', s, re.I): # medium
        Scale = 'medium'
    elif re.match(r'^l(a(r(g(e)?)?)?)?$', s, re.I): # large
        Scale = 'large'

def find_exec(exe):
    p = Popen('which %s' % exe, shell=True, stdout=PIPE, stderr=PIPE)
    found = p.communicate().strip()
    if len(found) == 0:
        return None
    else:
        return found

class TestCppUnitTests(unittest.TestCase):
    def test_units(self):
        self.srcdir = os.path.join(SourceRoot, 'cpp', 'tests')
        self.bindir = os.path.join(BuildRoot,  'cpp', 'tests')
        try:
            mkdir_p(self.bindir)
        except:
            pass
        check_call(['make', 'VERBSE=1', 'MODE=release',
                    '-C', self.srcdir, 'clean', 'all'], cwd = self.bindir, stdout=LogFile, stderr=LogFile)

        tests = glob.glob(os.path.join(SourceRoot, "cpp", "tests", "test_*"))
        for t in [t for t in tests if is_exe(t)]:
            try:
                retcode = check_call(t, stderr=LogFile, stdout=LogFile)
                self.assertEqual(retcode, 0, msg=t)
            except CalledProcessError as e:
                self.fail(msg = str(e))

class TestBH(unittest.TestCase):
    def setUp(self):
        self.srcdir = os.path.join(SourceRoot, 'sample', 'barnes-hut')
        self.bindir = os.path.join(BuildRoot,  'sample', 'barnes-hut')
        self.bin    = os.path.join(self.srcdir, 'bh_mpi')
        try:
            mkdir_p(self.bindir)
        except:
            pass
        check_call(['make', 'VERBSE=1', 'MODE=release',
                    '-C', self.srcdir, 'clean', 'all'], cwd = self.bindir, stdout=LogFile, stderr=LogFile)
    def test_bh(self):
        if Scale == 'small':
            NP = [1,6];
            NB = [1000]
        elif Scale == 'medium':
            NP = range(1,6)
            NB = [1000, 2000]
        elif Scale == 'large':
            NP = [1,2,4,8,16,32]
            NB = [1000, 2000, 4000, 8000, 16000]
            
        for np in NP:
            for nb in NB:
                nb_total = str(nb * np)
                p = Popen(['mpirun', '-n', str(np), self.bin, '-s', nb_total], stdout=PIPE, stderr=STDOUT, cwd = self.bindir)
                out,_ = p.communicate()

                if p.returncode != 0:
                    print out

                # Check the return code
                self.assertEqual(p.returncode, 0)
                
                # Check P ERR
                m = re.search(r'P ERR  \s*:\s*(\S+)$', out, re.MULTILINE)
                err = float(m.group(1))
                self.assertTrue(m != None)
                self.assertTrue(err < 1e-2, "P ERR check for NB=%d, NP=%d" % (nb, np))

                # Check F ERR
                m = re.search(r'F ERR  \s*:\s*(\S+)$', out, re.MULTILINE)
                err = float(m.group(1))
                self.assertTrue(m != None)
                self.assertTrue(err < 1e-2, "F ERR check for NB=%d, NP=%d" % (nb, np))

    def tearDown(self):
        check_call(['make', '-C', self.srcdir, 'clean'], cwd = self.bindir, stdout=LogFile, stderr=STDOUT)

class TestExaFMM(unittest.TestCase):
    def setUp(self):
        self.srcdir = os.path.join(SourceRoot, 'sample', 'exafmm-dev-13274dd4ac68', 'examples')
        self.bindir = os.path.join(BuildRoot,  'sample', 'exafmm-dev-13274dd4ac68', 'examples')
        try:
            mkdir_p(self.bindir)
        except:
            pass
        check_call(['make', 'VERBSE=1', 'MODE=release',
                    '-C', self.srcdir, 'clean', 'serial_tapas'], cwd = self.bindir, stdout=LogFile, stderr=LogFile)
        
        self.serial_tapas = os.path.join(self.srcdir, 'serial_tapas')

    def test_fmm(self):
        if Scale == 'small':
            D = ['c']
            NB = [1000]
            NCRIT = [16]
        elif Scale == 'medium':
            D = ['c', 'p', 's', 'l']
            NB = [1000, 2000]
            NCRIT = [16, 64]
        elif Scale == 'large':
            D = ['c', 'p', 's', 'l']
            NB = [1000, 2000, 4000, 8000, 16000]
            NCRIT = [16, 64, 128]
            
        # As of now Tapas port of ExaFMM only supports single process
        for dist in D:
            for nb in NB:
                for ncrit in ['16', '64']:
                    p = Popen([self.serial_tapas, '-n', str(nb), '-c', str(ncrit), '-d', dist], stdout=PIPE, stderr=STDOUT)
                    out,err = p.communicate()
                    out = out
                    self.assertEqual(p.returncode, 0)

                    m = re.search(r'Rel. L2 Error \(pot\)  : (\S*)$', out, re.MULTILINE)
                    if m is None: print out
                    self.assertTrue(m != None)
                    self.assertLess(float(m.group(1)), 1e-2)
                    
                    m = re.search(r'Rel. L2 Error \(acc\)  : (\S*)$', out, re.MULTILINE)
                    if m is None: print out
                    self.assertTrue(m != None)
                    self.assertLess(float(m.group(1)), 1e-2)
        
if __name__ == "__main__":
    sys.stderr.write("test.py: SourceRoot = " + SourceRoot + "\n")
    sys.stderr.write("test.py: BuildRoot = " + BuildRoot + "\n")
    sys.stderr.write("test.py: Test Scale = " + Scale + "\n")
    sys.stderr.write("test.py: CXX = " + os.environ["CXX"] + "\n")

    t = datetime.datetime.now()
    logfile_name = t.strftime("test-%Y%m%d-%H%M%S-%f.log")
    LogFile = open(logfile_name, 'w+')

    # Run tests
    try:
        unittest.main(verbosity=3)

    finally:
        LogFile.seek(0)
        print LogFile.read()
        
        sys.stderr.write("Removing temporary build directory: " + BuildRoot + "\n")
        shutil.rmtree(BuildRoot, True)
        if LogFile:
            LogFile.close()
            LogFile = None
    sys.exit(0)

