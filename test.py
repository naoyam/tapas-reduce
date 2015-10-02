# -*- coding: utf-8 -*-

import os,sys,re
import unittest
import tempfile
import shutil
import glob
from subprocess import Popen, PIPE, check_call, CalledProcessError, STDOUT

SourceRoot = os.path.dirname(os.path.abspath(__file__))
BuildRoot = tempfile.mkdtemp()

def find_exec(exe):
    p = Popen('which %s' % exe, shell=True, stdout=PIPE, stderr=PIPE)
    found = p.communicate().strip()
    if len(found) == 0:
        return None
    else:
        return found

class TestCppUnitTests(unittest.TestCase):
    def test_units(self):
        tests = glob.glob(os.path.join(BuildRoot, "cpp", "tests", "test_*"))
        for t in tests:
            try:
                retcode = check_call(t, stderr=PIPE, stdout=PIPE)
                self.assertEqual(retcode, 0, msg=t)
            except CalledProcessError as e:
                self.fail(msg = str(e))

class TestBH(unittest.TestCase):
    def setUp(self):
        self.srcdir = os.path.join(SourceRoot, 'sample', 'barnes-hut')
        self.bindir = os.path.join(BuildRoot,  'sample', 'barnes-hut')
        self.bin    = os.path.join(self.srcdir, 'bh_mpi')
        try:
            os.mkdir(self.bindir)
        except:
            pass
        check_call(['make', 'VERBSE=1', 'MODE=release',
                    '-C', self.srcdir, 'clean', 'all'], cwd = self.bindir)
    def test_bh(self):
        for np in range(1, 6):
            for nb in [1000, 2000]:
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
        check_call(['make', '-C', self.srcdir, 'clean'], cwd = self.bindir)

class TestExaFMM(unittest.TestCase):
    def setUp(self):
        self.serial_tapas = os.path.join(BuildRoot, 'sample', 'exafmm-dev-13274dd4ac68', 'examples', 'serial_tapas')

    def test_fmm(self):
        # As of now Tapas port of ExaFMM only supports single process
        for dist in ['c', 'p', 'l']:
            for nb in [1000]:
                for ncrit in ['16', '64']:
                    p = Popen([self.serial_tapas, '-n', str(nb), '-c', ncrit, '-d', dist], stdout=PIPE, stderr=PIPE)
                    out,err = p.communicate()
                    out = out + err
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
    sys.stderr.write("SourceRoot = " + SourceRoot + "\n")
    sys.stderr.write("BuildRoot = " + BuildRoot + "\n")

    try:
        # Build the source tree
        # We use P=6 to make the test clearer.
        check_call(['cmake', SourceRoot,
                    '-DCMAKE_BUILD_TYPE=Release',
                    '-DEXAFMM_EXPANSION=6',
                    '-DEXAFMM_ENABLE_MT=no'], cwd=BuildRoot)
        check_call(['make'], cwd=BuildRoot)

        # Run tests
        unittest.main(verbosity=3)
        
    finally:
        sys.stderr.write("Removing temporary build directory: " + BuildRoot + "\n")
        shutil.rmtree(BuildRoot, True)

