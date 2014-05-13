#
# Copyright (c) 2013, 2014, Oracle and/or its affiliates. All rights reserved.
# DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
#
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details (a copy is included in the LICENSE file that
# accompanied this code).
#
# You should have received a copy of the GNU General Public License version
# 2 along with this work; if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
# or visit www.oracle.com if you need additional information or have any
# questions.
#
import subprocess, tempfile, shutil, filecmp, platform
from os.path import join, sep
from argparse import ArgumentParser, REMAINDER
import mx
import mx_graal
import os

_fastr_suite = None

def _runR(args, className, nonZeroIsFatal=True):
    os.environ['R_HOME'] = _fastr_suite.dir
    # Set up path for Lapack libraries
    osname = platform.system()
    lib_base = join(_fastr_suite.dir, 'com.oracle.truffle.r.native', 'lib', osname.lower())
    lib_value = lib_base
    if osname == 'Darwin':
        lib_env = 'DYLD_FALLBACK_LIBRARY_PATH'
        lib_value = lib_value + os.pathsep + '/usr/lib'
    else:
        lib_env = 'LD_LIBRARY_PATH'
    os.environ[lib_env] = lib_value
    return mx_graal.vm(['-ea', '-esa', '-cp', mx.classpath("com.oracle.truffle.r.shell"), className] + args, nonZeroIsFatal=nonZeroIsFatal)

def runRCommand(args, nonZeroIsFatal=True):
    '''run R shell'''
    return _runR(args, "com.oracle.truffle.r.shell.RCommand", nonZeroIsFatal=nonZeroIsFatal)

def runRscriptCommand(args, nonZeroIsFatal=True):
    '''run Rscript file'''
    return _runR(args, "com.oracle.truffle.r.shell.RscriptCommand", nonZeroIsFatal=nonZeroIsFatal)

def _truffle_r_gate_body(args, tasks):
    _check_autogen_tests(False)

    # workaround for Hotspot Mac OS X build problem
    osname = platform.system()
    if osname == 'Darwin':
        os.environ['COMPILER_WARNINGS_FATAL'] = 'false'
        os.environ['USE_CLANG'] = 'true'
        os.environ['LFLAGS'] = '-Xlinker -lstdc++'

    t = mx_graal.Task('BuildHotSpotGraalServer: product')
    mx_graal.buildvms(['--vms', 'server', '--builds', 'product'])
    tasks.append(t.stop())

    # check that the expected test output file is up to date
    t = mx.GateTask('UnitTests: ExpectedTestOutput file check')
    junit(['--tests', _default_unit_tests(), '--check-expected-output'])
    tasks.append(t.stop())
    t = mx.GateTask('UnitTests: simple')
    rc = junit(['--tests', _default_unit_tests()])
    if rc != 0:
        mx.abort('unit tests failed')
    tasks.append(t.stop())

def gate(args):
    '''Run the R gate'''
    # suppress the download meter
    mx._opts.no_download_progress = True
    # ideally would be a standard gate task - we do it early
    t = mx.GateTask('Copyright check')
    rc = mx.checkcopyrights(['--primary'])
    t.stop()
    if rc != 0:
        mx.abort('copyright errors')
    _check_autogen_tests(True)
    mx.gate(args, _truffle_r_gate_body)

_tempdir = None

def _check_autogen_tests(copy):
    # make copies of AllTests and FailingTests, as these will be regenerated by the gate
    # and may be out of sync
    test_srcdir = _test_srcdir()
    all_tests = join(test_srcdir, 'all', 'AllTests.java')
    failing_tests = join(test_srcdir, 'failing', 'FailingTests.java')
    global _tempdir
    if copy:
        _tempdir = tempfile.mkdtemp()
        shutil.copy(all_tests, _tempdir)
        shutil.copy(failing_tests, _tempdir)
    else:
        files_equal = filecmp.cmp(all_tests, join(_tempdir, 'AllTests.java')) and filecmp.cmp(failing_tests, join(_tempdir, 'FailingTests.java'))
        shutil.rmtree(_tempdir)
        if not files_equal:
            mx.abort('AllTests.java and/or FailingTests.java are out of sync, regenerate with mx rtestgen')

def _test_srcdir():
    tp = 'com.oracle.truffle.r.test'
    return join(mx.project(tp).dir, 'src', tp.replace('.', sep))

def _junit_r_harness(args, vmArgs, junitArgs):
    # always pass the directory where the expected output file should reside
    runlistener_arg = 'expected=' + _test_srcdir()
    # there should not be any unparsed arguments at this stage
    if args.remainder:
        mx.abort('unexpected arguments: ' + str(args.remainder).strip('[]') + '; did you forget --tests')

    def add_arg_separator():
        # can't update in Python 2.7
        arg = runlistener_arg
        if len(arg) > 0:
            arg += ','
        return arg

    if args.gen_fastr_output:
        runlistener_arg = add_arg_separator()
        runlistener_arg += 'gen-fastr=' + args.gen_fastr_output

    if args.check_expected_output:
        args.gen_expected_output = True
        runlistener_arg = add_arg_separator()
        runlistener_arg += 'check-expected'

    if args.gen_expected_output:
        runlistener_arg = add_arg_separator()
        runlistener_arg += 'gen-expected'
        if args.keep_trailing_whitespace:
            runlistener_arg = add_arg_separator()
            runlistener_arg += 'keep-trailing-whitespace'

    if args.gen_diff_output:
        runlistener_arg = add_arg_separator()
        runlistener_arg += 'gen-diff=' + args.gen_diff_output

#    if args.test_methods:
#        runlistener_arg = add_arg_separator()
#        runlistener_arg = 'test-methods=' + args.test_methods

    # use a custom junit.RunListener
    runlistener = 'com.oracle.truffle.r.test.TestBase$RunListener'
    if len(runlistener_arg) > 0:
        runlistener += ':' + runlistener_arg

    junitArgs += ['--runlistener', runlistener]

    # suppress Truffle compilation by using a high threshold
    vmArgs += ['-G:TruffleCompilationThreshold=100000']

    return mx_graal.vm(vmArgs + junitArgs, vm="server", nonZeroIsFatal=False)

def junit(args):
    '''run R Junit tests'''
    parser = ArgumentParser(prog='r junit')
    parser.add_argument('--gen-expected-output', action='store_true', help='generate/update expected test output file')
    parser.add_argument('--keep-trailing-whitespace', action='store_true', help='keep trailing whitespace in expected test output file')
    parser.add_argument('--check-expected-output', action='store_true', help='check but do not update expected test output file')
    parser.add_argument('--gen-fastr-output', action='store', metavar='<path>', help='generate FastR test output file')
    parser.add_argument('--gen-diff-output', action='store', metavar='<path>', help='generate difference test output file ')
    # parser.add_argument('--test-methods', action='store', help='pattern to match test methods in test classes')

    return mx.junit(args, _junit_r_harness, parser=parser)

def junit_simple(args):
    junit(['--tests', 'com.oracle.truffle.r.test.simple'] + args)

def _default_unit_tests():
    return 'com.oracle.truffle.r.test.simple'

def testgen(args):
    '''generate the expected output for unit tests, and All/Failing test classes'''
    # clean the test project to invoke the test analyzer AP
    testOnly = ['--projects', 'com.oracle.truffle.r.test']
    mx.clean(testOnly)
    mx.build(testOnly)
    # now just invoke junit with the appropriate options
    junit(args + ['--tests', _default_unit_tests(), '--gen-expected-output'])

def rbench(args):
    '''run a one or more R benchmarks'''
    parser = ArgumentParser(prog='mx rbench')
    parser.add_argument('--path', action='store_true', help='print path to benchmark')
    parser.add_argument('--gnur', action='store_true', help='run under GnuR')
    parser.add_argument('--gnur-path', action='store', metavar='<path>', help='specify path to GnuR', default='R')
    parser.add_argument('--fail-fast', action='store_true', help='abort on first failure')
    parser.add_argument('--gnur-jit', action='store_true', help='enable GnuR JIT')
    parser.add_argument('benchmarks', nargs=REMAINDER, metavar='benchmarkgroup.name', help='list of benchmarks to run')
    args = parser.parse_args(args)

    # dynamically load the benchmarks suite
    hg_base = mx.get_env('HG_BASE')
    alternate = None if hg_base is None else join(hg_base, 'r_benchmarks')
    bm_suite = _fastr_suite.import_suite('r_benchmarks', version=None, alternate=alternate)
    mx.build_suite(bm_suite)

    failure = 0
    for bm in args.benchmarks:
        # Get the R script location via helper app
        # N.B. we do not use mx.run_java() as that might check options we don't want for the helper, e.g. debugging agent
        rc = 0
        javacmd = [mx.java().java, '-cp', mx.classpath('r.benchmarks'), 'r.benchmarks.RBenchmarks', bm]
        try:
            bmpath = subprocess.check_output(javacmd).rstrip()
            if args.path:
                print bmpath
            else:
                headline = ('GnuR' if args.gnur else 'FastR') + ' running ' + bm
                print headline
                command = ['-f', bmpath]
                if args.gnur:
                    env = os.environ
                    if args.gnur_jit:
                        env['R_ENABLE_JIT'] = '3'
                    rc = subprocess.call([args.gnur_path, '--slave'] + command, env=env)
                else:
                    # temporary: disable group generics as long as they impose a considerable performance overhead
                    command = ['--DisableGroupGenerics'] + c
                    rc = runRCommand(command, nonZeroIsFatal=False)
                if rc != 0:
                    print 'benchmark ' + bm + ' failed'
                    emsg = rc
        except subprocess.CalledProcessError:
            emsg = 'benchmark ' + bm + ' not found'
            rc = 1
        # check error and fail-fast option
        if rc != 0:
            failure = failure if failure > rc else rc
            if args.fail_fast:
                mx.abort(emsg)
    # if any failed
    if failure != 0:
        mx.abort('benchmark run failed')
    return failure

def _bench_harness_body(args, vmArgs):
    marks = ['shootout.binarytrees', 'shootout.fannkuchredux', 'shootout.fasta', 'shootout.fastaredux',
             'shootout.knucleotide', 'shootout.mandelbrot-ascii', 'shootout.nbody', 'shootout.pidigits',
             'shootout.regexdna', 'shootout.reversecomplement', 'shootout.spectralnorm',
             'b25.bench.prog-1', 'b25.bench.prog-2', 'b25.bench.prog-3', 'b25.bench.prog-4', 'b25.bench.prog-5']
    return rbench(marks)

def bench(args):
    '''Run a standard set of R benchmarks'''
    # In the automatic benchmark context, the vm will neither be built nor set.
    # In interactive (development) use, if it is set (interactive use) we use it, otherwise we choose the server variant.
    # The build component of mx.bench causes the vm to be built.

    # suppress the download meter
    mx._opts.no_download_progress = True

    vm = mx_graal.VM('server' if mx_graal._vm is None else mx_graal._vm)
    with vm:
        mx.bench(args, harness=_bench_harness_body)

def unittest(args):
    print "use 'junit --tests testclasses' or 'junitsimple' to run FastR unit tests"

def rbcheck(args):
    '''check FastR builtins against GnuR'''
    parser = ArgumentParser(prog='mx rbcheck')
    parser.add_argument('--check-internal', action='store_const', const='--check-internal', help='check .Internal functions')
    parser.add_argument('--unknown-to-gnur', action='store_const', const='--unknown-to-gnur', help='list builtins not in GnuR FUNCTAB')
    parser.add_argument('--todo', action='store_const', const='--todo', help='show unimplemented')
    parser.add_argument('--no-eval-args', action='store_const', const='--no-eval-args', help='list functions that do not evaluate their args')
    parser.add_argument('--visibility', action='store_const', const='--visibility', help='list visibility specification')
    parser.add_argument('--printGnuRFunctions', action='store', help='ask GnuR to "print" value of functions')
    args = parser.parse_args(args)

    class_map = mx.project('com.oracle.truffle.r.nodes').find_classes_with_matching_source_line(None, lambda line: "@RBuiltin" in line, True)
    classes = []
    for className, path in class_map.iteritems():
        classNameX = className.split("$")[0] if '$' in className else className

        if not classNameX.endswith('Factory'):
            classes.append([className, path])

    (_, testfile) = tempfile.mkstemp(".classes", "mx")
    os.close(_)
    with open(testfile, 'w') as f:
        for c in classes:
            f.write(c[0] + ',' + c[1] + '\n')
    analyzeArgs = []
    if args.check_internal:
        analyzeArgs.append(args.check_internal)
    if args.unknown_to_gnur:
        analyzeArgs.append(args.unknown_to_gnur)
    if args.todo:
        analyzeArgs.append(args.todo)
    if args.no_eval_args:
        analyzeArgs.append(args.no_eval_args)
    if args.visibility:
        analyzeArgs.append(args.visibility)
    if args.printGnuRFunctions:
        analyzeArgs.append('--printGnuRFunctions')
        analyzeArgs.append(args.printGnuRFunctions)
    analyzeArgs.append(testfile)
    cp = mx.classpath([pcp.name for pcp in mx.projects_opt_limit_to_suites()])
    mx.run_java(['-cp', cp, 'com.oracle.truffle.r.test.tools.AnalyzeRBuiltin'] + analyzeArgs)

def mx_init(suite):
    global _fastr_suite
    _fastr_suite = suite
    commands = {
        # new commands
        'r' : [runRCommand, '[options]'],
        'R' : [runRCommand, '[options]'],
        'rscript' : [runRscriptCommand, '[options]'],
        'Rscript' : [runRscriptCommand, '[options]'],
        'rtestgen' : [testgen, ''],
        'rbench' : [rbench, 'options'],
        # core overrides
        'gate' : [gate, ''],
        'bench' : [bench, 'options'],
        'junit' : [junit, ['options']],
        'junitsimple' : [junit_simple, ['options']],
        'unittest' : [unittest, ['options']],
        'rbcheck' : [rbcheck, ['options']],
    }
    mx.update_commands(suite, commands)
