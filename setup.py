import os
import re
import sys
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

with open("README.md", "r") as fh:
    long_description = fh.read()

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] +
                              cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] +
                              build_args, cwd=self.build_temp)


setup(
    name='mtspy',
    version='0.0.1',
    author='Igor Baratta',
    author_email='igorbaratta@gmail.com',
    description='Multi-threaded sparse matrix operations in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IgorBaratta/mtspy",
    ext_modules=[CMakeExtension('mtspy_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(),
    python_requires='>=3.6',
    zip_safe=False,
)
