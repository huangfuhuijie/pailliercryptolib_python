#! /usr/bin/env python3

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DIPCL_PYTHON_ENABLE_QAT=ON",
            "-DIPCL_PYTHON_DETECT_CPU_RUNTIME=ON",
            "-DIPCL_PYTHON_ENABLE_OMP=OFF",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        cpu_count = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        build_args += ["--", "-j" + str(cpu_count)]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="ipcl-python",
    version="2.0.0",
    author="Sejun Kim",
    author_email="sejun.kim@intel.com",
    description="Python wrapper for Intel Paillier Cryptosystem Library",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("ipcl_python/bindings/ipcl_bindings")],
    cmdclass={"build_ext": CMakeBuild},
    test_suite="tests",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "wheel",
        "numpy==1.23.1",
        "pycryptodomex==3.15.0",
        "gmpy2==2.1.5",
        "cachetools==3.0.0",
        "ruamel.yaml==0.16.10",
    ],
    zip_safe=False,
)
