#!/usr/bin/env python
import os
import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from distutils.extension import Extension
from Cython.Build import cythonize
import pyarrow
import numpy

include_dirs = [pyarrow.get_include(), numpy.get_include()]
library_dirs = pyarrow.get_library_dirs()

print ("include_dirs=", include_dirs)
print ("library_dirs=", library_dirs)

extra_compile_args = []
extra_link_args = []
debug = False,

if os.getenv('DEBUG', '') == 'ON':
    extra_compile_args = ["-O0", '-DDEBUG']
    extra_link_args = ["-debug:full"]
    debug = True,

# Define your extension
extensions = [
    Extension( "jollyjack.jollyjack_cython", ["jollyjack/jollyjack_cython.pyx", "jollyjack/jollyjack.cc"],
        include_dirs = include_dirs,  
        library_dirs = library_dirs,
        libraries=["arrow", "parquet"], 
        language = "c++",
        extra_compile_args = extra_compile_args + (['/std:c++17'] if sys.platform.startswith('win') else ['-std=c++17']),
        extra_link_args = extra_link_args,
    )
]

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives, gdb_debug=debug, emit_linenums=debug)

# Make default named pyarrow shared libs available.
pyarrow.create_library_symlinks()

# Custom build command to dynamically generate metadata file
class GenerateMetadata(build_py):
    def run(self):
        # Call the original build_py command
        super().run()

        # Get the distribution object
        dist = self.distribution

        package_name = dist.get_name()
        package_version = dist.get_version()
        package_dependencies = dist.install_requires
        
        print (f"package_name = {package_name}")
        print (f"package_version = {package_version}")
        print (f"package_dependencies = {package_dependencies}")
        
        output_dir = os.path.join(package_name)
        os.makedirs(output_dir, exist_ok=True)
        metadata_file = os.path.join(output_dir, "package_metadata.py")

        # Write metadata to the file
        with open(metadata_file, "w") as f:
            f.write("# Auto-generated package metadata\n")
            f.write(f"__package__ = '{package_name}'\n")
            f.write(f"__version__ = '{package_version}'\n")
            f.write(f"__dependencies__ = {package_dependencies}\n")

        print(f"Generated metadata file: {metadata_file}")
        print (os.listdir(output_dir))
        
setup(
    packages=["jollyjack"],
    package_dir={"": "."},
    zip_safe=False,
    ext_modules=extensions,
    project_urls={
        "Documentation": "https://github.com/marcin-krystianc/JollyJack",
        "Source": "https://github.com/marcin-krystianc/JollyJack",
    },
    cmdclass={
        "build_py": GenerateMetadata,  # Use the custom build command
    },
)
