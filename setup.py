import setuptools
from setuptools.extension import Extension
import numpy as np

from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

link_args = ["-std=c99"]
ext_modules = [
    Extension("arcticpy.trap_managers_utils", ["arcticpy/trap_managers_utils.pyx"],
            extra_compile_args=link_args,
            extra_link_args=link_args,
            include_dirs=[np.get_include()],
            define_macros=[('CYTHON_TRACE', '1')],
            language='c'),
    Extension("arcticpy.main_utils", ["arcticpy/main_utils.pyx"],
            extra_compile_args=link_args,
            extra_link_args=link_args,
            include_dirs=[np.get_include()],
            define_macros=[('CYTHON_TRACE', '1')],
            language='c'),
]
ext_modules = cythonize(
    ext_modules, compiler_directives={'language_level': 3})

setuptools.setup(
    name="arcticpy",
    packages=setuptools.find_packages(),
    version="1.1",
    description="AlgoRithm for Charge Transfer Inefficiency Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob Kegerreis, Richard Massey, James Nightingale",
    author_email="jacob.kegerreis@durham.ac.uk",
    url="https://github.com/jkeger/arcticpy",
    license="GNU GPLv3+",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    python_requires=">=3",
    install_requires=requirements,
    keywords=["charge transfer inefficiency correction"],
    ext_modules=ext_modules,
)
