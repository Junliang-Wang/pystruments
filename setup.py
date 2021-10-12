from distutils.version import StrictVersion
from importlib import import_module

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

short_description = """
Python package for instrument drivers.
"""

extras = {
    # 'matplotlib': ('matplotlib', '3.2', 'conda'),
}
extras_require = {k: '>='.join(v[0:2]) for k, v in extras.items()}

install_requires = [
    'numpy>=1.15',
    'pyvisa>=1.10',
]

setuptools.setup(
    name="pystruments",
    version="0.1.0",
    author="Junliang WANG",
    author_email="junliang.wang@gmail.com",
    license='BSD-3-Clause',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="",
    url="https://github.com/Junliang-Wang/pystruments",
    packages=setuptools.find_packages(exclude=["tests", "docs", "bechmark"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
    python_requires='>=2.7',
    install_requires=install_requires,
    extras_require=extras_require
)

# Code below adapted from QCoDeS (https://qcodes.github.io/)

version_template = '''
*****
***** package {0} must be at least version {1}.
***** Please upgrade it (pip install -U {0} or conda install {0})
***** in order to use {2}
***** Recommended method: {3}
*****
'''

missing_template = '''
*****
***** package {0} not found
***** Please install it (pip install {0} or conda install {0})
***** Recommended: {2} install {0}
***** in order to use {1}
*****
'''

valueerror_template = '''
*****
***** package {0} version not understood
***** Please make sure the installed version ({1})
***** is compatible with the minimum required version ({2})
***** in order to use {3}
*****
'''

othererror_template = '''
*****
***** could not import package {0}. Please try importing it from
***** the commandline to diagnose the issue.
*****
'''

# now test the versions of extras
for extra, (module_name, min_version, install_method) in extras.items():
    try:
        module = import_module(module_name.lower())
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra, install_method))
    except ImportError:
        print(missing_template.format(module_name, extra, install_method))
    except ValueError:
        print(valueerror_template.format(
            module_name, module.__version__, min_version, extra))
    except:
        print(othererror_template.format(module_name))
