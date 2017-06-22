
# GPU-LMfit - Library for parallel fitting of compartmental models to 4D medical imaging volumes
# Michele Scipioni
# University of Pisa
# Harvard University, Martinos Center for Biomedical Imaging
# 2015 - 2017, Pisa, Pi
# 2017, Boston, MA, USA


# Use old Python build system, otherwise the extension libraries cannot be found. FIXME
import sys
for arg in sys.argv:
    if arg == "install":
        sys.argv.append('--old-and-unmanageable')

from setuptools import setup, Extension
from glob import glob


setup(
    name='gpuKMfit',
    version='0.1.0',
    author='Michele Scipioni',
    author_email='scipioni.michele@gmail.com',
    packages=['gpuKMfit',
              'gpuKMfit.kernels',
              'gpuKMfit.python',
              ],
    package_data={'gpuKMfit': ['Data/*.pdf', 'Data/*.png', 'Data/*.jpg', 'Data/*.svg',
                               'Data/*.nii', 'Data/*.dcm', 'Data/*.h5', 'Data/*.txt', 'Data/*.dat']},
    scripts=[],
    url='https://github.com/mscipio/GPU_fitting_toolbox',
    license='LICENSE.txt',
    description='Compartmental models parallel GPU-Cuda fitting toolbox.',
    long_description=open('README.md').read(),
    keywords=["PET", "DCE-MRI", "emission tomography", "contrast enhanced mri",
              "kinetic modeling", "compartmental models", "cuda", "Nvidia"],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: CUDA",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Bio-Informatics"],
    install_requires=[
        "pycuda >= 2016.1.2",
        "scikit-cuda >= 0.5.1",
        "numpy >= 1.12.0",
        "matplotlib >= 1.4.0",
        "interfile >= 0.3.0",
        "ipy_table >= 1.11.0",
        "nibabel >= 2.0.0",
        "pydicom >= 0.9.0",
        "nipy >= 0.3.0",
        "jupyter >= 1.0.0",
        "h5py >= 2.3.0",
        "scipy >= 0.14.0",
        "pillow >= 2.8.0",
        "svgwrite >= 1.1.0"]
)
