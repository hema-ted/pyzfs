from setuptools import setup, find_packages

setup(
    name="pyzfs",
    version="1.2",
    author="He Ma, Marco Govoni, Giulia Galli",
    author_email="mahe@uchicago.edu, mgovoni@anl.gov, gagalli@uchicago.edu",
    description="A python code to compute zero-field splitting tensors",
    packages=find_packages(),
    classifiers=[],
    install_requires=[
        'numpy', 'scipy', 'mpi4py', 'h5py', 'ase', 'lxml',
    ],
    entry_points={
        "console_scripts": ["pyzfs = pyzfs.run:main"]
    }
)
