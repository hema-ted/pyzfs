from setuptools import setup

setup(
    name="pyzfs",
    version="dev",  # set as dev for development commits
    author="He Ma, Marco Govoni, Giulia Galli",
    author_email="mahe@uchicago.edu, mgovoni@anl.gov, gagalli@uchicago.edu",
    description="A python code to compute zero-field splitting tensors",
    packages=["pyzfs"],
    classifiers=[],
    install_requires=[
        'numpy', 'scipy', 'mpi4py', 'h5py', 'ase', 'lxml',
    ]
)

