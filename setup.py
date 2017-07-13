from setuptools import setup

setup(
    name = "hspin",
    version = "0.0.0",
    author = "He Ma",
    author_email = "mahe@uchicago.edu",
    description = "",
    packages=["hspin"],
    classifiers=[],
    install_requires=[
        'scipy', 'ase', 'mpi4py',
    ]
)
