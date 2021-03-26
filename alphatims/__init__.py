#!python


__project__ = "alphatims"
__version__ = "0.1.210323"
__license__ = "Apache"
__description__ = "A Python package for Bruker TimsTOF raw data accession and visualization"
__author__ = "Sander Willems, Eugenia Voytik"
__author_email__ = "willems@biochem.mpg.de"
__github__ = "https://github.com/MannLabs/alphatims"
__keywords__ = [
    "ms",
    "mass spectrometry",
    "bruker",
    "timsTOF",
    "proteomics",
    "bioinformatics"
]
__python_version__ = ">=3.8,<3.9"
__classifiers__ = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "alphatims=alphatims.cli:run",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "GitHub:": __github__
}
__requirements__ = {
    "": "requirements/requirements.txt",
    "plotting": "requirements/requirements_plotting.txt",
    "development": "requirements/requirements_development.txt",
}
__strict_requirements__ = True
