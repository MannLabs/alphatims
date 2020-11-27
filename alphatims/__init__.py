#!python


__project__ = "alphatims"
__version__ = "0.0.201125"
__license__ = "MIT"
__description__ = "A python package for Bruker TimsTOF raw data analysis and feature finding"
__author__ = "Sander Willems"
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
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "alphatims=alphatims.cli:run",
]
__urls__ = {
    "Mann Department at MPIB": "https://www.biochem.mpg.de/mann",
    "GitHub:": __github__
}
__requirements__ = {
    "": "requirements/requirements.txt",
    "gui": "requirements/requirements_gui.txt",
    "cli": "requirements/requirements_cli.txt",
    "nbs": "requirements/requirements_nbs.txt",
}
__strict_requirements__ = False
