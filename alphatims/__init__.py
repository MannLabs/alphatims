#!python


__project__ = "alphatims"
__version__ = "1.0.6"
__license__ = "Apache"
__description__ = "A Python package to index Bruker TimsTOF raw data for fast and easy accession and visualization"
__author__ = "Sander Willems, Eugenia Voytik"
__author_email__ = "opensource@alphapept.com"
__github__ = "https://github.com/MannLabs/alphatims"
__keywords__ = [
    "ms",
    "mass spectrometry",
    "bruker",
    "timsTOF",
    "proteomics",
    "bioinformatics",
    "data indexing",
]
__python_version__ = ">=3.8,<4"
__classifiers__ = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]
__console_scripts__ = [
    "alphatims=alphatims.cli:run",
]
__urls__ = {
    "Mann Labs at MPIB": "https://www.biochem.mpg.de/mann",
    "Mann Labs at CPR": "https://www.cpr.ku.dk/research/proteomics/mann/",
    "GitHub": __github__,
    "ReadTheDocs": "https://alphatims.readthedocs.io/en/latest/",
    "PyPi": "https://pypi.org/project/alphatims/",
    "Scientific paper": "https://doi.org/10.1016/j.mcpro.2021.100149",
}
__requirements__ = {
    "": "requirements/requirements.txt",
    "plotting": "requirements/requirements_plotting.txt",
    "development": "requirements/requirements_development.txt",
    "legacy": "requirements/requirements_legacy.txt",
}
