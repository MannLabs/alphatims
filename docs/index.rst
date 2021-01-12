.. AlphaTims documentation master file, created by
   sphinx-quickstart on Tue Jan 12 09:02:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for AlphaTims.
=====================================

With the introduction of the `Bruker TimsTOF <http://bruker.com/products/mass-spectrometry-and-separations/lc-ms/o-tof/timstof-pro.html/>`_ and `Parallel Accumulation–Serial Fragmentation (PASEF) <https://doi.org/10.1074/mcp.TIR118.000900/>`_, the inclusion of trapped ion mobility separation (TIMS) between liquid chromatography (LC) and tandem mass spectrometry (MSMS) instruments has gained popularity for both `DDA <https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.5b00932/>`_ and `DIA <https://www.nature.com/articles/s41592-020-00998-0/>`_. However, detection of such five dimensional points (chromatographic retention time (rt), ion mobility, quadrupole mass to charge (m/z), time-of-flight (TOF) m/z and intensity) at GHz results in an increased amount of data and complexity. Efficient accession, analysis and visualisation of Bruker TimsTOF data are therefore imperative. AlphaTims is an open-source Python package that allows such efficient access. It can be used with a graphical user interface (GUI), a command-line interface (CLI) or as a module directly within Python.

This documentation is intended as an API for direct Python use. For more information, see AlphaTims on `GitHub <https://github.com/MannLabs/alphatims/>`_.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   alphatims.utils
   alphatims.bruker


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
