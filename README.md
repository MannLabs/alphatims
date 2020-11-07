# AlphaTims

A python package for Bruker TimsTOF raw data analysis and feature finding from the [Mann department at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann).

## Table of contents

* [**AlphaTims**](#alphatims)
  * [**Table of contents**](#table-of-contents)
  * [**License**](#license)
  * [**Installation**](#installation)
     * [**One-click GUI**](#one-click-gui)
     * [**Python**](#python)
  * [**Test data**](#test-data)
  * [**Usage**](#usage)
  * [**Under the hood**](#under-the-hood)
  * [**Future perspectives**](#future-perspectives)

## License

Get a copy of the [MIT license here](LICENSE.txt).

## Installation

Two types of installation are possible:

* [**One-click GUI installer:**](#one-click-gui) Choose this installation if you only want the graphical user interface and/or keep things as simple as possible.
* [**Python installer:**](#python) Choose this installation if you are familiar with a terminal and/or python and want access to all available features.

*Since this software is dependent on Bruker libraries (available in the [alphatims/ext](alphatims/ext) folder) to read the raw data, it is only compatible with Windows and Linux. This is true for both the one-click GUI and python installer.*

### One-click GUI

* **Windows:** TODO
* **Linux:** TODO
* **MacOS:** Unavailable due to availability of Bruker libraries

### Python

It is strongly recommended to use a [conda virtual environment](https://docs.conda.io/en/latest/) to install AlphaTims. Install AlphaTims and all its [dependancy requirements](requirements.txt) with the following commands in a terminal:

```bash
# It is not advised to install AlphaTims directly in the home folder.
# Instead, create and move to another folder with e.g. the following commands:
# mkdir folder/where/to/install/downloaded/software
# cd folder/where/to/install/downloaded/software
conda create -n alphatims python=3.8 -y
conda activate alphatims
git clone https://github.com/swillems/alphatims.git
# For a standard version use:
pip install ./alphatims --use-feature=2020-resolver
# For an editable version with modifiable source code use:
# pip install -e ./alphatims --use-feature=2020-resolver
conda deactivate alphatims
```

If the editable flag `-e` is use, all modifications to the AlphaTims [source code folder](alphatims) are then directly incorporated. Note that the AlphaTims folder cannot be moved and/or renamed if an editable version is installed.

## Test data

A small Bruker TimsTOF HeLa DIA dataset with a 5 minute gradient is available for [download here](https://datashare.biochem.mpg.de/s/DyIenLA2SLDz2sc). Initial investigation of Bruker TimsTOF data can be done by opening the the .tdf file in the .d folder with an [SQL browser](https://sqlitebrowser.org/).

## Usage

There are three ways to use the software

* **GUI:** The GUI is accessible if you used the one-click GUI installer or by the following commands in a terminal:
```bash
conda activate alphatims
alphatims gui
conda deactivate alphatims
```
* **CLI:** The CLI can be run with the following commands in a terminal:
```bash
conda activate alphatims
alphatims
conda deactivate alphatims
```
* **Python:** AlphaTims can be imported as a python package into any python script or notebook with the command `import alphatims` if the conda environment is activated with `conda activate alphatims`. An [exemplary jupyter notebook](nbs/example_analysis.ipynb) is present in the [nbs folder](nbs).

## Under the hood

A connection to the .tdf and .tdf_bin in the bruker .d directory are made once and all data is read into memory as a TimsTOF object. This is done by opening the sql database (.tdf) and reading all individual scans from the binary data (.tdf_bin) with the function `bruker_dll.tims_read_scans_v2` from the Bruker library. The TimsTOF data object stores all TOF arrivals in two huge arrays: `tof_indices` and `intensities`. This data seems to be centroided on a 'per-scan' basis (i.e. per push), but are independent in the retention time and ion mobility domain.

Since the `tof_indices` array is quite sparse in the TOF domain, it is indexed with a `tof_indptr` array that similar to a a [compressed sparse row matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29). Herein a 'row' corresponds to a (`frame`, `scan`) tuple and the `tof_indptr` array thus has a length of `frame_max_index * scan_max_index`, which approximately equals `10 * gradient_length_in_seconds * 927`. Filtering in `rt`/`frame` and `mobility`/`scan` domain is thus just a slice of the `tof_indptr` array when represented as a 2D-matrix and is hence very performant. Filtering in `TOF`/`mz` domain unfortunately requires to loop over individual scans. Luckily this can be done with numba and with a performance of `log(n)` since the `tof_indices` are sorted per scan.

Slicing the total dataset happens with a magic `__getitem__` function and automatically converts any floating `rt`/`mobility`/`mz` values to the appropriate `frame`/`scan`/`TOF` indices and vice versa as well.

## Future perspectives

Precursor indices `quad_indptr` will be incorporated to complete the symmetry between instrumental components (LC, TIMS, quadrupole/collision cell, TOF) and acquired data. Note that `quad_indptr` are fully dependent on `frame`/`scan` indices, while all other coordinates can be considered orthogonal/independent (at least from a data accession perspective).

Implementation of feature finding has not been started yet.
