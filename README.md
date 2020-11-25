# AlphaTims

A python package for Bruker TimsTOF raw data analysis and feature finding from the [Mann department at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann).

## Table of contents

* [**AlphaTims**](#alphatims)
  * [**Table of contents**](#table-of-contents)
  * [**License**](#license)
  * [**Installation**](#installation)
     * [**One-click GUI**](#one-click-gui)
     * [**Jupyter notebook installer**](#jupyter-notebook)
     * [**Full installer**](#full)
  * [**Test data**](#test-data)
  * [**Usage**](#usage)
    * [**GUI**](#gui)
    * [**CLI**](#cli)
    * [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
  * [**Under the hood**](#under-the-hood)
  * [**Future perspectives**](#future-perspectives)

## License

Get a copy of the [MIT license](LICENSE.txt). Since AlphaTims is dependent on Bruker libraries (available in the [alphatims/ext](alphatims/ext) folder) and external python packages, additional [third-party licenses](LICENSE-THIRD-PARTY.txt) are applicable.

## Installation

Three types of installation are possible:

* [**One-click GUI installer:**](#one-click-gui) Choose this installation if you only want the graphical user interface (GUI) and/or keep things as simple as possible.
* [**Jupyter notebook installer:**](#jupyter-notebook) Choose this installation if you only work in Jupyter Notebooks and just want to use AlphaTims as an extension.
* [**Full installer:**](#full) Choose this installation if you are familiar with command line interface (CLI) tools and python and want access to all available features and/or require development mode with modifiable AlphaTims source code.

***Since this software is dependent on [Bruker libraries](alphatims/ext) to read the raw data, it is only compatible with Windows and Linux. This is true for all installation types.***

### One-click GUI

* **Windows:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/download/latest/alphatims_installer.exe).
* **Linux:** TODO.
* **MacOS:** Unavailable due to availability of Bruker libraries.

Older releases are available on the [release page](https://github.com/MannLabs/alphatims/releases). Note that even the latest release might be behind the latest [**Jupyter**](#jupyter-notebook) and [**full**](#full) installers. Furthermore, there is no guarantee about backwards compatibility between releases.

### Jupyter notebook

In an existing Jupyter notebook with Python 3, run the following:

```bash
# # If git is not installed,
# # install git manually or run the following command first:
# !conda install git -y
!pip install git+https://github.com/MannLabs/alphatims.git --use-feature=2020-resolver
# # Extras can be installed, but are normally not needed for jupyter notebooks
# pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui,cli,nbs]' --use-feature=2020-resolver
```

Once installed, the latest version can be downloaded with a simple upgrade:
```bash
!pip install git+https://github.com/MannLabs/alphatims.git --use-feature=2020-resolver --upgrade
```

### Full

It is highly recommended to use a [conda virtual environment](https://docs.conda.io/en/latest/) to install AlphaTims. Install AlphaTims and all its [core dependancy requirements](requirements.txt) (extra options include [cli](requirements_cli.txt), [gui](requirements_gui.txt) and [nbs](requirements_nbs.txt) dependancies) with the following commands in a terminal (copy-paste per individual line):

```bash
# # It is not advised to install alphatims in the home directory.
# # Navigate to the folder where you want to install it
# # An alphatims folder is created automatically,
# # so a general software folder suffices
# mkdir folder/where/to/install/downloaded/software
# cd folder/where/to/install/downloaded/software
conda create -n alphatims python=3.8 -y
conda activate alphatims
# # If git is not installed, run the following command:
# conda install git -y
git clone https://github.com/MannLabs/alphatims.git
# # While AlphaTims can be imported directly in other programs,
# # a standalone version often requires additional packages for
# # cli, gui and nbs usage. If not desired, they can be skipped.
pip install -e './alphatims[cli,gui,nbs]' --use-feature=2020-resolver
conda deactivate
```

By using the editable flag `-e`, all modifications to the AlphaTims [source code folder](alphatims) are directly reflected when running AlphaTims. Note that the AlphaTims folder cannot be moved and/or renamed if an editable version is installed.

To avoid calling `conda activate alphatims` and `conda deactivate` every time AlphaTims is used, the binary execution can be added as an alias. On linux, this can be done with e.g.:

```bash
conda activate alphatims
alphatims_bin="$(which alphatims)"
# # With bash
echo "alias alphatims='"${alphatims_bin}"'" >> ~/.bashrc
# # With zsh
# echo "alias alphatims='"${alphatims_bin}"'" >> ~/.zshrc
conda deactivate
```

On Windows, this can be done with e.g.:

```bash
conda activate alphatims
where alphatims
# # The result should be something like:
# # C:\Users\yourname\.conda\envs\alphatims\Scripts\alphatims.exe
# # This directory can then be permanently added to e.g. PATH with:
# setx PATH=%PATH%;C:\Users\yourname\.conda\envs\alphatims\Scripts\alphatims.exe
conda deactivate
```

Note that this binary still reflects all changes to the [source code folder](alphatims) if an editable version is installed with the `-e` flag.

When using Jupyter notebooks and multiple conda environments, it is recommended to `conda install nb_conda_kernels` in the conda base environment. The AlphaTims conda environment can then be installed as a kernel with `conda install ipykernel` in the AlphaTims environment. Hereafter, running a `jupyter notebook` from the conda base environment should have a `Python [conda env: alphatims]` kernel available.

## Test data

A small Bruker TimsTOF HeLa DIA dataset with a 5 minute gradient is available for [download](https://datashare.biochem.mpg.de/s/DyIenLA2SLDz2sc). Initial investigation of Bruker TimsTOF data files can be done by opening the the .tdf file in the .d folder with an [SQL browser](https://sqlitebrowser.org/).

## Usage

There are three ways to use the software:

* [**GUI**](#gui)
* [**CLI**](#cli)
* [**Python**](#python-and-jupyter-notebooks)

### GUI

The GUI is accessible if you used the one-click GUI installer or through the following commands in a terminal:

```bash
conda activate alphatims
alphatims gui
conda deactivate
```
### CLI

The CLI can be run with the following commands in a terminal:

```bash
conda activate alphatims
alphatims
conda deactivate
```
### Python and jupyter notebooks

AlphaTims can be imported as a python package into any python script or notebook with the command `import alphatims`. An [exemplary jupyter notebook](nbs/example_analysis.ipynb) (with the extra option `gui` activated for all plotting capabilities) is present in the [nbs folder](nbs).

## Under the hood

A connection to the .tdf and .tdf_bin in the bruker .d directory are made once and all data is read into memory as a TimsTOF object. This is done by opening the sql database (.tdf) and reading all individual scans from the binary data (.tdf_bin) with the function `bruker_dll.tims_read_scans_v2` from the Bruker library. The TimsTOF data object stores all TOF arrivals in two huge arrays: `tof_indices` and `intensities`. This data seems to be centroided on a 'per-scan' basis (i.e. per push), but are independent in the retention time and ion mobility domain.

Since the `tof_indices` array is quite sparse in the TOF domain, it is indexed with a `tof_indptr` array that is similar to a a [compressed sparse row matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29). Herein a 'row' corresponds to a (`frame`, `scan`) tuple and the `tof_indptr` array thus has a length of `frame_max_index * scan_max_index + 1`, which approximately equals `10 * gradient_length_in_seconds * 927`. Filtering in `rt`/`frame` and `mobility`/`scan` domain is thus just a slice of the `tof_indptr` array when represented as a 2D-matrix and is hence very performant. Filtering in `TOF`/`mz` domain unfortunately requires to loop over individual scans. Luckily this can be done with numba and with a performance of `log(n)` since the `tof_indices` are sorted per scan. Finally, a `quad_indptr` (sparse pointer) array and associated `quad_low_values` and `quad_high_values` arrays allow to determine which precursor values are filtered by the quadrupole for each (`frame`, `scan`) tuple.

Slicing the total dataset happens with a magic `__getitem__` function and automatically converts any floating `rt`/`mobility`/`fragment mz` values to the appropriate `frame`/`scan`/`TOF` indices and vice versa as well.

## Future perspectives

Implementation of feature finding has not been started yet.
