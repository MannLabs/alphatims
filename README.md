# AlphaTims

A python package for Bruker TimsTOF raw data analysis and feature finding from the [Mann department at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann).

## Table of contents

* [**AlphaTims**](#alphatims)
  * [**Table of contents**](#table-of-contents)
  * [**License**](#license)
  * [**Installation**](#installation)
     * [**One-click GUI**](#ne-click-gui)
     * [**Python**](#python)
  * [**Test data**](#test-data)
  * [**Usage**](#usage)

## License

Get a copy of the [MIT license here](LICENSE.txt).

## Installation

Two types of installation are possible:

* **One-click GUI installer:** Choose this installation if you only want the graphical user interface and/or keep things as simple as possible.
* **Python installer:** Choose this installation if you are familiar with a terminal and/or python and want access to all available features.

*Since this software is dependent on Bruker libraries (available in the [alphatims/ext](alphatims/ext) folder) to read the raw data, it is only compatible with Windows and Linux. This is true for both the one-click GUI and python installer.*

### One-click GUI

* **Windows:** TODO
* **Linux:** TODO
* **MacOS:** Unavailable due to availability of Bruker libraries ()

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

A small Bruker TimsTOF HeLa DIA dataset with a 5 minute gradient is available for [download here](https://datashare.biochem.mpg.de/s/DyIenLA2SLDz2sc).

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
