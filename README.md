![Pip installation](https://github.com/MannLabs/alphatims/workflows/Pip%20installation/badge.svg)

# AlphaTims

An open-source Python package for efficient accession and analysis of Bruker TimsTOF raw data from the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann).

* [**AlphaTims**](#alphatims)
  * [**About**](#about)
  * [**License**](#license)
  * [**Installation**](#installation)
     * [**One-click GUI**](#one-click-gui)
     * [**Pip installer**](#pip)
     * [**Full installer**](#full)
     * [**Installation issues**](#installation-issues)
  * [**Test data**](#test-data)
    * [**DDA**](#dda)
    * [**DIA**](#dia)
  * [**Usage**](#usage)
    * [**GUI**](#gui)
    * [**CLI**](#cli)
    * [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
  * [**How it works**](#how-it-works)
    * [**Bruker raw data**](#bruker-raw-data)
    * [**TimsTOF objects in Python**](#timstof-objects-in-python)
    * [**Slicing TimsTOF objects**](#slicing-timstof-objects)
  * [**Future perspectives**](#future-perspectives)
  * [**How to contribute**](#how-to-contribute)

## About

With the introduction of the [Bruker TimsTOF](bruker.com/products/mass-spectrometry-and-separations/lc-ms/o-tof/timstof-pro.html), the inclusion of ion mobility separation (IMS) between liquid chromatography (LC) and tandem mass spectrometry (MSMS) instruments has gained popularity for both [DDA](https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.5b00932) and [DIA](https://www.nature.com/articles/s41592-020-00998-0). However, detection of such five dimensional points (chromatographic retention time (rt), ion mobility, quadrupole mass to charge (m/z), time-of-flight (TOF) m/z and intensity) at GHz results in an increased amount of data and complexity. Efficient accession, analysis and visualisation of Bruker TimsTOF data is therefore imperative. AlphaTims is an open-source Python package that allows such efficient access. It can be used with a graphical user interface (GUI), a command-line interface (CLI) or as a module directly within Python.

## License

AlphaTims was developed at the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann) and is available with an [Apache License](LICENSE.txt). Since AlphaTims is dependent on Bruker libraries (available in the [alphatims/ext](alphatims/ext) folder) and external Python packages, additional [third-party licenses](LICENSE-THIRD-PARTY.txt) are applicable.

## Installation

Three types of installation are possible:

* [**One-click GUI installer:**](#one-click-gui) Choose this installation if you only want the graphical user interface (GUI) and/or keep things as simple as possible.
* [**Pip installer:**](#pip) Choose this installation if you only want to use AlphaTims as a Python module in an already existing Python 3.8 environment such as a Jupyter notebook.
* [**Full installer:**](#full) Choose this installation if you are familiar with command line interface (CLI) tools, [conda](https://docs.conda.io/en/latest/) and Python. This installation allows access to all available features and modifiable AlphaTims source code. Specific extensions (GUI, CLI and notebooks) can be included in this installation as well that generally outperform the precompiled versions.

***Since this software is dependent on [Bruker libraries](alphatims/ext) to read the raw data, reading raw data is only compatible with Windows and Linux. This is true for all installation types. All other functionality is platform independent.***

### One-click GUI

* **Windows:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims_installer_windows.exe) and follow the installation instructions. Note the following for Windows:
  * File download or launching might be disabled by your virus scanner.
  * Running with Internet Explorer might not update results properly. If so, copy-paste the `localhost:...` url to Google Chrome and continue working from there.
  * If you install AlphaTims for all users, you might need admin privileges to run it (right click AlphaTims logo and "run as admin").
* **Linux:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims). No installation is needed, just download the file to the desired location. To run it, drag-and-drop it in a terminal and the GUI will open as a tab in your default browser. ***By using the AlphaTims application you agree with the [license](LICENSE.txt) and [third-party licenses](LICENSE-THIRD-PARTY.txt)*** Note the following for Linux:
  * If permissions are wrong, run `chmod +x alphatims` in a terminal (at the right location).
* **MacOS:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims.app.zip). No installation is needed, just unzip it and move it to your applications folder. ***By using the AlphaTims application you agree with the [license](LICENSE.txt) and [third-party licenses](LICENSE-THIRD-PARTY.txt)***. Also note the following for MacOS:
  * The AlphaTims application takes a long time to load upon first opening, this should be significantly faster the second time.
  * Reading of raw data is not possible due to availability of Bruker libraries, we advise to export raw data as .hdf files on Windows or Linux and use those directly.
  * IMPORTANT WARNING! If you just close the browser tab and do not press the "Quit" button, AlphaTims will keep running in the background (potentially using a significant amount of RAM memory). This is because MacOS does not explicitly opens a terminal window when running the GUI.
  * If nothing happens when you launch AlphaTims, you might need to grant it permissions by going to the MacOS menu "System Preferences | Security & Privacy | General". If the problem still persists, it is possible that MacOS already quarantined the AlphaTims app. It can be removed from quarantine by running `xattr -dr com.apple.quarantine alphatims.app` in a terminal (in the applications folder where `alphatims.app` is located).

Older releases are available on the [release page](https://github.com/MannLabs/alphatims/releases). Note that the one-click GUI is only compiled periodically and therefore even the latest release might be behind the latest [pip](#pip) and [full](#full) installers.

### Pip

In an existing Python 3.8 environment AlphaTims can be installed with the command:

```bash
pip install git+https://github.com/MannLabs/alphatims.git
```

This assumes `git` is accessible to this environment. If this is not the case, it can often be installed in the environment with the command:

```bash
conda install git -y
```

Upgrading to a newer version is possible with the command:

```bash
pip install git+https://github.com/MannLabs/alphatims.git --upgrade
```

These commands can also be run directly in a Jupyter notebook by prepending them with a `!`:

```
!conda install git -y
!pip install git+https://github.com/MannLabs/alphatims.git
# !pip install git+https://github.com/MannLabs/alphatims.git --upgrade
```

### Full

It is highly recommended to use a [conda virtual environment](https://docs.conda.io/en/latest/) to install AlphaTims. Install AlphaTims and all its [core dependancy requirements](requirements/requirements.txt) (extra options include [cli](requirements/requirements_cli.txt), [gui](requirements/requirements_gui.txt) and [nbs](requirements/requirements_nbs.txt) dependancies) with the following commands in a terminal (copy-paste per individual line):

```bash
# # It is not advised to install alphatims in the home directory.
# # Navigate to the folder where you want to install it
# # An alphatims folder is created automatically,
# # so a general software folder suffices
# mkdir folder/where/to/install/downloaded/software
# cd folder/where/to/install/downloaded/software
conda create -n alphatims python=3.8 pip=20.2 -y
conda activate alphatims
# # If git is not installed, run the following command:
# conda install git -y
git clone https://github.com/MannLabs/alphatims.git
# # While AlphaTims can be imported directly in other programs,
# # a standalone version often requires additional packages for
# # cli, gui and nbs usage. If not desired, they can be skipped.
# # Note that no `cd alphatims` is required for the following
pip install -e './alphatims[cli,gui,nbs]'
conda deactivate
```

By using the editable flag `-e`, all modifications to the AlphaTims [source code folder](alphatims) are directly reflected when running AlphaTims. Note that the AlphaTims folder cannot be moved and/or renamed if an editable version is installed.

To avoid calling `conda activate alphatims` and `conda deactivate` every time AlphaTims is used, the binary execution can be added as an alias. On linux and MacOS, this can be done with e.g.:

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

When using Jupyter notebooks and multiple conda environments, it is recommended to `conda install nb_conda_kernels` in the conda base environment. Hereafter, running a `jupyter notebook` from the conda base environment should have a `python [conda env: alphatims]` kernel available.

### Installation issues

Common issues include:

* **Always make sure you have activate the alphatims environment with `conda activate alphatims`.** If this fails, make sure you have installed [conda](https://docs.conda.io/en/latest/) and have created an AlphaTims environment with `conda create -n alphatims python=3.8`.
* **No `git` command**. Make sure [git](https://git-scm.com/downloads) is installed. In a notebook `!conda install git -y` might work.
* **Wrong Python version.** AlphaTims is only compatible with Python 3.8. You can check if you have the right version with the command `python --version` (or `!python --version` in a notebook). If not, reinstall the AlphaTims environment with `conda create -n alphatims python=3.8`.
* **Dependancy conflicts/issues.** Pip changed their dependancy resolver with [pip version 20.3](https://pip.pypa.io/en/stable/news/). Downgrading pip to version 20.2 with `pip install pip==20.2` (before running `pip install ./alphatims`) could solve this issue.
* **Alphatims is not found.** Make sure you use the right folder. Local folders are best called by prefixing them with `./` (e.g. `pip install ./alphatims`). On some systems, installing require you to specifically (not) use single quotes `'` around the AlphaTims folder, e.g. `pip install './alphatims[gui, nbs]'`.
* **Modifications to the AlphaTims source code are not reflected.** Make sure you use the `-e` flag when using `pip install -e ./alphatims`.
* **Numpy not working properly.** On Windows, `numpy==1.19.4` has some issues. After installing AlphaTims, downgrade Numpy with `pip install numpy==1.19.3`.

## Test data

AlphaTims is compatible with both data-dependant acquisition (DDA) and data-independant acquisition (DIA). Initial investigation of Bruker TimsTOF data files can be done by opening the the .tdf file in the .d folder with an [SQL browser](https://sqlitebrowser.org/).

### DDA

A 21 min HeLA DDA dataset (803 Mb) is available for [download here](https://datashare.biochem.mpg.de/s/2sWNvImHwdELg55/download).

### DIA

A small Bruker TimsTOF DIA dataset with a 5 minute gradient containing tryptic HeLa peptides is available for [download here](https://datashare.biochem.mpg.de/s/s7zuTMilCOkYb2K/download).

## Usage

There are three ways to use the software:

* [**GUI:**](#gui) This is mostly used as a data browser.
* [**CLI:**](#cli) This is mostly used to process data and can be incorporated in automated workflows.
* [**Python:**](#python-and-jupyter-notebooks) This is mostly used as a Python package in other Python projects.

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

It is possible to get help about each function and their (required) parameters by using the `-h` flag. For instance,the command `alphatims export hdf -h` will produce the following output:

```
************************
* AlphaTims 0.0.201209 *
************************
Usage: alphatims export hdf [OPTIONS] BRUKER_D_FOLDER

  Export BRUKER_D_FOLDER as hdf file.

Options:
  --output_folder DIRECTORY  A directory for all output (blank means
                             `bruker_d_folder` root is used).

  --log_file PATH            Save all log data to a file (blank means
                             'log_[date].txt' with date format yymmddhhmmss in
                             'log' folder of AlphaTims directory).  [default:
                             ]

  --threads INTEGER          The number of threads to use (0 means all,
                             negative means how many threads to leave
                             available).  [default: -1]

  --disable_log_stream       Disable streaming of log data.  [default: False]
  --parameter_file FILE      A .json file with (non-required) parameters
                             (blank means default parameters are used). This
                             overrides all default and CLI parameters.

  --compress                 Compression of hdf files. If set, this roughly
                             halves files sizes (on-disk), at the cost of
                             taking 3-6 longer accession times.  [default:
                             False]

  -h, --help                 Show this message and exit.
```

### Python and jupyter notebooks

AlphaTims can be imported as a Python package into any Python script or notebook with the command `import alphatims`. An [exemplary jupyter notebook](nbs/example_analysis.ipynb) (with the extra option `gui` activated for all plotting capabilities) is present in the [nbs folder](nbs).

## How it works

The basic workflow of AlphaTims looks as follows:

* Read data from a [Bruker `.d` folder](#bruker-raw-data).
* Convert this to a [TimsTOF object in Python](#timstof-objects-in-python) and store this in a persistent HDF5 file.
* Use Python's [slicing mechanism](#slicing-timstof-objects) to retrieve data from this object for e.g. visualisation.

### Bruker raw data

Bruker stores TimsTOF raw data in a `.d` folder. The two main files in this folder are `analysis.tdf` and `analysis.tdf_bin`.

The `analysis.tdf` file is an sql database, in which all metadata is stored together with summarised information. This includes the `Frames` table, wherein information about each individual TIMS cycle is summarised including the retention time, number of scans (i.e. a single TOF push related to a single ion mobility value), summed intensity and total number of ions that have hit the detector. More details about individual scans of the frames is available in the tables `PasefFrameMSMSInfo` (for PASEF acquisition) or `DiaFrameMsMsWindows` (for diaPASEF acquisition). This includes quadrupole and collision settings of the frame/scan combinations.

The `analysis.tdf_bin` file is a binary file that contains the number of detected ions per individual scan, all detector arrival times and their intensity values. These values are grouped and compressed per frame (i.e. TIMS cycle), thereby allowing fast appendage during online acquisition.

### TimsTOF objects in Python

AlphaTims first reads relevant metadata from the `analysis.tdf` sql database and uses this to initialise a Python object of the `bruker.TimsTOF` class. Next, AlphaTims reads the summary information from the `Frames` table and uses this to create three empty arrays:

* An empty `tof_indices` array, in which all TOF arrival times of each individual detector hit can be stored. It's size is determined by summing the number of detector hits for all frames.
* An empty `intensities` array of the same size, in which all the intensity values of each individual detector hit can be stored.
* An empty `tof_indptr` array, that can store the number of detector hits per scan. It's size is equal to `(frame_max_index + 1) * scans_max_index + 1`. It includes one additional frame to compensate for the fact that Bruker arrays are 1-indexed, while Python uses 0-indexing. The final `+1` is because this array will be converted to an offset array, similar to the index pointer array of a [compressed sparse row matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29). Typical values are `scans_max_index = 1000` and `frame_max_index = gradient_length_in_seconds * 10`, resulting in approximately `len(tof_indptr) = 10000 * gradient_length_in_seconds`.

Hereafter the `PasefFrameMSMSInfo` or `DiaFrameMsMsWindows` table from the `analysis.tdf` sql database is read and four arrays are created:

* A `quad_indptr` array that indexes the `tof_indptr` array. Each element points to an index of the `tof_indptr` where the voltage on the quadrupole and collision cell is adjusted. For PASEF acquisitions, this is typically 20 times per MSMS frame (turning on and off a value for 10 precursor selections) and once per change from an MS (precursor) frame to an MSMS (fragment) frame. For diaPASEF, this is typically twice to 10 times per frame and with a repetitive pattern over the frame cycle. This results in an array of approximately `len(quad_indptr) = 100 * gradient_length_in_seconds`. As with the `tof_indptr` array, this array is converted to an offset array with size `+1`.
* A `quad_low_values` array of `len(quad_indptr) - 1`. This array stores the lower m/z boundary that is selected with the quadrupole. For precursors without quadrupole selection, this value is set to -1.
* A `quad_high_values` array, similar to `quad_low_values`.
* A `precursor_indices` array of `len(quad_indptr) - 1`. For PASEF this array stores the index of the selected precursor. For diaPASEF, this array stores the `WindowGroup` of the fragment frame. As with the `quad_low_values` and `quad_high_values`, a value of -1 indicates a precursor without quadrupole selection.

After processing this summarising information from the `analysis.tdf` sql database, the actual raw data from the `analysis.tdf_bin` binary file is read and stored in the empty `tof_indices`, `intensities` and `tof_indptr` arrays. This is done with the `tims_read_scans_v2` function from Bruker's `libtimsdata.dll` library (available in the [alphatims/ext](alphatims/ext) folder).

Finally, three arrays are created that allows translation of `frame_`, `scan_` and `tof_indices` to `rt_values`, `mobility_values` and `mz_values` arrays.
* The `rt_values` array is read directly read from the `Frames` table in `analysis.tdf` and has a length equal to `frame_max_index + 1`.
* The `mobility_values` array is defined by using the function `tims_scannum_to_oneoverk0` from `libtimsdata.dll` on the first frame and typically has a length of `1000`.
* Similarly, the `mz_values` array is defined by using the function `tims_index_to_mz` from `libtimsdata.dll` on the first frame. Typically this has a length of `400000`.

All these arrays can be loaded into memory, taking up roughly twice as much RAM memory as the `.d` fodler does in disk. This increase in RAM memory is mainly due to the compression used in the `analysis.tdf_bin` file. If the Python object is stored as an HDF5 file, the empty `tof_indices` and `intensity` arrays can be created and filled on-disk, thereby minimizing RAM memory usage to less than 1 GB even for files that take up several GB on-disk. The HDF5 file can also be compressed, so that its sizes is roughly halved and thereby has the same size as the Bruker `.d` folder, but (de)compression reduces accession times by 3-6 fold.

### Slicing TimsTOF objects

Once a Python TimsTOF object is available, it can be loaded into memory for ultrafast accession. Accession of the `data` object is done by simple Python slicing such as e.g. `selected_ion_indices = data[frame_selection, scan_selection, quad_selection, tof_selection]`. These ion indices are then easily parsed to a `pd.DataFrame` with the function `df = data.as_dataframe(selected_ion_indices)`. The columns of this dataframe contain all information, i.e. `frame`, `scan`, `precursor` and `tof` indices and `rt`, `mobility`, `quad_low`, `quad_high`, `mz` and `intensity` values.

## Future perspectives

* Detection of:
  * precursor and fragment ions
  * isotopic envelopes (i.e. features)
  * fragment clusters (i.e. pseudo MSMS spectra)

## How to contribute

All contributions are welcome. Feel free to post a new issue or clone the repository and create a PR with a new branch.
