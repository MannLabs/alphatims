![Pip installation](https://github.com/MannLabs/alphatims/workflows/Pip%20installation/badge.svg)

---
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
    * [**Test sample**](#test-sample)
    * [**LC**](#lc)  
    * [**DDA**](#dda)
    * [**DIA**](#dia)
  * [**Usage**](#usage)
    * [**GUI**](#gui)
    * [**CLI**](#cli)
    * [**Python and jupyter notebooks**](#python-and-jupyter-notebooks)
  * [**Performance**](#performance)
    * [**Speed**](#speed)
    * [**RAM**](#ram)
  * [**Troubleshooting**](#troubleshooting)
  * [**How it works**](#how-it-works)
    * [**Bruker raw data**](#bruker-raw-data)
    * [**TimsTOF objects in Python**](#timstof-objects-in-python)
    * [**Slicing TimsTOF objects**](#slicing-timstof-objects)
  * [**Future perspectives**](#future-perspectives)
  * [**How to contribute**](#how-to-contribute)

---
## About

With the introduction of the [Bruker TimsTOF](bruker.com/products/mass-spectrometry-and-separations/lc-ms/o-tof/timstof-pro.html) and [Parallel Accumulation–Serial Fragmentation (PASEF)](https://doi.org/10.1074/mcp.TIR118.000900), the inclusion of trapped ion mobility separation (TIMS) between liquid chromatography (LC) and tandem mass spectrometry (MSMS) instruments has gained popularity for both [DDA](https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.5b00932) and [DIA](https://www.nature.com/articles/s41592-020-00998-0). However, detection of such five dimensional points (chromatographic retention time (rt), ion mobility, quadrupole mass to charge (m/z), time-of-flight (TOF) m/z and intensity) at GHz results in an increased amount of data and complexity. Efficient accession, analysis and visualisation of Bruker TimsTOF data are therefore imperative. AlphaTims is an open-source Python package that allows such efficient access. It can be used with a graphical user interface (GUI), a command-line interface (CLI) or as a module directly within Python.

---
## License

AlphaTims was developed at the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann) and is available with an [Apache License](LICENSE.txt). Since AlphaTims is dependent on Bruker libraries (available in the [alphatims/ext](alphatims/ext) folder) and external Python packages (available in the [requirements](requirements) folder), additional [third-party licenses](LICENSE-THIRD-PARTY.txt) are applicable.

---
## Installation

Three types of installation are possible:

* [**One-click GUI installer:**](#one-click-gui) Choose this installation if you only want the GUI and/or keep things as simple as possible.
* [**Pip installer:**](#pip) Choose this installation if you only want to use AlphaTims as a Python module in an already existing Python 3.8 environment such as a Jupyter notebook.
* [**Full installer:**](#full) Choose this installation if you are familiar with CLI tools, [conda](https://docs.conda.io/en/latest/) and Python. This installation allows access to all available features and modifiable AlphaTims source code. Specific extensions (GUI, CLI and notebooks) can be included in this installation as well that generally outperform the precompiled versions.

***Since this software is dependent on [Bruker libraries](alphatims/ext), reading raw data is only compatible with Windows and Linux. This is true for all installation types. All other functionality is platform independent.***

### One-click GUI

* **Windows:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims_installer_windows.exe) and follow the installation instructions. Note the following for Windows:
  * File download or launching might be disabled by your virus scanner.
  * Running with Internet Explorer might not update results properly. If so, copy-paste the `localhost:...` url to an alternative browser (Google Chrome has been verified to work) and continue working from there.
  * If you install AlphaTims for all users, you might need admin privileges to run it (right click AlphaTims logo and "run as admin").
* **Linux:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims). No installation is needed, just download the file to the desired location. To run it, drag-and-drop it in a terminal and the GUI will open as a tab in your default browser. ***By using the AlphaTims application you agree with the [license](LICENSE.txt) and [third-party licenses](LICENSE-THIRD-PARTY.txt)*** Note the following for Linux:
  * If permissions are wrong, run `chmod +x alphatims` in a terminal (at the right location).
* **MacOS:** [Download the latest release](https://github.com/MannLabs/alphatims/releases/latest/download/alphatims.app.zip). No installation is needed, just unzip it and move it to your applications folder. ***By using the AlphaTims application you agree with the [license](LICENSE.txt) and [third-party licenses](LICENSE-THIRD-PARTY.txt)***. Also note the following for MacOS:
  * The AlphaTims application takes a long time to load upon first opening, this should be significantly faster the second time.
  * Reading of raw data is not possible due to availability of Bruker libraries, we advise to export raw data as .hdf files on Windows or Linux and use those directly.
  * If nothing happens when you launch AlphaTims, you might need to grant it permissions by going to the MacOS menu "System Preferences | Security & Privacy | General". If the problem still persists, it is possible that MacOS already quarantined the AlphaTims app. It can be removed from quarantine by running `xattr -dr com.apple.quarantine alphatims.app` in a terminal (in the applications folder where `alphatims.app` is located).

IMPORTANT WARNING! If you just close the browser tab and do not press the "Quit" button, AlphaTims will keep running in the background (potentially using a significant amount of RAM memory). This is especially important for MacOS, which does not explicitly open a terminal window when running the GUI.

Older releases are available on the [release page](https://github.com/MannLabs/alphatims/releases). Note that the one-click GUI is only compiled periodically and therefore even the latest release might be behind the latest [pip](#pip) and [full](#full) installers.

### Pip

In an existing Python 3.8 environment AlphaTims can be installed with the command:

```bash
pip install git+https://github.com/MannLabs/alphatims.git
```

If the plotting or development module are also required, use:
```bash
pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[plotting,devel]'
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

It is highly recommended to use a [conda virtual environment](https://docs.conda.io/en/latest/) to install AlphaTims. Install AlphaTims and all its [core dependancy requirements](requirements/requirements.txt) (extra options include [develop](requirements/requirements_develop.txt), and [plotting](requirements/requirements_plotting.txt) dependancies) with the following commands in a terminal (copy-paste per individual line):

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
pip install -e './alphatims[plotting,develop]'
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

See the general [troubleshooting](#troubleshooting) section.

---
## Test data

AlphaTims is compatible with both ddaPASEF and diaPASEF. Initial investigation of Bruker TimsTOF data files can be done by opening the .tdf file in the .d folder with an [SQL browser](https://sqlitebrowser.org/).

### Test sample

A test sample of human cervical cancer cells (HeLa, S3, ATCC) is provided for AlphaTims. These cells were cultured in Dulbecco's modified Eagle's medium (all Life Technologies Ltd., UK). Subsequently, the cells were collected, washed, flash-frozen, and stored at -80 °C.
Following the previously published [in-StageTip protocol](https://www.nature.com/articles/nmeth.2834), cell lysis, reduction, and alkylation with chloroacetamide were carried out simultaneously in a lysis buffer (PreOmics, Germany). The resultant dried peptides were reconstituted in water comprising 2 vol% acetonitrile and 0.1% vol% trifluoroacetic acid, yielding a 200 ng/µL solution. This solution was further diluted with water containing 0.1% vol% formic acid. The manufacturer's instructions were followed to load approximately 200ng peptides onto Evotips (Evosep, Denmark).

### LC

Single-run LC-MS analysis was executed via an [Evosep One LC system (Evosep)](https://doi.org/10.1074/mcp.TIR118.000853). This was coupled online with a hybrid [TIMS quadrupole TOF mass spectrometer (Bruker timsTOF Pro, Germany)](https://doi.org/10.1074/mcp.TIR118.000900). A silica emitter (Bruker) was placed inside a nano-electrospray ion source (Captive spray source, Bruker) and connected to an 8 cm x 150 µm reverse phase column to perform LC. The column was packed with 1.5 µm C18-beads (Pepsep, Denmark). Mobile phases were water and acetonitrile, buffered with 0.1% formic acid. The samples were separated with a predefined 60 samples per day method (Evosep).

### DDA

A ddaPASEF dataset (803 Mb) is available for [download here](https://datashare.biochem.mpg.de/s/s7zuTMilCOkYb2K/download). Each topN acquisition cycle consisted of 10 PASEF MS/MS scans, and the accumulation and ramp times were set to 100 ms. Single-charged precursors were excluded using a polygon filter in the m/z-ion mobility plane. Furthermore, all precursors, which reached the target value of 20000, were excluded for 0.4 min from the acquisition. Precursors were isolated with a quadrupole window of 2 Th for m/z <700 and 3 Th for m/z >700.

### DIA

The same sample was also acquired with diaPASEF (1.96 Gb) and is also available for [download here](https://datashare.biochem.mpg.de/s/jHph7AmaKivDSZJ/download). The "high-speed" method (mass range: m/z 400 to 1000, 1/K0: 0.6 – 1.6 Vs cm- 2, diaPASEF windows: 8 x 25 Th) was used, as described in [Meier et al](https://www.nature.com/articles/s41592-020-00998-0).

---
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

It is possible to get help about each function and their (required) parameters by using the `-h` flag. For instance, the command `alphatims export hdf -h` will produce the following output:

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

AlphaTims can be imported as a Python package into any Python script or notebook with the command `import alphatims`. Documentation for all functions is available in the [API](docs/_build/html/index.html). (NOTE: while the repo is private, html pages can not be safely rendered on e.g. GitHub pages or ReadTheDocs. For now it is best to download/clone/fork the AlphaTims repository and open `docs/_build/html/index.html` in a local browser.)

A brief [tutorial jupyter notebook](nbs/tutorial.ipynb) on how to use the API is also present in the [nbs folder](nbs). When running locally it provides interactive plot, which are not rendered on GitHub. Instead, they are available as individual html pages in the [nbs folder](nbs).

---
## Performance

Performance can be measured in function of [speed](#speed) or [RAM](#ram) usage.

### Speed

Typical performance statistics on data in-/output and slicing of standard [HeLa datasets](#test-sample) include:

| type | gradient | datapoints    | reading (raw/HDF) | export HDF| slicing (in ms)          |
|------|----------|---------------|-------------------|--------|--------------------------|
| DDA  | 6 min    | 214,172,697   | 1.55 s / 536 ms    | 571 ms | 1.64 / 45.7 / 27.0 / 78.8 |
| DIA  | 6 min    | 158,552,099   | 1.09 s / 381 ms    | 403 ms | 6.40 / 26.7 / 626 / 109     |
| DDA  | 21 min   | 295,251,252   | 3.07 s / 913 ms    | 757 ms | 1.74 / 72.5 / 122 / 186      |
| DIA  | 21 min   | 730,564,765   | 4.54 s / 2.20 s    | 1.85 s | 0.855 / 122 / 5040 / 404    |
| DDA  | 120 min  | 2,074,019,899 | 24.1 s / 10.6 s    | 5.70 s  | 0.709 / 371 / 609 / 1200    |

All slices were performed in a single dimension. Including more slices makes the analysis more stringent and hence faster. The considered dimensions were:

* **LC:** 100.0 <= retention_time < 100.5
* **TIMS:** scan_index = 450
* **Quadrupole:** 700.0 <= quad_mz_values < 710.0
* **TOF:** 621.9 <= tof_mz_values < 622.1

All of these analyses were timed with `timeit` and are the average of at least 7 runs. They were obtained on the following system:

* **MacBook Pro:** (13-inch, 2020, Four Thunderbolt 3 ports)
* **OS version:** macOS Catalina 10.15.7
* **Processor:** 2.3 GHz Quad-Core Intel Core i7
* **Memory:** 32 GB 3733 MHz LPDDR4X
* **Startup Disk:** Macintosh HD

Full details are available in the [perfomance notebook](nbs/performance.ipynb).

### RAM

On average, RAM usage is twice the size of a raw Bruker .d folder.

---
## Troubleshooting

Common issues include:

* **Always make sure you have activated the alphatims environment with `conda activate alphatims`.** If this fails, make sure you have installed [conda](https://docs.conda.io/en/latest/) and have created an AlphaTims environment with `conda create -n alphatims python=3.8`.
* **No `git` command**. Make sure [git](https://git-scm.com/downloads) is installed. In a notebook `!conda install git -y` might work.
* **Wrong Python version.** AlphaTims is only compatible with Python 3.8. You can check if you have the right version with the command `python --version` (or `!python --version` in a notebook). If not, reinstall the AlphaTims environment with `conda create -n alphatims python=3.8`.
* **Dependancy conflicts/issues.** Pip changed their dependancy resolver with [pip version 20.3](https://pip.pypa.io/en/stable/news/). Downgrading pip to version 20.2 with `pip install pip==20.2` (before running `pip install ./alphatims`) could solve this issue.
* **AlphaTims is not found.** Make sure you use the right folder. Local folders are best called by prefixing them with `./` (e.g. `pip install ./alphatims`). On some systems, installation specifically requires (not) to use single quotes `'` around the AlphaTims folder, e.g. `pip install './alphatims[plotting,develop]'`.
* **Modifications to the AlphaTims source code are not reflected.** Make sure you use the `-e` flag when using `pip install -e ./alphatims`.
* **Numpy does not work properly.** On Windows, `numpy==1.19.4` has some issues. After installing AlphaTims, downgrade Numpy with `pip install numpy==1.19.3`.
* Exporting PNG images with the CLI or Python package might not work out-of-the-box. If a conda environment is used, this can be fixed by running `conda install -c conda-forge firefox geckodriver` in the AlphaTims conda environment. Alternatively, a file can be exportes as html and opened in a browser, from where there is a save as png button available.

---
## How it works

The basic workflow of AlphaTims looks as follows:

* Read data from a [Bruker `.d` folder](#bruker-raw-data).
* Convert data to a [TimsTOF object in Python](#timstof-objects-in-python) and store them as a persistent [HDF5 file](https://www.hdfgroup.org/solutions/hdf5/).
* Use Python's [slicing mechanism](#slicing-timstof-objects) to retrieve data from this object e.g. for visualisation.

### Bruker raw data

Bruker stores TimsTOF raw data in a `.d` folder. The two main files in this folder are `analysis.tdf` and `analysis.tdf_bin`.

The `analysis.tdf` file is an SQL database, in which all metadata are stored together with summarised information. This includes the `Frames` table, wherein information about each individual TIMS cycle is summarised including the retention time, the number of scans (i.e. a single TOF push is related to a single ion mobility value), the summed intensity and the total number of ions that have hit the detector. More details about individual scans of the frames are available in the `PasefFrameMSMSInfo` (for PASEF acquisition) or `DiaFrameMsMsWindows` (for diaPASEF acquisition) tables. This includes quadrupole and collision settings of the frame/scan combinations.

The `analysis.tdf_bin` file is a binary file that contains the number of detected ions per individual scan, all detector arrival times and their intensity values. These values are grouped and compressed per frame (i.e. TIMS cycle), thereby allowing fast appendage during online acquisition.

### TimsTOF objects in Python

AlphaTims first reads relevant metadata from the `analysis.tdf` SQL database and creates a Python object of the `bruker.TimsTOF` class. Next, AlphaTims reads the summary information from the `Frames` table and creates three empty arrays:

* An empty `tof_indices` array, in which all TOF arrival times of each individual detector hit will be stored. Its size is determined by summing the number of detector hits for all frames.
* An empty `intensities` array of the same size, in which all intensity values of each individual detector hit will be stored.
* An empty `tof_indptr` array, that will store the number of detector hits per scan. Its size is equal to `(frame_max_index + 1) * scans_max_index + 1`. It includes one additional frame to compensate for the fact that Bruker arrays are 1-indexed, while Python uses 0-indexing. The final `+1` is because this array will be converted to an offset array, similar to the index pointer array of a [compressed sparse row matrix](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_%28CSR,_CRS_or_Yale_format%29). Typical values are `scans_max_index = 1000` and `frame_max_index = gradient_length_in_seconds * 10`, resulting in approximately `len(tof_indptr) = 10000 * gradient_length_in_seconds`.

After reading the `PasefFrameMSMSInfo` or `DiaFrameMsMsWindows` table from the `analysis.tdf` SQL database, four arrays are created:

* A `quad_indptr` array that indexes the `tof_indptr` array. Each element points to an index of the `tof_indptr` where the voltage on the quadrupole and collision cell is adjusted. For PASEF acquisitions, this is typically 20 times per MSMS frame (turning on and off a value for 10 precursor selections) and once per change from an MS (precursor) frame to an MSMS (fragment) frame. For diaPASEF, this is typically twice to 10 times per frame and with a repetitive pattern over the frame cycle. This results in an array of approximately `len(quad_indptr) = 100 * gradient_length_in_seconds`. As with the `tof_indptr` array, this array is converted to an offset array with size `+1`.
* A `quad_low_values` array of `len(quad_indptr) - 1`. This array stores the lower m/z boundary that is selected with the quadrupole. For precursors without quadrupole selection, this value is set to -1.
* A `quad_high_values` array, similar to `quad_low_values`.
* A `precursor_indices` array of `len(quad_indptr) - 1`. For PASEF this array stores the index of the selected precursor. For diaPASEF, this array stores the `WindowGroup` of the fragment frame. A value of 0 indicates an MS1 ion (i.e. precursor) without quadrupole selection.

After processing this summarising information from the `analysis.tdf` SQL database, the actual raw data from the `analysis.tdf_bin` binary file is read and stored in the empty `tof_indices`, `intensities` and `tof_indptr` arrays.

Finally, three arrays are defined that allow quick translation of `frame_`, `scan_` and `tof_indices` to `rt_values`, `mobility_values` and `mz_values` arrays.
* The `rt_values` array is read read directly from the `Frames` table in `analysis.tdf` and has a length equal to `frame_max_index + 1`. Note that an empty zeroth frame with `rt = 0` is created to make Python's 0-indexing compatible with Bruker's 1-indexing.
* The `mobility_values` array is defined by using the function `tims_scannum_to_oneoverk0` from `timsdata.dll` on the first frame and typically has a length of `1000`.
* Similarly, the `mz_values` array is defined by using the function `tims_index_to_mz` from `timsdata.dll` on the first frame. Typically this has a length of `400000`.

All these arrays can be loaded into memory, taking up roughly twice as much RAM as the `.d` folder on disk. This increase in RAM memory is mainly due to the compression used in the `analysis.tdf_bin` file. The HDF5 file can also be compressed so that its size is roughly halved and thereby has the same size as the Bruker `.d` folder, but (de)compression reduces accession times by 3-6 fold.

### Slicing TimsTOF objects

Once a Python TimsTOF object is available, it can be loaded into memory for ultrafast accession. Accession of the `data` object is done by simple Python slicing such as e.g. `selected_ion_indices = data[frame_selection, scan_selection, quad_selection, tof_selection]`. This slicing returns a `pd.DataFrame` for subsequent analysis. The columns of this dataframe contain all information for all selected ions, i.e. `frame`, `scan`, `precursor` and `tof` indices and `rt`, `mobility`, `quad_low`, `quad_high`, `mz` and `intensity` values. See the [tutorial jupyter notebook](nbs/tutorial.ipynb) for usage examples.

---
## Future perspectives

* Detection of:
  * precursor and fragment ions
  * isotopic envelopes (i.e. features)
  * fragment clusters (i.e. pseudo MSMS spectra)

---
## How to contribute

All contributions are welcome. Feel free to post a new issue or clone the repository and create a PR with a new branch. For more information see [the Contributors License Agreement](misc/CLA.md)
