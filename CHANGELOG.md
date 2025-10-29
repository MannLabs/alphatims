## Release History

For later release, see https://github.com/MannLabs/alphatims/releases

## 1.0.0

  * FEAT: tempmmap for large arrays by default.

## 0.3.2

  * FEAT: cli/gui allow bruker data as argument.
  * FEAT/FIX: Polarity included in frame table.
  * FIX: utils cleanup.
  * FIX: utils issues.
  * FEAT: by default use -1 threads in utils.
  * FIX: disable cla check.

## 0.3.1

  * FIX/FEAT: Intensity correction when ICC is used. Note that this is only for exported data, not for visualized data.
  * FEAT: By default, hdf files are now mmapped, making them much faster to initially load and use virtual memory in favor of residual memory.

## 0.3.0

  * FEAT: Introduction of global mz calibration.
  * FEAT: Introduction of dia_cycle for diaPASEF.
  * CHORE: Verified Python 3.9 compatibility.
  * FEAT: Included option to open Bruker raw data when starting the GUI.
  * FEAT: Provided hash for TimsTOF objects.
  * FEAT: Filter push indices.
  * CHORE: included stable and loose versions for all dependancies

## 0.2.8

  * FIX: Ensure stable version for one click GUI.
  * FIX: Do not require plotting dependancies for CLI export csv selection.
  * FIX: Import of very old diaPASEF samples where the analysis.tdf file still looks like ddaPASEF.
  * FIX: frame pointers of fragment_frame table.
  * FEAT: Include visual report in performance notebook.
  * FEAT: Include DIA 120 sample in performance tests.
  * FEAT: Show performance in README.
  * FIX: Move python-lzf dependancy (to decompress older Bruker files) to legacy requirements, as pip install on Windows requires visual c++ otherwise.
  * DOCS: BioRxiv paper link.
  * FEAT/FIX: RT in min column.
  * FEAT: CLI manual.
  * FEAT: Inclusion of more coordinates in CLI.

## 0.2.7

  * CHORE: Introduction of changelog.
  * CHORE: Automated publish_and_release action to parse version numbers.
  * FEAT/FIX: Include average precursor mz in MGF titles and set unknown precursor charges to 0.
  * FIX: Properly resolve set_global argument of `alphatims.utils.set_threads`.
  * FIX: Set nogil option for `alphatims.bruker.indptr_lookup`.
  * DOCS: GUI Manual typos.
  * FEAT: Include buttons to download test data and citation in GUI.
  * FEAT: Include option for progress_callback in alphatims.utils.pjit.
  * FIX/FEAT: Older samples with TimsCompressionType 1 can now also be read. This is at limited performance.
  * FEAT: By default use loose versioning for the base dependancies. Stable dependancy versions can be enforced with `pip install "alphatims[stable]"`. NOTE: This option is not guaranteed to be maintained. Future AlphaTims versions might opt for an intermediate solution with semi-strict dependancy versioning.
