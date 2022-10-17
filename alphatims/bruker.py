#!python
"""This module provides functions to handle Bruker data.
It primarily implements the TimsTOF class, that acts as an in-memory container
for Bruker data accession and storage.
"""

# builtin
import os
import sys
import contextlib
import logging
# external
import numpy as np
import pandas as pd
import h5py
# local
import alphatims
import alphatims.utils
import alphatims.tempmmap as tm

if sys.platform[:5] == "win32":
    BRUKER_DLL_FILE_NAME = os.path.join(
        alphatims.utils.EXT_PATH,
        "timsdata.dll"
    )
elif sys.platform[:5] == "linux":
    BRUKER_DLL_FILE_NAME = os.path.join(
        alphatims.utils.EXT_PATH,
        "timsdata.so"
    )
else:
    logging.warning(
        "WARNING: "
        "No Bruker libraries are available for this operating system. "
        "Mobility and m/z values need to be estimated. "
        "While this estimation often returns acceptable results with errors "
        "< 0.02 Th, huge errors (e.g. offsets of 6 Th) have already been "
        "observed for some samples!"
    )
    logging.info("")
    BRUKER_DLL_FILE_NAME = ""


def init_bruker_dll(bruker_dll_file_name: str = BRUKER_DLL_FILE_NAME):
    """Open a bruker.dll in Python.

    Five functions are defined for this dll:

        - tims_open: [c_char_p, c_uint32] -> c_uint64
        - tims_close: [c_char_p, c_uint32] -> c_uint64
        - tims_read_scans_v2: [c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32] -> c_uint32
        - tims_index_to_mz: [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32] -> None
        - tims_scannum_to_oneoverk0: Same as "tims_index_to_mz"

    Parameters
    ----------
    bruker_dll_file_name : str
        The absolute path to the timsdata.dll.
        Default is alphatims.utils.BRUKER_DLL_FILE_NAME.

    Returns
    -------
    : ctypes.cdll
        The Bruker dll library.
    """
    import ctypes
    bruker_dll = ctypes.cdll.LoadLibrary(
        os.path.realpath(bruker_dll_file_name)
    )
    bruker_dll.tims_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    bruker_dll.tims_open.restype = ctypes.c_uint64
    bruker_dll.tims_close.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_close.restype = None
    bruker_dll.tims_read_scans_v2.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32
    ]
    bruker_dll.tims_read_scans_v2.restype = ctypes.c_uint32
    bruker_dll.tims_index_to_mz.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_index_to_mz.restype = ctypes.c_uint32
    bruker_dll.tims_scannum_to_oneoverk0.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32
    ]
    bruker_dll.tims_scannum_to_oneoverk0.restype = ctypes.c_uint32
    bruker_dll.tims_set_num_threads.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_set_num_threads.restype = None
    bruker_dll.tims_set_num_threads(alphatims.utils.MAX_THREADS)
    # multiple threads is equally fast as just 1 for io?
    # bruker_dll.tims_set_num_threads(1)
    return bruker_dll


@contextlib.contextmanager
def open_bruker_d_folder(
    bruker_d_folder_name: str,
    bruker_dll_file_name=BRUKER_DLL_FILE_NAME,
) -> tuple:
    """A context manager for a bruker dll connection to a .d folder.

    Parameters
    ----------
    bruker_d_folder_name : str
        The name of a Bruker .d folder.
    bruker_dll_file_name : str, ctypes.cdll
        The path to Bruker' timsdata.dll library.
        Alternatively, the library itself can be passed as argument.
        Default is alphatims.utils.BRUKER_DLL_FILE_NAME,
        which in itself is dependent on the OS.

    Returns
    -------
    : tuple (ctypes.cdll, int).
        The opened bruker dll and identifier of the .d folder.
    """
    try:
        if isinstance(bruker_dll_file_name, str):
            bruker_dll = init_bruker_dll(bruker_dll_file_name)
        logging.info(f"Opening handle for {bruker_d_folder_name}")
        bruker_d_folder_handle = bruker_dll.tims_open(
            bruker_d_folder_name.encode('utf-8'),
            0
        )
        yield bruker_dll, bruker_d_folder_handle
    finally:
        logging.info(f"Closing handle for {bruker_d_folder_name}")
        bruker_dll.tims_close(bruker_d_folder_handle)


def read_bruker_sql(
    bruker_d_folder_name: str,
    add_zeroth_frame: bool = True,
    drop_polarity: bool = True,
    convert_polarity_to_int: bool = True,
) -> tuple:
    """Read metadata, (fragment) frames and precursors from a Bruker .d folder.

    Parameters
    ----------
    bruker_d_folder_name : str
        The name of a Bruker .d folder.
    add_zeroth_frame : bool
        Bruker uses 1-indexing for frames.
        If True, a zeroth frame is added without any TOF detections to
        make Python simulate this 1-indexing.
        If False, frames are 0-indexed.
        Default is True.
    drop_polarity : bool
        The polarity column of the frames table contains "+" or "-" and
        is not numerical.
        If True, the polarity column is dropped from the frames table.
        this ensures a fully numerical pd.DataFrame.
        If False, this column is kept, resulting in a pd.DataFrame with
        dtype=object.
        Default is True.
    convert_polarity_to_int : bool
        Convert the polarity to int (-1 or +1).
        This allows to keep it in numerical form.
        This is ignored if the polarity is dropped.
        Default is True.

    Returns
    -------
    : tuple
        (str, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame).
        The acquisition_mode, global_meta_data, frames, fragment_frames
        and precursors.
        For diaPASEF, precursors is None.
    """
    import sqlite3
    logging.info(f"Reading frame metadata for {bruker_d_folder_name}")
    with sqlite3.connect(
        os.path.join(bruker_d_folder_name, "analysis.tdf")
    ) as sql_database_connection:
        global_meta_data = pd.read_sql_query(
            "SELECT * from GlobalMetaData",
            sql_database_connection
        )
        frames = pd.read_sql_query(
            "SELECT * FROM Frames",
            sql_database_connection
        )
        if 9 in frames.MsMsType.values:
            acquisition_mode = "diaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * FROM DiaFrameMsMsInfo",
                sql_database_connection
            )
            fragment_frame_groups = pd.read_sql_query(
                "SELECT * from DiaFrameMsMsWindows",
                sql_database_connection
            )
            fragment_frames = fragment_frames.merge(
                fragment_frame_groups,
                how="left"
            )
            fragment_frames.rename(
                columns={"WindowGroup": "Precursor"},
                inplace=True
            )
            precursors = None
        elif 8 in frames.MsMsType.values:
            acquisition_mode = "ddaPASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * from PasefFrameMsMsInfo",
                sql_database_connection
            )
            precursors = pd.read_sql_query(
                "SELECT * from Precursors",
                sql_database_connection
            )
        else:
            acquisition_mode = "noPASEF"
            fragment_frames = pd.DataFrame(
                {
                    "Frame": np.array([0]),
                    "ScanNumBegin": np.array([0]),
                    "ScanNumEnd": np.array([0]),
                    "IsolationWidth": np.array([0]),
                    "IsolationMz": np.array([0]),
                    "Precursor": np.array([0]),
                }
            )
            precursors = None
            # raise ValueError("Scan mode is not ddaPASEF or diaPASEF")
        if add_zeroth_frame:
            frames = pd.concat(
                [
                    pd.DataFrame(frames.iloc[0]).T,
                    frames,
                ],
                ignore_index=True
            )
            frames.Id[0] = 0
            frames.Time[0] = 0
            frames.MaxIntensity[0] = 0
            frames.SummedIntensities[0] = 0
            frames.NumPeaks[0] = 0
            frames.MsMsType[0] = 0
        polarity_col = frames["Polarity"].copy()
        frames = pd.DataFrame(
            {
                col: pd.to_numeric(
                    frames[col]
                ) for col in frames if col != "Polarity"
            }
        )
        if not drop_polarity:
            if convert_polarity_to_int:
                frames['Polarity'] = polarity_col.apply(
                    lambda x: 1 if x == "+" else -1
                ).astype(np.int8)
            else:
                frames['Polarity'] = polarity_col
        return (
            acquisition_mode,
            global_meta_data,
            frames,
            fragment_frames,
            precursors
        )


@alphatims.utils.njit(nogil=True)
def parse_decompressed_bruker_binary_type2(decompressed_bytes: bytes) -> tuple:
    """Parse a Bruker binary frame buffer into scans, tofs and intensities.

    Parameters
    ----------
    decompressed_bytes : bytes
        A Bruker frame binary buffer that is already decompressed with pyzstd.

    Returns
    -------
    : tuple (np.uint32[:], np.uint32[:], np.uint32[:]).
        The scan_indices, tof_indices and intensities present in this binary
        array
    """
    temp = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    buffer = np.frombuffer(temp.reshape(4, -1).T.flatten(), dtype=np.uint32)
    scan_count = buffer[0]
    scan_indices = buffer[:scan_count].copy() // 2
    scan_indices[0] = 0
    tof_indices = buffer[scan_count::2].copy()
    index = 0
    for size in scan_indices:
        current_sum = 0
        for i in range(size):
            current_sum += tof_indices[index]
            tof_indices[index] = current_sum
            index += 1
    intensities = buffer[scan_count + 1::2]
    last_scan = len(intensities) - np.sum(scan_indices[1:])
    scan_indices[:-1] = scan_indices[1:]
    scan_indices[-1] = last_scan
    return scan_indices, tof_indices - 1, intensities


@alphatims.utils.njit(nogil=True)
def parse_decompressed_bruker_binary_type1(
    decompressed_bytes: bytes,
    scan_indices_: np.ndarray,
    tof_indices_: np.ndarray,
    intensities_: np.ndarray,
    scan_start: int,
    scan_index: int,
) -> int:
    """Parse a Bruker binary scan buffer into tofs and intensities.

    Parameters
    ----------
    decompressed_bytes : bytes
        A Bruker scan binary buffer that is already decompressed with lzf.
    scan_indices_ : np.ndarray
        The scan_indices_ buffer array.
    tof_indices_ : np.ndarray
        The tof_indices_ buffer array.
    intensities_ : np.ndarray
        The intensities_ buffer array.
    scan_start : int
        The offset where to start new tof_indices and intensity_values.
    scan_index : int
        The scan index.

    Returns
    -------
    : int
        The number of peaks in this scan.
    """
    buffer = np.frombuffer(decompressed_bytes, dtype=np.int32)
    tof_index = 0
    previous_was_intensity = True
    current_index = scan_start
    for value in buffer:
        if value >= 0:
            if previous_was_intensity:
                tof_index += 1
            tof_indices_[current_index] = tof_index
            intensities_[current_index] = value
            previous_was_intensity = True
            current_index += 1
        else:
            tof_index -= value
            previous_was_intensity = False
    scan_size = current_index - scan_start
    scan_indices_[scan_index] = scan_size
    return scan_size


def process_frame(
    frame_id: int,
    tdf_bin_file_name: str,
    tims_offset_values: np.ndarray,
    scan_indptr: np.ndarray,
    intensities: np.ndarray,
    tof_indices: np.ndarray,
    frame_indptr: np.ndarray,
    max_scan_count: int,
    compression_type: int,
    max_peaks_per_scan: int,
) -> None:
    """Read and parse a frame directly from a Bruker .d.analysis.tdf_bin.

    Parameters
    ----------
    frame_id : int
        The frame number that should be processed.
        Note that this is interpreted as 1-indixed instead of 0-indexed,
        so that it is compatible with Bruker.
    tdf_bin_file_name : str
        The full file name of the SQL database "analysis.tdf_bin" in a Bruker
        .d folder.
    tims_offset_values : np.int64[:]
        The offsets that indicate the starting indices of each frame in the
        binary.
        These are contained in the "TimsId" column of the frames table in
        "analysis.tdf_bin".
    scan_indptr : np.int64[:]
        A buffer containing zeros that can store the cumulative number of
        detections per scan.
        The size should be equal to max_scan_count * len(frames) + 1.
        A dummy 0-indexed frame is required to be present for len(frames).
        The last + 1 allows to explicitly interpret the end of a scan as
        the start of a subsequent scan.
    intensities : np.uint16[:]
        A buffer that can store the intensities of all detections.
        It's size can be determined by summing the "NumPeaks" column from
        the frames table in "analysis.tdf_bin".
    tof_indices : np.uint32[:]
        A buffer that can store the tof indices of all detections.
        It's size can be determined by summing the "NumPeaks" column from
        the frames table in "analysis.tdf_bin".
    frame_indptr : np.int64[:]
        The cumulative sum of the number of detections per frame.
        The size should be equal to len(frames) + 1.
        A dummy 0-indexed frame is required to be present for len(frames).
        The last + 1 allows to explicitly interpret the end of a frame as
        the start of a subsequent frame.
    max_scan_count : int
        The maximum number of scans a single frame can have.
    compression_type : int
        The compression type. This must be either 1 or 2.
        Should be treieved from the global metadata.
    max_peaks_per_scan : int
        The maximum number of peaks per scan.
        Should be retrieved from the global metadata.
    """
    with open(tdf_bin_file_name, "rb") as infile:
        frame_start = frame_indptr[frame_id]
        frame_end = frame_indptr[frame_id + 1]
        if frame_start != frame_end:
            offset = tims_offset_values[frame_id]
            infile.seek(offset)
            bin_size = int.from_bytes(infile.read(4), "little")
            scan_count = int.from_bytes(infile.read(4), "little")
            max_peak_count = min(
                max_peaks_per_scan,
                frame_end - frame_start
            )
            if compression_type == 1:
                import lzf
                compression_offset = 8 + (scan_count + 1) * 4
                scan_offsets = np.frombuffer(
                    infile.read((scan_count + 1) * 4),
                    dtype=np.int32
                ) - compression_offset
                compressed_data = infile.read(bin_size - compression_offset)
                scan_indices_ = np.zeros(scan_count, dtype=np.int64)
                tof_indices_ = np.empty(
                    frame_end - frame_start,
                    dtype=np.uint32
                )
                intensities_ = np.empty(
                    frame_end - frame_start,
                    dtype=np.uint16
                )
                scan_start = 0
                for scan_index in range(scan_count):
                    start = scan_offsets[scan_index]
                    end = scan_offsets[scan_index + 1]
                    if start == end:
                        continue
                    decompressed_bytes = lzf.decompress(
                        compressed_data[start: end],
                        max_peak_count * 4 * 2
                    )
                    scan_start += parse_decompressed_bruker_binary_type1(
                        decompressed_bytes,
                        scan_indices_,
                        tof_indices_,
                        intensities_,
                        scan_start,
                        scan_index,
                    )
            elif compression_type == 2:
                import pyzstd
                compressed_data = infile.read(bin_size - 8)
                decompressed_bytes = pyzstd.decompress(compressed_data)
                (
                    scan_indices_,
                    tof_indices_,
                    intensities_
                ) = parse_decompressed_bruker_binary_type2(decompressed_bytes)
            else:
                raise ValueError("TimsCompressionType is not 1 or 2.")
            scan_start = frame_id * max_scan_count
            scan_end = scan_start + scan_count
            scan_indptr[scan_start: scan_end] = scan_indices_
            tof_indices[frame_start: frame_end] = tof_indices_
            intensities[frame_start: frame_end] = intensities_


def read_bruker_binary(
    frames: np.ndarray,
    bruker_d_folder_name: str,
    compression_type: int,
    max_peaks_per_scan: int,
    mmap_detector_events: bool = None,
) -> tuple:
    """Read all data from an "analysis.tdf_bin" of a Bruker .d folder.

    Parameters
    ----------
    frames : pd.DataFrame
        The frames from the "analysis.tdf" SQL database of a Bruker .d folder.
        These can be acquired with e.g. alphatims.bruker.read_bruker_sql.
    bruker_d_folder_name : str
        The full path to a Bruker .d folder.
    compression_type : int
        The compression type. This must be either 1 or 2.
    max_peaks_per_scan : int
        The maximum number of peaks per scan.
        Should be treieved from the global metadata.
    mmap_detector_events : bool
        Do not save the intensity_values and tof_indices in memory,
        but use an mmap instead.
        Default is True

    Returns
    -------
    : tuple (np.int64[:], np.uint32[:], np.uint16[:]).
        The scan_indptr, tof_indices and intensities.
    """
    frame_indptr = np.empty(frames.shape[0] + 1, dtype=np.int64)
    frame_indptr[0] = 0
    frame_indptr[1:] = np.cumsum(frames.NumPeaks.values)
    max_scan_count = frames.NumScans.max() + 1
    scan_count = max_scan_count * frames.shape[0]
    scan_indptr = np.zeros(scan_count + 1, dtype=np.int64)
    if mmap_detector_events:
        intensities = tm.empty(int(frame_indptr[-1]), dtype=np.uint16)
        tof_indices = tm.empty(int(frame_indptr[-1]), dtype=np.uint32)
    else:
        intensities = np.empty(int(frame_indptr[-1]), dtype=np.uint16)
        tof_indices = np.empty(int(frame_indptr[-1]), dtype=np.uint32)
    tdf_bin_file_name = os.path.join(bruker_d_folder_name, "analysis.tdf_bin")
    tims_offset_values = frames.TimsId.values
    logging.info(
        f"Reading {frame_indptr.size - 2:,} frames with "
        f"{frame_indptr[-1]:,} detector events for {bruker_d_folder_name}"
    )
    if compression_type == 1:
        process_frame_func = alphatims.utils.threadpool(
            process_frame,
            thread_count=1
        )
    else:
        process_frame_func = alphatims.utils.threadpool(process_frame)
    process_frame_func(
        range(1, len(frames)),
        tdf_bin_file_name,
        tims_offset_values,
        scan_indptr,
        intensities,
        tof_indices,
        frame_indptr,
        max_scan_count,
        compression_type,
        max_peaks_per_scan,
    )
    scan_indptr[1:] = np.cumsum(scan_indptr[:-1])
    scan_indptr[0] = 0
    return scan_indptr, tof_indices, intensities


class TimsTOF(object):
    """A class that stores Bruker TimsTOF data in memory for fast access.

    Data can be read directly from a Bruker .d folder.
    All OS's are supported,
    but reading mz_values and mobility_values from a .d folder
    requires Windows or Linux due to availability of Bruker libraries.
    On MacOS, they are estimated based on metadata,
    but these values are not guaranteed to be correct.
    Often they fall within 0.02 Th, but errors up to 6 Th have already
    been observed!

    A TimsTOF object can also be exported to HDF for subsequent access.
    This file format is portable to all OS's.
    As such, initial reading on Windows with correct mz_values and
    mobility_values can be done and the resulting HDF file can
    safely be read on MacOS.
    This HDF file also provides improved accession times for subsequent use.

    After reading, data can be accessed with traditional Python slices.
    As TimsTOF data is 5-dimensional, the data can be sliced in 5 dimensions
    as well. These dimensions follows the design of the TimsTOF Pro:

        1 LC: rt_values, frame_indices
            The first dimension allows to slice retention_time values
            or frames indices. These values and indices
            have a one-to-one relationship.
        2 TIMS: mobility_values, scan_indices
            The second dimension allows to slice mobility values or
            scan indices (i.e. a single push).
            These values and indices have a one-to-one relationship.
        3 QUAD: quad_mz_values, precursor_indices
            The third dimension focusses on the quadrupole and indirectly
            on the collision cell. It allows to slice lower and upper
            quadrupole mz values (e.g. the m/z of
            unfragmented ions / precursors). If set to -1, the quadrupole and
            collision cell are assumed to be inactive, i.e. precursor ions
            are detected instead of fragments.
            Equally, this dimension allows to slice precursor indices.
            Precursor index 0 defaults to all precusors (i.e. quad mz values
            equal to -1). In DDA, precursor indices larger than 0 point
            to ddaPASEF MSMS spectra.
            In DIA, precursor indices larger than 0 point to windows,
            i.e. all scans in a frame with equal quadrupole and collision
            settings that is repeated once per full cycle.
            Note that these values do not have a one-to-one relationship.
        4 TOF: mz_values, tof_indices
            The fourth dimension allows to slice (fragment) mz_values
            or tof indices. Note that the quadrupole dimension determines
            if precursors are detected or fragments.
            These values and indices have a one-to-one relationship.
        5 DETECTOR: intensity_values
            The fifth dimension allows to slice intensity values.

    Note that all dimensions except for the detector have both
    (float) values and (integer) indices.
    For each dimension, slices can be provided in several different ways:

        - int:
            A single int can be used to select a single index.
            If used in the fifth dimension, it still allows to select
            intensity_values
        - float:
            A single float can be used to select a single value.
            As the values arrays are discrete, the smallest index with a value
            equal to or larger than this value is actually selected.
            For intensity_value slicing, the exact value is used.
        - slice:
            A Python slice with start, stop and step can be provided.
            Start and stop values can independently be set to int or float.
            If a float is provided it conversed to an int as previously
            described.
            The step always needs to be provided as an int.
            Since there is not one-to-one relation from values to indices for
            QUAD and DETECTOR, the step value is ignored in these cases and
            only start and stop can be used.

            **IMPORTANT NOTE:** negative start, step and stop integers are not
            supported!
        - iterable:
            An iterable with (mixed) floats and ints can also be provided,
            in a similar fashion as Numpy's fancy indexing.

            **IMPORTANT NOTE:** The resulting integers after float->int
            conversion need to be sorted in ascending order!
        - np.ndarray:
            Multiple slicing is supported by providing either a
            np.int64[:, 3] array, where each row is assumed to be a
            (start, stop, step) tuple or np.float64[:, 2] where each row
            is assumed to be a (start, stop) tuple.

            **IMPORTANT NOTE:** These arrays need to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0)
            = True).

    Alternatively, a dictionary can be used to define filters for each
    dimension (see examples).

    The result of such slicing is a pd.DataFrame with the following columns:

        - raw_indices
        - frame_indices
        - scan_indices
        - precursor_indices
        - tof_indices
        - rt_values
        - mobility_values
        - quad_low_mz_values
        - quad_high_mz_values
        - mz_values
        - intensity_values

    Instead of returning a pd.DataFrame, raw indices can be returned by
    setting the last slice element to "raw".

    Examples
    --------
    >>> data[:100.0]
    # Return all datapoints with rt_values < 100.0 seconds

    >>> data[:, 450]
    # Return all datapoints with scan_index = 450

    >>> data[:, :, 700.: 710.]
    # Return all datapoints with 700.0 <= quad_mz_values < 710.0

    >>> data[:, :, :, 621.9: 191000]
    # Return all datapoints with 621.9 <= mz_values and
    # tof_indices < 191000

    >>> data[[1, 8, 10], :, 0, 621.9: np.inf]
    # Return all datapoints from frames 1, 8 and 10, which are unfragmented
    # (precursor_index = 0) and with 621.9 <= mz_values < np.inf

    >>> data[:, :, 999]
    # Return all datapoints from precursor 999
    # (for diaPASEF this is a traditional MSMS spectrum)

    >>> scan_slices = np.array([[10, 20, 1], [100, 200, 10]])
    >>> data[:, scan_slices, :, :, :]
    # Return all datapoints with scan_indices in range(10, 20) or
    # range(100, 200, 10)

    >>> df = data[
    ...     {
    ...         "frame_indices": [1, 191],
    ...         "scan_indices": slice(300, 800, 10),
    ...         "mz_values": slice(None, 400.5),
    ...         "intensity_values": 50,
    ...     }
    ... ]
    # Slice by using a dictionary

    >>> data[:, :, 999, "raw"]
    # Return the raw indices of datapoints from precursor 999
    """

    @property
    def sample_name(self):
        """: str : The sample name of this TimsTOF object."""
        file_name = os.path.basename(self.bruker_d_folder_name)
        return '.'.join(file_name.split('.')[:-1])

    @property
    def directory(self):
        """: str : The directory of this TimsTOF object."""
        return os.path.dirname(self.bruker_d_folder_name)

    @property
    def is_compressed(self):
        """: bool : HDF array is compressed or not."""
        return self._compressed

    @property
    def version(self):
        """: str : AlphaTims version used to create this TimsTOF object."""
        return self._version

    @property
    def acquisition_mode(self):
        """: str : The acquisition mode."""
        return self._acquisition_mode

    @property
    def meta_data(self):
        """: dict : The metadata for the acquisition."""
        return self._meta_data

    @property
    def rt_values(self):
        """: np.ndarray : np.float64[:] : The rt values."""
        return self._rt_values

    @property
    def mobility_values(self):
        """: np.ndarray : np.float64[:] : The mobility values."""
        return self._mobility_values

    @property
    def cycle(self):
        """: np.ndarray : np.float64[:,:,:,:] : The quad values."""
        return self._cycle

    @property
    def mz_values(self):
        """: np.ndarray : np.float64[:] : The mz values."""
        if self._use_calibrated_mz_values_as_default:
            return self._calibrated_mz_values
        else:
            return self._mz_values

    @property
    def calibrated_mz_values(self):
        """: np.ndarray : np.float64[:] : The global calibrated mz values."""
        return self._calibrated_mz_values

    @property
    def quad_mz_values(self):
        """: np.ndarray : np.float64[:, 2] : The (low, high) quad mz values."""
        return self._quad_mz_values

    @property
    def intensity_values(self):
        """: np.ndarray : np.uint16[:] : The intensity values."""
        return self._intensity_values

    @property
    def frame_max_index(self):
        """: int : The maximum frame index."""
        return self._frame_max_index

    @property
    def scan_max_index(self):
        """: int : The maximum scan index."""
        return self._scan_max_index

    @property
    def tof_max_index(self):
        """: int : The maximum tof index."""
        return self._tof_max_index

    @property
    def precursor_max_index(self):
        """: int : The maximum precursor index."""
        return self._precursor_max_index

    @property
    def mz_min_value(self):
        """: float : The minimum mz value."""
        return self.mz_values[0]

    @property
    def mz_max_value(self):
        """: float : The maximum mz value."""
        return self.mz_values[-1]

    @property
    def calibrated_mz_min_value(self):
        """: float : The minimum calibrated mz value."""
        return self.calibrated_mz_values[0]

    @property
    def calibrated_mz_max_value(self):
        """: float : The maximum calibrated mz value."""
        return self.calibrated_mz_values[-1]

    @property
    def rt_max_value(self):
        """: float : The maximum rt value."""
        return self.rt_values[-1]

    @property
    def quad_mz_min_value(self):
        """: float : The minimum quad mz value."""
        return self._quad_min_mz_value

    @property
    def quad_mz_max_value(self):
        """: float : The maximum quad mz value."""
        return self._quad_max_mz_value

    @property
    def mobility_min_value(self):
        """: float : The minimum mobility value."""
        return self._mobility_min_value

    @property
    def mobility_max_value(self):
        """: float : The maximum mobility value."""
        return self._mobility_max_value

    @property
    def intensity_min_value(self):
        """: float : The minimum intensity value."""
        return self._intensity_min_value

    @property
    def intensity_max_value(self):
        """: float : The maximum intensity value."""
        return self._intensity_max_value

    @property
    def frames(self):
        """: pd.DataFrame : The frames table of the analysis.tdf SQL."""
        return self._frames

    @property
    def fragment_frames(self):
        """: pd.DataFrame : The fragment frames table."""
        return self._fragment_frames

    @property
    def precursors(self):
        """: pd.DataFrame : The precursor table."""
        return self._precursors

    @property
    def tof_indices(self):
        """: np.ndarray : np.uint32[:] : The tof indices."""
        return self._tof_indices

    @property
    def push_indptr(self):
        """: np.ndarray : np.int64[:] : The tof indptr."""
        return self._push_indptr

    @property
    def quad_indptr(self):
        """: np.ndarray : np.int64[:] : The quad indptr (tof_indices)."""
        return self._quad_indptr

    @property
    def raw_quad_indptr(self):
        """: np.ndarray : np.int64[:] : The raw quad indptr (push indices)."""
        return self._raw_quad_indptr

    @property
    def precursor_indices(self):
        """: np.ndarray : np.int64[:] : The precursor indices."""
        return self._precursor_indices

    @property
    def dia_precursor_cycle(self):
        """: np.ndarray : np.int64[:] : The precursor indices of a DIA cycle."""
        return self._dia_precursor_cycle

    @property
    def dia_mz_cycle(self):
        """: np.ndarray : np.float64[:, 2] : The mz_values of a DIA cycle."""
        return self._dia_mz_cycle

    @property
    def zeroth_frame(self):
        """: bool : A blank zeroth frame is present so frames are 1-indexed."""
        return self._zeroth_frame

    @property
    def max_accumulation_time(self):
        """: float : The maximum accumulation time of all frames."""
        return self._max_accumulation_time

    @property
    def accumulation_times(self):
        """: np.ndarray : The accumulation times of all frames."""
        return self._accumulation_times

    @property
    def intensity_corrections(self):
        """: np.ndarray : The intensity_correction per frame."""
        return self._intensity_corrections

    def __init__(
        self,
        bruker_d_folder_name: str,
        *,
        mz_estimation_from_frame: int = 1,
        mobility_estimation_from_frame: int = 1,
        slice_as_dataframe: bool = True,
        use_calibrated_mz_values_as_default: int = 0,
        use_hdf_if_available: bool = True,
        mmap_detector_events: bool = True,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
    ):
        """Create a Bruker TimsTOF object that contains all data in-memory.

        Parameters
        ----------
        bruker_d_folder_name : str
            The full file name to a Bruker .d folder.
            Alternatively, the full file name of an already exported .hdf
            can be provided as well.
        mz_estimation_from_frame : int
            If larger than 0, mz_values from this frame are read as
            default mz_values with the Bruker library.
            If 0, mz_values are being estimated with the metadata
            based on "MzAcqRangeLower" and "MzAcqRangeUpper".
            IMPORTANT NOTE: MacOS defaults to 0, as no Bruker library
            is available.
            Default is 1.
        mobility_estimation_from_frame : int
            If larger than 0, mobility_values from this frame are read as
            default mobility_values with the Bruker library.
            If 0, mobility_values are being estimated with the metadata
            based on "OneOverK0AcqRangeLower" and "OneOverK0AcqRangeUpper".
            IMPORTANT NOTE: MacOS defaults to 0, as no Bruker library
            is available.
            Default is 1.
        slice_as_dataframe : bool
            If True, slicing returns a pd.DataFrame by default.
            If False, slicing provides a np.int64[:] with raw indices.
            This value can also be modified after creation.
            Default is True.
        use_calibrated_mz_values : int
            If not 0, the mz_values are overwritten with global
            calibrated_mz_values.
            If 1, calibration at the MS1 level is performed.
            If 2, calibration at the MS2 level is performed.
            Default is 0.
        use_hdf_if_available : bool
            If an HDF file is available, use this instead of the .d folder.
            Default is True.
        mmap_detector_events : bool
            Do not save the intensity_values and tof_indices in memory,
            but use an mmap instead.
            Default is True
        drop_polarity : bool
            The polarity column of the frames table contains "+" or "-" and
            is not numerical.
            If True, the polarity column is dropped from the frames table.
            this ensures a fully numerical pd.DataFrame.
            If False, this column is kept, resulting in a pd.DataFrame with
            dtype=object.
            Default is True.
        convert_polarity_to_int : bool
            Convert the polarity to int (-1 or +1).
            This allows to keep it in numerical form.
            This is ignored if the polarity is dropped.
            Default is True.
        """
        if bruker_d_folder_name.endswith("/"):
            bruker_d_folder_name = bruker_d_folder_name[:-1]
        logging.info(f"Importing data from {bruker_d_folder_name}")
        if bruker_d_folder_name.endswith(".d"):
            bruker_hdf_file_name = f"{bruker_d_folder_name[:-2]}.hdf"
            hdf_file_exists = os.path.exists(bruker_hdf_file_name)
            if use_hdf_if_available and hdf_file_exists:
                self._import_data_from_hdf_file(
                    bruker_hdf_file_name,
                    mmap_detector_events,
                )
                self.bruker_hdf_file_name = bruker_hdf_file_name
            else:
                self.bruker_d_folder_name = os.path.abspath(
                    bruker_d_folder_name
                )
                self._import_data_from_d_folder(
                    bruker_d_folder_name,
                    mz_estimation_from_frame,
                    mobility_estimation_from_frame,
                    drop_polarity,
                    convert_polarity_to_int,
                    mmap_detector_events,
                )
        elif bruker_d_folder_name.endswith(".hdf"):
            self._import_data_from_hdf_file(
                bruker_d_folder_name,
                mmap_detector_events,
            )
            self.bruker_hdf_file_name = bruker_d_folder_name
        else:
            raise NotImplementedError(
                "WARNING: file extension not understood"
            )
        if not hasattr(self, "version"):
            self._version = "N.A."
        if self.version != alphatims.__version__:
            logging.info(
                "WARNING: "
                f"AlphaTims version {self.version} was used to initialize "
                f"{bruker_d_folder_name}, while the current version of "
                f"AlphaTims is {alphatims.__version__}."
            )
        self.slice_as_dataframe = slice_as_dataframe
        self.use_calibrated_mz_values_as_default(
            use_calibrated_mz_values_as_default
        )
        # Precompile
        self[0, "raw"]
        logging.info(f"Successfully imported data from {bruker_d_folder_name}")

    def __len__(self):
        return len(self.intensity_values)

    def __hash__(self):
        return hash(self.bruker_d_folder_name)

    def _import_data_from_d_folder(
        self,
        bruker_d_folder_name: str,
        mz_estimation_from_frame: int,
        mobility_estimation_from_frame: int,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
        mmap_detector_events: bool = True
    ):
        logging.info(f"Using .d import for {bruker_d_folder_name}")
        self._version = alphatims.__version__
        self._zeroth_frame = True
        (
            self._acquisition_mode,
            global_meta_data,
            self._frames,
            self._fragment_frames,
            self._precursors,
        ) = read_bruker_sql(
            bruker_d_folder_name,
            self._zeroth_frame,
            drop_polarity,
            convert_polarity_to_int,
        )
        self._meta_data = dict(
            zip(global_meta_data.Key, global_meta_data.Value)
        )
        (
            self._push_indptr,
            self._tof_indices,
            self._intensity_values,
        ) = read_bruker_binary(
            self.frames,
            bruker_d_folder_name,
            int(self._meta_data["TimsCompressionType"]),
            int(self._meta_data["MaxNumPeaksPerScan"]),
        )
        logging.info(f"Indexing {bruker_d_folder_name}...")
        self._use_calibrated_mz_values_as_default = False
        self._frame_max_index = self.frames.shape[0]
        self._scan_max_index = int(self.frames.NumScans.max()) + 1
        self._tof_max_index = int(self.meta_data["DigitizerNumSamples"]) + 1
        self._rt_values = self.frames.Time.values.astype(np.float64)
        self._mobility_min_value = float(
            self.meta_data["OneOverK0AcqRangeLower"]
        )
        self._mobility_max_value = float(
            self.meta_data["OneOverK0AcqRangeUpper"]
        )
        self._accumulation_times = self.frames.AccumulationTime.values.astype(
            np.float64
        )
        self._max_accumulation_time = np.max(self._accumulation_times)
        self._intensity_corrections = self._max_accumulation_time / self._accumulation_times
        bruker_dll_available = BRUKER_DLL_FILE_NAME != ""
        if (mobility_estimation_from_frame != 0) and bruker_dll_available:
            import ctypes
            with alphatims.bruker.open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mobility values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.scan_max_index).astype(np.float64)
                self._mobility_values = np.empty_like(indices)
                bruker_dll.tims_scannum_to_oneoverk0(
                    bruker_d_folder_handle,
                    mobility_estimation_from_frame,
                    indices.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.mobility_values.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.scan_max_index
                )
        else:
            if (mobility_estimation_from_frame != 0):
                logging.info(
                    "Bruker DLL not available, estimating mobility values"
                )
            self._mobility_values = self.mobility_max_value - (
                self.mobility_max_value - self.mobility_min_value
            ) / self.scan_max_index * np.arange(self.scan_max_index)
        mz_min_value = float(self.meta_data["MzAcqRangeLower"])
        mz_max_value = float(self.meta_data["MzAcqRangeUpper"])
        if self.meta_data["AcquisitionSoftware"] == "Bruker otofControl":
            logging.warning(
                "WARNING: Acquisition software is Bruker otofControl, "
                "mz min/max values are assumed to be 5 m/z wider than "
                "defined in analysis.tdf!"
            )
            mz_min_value -= 5
            mz_max_value += 5
        tof_intercept = np.sqrt(mz_min_value)
        tof_slope = (
            np.sqrt(mz_max_value) - tof_intercept
        ) / self.tof_max_index
        if (mz_estimation_from_frame != 0) and bruker_dll_available:
            import ctypes
            with alphatims.bruker.open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mz values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.tof_max_index).astype(np.float64)
                self._mz_values = np.empty_like(indices)
                bruker_dll.tims_index_to_mz(
                    bruker_d_folder_handle,
                    mz_estimation_from_frame,
                    indices.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self._mz_values.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.tof_max_index
                )
        else:
            if (mz_estimation_from_frame != 0):
                logging.info(
                    "Bruker DLL not available, estimating mz values"
                )
            self._mz_values = (
                tof_intercept + tof_slope * np.arange(self.tof_max_index)
            )**2
        self._parse_quad_indptr()
        self._intensity_min_value = int(np.min(self.intensity_values))
        self._intensity_max_value = int(np.max(self.intensity_values))
        if self.acquisition_mode == "diaPASEF":
            self.set_cycle()

    def save_as_hdf(
        self,
        directory: str = None,
        file_name: str = None,
        overwrite: bool = False,
        compress: bool = False,
        return_as_bytes_io: bool = False,
    ):
        """Save the TimsTOF object as an hdf file.

        Parameters
        ----------
        directory : str
            The directory where to save the HDF file.
            Ignored if return_as_bytes_io == True.
            Default is None, meaning it is parsed from input file.
        file_name : str
            The file name of the  HDF file.
            Ignored if return_as_bytes_io == True.
            Default is None, meaning it is parsed from input file.
        overwrite : bool
            If True, an existing file is truncated.
            If False, the existing file is appended to only if the original
            group, array or property does not exist yet.
            Default is False.
        compress : bool
            If True, compression is used.
            This roughly halves files sizes (on-disk),
            at the cost of taking 3-6 longer accession times.
            See also alphatims.utils.create_hdf_group_from_dict.
            If False, no compression is used
            Default is False.
        return_as_bytes_io
            If True, the HDF file is only created in memory and returned
            as a bytes stream.
            If False, the file is written to disk.
            Default is False.

        Returns
        -------
        str, io.BytesIO
            The full file name or a bytes stream containing the HDF file.
        """
        import io
        if directory is None:
            directory = self.directory
        if file_name is None:
            file_name = f"{self.sample_name}.hdf"
        if overwrite:
            hdf_mode = "w"
        else:
            hdf_mode = "a"
        if return_as_bytes_io:
            full_file_name = io.BytesIO()
        else:
            full_file_name = os.path.join(
                directory,
                file_name
            )
        logging.info(
            f"Writing TimsTOF data to {full_file_name}."
        )
        self._compressed = compress
        with h5py.File(full_file_name, hdf_mode) as hdf_root:
            # hdf_root.swmr_mode = True
            alphatims.utils.create_hdf_group_from_dict(
                hdf_root.create_group("raw"),
                self.__dict__,
                overwrite=overwrite,
                compress=compress,
            )
        if return_as_bytes_io:
            full_file_name.seek(0)
        else:
            logging.info(
                f"Successfully wrote TimsTOF data to {full_file_name}."
            )
        return full_file_name

    def _import_data_from_hdf_file(
        self,
        bruker_d_folder_name: str,
        mmap_detector_events: bool = False,
    ):
        logging.info(f"Using HDF import for {bruker_d_folder_name}")
        with h5py.File(bruker_d_folder_name, "r") as hdf_root:
            mmap_arrays = []
            if mmap_detector_events:
                mmap_arrays.append("/raw/_tof_indices")
                mmap_arrays.append("/raw/_intensity_values")
            self.__dict__ = alphatims.utils.create_dict_from_hdf_group(
                hdf_root["raw"],
                mmap_arrays,
                bruker_d_folder_name,
            )

    def convert_from_indices(
        self,
        raw_indices,
        *,
        frame_indices=None,
        quad_indices=None,
        scan_indices=None,
        tof_indices=None,
        return_raw_indices: bool = False,
        return_frame_indices: bool = False,
        return_scan_indices: bool = False,
        return_quad_indices: bool = False,
        return_tof_indices: bool = False,
        return_precursor_indices: bool = False,
        return_rt_values: bool = False,
        return_rt_values_min: bool = False,
        return_mobility_values: bool = False,
        return_quad_mz_values: bool = False,
        return_push_indices: bool = False,
        return_mz_values: bool = False,
        return_intensity_values: bool = False,
        return_corrected_intensity_values: bool = False,
        raw_indices_sorted: bool = False,
    ) -> dict:
        """Convert selected indices to a dict.

        Parameters
        ----------
        raw_indices : np.int64[:], None
            The raw indices for which coordinates need to be retrieved.
        frame_indices : np.int64[:], None
            The frame indices for which coordinates need to be retrieved.
        quad_indices : np.int64[:], None
            The quad indices for which coordinates need to be retrieved.
        scan_indices : np.int64[:], None
            The scan indices for which coordinates need to be retrieved.
        tof_indices : np.int64[:], None
            The tof indices for which coordinates need to be retrieved.
        return_raw_indices : bool
            If True, include "raw_indices" in the dict.
            Default is False.
        return_frame_indices : bool
            If True, include "frame_indices" in the dict.
            Default is False.
        return_scan_indices : bool
            If True, include "scan_indices" in the dict.
            Default is False.
        return_quad_indices : bool
            If True, include "quad_indices" in the dict.
            Default is False.
        return_tof_indices : bool
            If True, include "tof_indices" in the dict.
            Default is False.
        return_precursor_indices : bool
            If True, include "precursor_indices" in the dict.
            Default is False.
        return_rt_values : bool
            If True, include "rt_values" in the dict.
            Default is False.
        return_rt_values_min : bool
            If True, include "rt_values_min" in the dict.
            Default is False.
        return_mobility_values : bool
            If True, include "mobility_values" in the dict.
            Default is False.
        return_quad_mz_values : bool
            If True, include "quad_low_mz_values" and
            "quad_high_mz_values" in the dict.
            Default is False.
        return_push_indices : bool
            If True, include "push_indices" in the dict.
            Default is False.
        return_mz_values : bool
            If True, include "mz_values" in the dict.
            Default is False.
        return_intensity_values : bool
            If True, include "intensity_values" in the dict.
            Default is False.
        return_corrected_intensity_values : bool
            If True, include "corrected_intensity_values" in the dict.
            Default is False.
        raw_indices_sorted : bool
            If True, raw_indices are assumed to be sorted,
            resulting in a faster conversion.
            Default is False.

        Returns
        -------
        dict
            A dict with all requested columns.
        """
        try:
            iter(raw_indices)
        except TypeError:
            raw_indices = [raw_indices]
        if not isinstance(raw_indices, np.ndarray):
            raw_indices = np.array(raw_indices)
        result = {}
        if (raw_indices is not None) and any(
            [
                return_frame_indices,
                return_scan_indices,
                return_quad_indices,
                return_rt_values,
                return_rt_values_min,
                return_mobility_values,
                return_quad_mz_values,
                return_precursor_indices,
                return_push_indices,
                return_corrected_intensity_values,
            ]
        ):
            if raw_indices_sorted:
                push_indices = indptr_lookup(
                    self.push_indptr,
                    raw_indices,
                )
            else:
                push_indices = np.searchsorted(
                    self.push_indptr,
                    raw_indices,
                    "right"
                ) - 1
        if (
            any(
                [
                    return_frame_indices,
                    return_rt_values,
                    return_rt_values_min,
                    return_corrected_intensity_values,
                ]
            )
        ) and (
            frame_indices is None
        ):
            frame_indices = push_indices // self.scan_max_index
        if (return_scan_indices or return_mobility_values) and (
            scan_indices is None
        ):
            scan_indices = push_indices % self.scan_max_index
        if any(
            [
                return_quad_indices,
                return_quad_mz_values,
                return_precursor_indices
            ]
        ) and (
            quad_indices is None
        ):
            if raw_indices_sorted:
                quad_indices = indptr_lookup(
                    self.quad_indptr,
                    raw_indices,
                )
            else:
                quad_indices = np.searchsorted(
                    self.quad_indptr,
                    raw_indices,
                    "right"
                ) - 1
        if (return_tof_indices or return_mz_values) and (tof_indices is None):
            tof_indices = self.tof_indices[raw_indices]
        if return_raw_indices:
            result["raw_indices"] = raw_indices
        if return_frame_indices:
            result["frame_indices"] = frame_indices
        if return_scan_indices:
            result["scan_indices"] = scan_indices
        if return_quad_indices:
            result["quad_indices"] = quad_indices
        if return_precursor_indices:
            result["precursor_indices"] = self.precursor_indices[quad_indices]
        if return_push_indices:
            result["push_indices"] = push_indices
        if return_tof_indices:
            result["tof_indices"] = tof_indices
        if return_rt_values:
            result["rt_values"] = self.rt_values[frame_indices]
        if return_rt_values_min:
            if "rt_values" in result:
                result['rt_values_min'] = result["rt_values"] / 60
            else:
                result['rt_values_min'] = self.rt_values[frame_indices] / 60
        if return_mobility_values:
            result["mobility_values"] = self.mobility_values[scan_indices]
        if return_quad_mz_values:
            selected_quad_values = self.quad_mz_values[quad_indices]
            low_mz_values = selected_quad_values[:, 0]
            high_mz_values = selected_quad_values[:, 1]
            result["quad_low_mz_values"] = low_mz_values
            result["quad_high_mz_values"] = high_mz_values
        if return_mz_values:
            result["mz_values"] = self.mz_values[tof_indices]
        if return_intensity_values:
            result["intensity_values"] = self.intensity_values[raw_indices]
        if return_corrected_intensity_values:
            result["corrected_intensity_values"] = (
                self.intensity_values[raw_indices] * self.intensity_corrections[frame_indices]
            ).astype(np.uint32)
        return result

    def convert_to_indices(
        self,
        values: np.ndarray,
        *,
        return_frame_indices: bool = False,
        return_scan_indices: bool = False,
        return_tof_indices: bool = False,
        side: str = "left",
        return_type: str = "",
    ):
        """Convert selected values to an array in the requested dimension.

        Parameters
        ----------
        values : float, np.float64[...], iterable
            The raw values for which indices need to be retrieved.
        return_frame_indices : bool
            If True, convert the values to "frame_indices".
            Default is False.
        return_scan_indices : bool
            If True, convert the values to "scan_indices".
            Default is False.
        return_tof_indices : bool
            If True, convert the values to "tof_indices".
            Default is False.
        side : str
            If there is an exact match between the values and reference array,
            which index should be chosen. See also np.searchsorted.
            Options are "left" or "right".
            Default is "left".
        return_type : str
            Alternative way to define the return type.
            Options are "frame_indices", "scan_indices" or "tof_indices".
            Default is "".

        Returns
        -------
        np.int64[...], int
            An array with the same shape as values or iterable or an int
            which corresponds to the requested value.

        Raises
        ------
        PrecursorFloatError
            When trying to convert a quad float other than np.inf or -np.inf
            to precursor index.
        """
        if return_frame_indices:
            return_type = "frame_indices"
        elif return_scan_indices:
            return_type = "scan_indices"
        elif return_tof_indices:
            return_type = "tof_indices"
        if return_type == "frame_indices":
            return np.searchsorted(self.rt_values, values, side)
        elif return_type == "scan_indices":
            return self.scan_max_index - np.searchsorted(
                self.mobility_values[::-1],
                values,
                "left" if side == "right" else "right"
            )
        elif return_type == "tof_indices":
            return np.searchsorted(self.mz_values, values, side)
        elif return_type == "precursor_indices":
            try:
                if values not in [-np.inf, np.inf]:
                    raise PrecursorFloatError(
                        "Can not convert values to precursor_indices"
                    )
            except ValueError:
                raise PrecursorFloatError(
                    "Can not convert values to precursor_indices"
                )
            if values == -np.inf:
                return 0
            elif values == np.inf:
                return self.precursor_max_index
        else:
            raise KeyError(f"return_type '{return_type}' is invalid")

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = tuple([keys])
        if isinstance(keys[-1], str):
            if keys[-1] == "df":
                as_dataframe = True
            elif keys[-1] == "raw":
                as_dataframe = False
            else:
                raise ValueError(f"Cannot use {keys[-1]} as a key")
            keys = keys[:-1]
        else:
            as_dataframe = self.slice_as_dataframe
        parsed_keys = parse_keys(self, keys)
        raw_indices = filter_indices(
            frame_slices=parsed_keys["frame_indices"],
            scan_slices=parsed_keys["scan_indices"],
            precursor_slices=parsed_keys["precursor_indices"],
            tof_slices=parsed_keys["tof_indices"],
            quad_slices=parsed_keys["quad_values"],
            intensity_slices=parsed_keys["intensity_values"],
            frame_max_index=self.frame_max_index,
            scan_max_index=self.scan_max_index,
            push_indptr=self.push_indptr,
            precursor_indices=self.precursor_indices,
            quad_mz_values=self.quad_mz_values,
            quad_indptr=self.quad_indptr,
            tof_indices=self.tof_indices,
            intensities=self.intensity_values
        )
        if as_dataframe:
            return self.as_dataframe(raw_indices)
        else:
            return raw_indices

    def estimate_strike_count(
        self,
        frame_slices: np.ndarray,
        scan_slices: np.ndarray,
        precursor_slices: np.ndarray,
        tof_slices: np.ndarray,
        quad_slices: np.ndarray,
    ) -> int:
        """Estimate the number of detector events, given a set of slices.

        Parameters
        ----------
        frame_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).
        scan_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).
        precursor_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0)
            = True).
        tof_slices : np.int64[:, 3]
            Each row of the array is assumed to be a (start, stop, step) tuple.
            This array is assumed to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
        quad_slices : np.float64[:, 2]
            Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
            This array is assumed to be sorted,
            disjunct and strictly increasing
            (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).

        Returns
        -------
        int
            The estimated number of detector events given these slices.
        """
        frame_count = 0
        for frame_start, frame_end, frame_stop in frame_slices:
            frame_count += len(range(frame_start, frame_end, frame_stop))
        scan_count = 0
        for scan_start, scan_end, scan_stop in scan_slices:
            scan_count += len(range(scan_start, scan_end, scan_stop))
        tof_count = 0
        for tof_start, tof_end, tof_stop in tof_slices:
            tof_count += len(range(tof_start, tof_end, tof_stop))
        precursor_count = 0
        precursor_index_included = False
        for precursor_start, precursor_end, precursor_stop in precursor_slices:
            precursor_count += len(
                range(precursor_start, precursor_end, precursor_stop)
            )
            if 0 in range(precursor_start, precursor_end, precursor_stop):
                precursor_index_included = True
        quad_count = 0
        precursor_quad_included = False
        for quad_start, quad_end in quad_slices:
            if quad_start < 0:
                precursor_quad_included = True
            if quad_start < self.quad_mz_min_value:
                quad_start = self.quad_mz_min_value
            if quad_end > self.quad_mz_max_value:
                quad_end = self.quad_mz_max_value
            if quad_start < quad_end:
                quad_count += quad_end - quad_start
        estimated_count = len(self)
        estimated_count *= frame_count / self.frame_max_index
        estimated_count *= scan_count / self.scan_max_index
        estimated_count *= tof_count / self.tof_max_index
        fragment_multiplier = 0.5 * min(
            precursor_count / (self.precursor_max_index),
            quad_count / (
                self.quad_mz_max_value - self.quad_mz_min_value
            )
        )
        if fragment_multiplier < 0:
            fragment_multiplier = 0
        if precursor_index_included and precursor_quad_included:
            fragment_multiplier += 0.5
        estimated_count *= fragment_multiplier
        return int(estimated_count)

    def bin_intensities(self, indices: np.ndarray, axis: tuple):
        """Sum and project the intensities of the indices along 1 or 2 axis.

        Parameters
        ----------
        indices : np.int64[:]
            The selected indices whose coordinates need to be summed along
            the selected axis.
        axis : tuple
            Must be length 1 or 2 and can only contain the elements
            "rt_values", "mobility_values" and "mz_values".

        Returns
        -------
        np.float64[:], np.float64[:, 2]
            An array or heatmap that express the summed intensity along
            the selected axis.
        """
        intensities = self.intensity_values[indices].astype(np.float64)
        max_index = {
            "rt_values": self.frame_max_index,
            "mobility_values": self.scan_max_index,
            "mz_values": self.tof_max_index,
        }
        parsed_indices = self.convert_from_indices(
            indices,
            return_frame_indices="rt_values" in axis,
            return_scan_indices="mobility_values" in axis,
            return_tof_indices="mz_values" in axis,
        )
        binned_intensities = np.zeros(tuple([max_index[ax] for ax in axis]))
        parse_dict = {
            "rt_values": "frame_indices",
            "mobility_values": "scan_indices",
            "mz_values": "tof_indices",
        }
        add_intensity_to_bin(
            range(indices.size),
            intensities,
            tuple(
                [
                    parsed_indices[parse_dict[ax]] for ax in axis
                ]
            ),
            binned_intensities
        )
        return binned_intensities

    def as_dataframe(
        self,
        indices: np.ndarray,
        *,
        raw_indices: bool = True,
        frame_indices: bool = True,
        scan_indices: bool = True,
        quad_indices: bool = False,
        tof_indices: bool = True,
        precursor_indices: bool = True,
        rt_values: bool = True,
        rt_values_min: bool = True,
        mobility_values: bool = True,
        quad_mz_values: bool = True,
        push_indices: bool = True,
        mz_values: bool = True,
        intensity_values: bool = True,
        corrected_intensity_values: bool = True,
        raw_indices_sorted: bool = False,
    ):
        """Convert raw indices to a pd.DataFrame.

        Parameters
        ----------
        indices : np.int64[:]
            The raw indices for which coordinates need to be retrieved.
        raw_indices : bool
            If True, include "raw_indices" in the dataframe.
            Default is True.
        frame_indices : bool
            If True, include "frame_indices" in the dataframe.
            Default is True.
        scan_indices : bool
            If True, include "scan_indices" in the dataframe.
            Default is True.
        quad_indices : bool
            If True, include "quad_indices" in the dataframe.
            Default is False.
        tof_indices : bool
            If True, include "tof_indices" in the dataframe.
            Default is True.
        precursor_indices : bool
            If True, include "precursor_indices" in the dataframe.
            Default is True.
        rt_values : bool
            If True, include "rt_values" in the dataframe.
            Default is True.
        rt_values_min : bool
            If True, include "rt_values_min" in the dataframe.
            Default is True.
        mobility_values : bool
            If True, include "mobility_values" in the dataframe.
            Default is True.
        quad_mz_values : bool
            If True, include "quad_low_mz_values" and
            "quad_high_mz_values" in the dict.
            Default is True.
        push_indices : bool
            If True, include "push_indices" in the dataframe.
            Default is True.
        mz_values : bool
            If True, include "mz_values" in the dataframe.
            Default is True.
        intensity_values : bool
            If True, include "intensity_values" in the dataframe.
            Default is True.
        corrected_intensity_values : bool
            If True, include "corrected_intensity_values" in the dataframe.
            Default is True.
        raw_indices_sorted : bool
            If True, raw_indices are assumed to be sorted,
            resulting in a faster conversion.
            Default is False.

        Returns
        -------
        pd.DataFrame
            A dataframe with all requested columns.
        """
        return pd.DataFrame(
           self.convert_from_indices(
                indices,
                return_raw_indices=raw_indices,
                return_frame_indices=frame_indices,
                return_scan_indices=scan_indices,
                return_quad_indices=quad_indices,
                return_precursor_indices=precursor_indices,
                return_tof_indices=tof_indices,
                return_rt_values=rt_values,
                return_rt_values_min=rt_values_min,
                return_mobility_values=mobility_values,
                return_quad_mz_values=quad_mz_values,
                return_push_indices=push_indices,
                return_mz_values=mz_values,
                return_intensity_values=intensity_values,
                return_corrected_intensity_values=corrected_intensity_values,
                raw_indices_sorted=raw_indices_sorted,
            )
        )

    def _parse_quad_indptr(self) -> None:
        logging.info("Indexing quadrupole dimension")
        frame_ids = self.fragment_frames.Frame.values + 1
        scan_begins = self.fragment_frames.ScanNumBegin.values
        scan_ends = self.fragment_frames.ScanNumEnd.values
        isolation_mzs = self.fragment_frames.IsolationMz.values
        isolation_widths = self.fragment_frames.IsolationWidth.values
        precursors = self.fragment_frames.Precursor.values
        if (precursors[0] is None):
            if self.zeroth_frame:
                frame_groups = self.frames.MsMsType.values[1:]
            else:
                frame_groups = self.frames.MsMsType.values
            precursor_frames = np.flatnonzero(frame_groups == 0)
            group_sizes = np.diff(precursor_frames)
            group_size = group_sizes[0]
            if np.any(group_sizes != group_size):
                raise ValueError("Sample type not understood")
            precursors = (1 + frame_ids - frame_ids[0]) % group_size
            if self.zeroth_frame:
                precursors[0] = 0
            self.fragment_frames.Precursor = precursors
            self._acquisition_mode = "diaPASEF"
        scan_max_index = self.scan_max_index
        frame_max_index = self.frame_max_index
        quad_indptr = [0]
        quad_low_values = []
        quad_high_values = []
        precursor_indices = []
        high = -1
        for (
            frame_id,
            scan_begin,
            scan_end,
            isolation_mz,
            isolation_width,
            precursor
        ) in zip(
            frame_ids - 1,
            scan_begins,
            scan_ends,
            isolation_mzs,
            isolation_widths / 2,
            precursors
        ):
            low = frame_id * scan_max_index + scan_begin
            # TODO: CHECK?
            # if low < high:
            #     print(frame_id, low, frame_id * scan_max_index + scan_end, high, low - high)
            if low != high:
                quad_indptr.append(low)
                quad_low_values.append(-1)
                quad_high_values.append(-1)
                precursor_indices.append(0)
            high = frame_id * scan_max_index + scan_end
            quad_indptr.append(high)
            quad_low_values.append(isolation_mz - isolation_width)
            quad_high_values.append(isolation_mz + isolation_width)
            precursor_indices.append(precursor)
        quad_max_index = scan_max_index * frame_max_index
        if high < quad_max_index:
            quad_indptr.append(quad_max_index)
            quad_low_values.append(-1)
            quad_high_values.append(-1)
            precursor_indices.append(0)
        self._quad_mz_values = np.stack([quad_low_values, quad_high_values]).T
        self._precursor_indices = np.array(precursor_indices)
        self._raw_quad_indptr = np.array(quad_indptr)
        self._quad_indptr = self.push_indptr[self._raw_quad_indptr]
        self._quad_max_mz_value = np.max(self.quad_mz_values[:, 1])
        self._quad_min_mz_value = np.min(
            self.quad_mz_values[
                self.quad_mz_values[:, 0] >= 0,
                0
            ]
        )
        self._precursor_max_index = int(np.max(self.precursor_indices)) + 1
        if self._acquisition_mode == "diaPASEF":
            offset = int(self.zeroth_frame)
            cycle_index = np.searchsorted(
                self.raw_quad_indptr,
                (self.scan_max_index) * (self.precursor_max_index + offset),
                "r"
            ) + 1
            repeats = np.diff(self.raw_quad_indptr[: cycle_index])
            if self.zeroth_frame:
                repeats[0] -= self.scan_max_index
            cycle_length = self.scan_max_index * self.precursor_max_index
            repeat_length = np.sum(repeats)
            if repeat_length != cycle_length:
                repeats[-1] -= repeat_length - cycle_length
            self._dia_mz_cycle = np.empty((cycle_length, 2))
            self._dia_mz_cycle[:, 0] = np.repeat(
                self.quad_mz_values[: cycle_index - 1, 0],
                repeats
            )
            self._dia_mz_cycle[:, 1] = np.repeat(
                self.quad_mz_values[: cycle_index - 1, 1],
                repeats
            )
            self._dia_precursor_cycle = np.repeat(
                self.precursor_indices[: cycle_index - 1],
                repeats
            )
        else:
            self._dia_mz_cycle = np.empty((0, 2))
            self._dia_precursor_cycle = np.empty(0, dtype=np.int64)

    def index_precursors(
        self,
        centroiding_window: int = 0,
        keep_n_most_abundant_peaks: int = -1,
    ) -> tuple:
        """Retrieve all MS2 spectra acquired with DDA.

        IMPORTANT NOTE: This function is intended for DDA samples.
        While it in theory works for DIA sample too, this probably has little
        value.

        Parameters
        ----------
        centroiding_window : int
            The centroiding window to use.
            If 0, no centroiding is performed.
            Default is 0.
        keep_n_most_abundant_peaks : int
            Keep the n most abundant peaks.
            If -1, all peaks are retained.
            Default is -1.

        Returns
        -------
        : tuple (np.int64[:], np.uint32[:], np.uint32[:])
            The spectrum_indptr array, spectrum_tof_indices array and
            spectrum_intensity_values array.
        """
        precursor_order = np.argsort(self.precursor_indices)
        precursor_offsets = np.empty(
            self.precursor_max_index + 1,
            dtype=np.int64
        )
        precursor_offsets[0] = 0
        precursor_offsets[1:-1] = np.flatnonzero(
            np.diff(self.precursor_indices[precursor_order]) > 0) + 1
        precursor_offsets[-1] = len(precursor_order)
        offset = precursor_offsets[1]
        offsets = precursor_order[offset:]
        counts = np.empty(len(offsets) + 1, dtype=np.int)
        counts[0] = 0
        counts[1:] = np.cumsum(
            self.quad_indptr[offsets + 1] - self.quad_indptr[offsets]
        )
        spectrum_indptr = np.empty(
            self.precursor_max_index + 1,
            dtype=np.int64
        )
        spectrum_indptr[1:] = counts[
            precursor_offsets[1:] - precursor_offsets[1]
        ]
        spectrum_indptr[0] = 0
        spectrum_counts = np.zeros_like(spectrum_indptr)
        spectrum_tof_indices = np.empty(spectrum_indptr[-1], dtype=np.uint32)
        spectrum_intensity_values = np.empty(
            len(spectrum_tof_indices),
            dtype=np.float64
        )
        set_precursor(
            range(1, self.precursor_max_index),
            precursor_order,
            precursor_offsets,
            self.quad_indptr,
            self.tof_indices,
            self.intensity_values,
            spectrum_tof_indices,
            spectrum_intensity_values,
            spectrum_indptr,
            spectrum_counts,
        )
        if centroiding_window > 0:
            centroid_spectra(
                range(1, self.precursor_max_index),
                spectrum_indptr,
                spectrum_counts,
                spectrum_tof_indices,
                spectrum_intensity_values,
                centroiding_window,
            )
        if keep_n_most_abundant_peaks > -1:
            filter_spectra_by_abundant_peaks(
                range(1, self.precursor_max_index),
                spectrum_indptr,
                spectrum_counts,
                spectrum_tof_indices,
                spectrum_intensity_values,
                keep_n_most_abundant_peaks,
            )
        new_spectrum_indptr = np.empty_like(spectrum_counts)
        new_spectrum_indptr[1:] = np.cumsum(spectrum_counts[:-1])
        new_spectrum_indptr[0] = 0
        trimmed_spectrum_tof_indices = np.empty(
            new_spectrum_indptr[-1],
            dtype=np.uint32
        )
        trimmed_spectrum_intensity_values = np.empty(
            len(trimmed_spectrum_tof_indices),
            dtype=np.float64
        )
        spectrum_intensity_values
        trim_spectra(
            range(1, self.precursor_max_index),
            spectrum_tof_indices,
            spectrum_intensity_values,
            spectrum_indptr,
            trimmed_spectrum_tof_indices,
            trimmed_spectrum_intensity_values,
            new_spectrum_indptr,
        )
        return (
            new_spectrum_indptr,
            trimmed_spectrum_tof_indices,
            trimmed_spectrum_intensity_values
        )

    def save_as_mgf(
        self,
        directory: str,
        file_name: str,
        overwrite: bool = False,
        centroiding_window: int = 5,
        keep_n_most_abundant_peaks: int = -1,
    ):
        """Save profile spectra from this TimsTOF object as an mgf file.

        Parameters
        ----------
        directory : str
            The directory where to save the mgf file.
        file_name : str
            The file name of the  mgf file.
        overwrite : bool
            If True, an existing file is truncated.
            If False, nothing happens if a file already exists.
            Default is False.
        centroiding_window : int
            The centroiding window to use.
            If 0, no centroiding is performed.
            Default is 5.
        keep_n_most_abundant_peaks : int
            Keep the n most abundant peaks.
            If -1, all peaks are retained.
            Default is -1.

        Returns
        -------
        str
            The full file name of the mgf file.
        """
        full_file_name = os.path.join(
            directory,
            file_name
        )
        if self.acquisition_mode != "ddaPASEF":
            logging.info(
                f"File {self.bruker_d_folder_name} is not "
                "a ddaPASEF file, nothing to do."
            )
            return full_file_name
        if os.path.exists(full_file_name):
            if not overwrite:
                logging.info(
                    f"File {full_file_name} already exists, nothing to do."
                )
                return full_file_name
        logging.info(f"Indexing spectra of {self.bruker_d_folder_name}...")
        (
            spectrum_indptr,
            spectrum_tof_indices,
            spectrum_intensity_values,
        ) = self.index_precursors(
            centroiding_window=centroiding_window,
            keep_n_most_abundant_peaks=keep_n_most_abundant_peaks,
        )
        mono_mzs = self.precursors.MonoisotopicMz.values
        average_mzs = self.precursors.AverageMz.values
        charges = self.precursors.Charge.values
        charges[np.flatnonzero(np.isnan(charges))] = 0
        charges = charges.astype(np.int64)
        rtinseconds = self.rt_values[self.precursors.Parent.values]
        intensities = self.precursors.Intensity.values
        mobilities = self.mobility_values[
            self.precursors.ScanNumber.values.astype(np.int64)
        ]
        with open(full_file_name, "w") as infile:
            logging.info(f"Exporting profile spectra to {full_file_name}...")
            for index in alphatims.utils.progress_callback(
                range(1, self.precursor_max_index)
            ):
                start = spectrum_indptr[index]
                end = spectrum_indptr[index + 1]
                title = (
                    f"index: {index}, "
                    f"intensity: {intensities[index - 1]:.1f}, "
                    f"mobility: {mobilities[index - 1]:.3f}, "
                    f"average_mz: {average_mzs[index - 1]:.3f}"
                )
                infile.write("BEGIN IONS\n")
                infile.write(f'TITLE="{title}"\n')
                infile.write(f"PEPMASS={mono_mzs[index - 1]:.6f}\n")
                infile.write(f"CHARGE={charges[index - 1]}\n")
                infile.write(f"RTINSECONDS={rtinseconds[index - 1]:.2f}\n")
                for mz, intensity in zip(
                    self.mz_values[spectrum_tof_indices[start: end]],
                    spectrum_intensity_values[start: end],
                ):
                    infile.write(f"{mz:.6f} {intensity}\n")
                infile.write("END IONS\n")
        logging.info(
            f"Succesfully wrote {self.precursor_max_index - 1:,} "
            f"spectra to {full_file_name}."
        )
        return full_file_name

    def calculate_global_calibrated_mz_values(
        self,
        calibrant1: tuple = (922.009798, 1.1895, slice(0, 1)),
        calibrant2: tuple = (1221.990637, 1.3820, slice(0, 1)),
        mz_tolerance: float = 10,  # in Th
        mobility_tolerance: float = 0.1,  # in 1/k0,
    ) -> None:
        """Calculate global calibrated_mz_values based on two calibrant ions.

        Parameters
        ----------
        calibrant1 : tuple
            The first calibrant ion.
            This is a tuple with (mz, mobility, precursor_slice) foat values.
            Default is (922.009798, 1.1895, slice(0, 1)).
        calibrant2 : tuple
            The first calibrant ion.
            This is a tuple with (mz, mobility, precursor_slice) foat values.
            Default is (1221.990637, 1.3820, slice(0, 1)).
        mz_tolerance : float
            The tolerance window (in Th) with respect to the
            uncalibrated mz_values. If this is too large,
            the calibrant ion might not be the most intense ion anymore.
            If this is too small, the calibrant ion might not be contained.
            Default is 10.
        mobility_tolerance : float
            The tolerance window with respect to the
            uncalibrated mobility_values. If this is too large,
            the calibrant ion might not be the most intense ion anymore.
            If this is too small, the calibrant ion might not be contained.
            Default is 0.1.
        """
        logging.info("Calculating global calibrated mz values...")
        if calibrant1[0] > calibrant2[0]:
            calibrant1, calibrant2 = calibrant2, calibrant1
        calibrant1_lower_mz = calibrant1[0] - mz_tolerance
        calibrant1_upper_mz = calibrant1[0] + mz_tolerance
        calibrant1_lower_mobility = calibrant1[1] - mobility_tolerance
        calibrant1_upper_mobility = calibrant1[1] + mobility_tolerance
        calibrant1_tof = np.argmax(
            np.bincount(
                self.tof_indices[
                    self[
                        :,
                        calibrant1_lower_mobility: calibrant1_upper_mobility,
                        calibrant1[2],
                        calibrant1_lower_mz: calibrant1_upper_mz,
                        "raw"
                    ]
                ]
            )
        )
        calibrant2_lower_mz = calibrant2[0] - mz_tolerance
        calibrant2_upper_mz = calibrant2[0] + mz_tolerance
        calibrant2_lower_mobility = calibrant2[1] - mobility_tolerance
        calibrant2_upper_mobility = calibrant2[1] + mobility_tolerance
        calibrant2_tof = np.argmax(
            np.bincount(
                self.tof_indices[
                    self[
                        :,
                        calibrant2_lower_mobility: calibrant2_upper_mobility,
                        calibrant2[2],
                        calibrant2_lower_mz: calibrant2_upper_mz,
                        "raw"
                    ]
                ]
            )
        )
        tof_slope = (
            np.sqrt(calibrant2[0]) - np.sqrt(calibrant1[0])
        ) / (calibrant2_tof - calibrant1_tof)
        tof_intercept = np.sqrt(calibrant1[0]) - tof_slope * calibrant1_tof
        self._calibrated_mz_values = (
            tof_intercept + tof_slope * np.arange(self.tof_max_index)
        )**2
        ppms = 10**6 * (
            self._mz_values - self._calibrated_mz_values
        ) / self._mz_values
        logging.info(
            "Global calibration of mz values yielded differences between "
            f"{np.min(ppms):.2f} and {np.max(ppms):.2f} ppm."
        )

    def use_calibrated_mz_values_as_default(
        self,
        use_calibrated_mz_values: int
    ) -> None:
        """Override the default mz_values with the global calibrated_mz_values.

        Calibrated_mz_values will be calculated if they do not exist yet.

        Parameters
        ----------
        use_calibrated_mz_values : int
            If not 0, the mz_values are overwritten with global
            calibrated_mz_values.
            If 1, calibration at the MS1 level is performed.
            If 2, calibration at the MS2 level is performed.
        """
        if use_calibrated_mz_values != 0:
            if not hasattr(self, "_calibrated_mz_values"):
                if use_calibrated_mz_values == 1:
                    ms_level = 0
                if use_calibrated_mz_values == 2:
                    ms_level = slice(1, None)
                self.calculate_global_calibrated_mz_values(
                    calibrant1=(922.009798, 1.1895, ms_level),
                    calibrant2=(1221.990637, 1.3820, ms_level),
                    mz_tolerance=1
                )
        self._use_calibrated_mz_values_as_default = use_calibrated_mz_values

    def set_cycle(self) -> None:
        """Set the quad cycle for diaPASEF data.
        """
        ms1_diffs = np.diff(
            np.flatnonzero(self.frames.MsMsType[int(self.zeroth_frame):]==0)
        )
        subcycle_length_count = np.bincount(ms1_diffs)
        if np.all(subcycle_length_count[:-1]!=0):
            raise ValueError("No consistent subcycle length")
        subcycle_length = len(subcycle_length_count) - 1
        max_precursor = len(self.fragment_frames.Precursor.unique())
        subcycle_count = max_precursor // (subcycle_length - 1)
        frame_count = subcycle_length * subcycle_count
        cycle = np.zeros(
            (
                frame_count,
                self.scan_max_index,
                2,
            )
        )
        precursor_frames = np.ones(frame_count, dtype=np.bool_)

        subframes = self.fragment_frames.drop("Frame", axis=1)
        for max_index in range(1, len(subframes)):
            subframe = subframes.iloc[max_index]
            if subframe.equals(subframes.iloc[0]):
                break
        for index, row in self.fragment_frames[:max_index].iterrows():
            frame = int(row.Frame - self.zeroth_frame)
            scan_begin = int(row.ScanNumBegin)
            scan_end = int(row.ScanNumEnd)
            low_mz = row.IsolationMz - row.IsolationWidth / 2
            high_mz = row.IsolationMz + row.IsolationWidth / 2
            cycle[
                frame,
                scan_begin: scan_end,
            ] = (low_mz, high_mz)
            precursor_frames[frame] = False

        cycle[precursor_frames] = (-1, -1)
        cycle = cycle.reshape(
            (
                subcycle_count,
                subcycle_length,
                *cycle.shape[1:]
            )
        )
        self._cycle = cycle


class PrecursorFloatError(TypeError):
    """Used to indicate that a precursor value is not an int but a float."""
    pass


@alphatims.utils.pjit(
    signature_or_function="void(i8,i8[:],i8[:],i8[:],u4[:],u2[:],u4[:],f8[:],i8[:],i8[:])"
)
def set_precursor(
    precursor_index: int,
    offset_order: np.ndarray,
    precursor_offsets: np.ndarray,
    quad_indptr: np.ndarray,
    tof_indices: np.ndarray,
    intensities: np.ndarray,
    spectrum_tof_indices: np.ndarray,
    spectrum_intensity_values: np.ndarray,
    spectrum_indptr: np.ndarray,
    spectrum_counts: np.ndarray,
) -> None:
    """Sum the intensities of all pushes belonging to a single precursor.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.pjit.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    precursor_index : int
        The precursor index indicating which MS2 spectrum to determine.
    offset_order : np.int64[:]
        The order of self.precursor_indices, obtained with np.argsort.
    precursor_offsets : np.int64[:]
        An index pointer array for precursor offsets.
    quad_indptr : np.int64[:]
        The self.quad_indptr array of a TimsTOF object.
    tof_indices : np.uint32[:]
        The self.tof_indices array of a TimsTOF object.
    intensities : np.uint16[:]
        The self.intensity_values array of a TimsTOF object.
    spectrum_tof_indices : np.uint32[:]
        A buffer array to store tof indices of the new spectrum.
    spectrum_intensity_values : np.float64[:]
        A buffer array to store intensity values of the new spectrum.
    spectrum_indptr : np.int64[:]
        An index pointer array defining the original spectrum boundaries.
    spectrum_counts : np. int64[:]
        An buffer array defining how many distinct tof indices the new
        spectrum has.
    """
    offset = spectrum_indptr[precursor_index]
    precursor_offset_lower = precursor_offsets[precursor_index]
    precursor_offset_upper = precursor_offsets[precursor_index + 1]
    selected_offsets = offset_order[
        precursor_offset_lower: precursor_offset_upper
    ]
    starts = quad_indptr[selected_offsets]
    ends = quad_indptr[selected_offsets + 1]
    offset_index = offset
    for start, end in zip(starts, ends):
        spectrum_tof_indices[
            offset_index: offset_index + end - start
        ] = tof_indices[start: end]
        spectrum_intensity_values[
            offset_index: offset_index + end - start
            ] = intensities[start: end]
        offset_index += end - start
    offset_end = spectrum_indptr[precursor_index + 1]
    order = np.argsort(spectrum_tof_indices[offset: offset_end])
    current_index = offset - 1
    previous_tof_index = -1
    for tof_index, intensity in zip(
        spectrum_tof_indices[offset: offset_end][order],
        spectrum_intensity_values[offset: offset_end][order],
    ):
        if tof_index != previous_tof_index:
            current_index += 1
            spectrum_tof_indices[current_index] = tof_index
            spectrum_intensity_values[current_index] = intensity
            previous_tof_index = tof_index
        else:
            spectrum_intensity_values[current_index] += intensity
    spectrum_tof_indices[current_index + 1: offset_end] = 0
    spectrum_intensity_values[current_index + 1: offset_end] = 0
    spectrum_counts[precursor_index] = current_index + 1 - offset


@alphatims.utils.pjit
def centroid_spectra(
    index: int,
    spectrum_indptr: np.ndarray,
    spectrum_counts: np.ndarray,
    spectrum_tof_indices: np.ndarray,
    spectrum_intensity_values: np.ndarray,
    window_size: int,
):
    """Smoothen and centroid a profile spectrum (inplace operation).

    IMPORTANT NOTE: This function will overwrite all input arrays.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.pjit.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    index : int
        The push index whose intensity_values and tof_indices will be
        centroided.
    spectrum_indptr : np.int64[:]
        An index pointer array defining the (untrimmed) spectrum boundaries.
    spectrum_counts : np. int64[:]
        The original array defining how many distinct tof indices each
        spectrum has.
    spectrum_tof_indices : np.uint32[:]
        The original array containing tof indices.
    spectrum_intensity_values : np.float64[:]
        The original array containing intensity values.
    window_size : int
        The window size to use for smoothing and centroiding peaks.
    """
    start = spectrum_indptr[index]
    end = start + spectrum_counts[index]
    if start == end:
        return
    mzs = spectrum_tof_indices[start: end]
    ints = spectrum_intensity_values[start: end]
    smooth_ints = ints.copy()
    for i, self_mz in enumerate(mzs[:-1]):
        for j in range(i + 1, len(mzs)):
            other_mz = mzs[j]
            diff = other_mz - self_mz + 1
            if diff >= window_size:
                break
            smooth_ints[i] += ints[j] / diff
            smooth_ints[j] += ints[i] / diff
    pre_apex = True
    maxima = [mzs[0]]
    intensities = [ints[0]]
    for i, self_mz in enumerate(mzs[1:], 1):
        if self_mz > mzs[i - 1] + window_size:
            maxima.append(mzs[i])
            intensities.append(0)
            pre_apex = True
        elif pre_apex:
            if smooth_ints[i] < smooth_ints[i - 1]:
                pre_apex = False
                maxima[-1] = mzs[i - 1]
        elif smooth_ints[i] > smooth_ints[i - 1]:
            maxima.append(mzs[i])
            intensities.append(0)
            pre_apex = True
        intensities[-1] += ints[i]
    spectrum_tof_indices[start: start + len(maxima)] = np.array(
        maxima,
        dtype=spectrum_tof_indices.dtype
    )
    spectrum_intensity_values[start: start + len(maxima)] = np.array(
        intensities,
        dtype=spectrum_intensity_values.dtype
    )
    spectrum_counts[index] = len(maxima)


@alphatims.utils.pjit
def filter_spectra_by_abundant_peaks(
    index: int,
    spectrum_indptr: np.ndarray,
    spectrum_counts: np.ndarray,
    spectrum_tof_indices: np.ndarray,
    spectrum_intensity_values: np.ndarray,
    keep_n_most_abundant_peaks: int,
):
    """Filter a spectrum to retain only the most abundant peaks.

    IMPORTANT NOTE: This function will overwrite all input arrays.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.pjit.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    index : int
        The push index whose intensity_values and tof_indices will be
        centroided.
    spectrum_indptr : np.int64[:]
        An index pointer array defining the (untrimmed) spectrum boundaries.
    spectrum_counts : np. int64[:]
        The original array defining how many distinct tof indices each
        spectrum has.
    spectrum_tof_indices : np.uint32[:]
        The original array containing tof indices.
    spectrum_intensity_values : np.float64[:]
        The original array containing intensity values.
    keep_n_most_abundant_peaks : int
        Keep only this many abundant peaks.
    """
    start = spectrum_indptr[index]
    end = start + spectrum_counts[index]
    if end - start <= keep_n_most_abundant_peaks:
        return
    mzs = spectrum_tof_indices[start: end]
    ints = spectrum_intensity_values[start: end]
    selected_indices = np.sort(
        np.argsort(ints)[-keep_n_most_abundant_peaks:]
    )
    count = len(selected_indices)
    spectrum_tof_indices[start: start + count] = mzs[selected_indices]
    spectrum_intensity_values[start: start + count] = ints[selected_indices]
    spectrum_counts[index] = count


@alphatims.utils.pjit
def trim_spectra(
    index: int,
    spectrum_tof_indices: np.ndarray,
    spectrum_intensity_values: np.ndarray,
    spectrum_indptr: np.ndarray,
    trimmed_spectrum_tof_indices: np.ndarray,
    trimmed_spectrum_intensity_values: np.ndarray,
    new_spectrum_indptr: np.ndarray,
) -> None:
    """Trim remaining bytes after merging of multiple pushes.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.pjit.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    index : int
        The push index whose intensity_values and tof_indices will be trimmed.
    spectrum_tof_indices : np.uint32[:]
        The original array containing tof indices.
    spectrum_intensity_values : np.float64[:]
        The original array containing intensity values.
    spectrum_indptr : np.int64[:]
        An index pointer array defining the original spectrum boundaries.
    trimmed_spectrum_tof_indices : np.uint32[:]
        A buffer array to store new tof indices.
    trimmed_spectrum_intensity_values : np.float64[:]
        A buffer array to store new intensity values.
    new_spectrum_indptr : np.int64[:]
        An index pointer array defining the trimmed spectrum boundaries.
    """
    start = spectrum_indptr[index]
    new_start = new_spectrum_indptr[index]
    new_end = new_spectrum_indptr[index + 1]
    trimmed_spectrum_tof_indices[new_start: new_end] = spectrum_tof_indices[
        start: start + new_end - new_start
    ]
    trimmed_spectrum_intensity_values[
        new_start: new_end
    ] = spectrum_intensity_values[
        start: start + new_end - new_start
    ]


def parse_keys(data: TimsTOF, keys) -> dict:
    """Convert different keys to a key dict with defined types.

    NOTE: Negative slicing is not supported and all indiviudal keys
    are assumed to be sorted, disjunct and strictly increasing

    Parameters
    ----------
    data : alphatims.bruker.TimsTOF
        The TimsTOF objext for which to get slices.
    keys : tuple
        A tuple of at most 5 elemens, containing
        slices, ints, floats, Nones, and/or iterables.
        See `alphatims.bruker.convert_slice_key_to_int_array` and
        `alphatims.bruker.convert_slice_key_to_float_array` for more details.

    Returns
    -------
    : dict
        The resulting dict always has the following items:
            - "frame_indices": np.int64[:, 3]
            - "scan_indices": np.int64[:, 3]
            - "tof_indices": np.int64[:, 3]
            - "precursor_indices": np.int64[:, 3]
            - "quad_values": np.float64[:, 2]
            - "intensity_values": np.float64[:, 2]
    """
    dimensions = [
        "frame_indices",
        "scan_indices",
        "precursor_indices",
        "tof_indices",
    ]
    dimension_slices = {}
    if len(keys) > (len(dimensions) + 1):
        raise KeyError(
            "LC-IMS-MSMS data can be sliced in maximum 5 dimensions. "
            "Integers are assumed to be indices, while "
            "floats are assumed as values. Intensity is always casted "
            "to integer values, regardless of input type."
        )
    if isinstance(keys[0], dict):
        new_keys = []
        dimension_translations = {
            "frame_indices": "rt_values",
            "scan_indices": "mobility_values",
            "precursor_indices": "quad_mz_values",
            "tof_indices": "mz_values",
        }
        for indices, values in dimension_translations.items():
            if indices in keys[0]:
                new_keys.append(keys[0][indices])
            elif values in keys[0]:
                new_keys.append(keys[0][values])
            else:
                new_keys.append(slice(None))
        if "intensity_values" in keys[0]:
            new_keys.append(keys[0]["intensity_values"])
        keys = new_keys
    for i, dimension in enumerate(dimensions):
        try:
            dimension_slices[
                dimension
            ] = convert_slice_key_to_int_array(
                data,
                keys[i] if (i < len(keys)) else slice(None),
                dimension
            )
        except PrecursorFloatError:
            dimension_slices[
                "precursor_indices"
            ] = convert_slice_key_to_int_array(
                data,
                slice(None),
                "precursor_indices"
            )
            dimension_slices[
                "quad_values"
            ] = convert_slice_key_to_float_array(keys[i])
    dimension_slices[
        "intensity_values"
    ] = convert_slice_key_to_float_array(
        keys[-1] if (len(keys) > len(dimensions)) else slice(None)
    )
    if "quad_values" not in dimension_slices:
        dimension_slices["quad_values"] = np.array(
            [[-np.inf, np.inf]],
            dtype=np.float64
        )
    return dimension_slices


def convert_slice_key_to_float_array(key):
    """Convert a key to a slice float array.

    NOTE: This function should only be used for QUAD or DETECTOR dimensions.

    Parameters
    ----------
    key : slice, int, float, None, iterable
        The key that needs to be converted.

    Returns
    -------
    : np.float64[:, 2]
        Each row represent a a (start, stop) slice.

    Raises
    ------
    ValueError
        When the key is an np.ndarray with more than 2 columns.
    """
    try:
        iter(key)
    except TypeError:
        if key is None:
            key = slice(None)
        if isinstance(key, slice):
            start = key.start
            if start is None:
                start = -np.inf
            stop = key.stop
            if stop is None:
                stop = np.inf
        else:
            start = key
            stop = key
        return np.array([[start, stop]], dtype=np.float64)
    else:
        if not isinstance(key, np.ndarray):
            key = np.array(key, dtype=np.float64)
        key = key.astype(np.float64)
        if len(key.shape) == 1:
            return np.array([key, key]).T
        elif len(key.shape) == 2:
            if key.shape[1] != 2:
                raise ValueError
            return key
        else:
            raise ValueError


def convert_slice_key_to_int_array(data: TimsTOF, key, dimension: str):
    """Convert a key of a data dimension to a slice integer array.

    Parameters
    ----------
    data : alphatims.bruker.TimsTOF
        The TimsTOF objext for which to get slices.
    key : slice, int, float, None, iterable
        The key that needs to be converted.
    dimension : str
        The dimension for which the key needs to be retrieved

    Returns
    -------
    : np.int64[:, 3]
        Each row represent a a (start, stop, step) slice.

    Raises
    ------
    ValueError
        When the key contains elements other than int or float.
    PrecursorFloatError
        When trying to convert a quad float to precursor index.
    """
    result = np.empty((0, 3), dtype=np.int64)
    inverse_of_scans = False
    try:
        iter(key)
    except TypeError:
        if key is None:
            key = slice(None)
        if isinstance(key, slice):
            if dimension == "scan_indices":
                if isinstance(key.start, (np.inexact, float)) or isinstance(
                    key.stop,
                    (np.inexact, float)
                ):
                    key = slice(key.stop, key.start, key.step)
            start = key.start
            if not isinstance(start, (np.integer, int)):
                if start is None:
                    if dimension == "scan_indices":
                        start = np.inf
                    else:
                        start = -np.inf
                if not isinstance(start, (np.inexact, float)):
                    raise ValueError
                start = data.convert_to_indices(
                    start,
                    return_type=dimension
                )
            stop = key.stop
            if not isinstance(stop, (np.integer, int)):
                if stop is None:
                    if dimension == "scan_indices":
                        stop = -np.inf
                    else:
                        stop = np.inf
                if not isinstance(stop, (np.inexact, float)):
                    raise ValueError
                stop = data.convert_to_indices(
                    stop,
                    return_type=dimension,
                )
            step = key.step
            if not isinstance(step, (np.integer, int)):
                if step is not None:
                    raise ValueError
                step = 1
            result = np.array([[start, stop, step]])
        elif isinstance(key, (np.integer, int)):
            result = np.array([[key, key + 1, 1]])
        elif isinstance(key, (np.inexact, float)):
            key = data.convert_to_indices(key, return_type=dimension)
            if dimension == "scan_indices":
                result = np.array([[key - 1, key, 1]])
            else:
                result = np.array([[key, key + 1, 1]])
        else:
            raise ValueError
    else:
        if not isinstance(key, np.ndarray):
            key = np.array(key)
        step = 1
        if not isinstance(key.ravel()[0], np.integer):
            key = data.convert_to_indices(key, return_type=dimension)
            if dimension == "scan_indices":
                key -= 1
        if len(key.shape) == 1:
            result = np.array([key, key + 1, np.repeat(step, key.size)]).T
        elif len(key.shape) == 2:
            if key.shape[1] != 3:
                raise ValueError
            result = key
        else:
            raise ValueError
    if inverse_of_scans:
        return result[:, [1, 0, 2]]
    else:
        return result


@alphatims.utils.njit
def calculate_dia_cycle_mask(
    dia_mz_cycle: np.ndarray,
    quad_slices: np.ndarray,
    dia_precursor_cycle: np.ndarray = None,
    precursor_slices: np.ndarray = None,
):
    """Calculate a boolean mask for cyclic push indices satisfying queries.

    Parameters
    ----------
    dia_mz_cycle : np.float64[:, 2]
        An array with (upper, lower) mz values of a DIA cycle (per push).
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
    dia_precursor_cycle : np.int64[:]
        An array with precursor indices of a DIA cycle (per push).
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).

    Returns
    -------
    : np.bool_[:]
        A mask that determines if a cyclic push index is valid given the
        requested slices.
    """
    mz_mask = np.zeros(len(dia_mz_cycle), dtype=np.bool_)
    for i, (mz_start, mz_stop) in enumerate(dia_mz_cycle):
        for quad_mz_start, quad_mz_stop in quad_slices:
            if (quad_mz_start <= mz_stop) and (quad_mz_stop >= mz_start):
                mz_mask[i] = True
                break
    if precursor_slices is not None:
        precursor_mask = np.zeros(len(dia_mz_cycle), dtype=np.bool_)
        for i, precursor_index in enumerate(dia_precursor_cycle):
            for (start, stop, step) in precursor_slices:
                if precursor_index in range(start, stop, step):
                    precursor_mask[i] = True
                    break
        return mz_mask & precursor_mask
    else:
        return mz_mask


@alphatims.utils.njit
def valid_quad_mz_values(
    low_mz_value: float,
    high_mz_value: float,
    quad_slices: np.ndarray,
) -> bool:
    """Check if the low and high quad mz values are included in the slices.

    NOTE: Just a part of the quad range needs to overlap with a part
    of a single slice.

    Parameters
    ----------
    low_mz_value : float
        The lower mz value of the current quad selection.
    high_mz_value : float
        The upper mz value of the current quad selection.
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).

    Returns
    -------
    : bool
        True if some part of the quad overlaps with some part of some slice.
        False if there is no overlap in the range.
    """
    slice_index = np.searchsorted(
        quad_slices[:, 0].ravel(),
        high_mz_value,
        "right"
    )
    if slice_index == 0:
        return False
    if low_mz_value <= quad_slices[slice_index - 1, 1]:
        return True
    return False


@alphatims.utils.njit
def valid_precursor_index(
    precursor_index: int,
    precursor_slices: np.ndarray
) -> bool:
    """Check if a precursor index is included in the slices.

    Parameters
    ----------
    precursor_index : int
        The precursor index to validate.
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).

    Returns
    -------
    : bool
        True if the precursor index is present in any of the slices.
        False otherwise.
    """
    slice_index = np.searchsorted(
        precursor_slices[:, 0].ravel(),
        precursor_index,
        side="right"
    )
    if slice_index == 0:
        return False
    return precursor_index in range(
        precursor_slices[slice_index - 1, 0],
        precursor_slices[slice_index - 1, 1],
        precursor_slices[slice_index - 1, 2],
    )


@alphatims.utils.njit
def filter_indices(
    frame_slices: np.ndarray,
    scan_slices: np.ndarray,
    precursor_slices: np.ndarray,
    tof_slices: np.ndarray,
    quad_slices: np.ndarray,
    intensity_slices: np.ndarray,
    frame_max_index: int,
    scan_max_index: int,
    push_indptr: np.ndarray,
    precursor_indices: np.ndarray,
    quad_mz_values: np.ndarray,
    quad_indptr: np.ndarray,
    tof_indices: np.ndarray,
    intensities: np.ndarray,
):
    """Filter raw indices by slices from all dimensions.

    Parameters
    ----------
    frame_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).
    scan_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).
    tof_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
    intensity_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(intensity_slices.ravel()) >= 0) = True).
    frame_max_index : int
        The maximum frame index of a TimsTOF object.
    scan_max_index : int
        The maximum scan index of a TimsTOF object.
    push_indptr : np.int64[:]
        The self.push_indptr array of a TimsTOF object.
    precursor_indices : np.int64[:]
        The self.precursor_indices array of a TimsTOF object.
    quad_mz_values : np.float64[:, 2]
        The self.quad_mz_values array of a TimsTOF object.
    quad_indptr : np.int64[:]
        The self.quad_indptr array of a TimsTOF object.
    tof_indices : np.uint32[:]
        The self.tof_indices array of a TimsTOF object.
    intensities : np.uint16[:]
        The self.intensity_values array of a TimsTOF object.

    Returns
    -------
    : np.int64[:]
        The raw indices that satisfy all the slices.
    """
    result = []
    quad_index = -1
    new_quad_index = -1
    quad_end = -1
    is_valid_quad_index = True
    starts = push_indptr[:-1].reshape(
        frame_max_index,
        scan_max_index
    )
    ends = push_indptr[1:].reshape(
        frame_max_index,
        scan_max_index
    )
    for frame_start, frame_stop, frame_step in frame_slices:
        for frame_start_slice, frame_end_slice in zip(
            starts[slice(frame_start, frame_stop, frame_step)],
            ends[slice(frame_start, frame_stop, frame_step)]
        ):
            for scan_start, scan_stop, scan_step in scan_slices:
                for sparse_start, sparse_end in zip(
                    frame_start_slice[slice(scan_start, scan_stop, scan_step)],
                    frame_end_slice[slice(scan_start, scan_stop, scan_step)]
                ):
                    if (sparse_start == sparse_end):
                        continue
                    while quad_end < sparse_end:
                        new_quad_index += 1
                        quad_end = quad_indptr[new_quad_index + 1]
                    if quad_index != new_quad_index:
                        quad_index = new_quad_index
                        if not valid_quad_mz_values(
                            quad_mz_values[quad_index, 0],
                            quad_mz_values[quad_index, 1],
                            quad_slices
                        ):
                            is_valid_quad_index = False
                        elif not valid_precursor_index(
                            precursor_indices[quad_index],
                            precursor_slices,
                        ):
                            is_valid_quad_index = False
                        else:
                            is_valid_quad_index = True
                    if not is_valid_quad_index:
                        continue
                    idx = sparse_start
                    for tof_start, tof_stop, tof_step in tof_slices:
                        idx += np.searchsorted(
                            tof_indices[idx: sparse_end],
                            tof_start
                        )
                        tof_value = tof_indices[idx]
                        while (tof_value < tof_stop) and (idx < sparse_end):
                            if tof_value in range(
                                tof_start,
                                tof_stop,
                                tof_step
                            ):
                                intensity = intensities[idx]
                                for (
                                    low_intensity,
                                    high_intensity
                                ) in intensity_slices:
                                    if (low_intensity <= intensity):
                                        if (intensity <= high_intensity):
                                            result.append(idx)
                                            break
                            idx += 1
                            tof_value = tof_indices[idx]
    return np.array(result)


# Overhead of using more than 1 threads is actually slower
@alphatims.utils.pjit(thread_count=1, include_progress_callback=False)
def add_intensity_to_bin(
    query_index: int,
    intensities: np.ndarray,
    parsed_indices: np.ndarray,
    intensity_bins: np.ndarray,
) -> None:
    """Add the intensity of a query to the appropriate bin.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.pjit.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

    Parameters
    ----------
    query_index : int
        The query whose intensity needs to be binned
        The first argument is thus expected to be provided as an iterable
        containing ints instead of a single int.
    intensities : np.float64[:]
        An array with intensities that need to be binned.
    parsed_indices : np.int64[:], np.int64[:, :]
        Description of parameter `parsed_indices`.
    intensity_bins : np.float64[:]
        A buffer with intensity bins to which the current query will be added.
    """
    intensity = intensities[query_index]
    if len(parsed_indices) == 1:
        intensity_bins[parsed_indices[0][query_index]] += intensity
    elif len(parsed_indices) == 2:
        intensity_bins[
            parsed_indices[0][query_index],
            parsed_indices[1][query_index]
        ] += intensity


@alphatims.utils.njit(nogil=True)
def indptr_lookup(
    targets: np.ndarray,
    queries: np.ndarray,
    momentum_amplifier: int = 2
):
    """Find the indices of queries in targets.

    This function is equivalent to
    "np.searchsorted(targets, queries, "right") - 1".
    By utilizing the fact that queries are also sorted,
    it is significantly faster though.

    Parameters
    ----------
    targets : np.int64[:]
        A sorted list of index pointers where queries needs to be looked up.
    queries : np.int64[:]
        A sorted list of queries whose index pointers needs to be looked up.
    momentum_amplifier : int
        Factor to add momentum to linear searching, attempting to quickly
        discard empty range without hits.
        Invreasing it can speed up searching of queries if they are sparsely
        spread out in targets.

    Returns
    -------
    : np.int64[:]
        The indices of queries in targets.
    """
    hits = np.empty_like(queries)
    target_index = 0
    no_target_overflow = True
    for i, query_index in enumerate(queries):
        while no_target_overflow:
            momentum = 1
            while targets[target_index] <= query_index:
                target_index += momentum
                if target_index >= len(targets):
                    break
                momentum *= momentum_amplifier
            else:
                if momentum <= momentum_amplifier:
                    break
                else:
                    target_index -= momentum // momentum_amplifier - 1
                    continue
            if momentum == 1:
                no_target_overflow = False
            else:
                target_index -= momentum
        hits[i] = target_index - 1
    return hits


@alphatims.utils.njit(nogil=True)
def get_dia_push_indices(
    frame_slices: np.ndarray,
    scan_slices: np.ndarray,
    quad_slices: np.ndarray,
    scan_max_index: int,
    dia_mz_cycle: np.ndarray,
    dia_precursor_cycle: np.ndarray = None,
    precursor_slices: np.ndarray = None,
    zeroth_frame: bool = True,
):
    """Filter DIA push indices by slices from LC, TIMS and QUAD.

    Parameters
    ----------
    frame_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).
    scan_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).
    quad_slices : np.float64[:, 2]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
    scan_max_index : int
        The maximum scan index of a TimsTOF object.
    dia_mz_cycle : np.float64[:, 2]
        An array with (upper, lower) mz values of a DIA cycle (per push).
    dia_precursor_cycle : np.int64[:]
        An array with precursor indices of a DIA cycle (per push).
    precursor_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).
    zeroth_frame : bool
        Indicates if a zeroth frame was used before a DIA cycle.

    Returns
    -------
    : np.int64[:]
        The raw push indices that satisfy all the slices.
    """
    result = []
    quad_mask = calculate_dia_cycle_mask(
        dia_mz_cycle=dia_mz_cycle,
        quad_slices=quad_slices,
        dia_precursor_cycle=dia_precursor_cycle,
        precursor_slices=precursor_slices
    )
    for frame_start, frame_stop, frame_step in frame_slices:
        for frame_index in range(frame_start, frame_stop, frame_step):
            for scan_start, scan_stop, scan_step in scan_slices:
                for scan_index in range(scan_start, scan_stop, scan_step):
                    push_index = frame_index * scan_max_index + scan_index
                    if zeroth_frame:
                        cyclic_push_index = push_index - scan_max_index
                    else:
                        cyclic_push_index = push_index
                    if quad_mask[cyclic_push_index % len(dia_mz_cycle)]:
                        result.append(push_index)
    return np.array(result)


@alphatims.utils.njit(nogil=True)
def filter_tof_to_csr(
    tof_slices: np.ndarray,
    push_indices: np.ndarray,
    tof_indices: np.ndarray,
    push_indptr: np.ndarray,
) -> tuple:
    """Get a CSR-matrix with raw indices satisfying push indices and tof slices.

    Parameters
    ----------
    tof_slices : np.int64[:, 3]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
    push_indices : np.int64[:]
        The push indices from where to retrieve the TOF slices.
    tof_indices : np.uint32[:]
        The self.tof_indices array of a TimsTOF object.
    push_indptr : np.int64[:]
        The self.push_indptr array of a TimsTOF object.

    Returns
    -------
    (np.int64[:], np.int64[:], np.int64[:],)
        An (indptr, values, columns) tuple, where indptr are push indices,
        values raw indices, and columns the tof_slices.
    """
    indptr = [0]
    values = []
    columns = []
    for push_index in push_indices:
        start = push_indptr[push_index]
        end = push_indptr[push_index + 1]
        idx = start
        for i, (tof_start, tof_stop, tof_step) in enumerate(tof_slices):
            idx += np.searchsorted(tof_indices[idx: end], tof_start)
            tof_value = tof_indices[idx]
            while (tof_value < tof_stop) and (idx < end):
                if tof_value in range(tof_start, tof_stop, tof_step):
                    values.append(idx)
                    columns.append(i)
                    break  # TODO what if multiple hits?
                idx += 1
                tof_value = tof_indices[idx]
        indptr.append(len(values))
    return np.array(indptr), np.array(values), np.array(columns)
