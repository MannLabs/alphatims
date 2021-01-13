#!python

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
        "No Bruker libraries are available for this operating system. "
        "Intensities are uncalibrated, resulting in (very) small differences. "
        "However, mobility and m/z values need to be estimated. "
        "Possibly these have huge errors (e.g. offsets of 6 Th have "
        "already been observed)!"
    )
    logging.info("")
    BRUKER_DLL_FILE_NAME = ""


def init_bruker_dll(bruker_dll_file_name: str = BRUKER_DLL_FILE_NAME):
    """Open a bruker.dll in Python.

    Five functions are defined for this dll:

        - tims_open:
            [c_char_p, c_uint32] -> c_uint64
        - tims_close:
            [c_char_p, c_uint32] -> c_uint64
        - tims_read_scans_v2:
            [c_uint64, c_int64, c_uint32, c_uint32, c_void_p,
            c_uint32] -> c_uint32
        - tims_index_to_mz:
            [c_uint64, c_int64, POINTER(c_double), POINTER(c_double),
            c_uint32] -> None
        - tims_scannum_to_oneoverk0:
            Same as "tims_index_to_mz"

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
) -> tuple:
    """Read metadata and (fragment) frames from a Bruker .d folder.

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

    Returns
    -------
    : tuple
        (str, dict, pd.DataFrame, pd.DataFrame).
        The acquisition_mode, global_meta_data, frames and fragment_frames.

    Raises
    ------
    ValueError
        When table "MsMsType" is not 8 or 9.
        In this case it is unclear if it is ddaPASEF or diaPASEF.
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
        elif 8 in frames.MsMsType.values:
            acquisition_mode = "PASEF"
            fragment_frames = pd.read_sql_query(
                "SELECT * from PasefFrameMsMsInfo",
                sql_database_connection
            )
        else:
            raise ValueError("Scan mode is not PASEF or diaPASEF")
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
            fragment_frames.Frame += 1
        frames = pd.DataFrame(
            {
                col: pd.to_numeric(
                    frames[col]
                ) for col in frames if col != "Polarity"
            }
        )
        return (
            acquisition_mode,
            global_meta_data,
            frames,
            fragment_frames,
        )


@alphatims.utils.njit(nogil=True)
def parse_decompressed_bruker_binary(decomp_data: bytes) -> tuple:
    """Parse a Bruker binary frame buffer into scans, tofs and intensities.

    Parameters
    ----------
    decomp_data : bytes
        A Bruker frame binary buffer that is already decompressed with pyzstd.

    Returns
    -------
    : tuple (np.uint32[:], np.uint32[:], np.uint32[:]).
        The scan_indices, tof_indices and intensities present in this binary
        array
    """
    temp = np.frombuffer(decomp_data, dtype=np.uint8)
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
    return scan_indices, tof_indices, intensities


@alphatims.utils.threadpool(progress_callback=True)
def process_frame(
    frame_id: int,
    tdf_bin_file_name: str,
    tims_offset_values,
    scan_indptr,
    intensities,
    tof_indices,
    frame_indptr,
    max_scan_count: int,
) -> None:
    """Read and parse a frame directly from a Bruker .d.analysis.tdf_bin.

    IMPORTANT NOTE: This function is decorated with alphatims.utils.threadpool.
    The first argument is thus expected to be provided as an iterable
    containing ints instead of a single int.

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
    """
    import pyzstd
    with open(tdf_bin_file_name, "rb") as infile:
        offset = tims_offset_values[frame_id]
        infile.seek(offset)
        bin_size = int.from_bytes(infile.read(4), "little")
        scan_count = int.from_bytes(infile.read(4), "little")
        if bin_size != 8:
            comp_data = infile.read(bin_size - 8)
            decomp_data = pyzstd.decompress(comp_data)
            (
                scan_indices_,
                tof_indices_,
                intensities_
            ) = parse_decompressed_bruker_binary(decomp_data)
            frame_start = frame_indptr[frame_id]
            frame_end = frame_indptr[frame_id + 1]
            scan_start = frame_id * max_scan_count
            scan_end = scan_start + scan_count
            scan_indptr[scan_start: scan_end] = scan_indices_
            tof_indices[frame_start: frame_end] = tof_indices_
            intensities[frame_start: frame_end] = intensities_


def read_bruker_binary(frames, bruker_d_folder_name: str) -> tuple:
    """Read all data from an "analysis.tdf_bin" of a Bruker .d folder.

    Parameters
    ----------
    frames : pd.DataFrame
        The frames from the "analysis.tdf" SQL database of a Bruker .d folder.
        These can be acquired with e.g. alphatims.bruker.read_bruker_sql.
    bruker_d_folder_name : str
        The full path to a Bruker .d folder.

    Returns
    -------
    : tuple (np.int64[:], np.uint32[:], np.uint16[:]).
        The scan_indptr, tof_indices and intensities.
    """
    # TODO: colon (:) in returns is not parsed properly in readthedocs.
    frame_indptr = np.empty(frames.shape[0] + 1, dtype=np.int64)
    frame_indptr[0] = 0
    frame_indptr[1:] = np.cumsum(frames.NumPeaks.values)
    max_scan_count = frames.NumScans.max()
    scan_count = max_scan_count * frames.shape[0]
    scan_indptr = np.zeros(scan_count + 1, dtype=np.int64)
    intensities = np.empty(frame_indptr[-1], dtype=np.uint16)
    tof_indices = np.empty(frame_indptr[-1], dtype=np.uint32)
    tdf_bin_file_name = os.path.join(bruker_d_folder_name, "analysis.tdf_bin")
    tims_offset_values = frames.TimsId.values
    logging.info(
        f"Reading {frame_indptr.size - 2:,} frames with "
        f"{frame_indptr[-1]:,} TOF arrivals for {bruker_d_folder_name}"
    )
    process_frame(
        range(1, len(frames)),
        tdf_bin_file_name,
        tims_offset_values,
        scan_indptr,
        intensities,
        tof_indices,
        frame_indptr,
        max_scan_count,
    )
    scan_indptr[1:] = np.cumsum(scan_indptr[:-1])
    scan_indptr[0] = 0
    return scan_indptr, tof_indices, intensities


class TimsTOF(object):

    def __init__(
        self,
        bruker_d_folder_name: str,
        bruker_calibrated_mz_values: bool = False,
        bruker_calibrated_mobility_values: bool = False,
        mz_estimation_from_frame: int = 1,
        mobility_estimation_from_frame: int = 1,
    ):
        bruker_d_folder_name = os.path.abspath(bruker_d_folder_name)
        if bruker_d_folder_name.endswith(".d"):
            bruker_dll_available = BRUKER_DLL_FILE_NAME != ""
            self.import_data_from_d_folder(
                bruker_d_folder_name,
                bruker_calibrated_mz_values,
                bruker_calibrated_mobility_values,
                mz_estimation_from_frame and bruker_dll_available,
                mobility_estimation_from_frame and bruker_dll_available,
            )
        elif bruker_d_folder_name.endswith(".hdf"):
            self.import_data_from_hdf_file(
                bruker_d_folder_name,
            )
        if not hasattr(self, "version"):
            self.version = "none"
        if self.version != alphatims.__version__:
            logging.info(
                f"AlphaTims version {self.version} was used to initialize "
                f"{bruker_d_folder_name}, while the current version of "
                f"AlphaTims is {alphatims.__version__}."
            )

    def import_data_from_d_folder(
        self,
        bruker_d_folder_name: str,
        bruker_calibrated_mz_values: bool,
        bruker_calibrated_mobility_values: bool,
        mz_estimation_from_frame: int,
        mobility_estimation_from_frame: int,
    ):
        logging.info(f"Importing data from {bruker_d_folder_name}")
        self.bruker_d_folder_name = bruker_d_folder_name
        self.version = alphatims.__version__
        (
            self.acquisition_mode,
            global_meta_data,
            self.frames,
            self.fragment_frames,
        ) = read_bruker_sql(bruker_d_folder_name)
        (
            self.tof_indptr,
            self.tof_indices,
            self.intensities,
        ) = read_bruker_binary(
            self.frames,
            bruker_d_folder_name,
        )
        self.meta_data = dict(
            zip(global_meta_data.Key, global_meta_data.Value)
        )
        self.frame_max_index = self.frames.shape[0]
        self.scan_max_index = int(self.frames.NumScans.max())
        self.tof_max_index = int(self.meta_data["DigitizerNumSamples"])
        self.rt_values = self.frames.Time.values.astype(np.float64)
        self.mobility_min_value = float(
            self.meta_data["OneOverK0AcqRangeLower"]
        )
        self.mobility_max_value = float(
            self.meta_data["OneOverK0AcqRangeUpper"]
        )
        if mobility_estimation_from_frame == 0:
            self.mobility_values = self.mobility_max_value - (
                self.mobility_max_value - self.mobility_min_value
            ) / self.scan_max_index * np.arange(self.scan_max_index)
        else:
            import ctypes
            with alphatims.bruker.open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mobility values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.scan_max_index).astype(np.float64)
                self.mobility_values = np.empty_like(indices)
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
        self.mz_min_value = float(self.meta_data["MzAcqRangeLower"])
        self.mz_max_value = float(self.meta_data["MzAcqRangeUpper"])
        self.tof_intercept = np.sqrt(self.mz_min_value)
        self.tof_slope = (
            np.sqrt(self.mz_max_value) - self.tof_intercept
        ) / self.tof_max_index
        if mz_estimation_from_frame == 0:
            self.mz_values = (
                self.tof_intercept + self.tof_slope * np.arange(
                    self.tof_max_index
                )
            )**2
        else:
            import ctypes
            with alphatims.bruker.open_bruker_d_folder(
                bruker_d_folder_name
            ) as (bruker_dll, bruker_d_folder_handle):
                logging.info(
                    f"Fetching mz values from {bruker_d_folder_name}"
                )
                indices = np.arange(self.tof_max_index).astype(np.float64)
                self.mz_values = np.empty_like(indices)
                bruker_dll.tims_index_to_mz(
                    bruker_d_folder_handle,
                    mz_estimation_from_frame,
                    indices.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.mz_values.ctypes.data_as(
                        ctypes.POINTER(ctypes.c_double)
                    ),
                    self.tof_max_index
                )
        (
            quad_indptr,
            self.quad_mz_values,
            self.precursor_indices
        ) = parse_quad_indptr(
            self.fragment_frames.Frame.values,
            self.fragment_frames.ScanNumBegin.values,
            self.fragment_frames.ScanNumEnd.values,
            self.fragment_frames.IsolationMz.values,
            self.fragment_frames.IsolationWidth.values,
            self.fragment_frames.Precursor.values,
            self.scan_max_index,
            self.frame_max_index,
        )
        self.quad_indptr = self.tof_indptr[quad_indptr]
        self.quad_max_mz_value = int(np.max(self.quad_mz_values[:, 1]))
        self.precursor_max_index = int(np.max(self.precursor_indices))

    def save_as_hdf(
        self,
        directory: str,
        file_name: str,
        overwrite: bool = False,
        compress: bool = False
    ):
        full_file_name = os.path.join(
            directory,
            file_name
        )
        if overwrite:
            hdf_mode = "w"
        else:
            hdf_mode = "a"
        logging.info(
            f"Writing TimsTOF data to {full_file_name}"
        )
        self.compress = compress
        with h5py.File(full_file_name, hdf_mode, swmr=True) as hdf_root:
            alphatims.utils.create_hdf_group_from_dict(
                hdf_root.create_group("raw"),
                self.__dict__,
                overwrite=overwrite,
                compress=compress,
            )

    def import_data_from_hdf_file(
        self,
        bruker_d_folder_name: str,
    ):
        with h5py.File(bruker_d_folder_name, "r") as hdf_root:
            self.__dict__ = alphatims.utils.create_dict_from_hdf_group(
                hdf_root["raw"]
            )

    def convert_from_indices(
        self,
        raw_indices=None,
        *,
        frame_indices=None,
        quad_indices=None,
        scan_indices=None,
        tof_indices=None,
        return_frame_indices: bool = False,
        return_scan_indices: bool = False,
        return_quad_indices: bool = False,
        return_tof_indices: bool = False,
        return_rt_values: bool = False,
        return_mobility_values: bool = False,
        return_quad_mz_values: bool = False,
        return_precursor_indices: bool = False,
        return_mz_values: bool = False,
        return_intensity_values: bool = False,
        return_as_dict: bool = False,
    ):
        result = {}
        if (raw_indices is not None) and any(
            [
                return_frame_indices,
                return_scan_indices,
                return_quad_indices,
                return_rt_values,
                return_mobility_values,
                return_quad_mz_values,
                return_precursor_indices
            ]
        ):
            parsed_indices = indptr_lookup(
                self.tof_indptr,
                raw_indices,
            )
        if (return_frame_indices or return_rt_values) and (
            frame_indices is None
        ):
            frame_indices = parsed_indices // self.scan_max_index
        if (return_scan_indices or return_mobility_values) and (
            scan_indices is None
        ):
            scan_indices = parsed_indices % self.scan_max_index
        if (
            return_quad_indices or return_quad_mz_values or return_precursor_indices
        ) and (
            quad_indices is None
        ):
            quad_indices = indptr_lookup(
                self.quad_indptr,
                raw_indices,
            )
        if (return_tof_indices or return_mz_values) and (tof_indices is None):
            tof_indices = self.tof_indices[raw_indices]
        if return_frame_indices:
            result["frame_indices"] = frame_indices
        if return_scan_indices:
            result["scan_indices"] = scan_indices
        if return_quad_indices:
            result["quad_indices"] = quad_indices
        if return_precursor_indices:
            result["precursor_indices"] = self.precursor_indices[quad_indices]
        if return_tof_indices:
            result["tof_indices"] = tof_indices
        if return_rt_values:
            result["rt_values"] = self.rt_values[frame_indices]
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
            result["intensity_values"] = self.intensities[raw_indices]
        if not return_as_dict:
            # python >= 3.7 maintains dict insertion order
            return list(result.values())
        else:
            return result

    def convert_to_indices(
        self,
        values,
        *,
        return_frame_indices: bool = False,
        return_scan_indices: bool = False,
        return_tof_indices: bool = False,
        side: str = "right",
        return_type: str = "",
    ):
        if return_frame_indices:
            return_type = "frame_indices"
        elif return_scan_indices:
            return_type = "scan_indices"
        elif return_tof_indices:
            return_type = "tof_indices"
        if return_type == "frame_indices":
            return np.searchsorted(self.rt_values, values, side)
        elif return_type == "scan_indices":
            return np.searchsorted(
                self.mobility_values[::-1],
                values,
                side
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
                return self.precursor_max_index + 1
        else:
            raise KeyError(f"return_type '{return_type}' is invalid")

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = tuple([keys])
        if isinstance(keys[-1], str):
            if keys[-1] == "df":
                as_dataframe = True
            else:
                as_dataframe = False
            keys = keys[:-1]
        else:
            as_dataframe = False
        parsed_keys = self.parse_keys(keys)
        raw_indices = filter_indices(
            frame_slices=parsed_keys["frame_indices"],
            scan_slices=parsed_keys["scan_indices"],
            precursor_slices=parsed_keys["precursor_indices"],
            tof_slices=parsed_keys["tof_indices"],
            quad_slices=parsed_keys["quad_values"],
            intensity_slices=parsed_keys["intensity_values"],
            frame_max_index=self.frame_max_index,
            scan_max_index=self.scan_max_index,
            tof_indptr=self.tof_indptr,
            precursor_indices=self.precursor_indices,
            quad_mz_values=self.quad_mz_values,
            quad_indptr=self.quad_indptr,
            tof_indices=self.tof_indices,
            intensities=self.intensities,
        )
        if as_dataframe:
            return self.as_dataframe(raw_indices)
        else:
            return raw_indices

    def bin_intensities(self, indices, axis):
        intensities = self.intensities[indices].astype(np.float64)
        max_index = {
            "rt": self.frame_max_index,
            "mobility": self.scan_max_index,
            "mz": self.tof_max_index,
        }
        parsed_indices = self.convert_from_indices(
            indices,
            return_frame_indices="rt" in axis,
            return_scan_indices="mobility" in axis,
            return_tof_indices="mz" in axis,
            return_as_dict=True,
        )
        binned_intensities = np.zeros(tuple([max_index[ax] for ax in axis]))
        parse_dict = {
            "rt": "frame_indices",
            "mobility": "scan_indices",
            "mz": "tof_indices",
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
        indices,
        *,
        frame_indices=True,
        scan_indices=True,
        quad_indices=False,
        tof_indices=True,
        precursor_indices=True,
        rt_values=True,
        mobility_values=True,
        quad_mz_values=True,
        mz_values=True,
        intensity_values=True
    ):
        return pd.DataFrame(
           self.convert_from_indices(
                indices,
                return_frame_indices=frame_indices,
                return_scan_indices=scan_indices,
                return_quad_indices=quad_indices,
                return_precursor_indices=precursor_indices,
                return_tof_indices=tof_indices,
                return_rt_values=rt_values,
                return_mobility_values=mobility_values,
                return_quad_mz_values=quad_mz_values,
                return_mz_values=mz_values,
                return_intensity_values=intensity_values,
                return_as_dict=True,
            )
        )

    def parse_keys(self, keys):
        # """
        # Keys is at most a 5-tuple, with selection criteria for the
        # LC-IMS-MSMS dimensions:
        #     (
        #         'frame_index/rt_value',
        #         'scan_index/mobility_value',
        #         'precursor_index/quad_mz_value',
        #         'TOF_index/tof_mz_value',
        #         'intensity_values',
        #     )
        # Each element of this tuple can be either:
        #     A slice
        #         slice.start and slice.stop can be either integer,
        #         float or None,
        #         representing respectively a value or an index.
        #         For the intensity dimension, both integers and floats are
        #         interpreted as a value
        #         slice.step can only be None or an integer.
        #     An iterable with sorted indices
        # NOTE: Negative slicing is not supported
        # """
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
            for dimension in dimensions:
                if dimension in keys[0]:
                    new_keys.append(keys[0][dimension])
                else:
                    new_keys.append(slice(None))
            if "intensity_values" in keys[0]:
                new_keys.append(keys[0]["intensity_values"])
            keys = new_keys
        for i, dimension in enumerate(dimensions):
            try:
                dimension_slices[
                    dimension
                ] = self.convert_slice_key_to_integer(
                    keys[i] if (i < len(keys)) else slice(None),
                    dimension
                )
            except PrecursorFloatError:
                dimension_slices[
                    "precursor_indices"
                ] = self.convert_slice_key_to_integer(
                    slice(None),
                    "precursor_indices"
                )
                dimension_slices[
                    "quad_values"
                ] = self.convert_slice_key_to_float(keys[i])
        dimension_slices["intensity_values"] = self.convert_slice_key_to_float(
            keys[-1] if (len(keys) > len(dimensions)) else slice(None)
        )
        if "quad_values" not in dimension_slices:
            dimension_slices["quad_values"] = np.array(
                [[-np.inf, np.inf]],
                dtype=np.float
            )
        return dimension_slices

    def convert_slice_key_to_float(self, key):
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
            return np.array([[start, stop]], dtype=np.float)
        else:
            if not isinstance(key, np.ndarray):
                key = np.array(key, dtype=np.float)
            if not isinstance(key.ravel()[0], np.float):
                raise ValueError
            if len(key.shape) == 1:
                return np.array([[key, key]]).T
            elif len(key.shape) == 2:
                if key.shape[1] != 2:
                    raise ValueError
                return key
            else:
                raise ValueError

    def convert_slice_key_to_integer(self, key, dimension):
        # TODO: BUG? 0-value sometimes interpreted as float?
        try:
            iter(key)
        except TypeError:
            if key is None:
                key = slice(None)
            if isinstance(key, slice):
                start = key.start
                if not isinstance(start, (np.integer, int)):
                    if start is None:
                        start = -np.inf
                    if not isinstance(start, (np.inexact, float)):
                        raise ValueError
                    start = self.convert_to_indices(
                        start,
                        return_type=dimension
                    )
                stop = key.stop
                if not isinstance(stop, (np.integer, int)):
                    if stop is None:
                        stop = np.inf
                    if not isinstance(stop, (np.inexact, float)):
                        raise ValueError
                    stop = self.convert_to_indices(
                        stop,
                        return_type=dimension
                    )
                step = key.step
                if not isinstance(step, (np.integer, int)):
                    if step is not None:
                        raise ValueError
                    step = 1
                return np.array([[start, stop, step]])
            elif isinstance(key, (np.integer, int)):
                return np.array([[key, key + 1, 1]])
            else:
                raise ValueError
        else:
            if not isinstance(key, np.ndarray):
                key = np.array(key)
            if not isinstance(key.ravel()[0], np.integer):
                key = self.convert_to_indices(key, return_type=dimension)
            if len(key.shape) == 1:
                return np.array([key, key + 1, np.repeat(1, key.size)]).T
            elif len(key.shape) == 2:
                if key.shape[1] != 3:
                    raise ValueError
                return key
            else:
                raise ValueError


class PrecursorFloatError(TypeError):
    """Used to indicate that a precursor value is not an int but a float."""
    pass


@alphatims.utils.njit
def valid_quad_mz_values(
    low_mz_value: float,
    high_mz_value: float,
    quad_slices,
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
    quad_slices : np.float64[:, :]
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
def valid_precursor_index(precursor_index: int, precursor_slices) -> bool:
    """Check if a precursor index is included in the slices.

    Parameters
    ----------
    precursor_index : int
        The precursor index to validate.
    precursor_slices : np.int64[:, :, :]
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
    frame_slices,
    scan_slices,
    precursor_slices,
    tof_slices,
    quad_slices,
    intensity_slices,
    frame_max_index: int,
    scan_max_index: int,
    tof_indptr,
    precursor_indices,
    quad_mz_values,
    quad_indptr,
    tof_indices,
    intensities,
):
    """Filter raw indices by slices from all dimensions.

    Parameters
    ----------
    frame_slices : np.int64[:, :, :]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(frame_slices[:, :2].ravel()) >= 0) = True).
    scan_slices : np.int64[:, :, :]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(scan_slices[:, :2].ravel()) >= 0) = True).
    precursor_slices : np.int64[:, :, :]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(precursor_slices[:, :2].ravel()) >= 0) = True).
    tof_slices : np.int64[:, :, :]
        Each row of the array is assumed to be a (start, stop, step) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(tof_slices[:, :2].ravel()) >= 0) = True).
    quad_slices : np.float64[:, :]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(quad_slices.ravel()) >= 0) = True).
    intensity_slices : np.float64[:, :]
        Each row of the array is assumed to be (lower_mz, upper_mz) tuple.
        This array is assumed to be sorted, disjunct and strictly increasing
        (i.e. np.all(np.diff(intensity_slices.ravel()) >= 0) = True).
    frame_max_index : int
        The maximum frame index of a TimsTOF object.
    scan_max_index : int
        The maximum scan index of a TimsTOF object.
    tof_indptr : np.int64[:]
        The self.tof_indptr array of a TimsTOF object.
    precursor_indices : np.int64[:]
        The self.precursor_indices array of a TimsTOF object.
    quad_mz_values : np.float64[:, :]
        The self.quad_mz_values array of a TimsTOF object.
    quad_indptr : np.int64[:]
        The self.quad_indptr array of a TimsTOF object.
    tof_indices : np.uint32[:]
        The self.tof_indices array of a TimsTOF object.
    intensities : np.uint16[:]
        The self.intensities array of a TimsTOF object.

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
    starts = tof_indptr[:-1].reshape(
        frame_max_index,
        scan_max_index
    )
    ends = tof_indptr[1:].reshape(
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


# @alphatims.utils.njit
def parse_quad_indptr(
    frame_ids,
    scan_begins,
    scan_ends,
    isolation_mzs,
    isolation_widths,
    precursors,
    scan_max_index: int,
    frame_max_index: int,
) -> tuple:
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
        low = frame_id * scan_max_index + scan_begin - 1
        if low > high:
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
    return (
        np.array(quad_indptr),
        np.stack([quad_low_values, quad_high_values]).T,
        np.array(precursor_indices),
    )


# TODO: Overhead of using multiple threads is slower
@alphatims.utils.pjit(thread_count=1)
def add_intensity_to_bin(
    query_index: int,
    intensities,
    parsed_indices,
    intensity_bins
) -> None:
    """Add the intensity of a query to the appropriate bin.

    IMPORTANT NOTE: This function is decorate with alphatims.utils.pjit.
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


@alphatims.utils.njit
def indptr_lookup(targets, queries, momentum_amplifier: int = 2):
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
