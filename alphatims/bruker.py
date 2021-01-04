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
    BRUKER_DLL_FILE_NAME = "timsdata.dll"
elif sys.platform[:5] == "linux":
    BRUKER_DLL_FILE_NAME = "timsdata.so"
else:
    logging.warning(
        "No Bruker libraries are available for MacOS. "
        "Raw data import will not be possible."
    )
    logging.info("")
    BRUKER_DLL_FILE_NAME = ""
BRUKER_DLL_FILE_NAME = os.path.join(
    alphatims.utils.EXT_PATH,
    BRUKER_DLL_FILE_NAME
)
MSMSTYPE_PASEF = 8
MSMSTYPE_DIAPASEF = 9


def init_bruker_dll(bruker_dll_file_name):
    import ctypes
    # logging.info(f"Reading bruker dll file {bruker_dll_file_name}")
    bruker_dll = ctypes.cdll.LoadLibrary(
        os.path.realpath(bruker_dll_file_name)
    )
    bruker_dll.tims_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    bruker_dll.tims_open.restype = ctypes.c_uint64
    bruker_dll.tims_close.argtypes = [ctypes.c_uint64]
    bruker_dll.tims_close.restype = None
    # dia_pasef_dll.tims_has_recalibrated_state.argtypes = [ctypes.c_uint64 ]
    # dia_pasef_dll.tims_has_recalibrated_state.restype = ctypes.c_uint32
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
    # TODO: multiple threads is equally fast as just 1 for io?
    bruker_dll.tims_set_num_threads(alphatims.utils.MAX_THREADS)
    # bruker_dll.tims_set_num_threads(1)
    return bruker_dll


@contextlib.contextmanager
def open_bruker_d_folder(bruker_dll, bruker_d_folder_name):
    try:
        if isinstance(bruker_dll, str):
            bruker_dll = init_bruker_dll(bruker_dll)
        logging.info(f"Opening handle for {bruker_d_folder_name}")
        bruker_d_folder_handle = bruker_dll.tims_open(
            bruker_d_folder_name.encode('utf-8'),
            0
        )
        yield bruker_dll, bruker_d_folder_handle
    finally:
        logging.info(f"Closing handle for {bruker_d_folder_name}")
        bruker_dll.tims_close(bruker_d_folder_handle)


def read_bruker_frames(
    bruker_d_folder_name,
    add_zeroth_frame=True,
    drop_polarity=True,
):
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
        if MSMSTYPE_DIAPASEF in frames.MsMsType.values:
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
        elif MSMSTYPE_PASEF in frames.MsMsType.values:
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
def parse_frame_buffer(
    scan_offset,
    buffer,
    scan_count,
    peak_start,
    scan_indptr,
    mz_indices,
    intensities,
):
    scan_indptr[scan_offset: scan_offset + scan_count] = buffer[:scan_count]
    end_indices = np.cumsum(buffer[:scan_count])
    for end, size in zip(end_indices, buffer[:scan_count]):
        start = end - size
        mz_start = scan_count + start * 2
        int_start = scan_count + start * 2 + size
        buffer_start = peak_start + start
        buffer_end = peak_start + end
        mz_indices[buffer_start: buffer_end] = buffer[
            mz_start: mz_start + size
        ]
        intensities[buffer_start: buffer_end] = buffer[
            int_start: int_start + size
        ]


def read_scans_of_frame(
    frame_id,
    frames,
    frame_indptr,
    intensities,
    mz_indices,
    scan_indptr,
    bruker_dll,
    bruker_d_folder_handle,
    calibrated_mzs=None,
    calibrated_ccs=None,
):
    import ctypes
    scan_start = 0
    scan_end = frames.NumScans[frame_id]
    scan_count = scan_end - scan_start
    peak_start = frame_indptr[frame_id]
    peak_end = frame_indptr[frame_id + 1]
    peak_count = peak_end - peak_start
    buffer = np.empty(
        shape=scan_count + 2 * peak_count,
        dtype=np.uint32
    )
    bruker_dll.tims_read_scans_v2(
        bruker_d_folder_handle,
        frame_id,
        scan_start,
        scan_end,
        buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(buffer) * 4
    )
    scan_offset = scan_count * (frame_id)
    parse_frame_buffer(
        scan_offset,
        buffer,
        scan_count,
        peak_start,
        scan_indptr,
        mz_indices,
        intensities,
    )
    if calibrated_mzs is not None:
        bruker_dll.tims_index_to_mz(
            bruker_d_folder_handle,
            frame_id,
            mz_indices[peak_start: peak_end].astype(np.float64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            calibrated_mzs[peak_start: peak_end].ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            peak_count
        )
    if calibrated_ccs is not None:
        bruker_dll.tims_scannum_to_oneoverk0(
            bruker_d_folder_handle,
            frame_id,
            np.arange(scan_end, dtype=np.float64).ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            calibrated_ccs[
                scan_offset: scan_offset + scan_count
            ].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            scan_count
        )


def read_bruker_scans(
    frames,
    bruker_d_folder_name:str,
    bruker_calibrated_mz_values:bool=False,
    bruker_calibrated_mobility_values:bool=False,
):
    frame_indptr = np.empty(frames.shape[0] + 1, dtype=np.int64)
    frame_indptr[0] = 0
    frame_indptr[1:] = np.cumsum(frames.NumPeaks.values)
    scan_count = frames.NumScans.max() * frames.shape[0]
    scan_indptr = np.empty(scan_count + 1, dtype=np.int64)
    if bruker_calibrated_mz_values:
        calibrated_mzs = np.empty(frame_indptr[-1], dtype=np.float64)
    else:
        calibrated_mzs = None
    if bruker_calibrated_mobility_values:
        calibrated_ccs = np.empty(scan_count, dtype=np.float64)
    else:
        calibrated_ccs = None
    intensities = np.empty(frame_indptr[-1], dtype=np.uint16)
    tof_indices = np.empty(frame_indptr[-1], dtype=np.uint32)
    with open_bruker_d_folder(
        BRUKER_DLL_FILE_NAME,
        bruker_d_folder_name
    ) as (bruker_dll, bruker_d_folder_handle):
        logging.info(
            f"Reading {frame_indptr[-1]:,} TOF arrivals for "
            f"{bruker_d_folder_name}"
        )
        for frame_id in alphatims.utils.progress_callback(
            range(1, frame_indptr.shape[0] - 1)
        ):
            read_scans_of_frame(
                frame_id,
                frames,
                frame_indptr,
                intensities,
                tof_indices,
                scan_indptr,
                bruker_dll,
                bruker_d_folder_handle,
                calibrated_mzs=calibrated_mzs,
                calibrated_ccs=calibrated_ccs,
            )
    scan_indptr[1:] = np.cumsum(scan_indptr[:-1])
    scan_indptr[0] = 0
    return (
        scan_indptr,
        tof_indices,
        intensities,
        calibrated_mzs,
        calibrated_ccs,
    )


class TimsTOF(object):

    def __init__(
        self,
        bruker_d_folder_name:str,
        bruker_calibrated_mz_values:bool=False,
        bruker_calibrated_mobility_values:bool=False,
        mz_estimation_from_frame:int=1,
        mobility_estimation_from_frame:int=1,
    ):
        bruker_d_folder_name = os.path.abspath(bruker_d_folder_name)
        if bruker_d_folder_name.endswith(".d"):
            self.import_data_from_d_folder(
                bruker_d_folder_name,
                bruker_calibrated_mz_values,
                bruker_calibrated_mobility_values,
                mz_estimation_from_frame,
                mobility_estimation_from_frame,
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
        bruker_d_folder_name:str,
        bruker_calibrated_mz_values:bool,
        bruker_calibrated_mobility_values:bool,
        mz_estimation_from_frame:int,
        mobility_estimation_from_frame:int,
    ):
        logging.info(f"Importing data from {bruker_d_folder_name}")
        self.bruker_d_folder_name = bruker_d_folder_name
        self.version = alphatims.__version__
        (
            self.acquisition_mode,
            global_meta_data,
            self.frames,
            self.fragment_frames,
        ) = read_bruker_frames(bruker_d_folder_name)
        (
            self.tof_indptr,
            self.tof_indices,
            self.intensities,
            calibrated_mz_values,
            calibrated_mobility_values,
        ) = read_bruker_scans(
            self.frames,
            bruker_d_folder_name,
            bruker_calibrated_mz_values,
            bruker_calibrated_mobility_values,
        )
        if bruker_calibrated_mz_values:
            self.calibrated_mz_values = calibrated_mz_values
        if bruker_calibrated_mobility_values:
            self.calibrated_mobility_values = calibrated_mobility_values
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
                alphatims.bruker.BRUKER_DLL_FILE_NAME,
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
                alphatims.bruker.BRUKER_DLL_FILE_NAME,
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
            self.quad_indptr,
            self.quad_low_values,
            self.quad_high_values,
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
        self.quad_max_index = int(np.max(self.quad_high_values))
        self.precursor_max_index = int(np.max(self.precursor_indices))

    def save_as_hdf(
        self,
        directory:str,
        file_name:str,
        overwrite:bool=False,
        compress:bool=False
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
        bruker_d_folder_name:str,
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
        return_frame_indices:bool=False,
        return_scan_indices:bool=False,
        return_quad_indices:bool=False,
        return_tof_indices:bool=False,
        return_rt_values:bool=False,
        return_mobility_values:bool=False,
        return_quad_mz_values:bool=False,
        return_precursor_indices:bool=False,
        return_mz_values:bool=False,
        return_intensity_values:bool=False,
        return_as_dict:bool=False,
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
            parsed_indices = np.searchsorted(
                self.tof_indptr,
                raw_indices,
                "right"
            ) - 1
        if (return_frame_indices or return_rt_values) and frame_indices is None:
            frame_indices = parsed_indices // self.scan_max_index
        if (return_scan_indices or return_mobility_values) and scan_indices is None:
            scan_indices = parsed_indices % self.scan_max_index
        if (
            return_quad_indices or return_quad_mz_values or return_precursor_indices
        ) and quad_indices is None:
            quad_indices = np.searchsorted(
                self.quad_indptr,
                parsed_indices,
                "right"
            ) - 1
        if (return_tof_indices or return_mz_values) and tof_indices is None:
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
            result["quad_low_mz_values"] = self.quad_low_values[quad_indices]
            result["quad_high_mz_values"] = self.quad_high_values[quad_indices]
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
        return_frame_indices:bool=False,
        return_scan_indices:bool=False,
        return_tof_indices:bool=False,
        side:str="right",
        return_type:str="",
    ):
        if return_frame_indices:
            return_type = "frame"
        elif return_scan_indices:
            return_type = "scan"
        elif return_tof_indices:
            return_type = "tof"
        if return_type == "frame":
            return np.searchsorted(self.rt_values, values, side)
        elif return_type == "scan":
            return self.mobility_values.size - np.searchsorted(
                self.mobility_values[::-1],
                values,
                side
            )
        elif return_type == "tof":
            return np.searchsorted(self.mz_values, values, side)
        elif return_type == "precursor":
            try:
                if values not in [-np.inf, np.inf]:
                    raise PrecursorValueError(
                        "Can not convert values to precursor_indices"
                    )
            except ValueError:
                raise PrecursorValueError(
                    "Can not convert values to precursor_indices"
                )
            if values == -np.inf:
                return 0
            elif values == np.inf:
                return self.precursor_max_index + 1
        else:
            raise KeyError(f"return_type '{return_type}' is invalid")

    def __getitem__(self, keys):
        if len(keys) > 5:
            raise ValueError(
                "Slicing LC-IMS-MSMS can be done in maximum 5 dimensions: "
                "frame_index/rt_value, scan_index/mobility_value, "
                "precursor_index/quad_mz_value, TOF_index/tof_mz_value "
                "and intensity. Integers are assumed to be indices, while "
                "floats are assumed as values. Intensity is always assumed "
                "as a float"
            )
        else:
            keys = list(keys)
            while len(keys) < 5:
                keys.append(slice(None))
        stretch_starts = self.tof_indptr[:-1].reshape(
            self.frame_max_index,
            self.scan_max_index
        )
        stretch_ends = self.tof_indptr[1:].reshape(
            self.frame_max_index,
            self.scan_max_index
        )
        new_keys = []
        for i, key in enumerate(keys):
            if (i in [0, 1, 3]) and isinstance(key, slice):
                slice_start = key.start
                slice_stop = key.stop
                slice_step = key.step
                if isinstance(slice_start, float):
                    slice_start = self.convert_to_indices(
                        slice_start,
                        return_frame_indices=(i == 0),
                        return_scan_indices=(i == 1),
                        return_tof_indices=(i == 3),
                    )
                if isinstance(slice_stop, float):
                    slice_stop = self.convert_to_indices(
                        slice_stop,
                        return_frame_indices=(i == 0),
                        return_scan_indices=(i == 1),
                        return_tof_indices=(i == 3),
                        side="right"
                    )
                new_keys.append(slice(slice_start, slice_stop, slice_step))
            else:
                new_keys.append(key)
        keys = tuple(new_keys)
        slice_start = keys[-2].start
        if slice_start is None:
            slice_start = -np.inf
        slice_stop = keys[-2].stop
        if slice_stop is None:
            slice_stop = np.inf
        slice_step = keys[-2].step
        if slice_step is None:
            slice_step = 1
        quad_start = keys[-3].start
        if quad_start is None:
            quad_start = -np.inf
        quad_stop = keys[-3].stop
        if quad_stop is None:
            quad_stop = np.inf
        if isinstance(quad_start, int):
            precursor_low_index = quad_start
            quad_start = -np.inf
        else:
            precursor_low_index = -1
        if isinstance(quad_stop, int):
            precursor_high_index = quad_stop
            quad_stop = np.inf
        else:
            precursor_high_index = self.precursor_max_index + 1
        raw_indices = tof_slicer(
            self.tof_indices,
            slice_start,
            slice_stop,
            slice_step,
            stretch_starts[keys[:2]].flatten(),
            stretch_ends[keys[:2]].flatten(),
            self.tof_indptr[self.quad_indptr],
            self.quad_low_values,
            self.quad_high_values,
            quad_start,
            quad_stop,
            self.precursor_indices,
            precursor_low_index,
            precursor_high_index,
        )
        if keys[-1].start is not None:
            raw_indices = raw_indices[
                self.intensities[raw_indices] >= keys[-1].start
            ]
        if keys[-1].stop is not None:
            raw_indices = raw_indices[
                self.intensities[raw_indices] < keys[-1].stop
            ]
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
        bin_intensities(
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
        """
        Keys is at most a 5-tuple, with selection criteria for the
        LC-IMS-MSMS dimensions:
            (
                'frame_index/rt_value',
                'scan_index/mobility_value',
                'precursor_index/quad_mz_value',
                'TOF_index/tof_mz_value',
                'intensity_values',
            )
        Each element of this tuple can be either:
            A slice
                slice.start and slice.stop can be either integer, float or None,
                representing respectively a value or an index.
                For the intensity dimension, both integers and floats are
                interpreted as a value
                slice.step can only be None or an integer.
            An iterable with sorted indices
        NOTE: Negative slicing is not supported
        """
        dimensions = [
            "frame",
            "scan",
            "precursor",
            "tof",
        ]
        dimension_slices = {}
        if len(keys) > (len(dimensions) + 1):
            raise KeyError(
                "LC-IMS-MSMS data can be sliced in maximum 5 dimensions. "
                "Integers are assumed to be indices, while "
                "floats are assumed as values. Intensity is always casted "
                "to integer values, regardless of input type."
            )
        for i, dimension in enumerate(dimensions):
            try:
                dimension_slices[dimension] = self.convert_slice_key_to_integer(
                    keys[i] if (i < len(keys)) else slice(None),
                    dimension
                )
            except PrecursorValueError:
                dimension_slices["precursor"] = self.convert_slice_key_to_integer(
                    slice(None),
                    "precursor"
                )
                dimension_slices["quad_values"] = self.convert_slice_key_to_float(
                    keys[i]
                )
        dimension_slices["intensity_values"] = self.convert_slice_key_to_float(
            keys[-1] if (len(keys) > len(dimensions)) else slice(None)
        )
        if "quad_values" not in dimension_slices:
            dimension_slices["quad_values"] = np.array(
                [-np.inf, np.inf],
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
            return np.array([start, stop], dtype=np.float)
        else:
            if not isinstance(key, np.ndarray):
                key = np.array(key, dtype=np.float)
            if not isinstance(key.ravel()[0], np.float):
                raise ValueError
            if len(key.shape) == 1:
                return np.array([key, key]).T
            elif len(key.shape) == 2:
                if key.shape[1] != 2:
                    raise ValueError
                return key
            else:
                raise ValueError

    def convert_slice_key_to_integer(self, key, dimension):
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


class PrecursorValueError(ValueError):
    pass


@alphatims.utils.njit
def filter_indices(
    frame_slices,
    scan_slices,
    precursor_slices,
    tof_slices,
    quad_slices,
    intensity_slices,
    frame_max_index,
    scan_max_index,
    tof_indptr,
    precursor_indices,
    quad_values,
    quad_indptr,
    tof_indices,
    intensities,
):
    result = []
    starts = tof_indptr[:-1].reshape(
        frame_max_index,
        scan_max_index
    )
    ends = tof_indptr[1:].reshape(
        frame_max_index,
        scan_max_index
    )
    for frame_start, frame_stop, frame_step in frame_slices:
        for scan_start, scan_stop, scan_step in scan_slices:
            sparse_starts = starts[
                slice(frame_start, frame_stop, frame_step),
                slice(scan_start, scan_stop, scan_step)
            ]
            sparse_ends = ends[
                slice(frame_start, frame_stop, frame_step),
                slice(scan_start, scan_stop, scan_step)
            ]
            quad_index = 0
            for sparse_start, sparse_end in zip(sparse_starts, sparse_ends):
                while quad_indptr[quad_index + 1] < sparse_end:
                    quad_index += 1
                if quad_high_values[quad_index] < quad_low:
                    continue
                if quad_low_values[quad_index] >= quad_high:
                    continue
                if precursor_values[quad_index] < precursor_low_value:
                    continue
                if precursor_values[quad_index] >= precursor_high_value:
                    continue
                if (
                    sparse_start == sparse_end
                ) or (
                    index_array[sparse_end - 1] < slice_start
                ) or (
                    index_array[sparse_start] > slice_stop
                ):
                    continue
                if slice_start == -np.inf:
                    idx_start = sparse_start
                else:
                    idx_start = sparse_start + np.searchsorted(
                        index_array[sparse_start: sparse_end],
                        slice_start,
                        "left"
                    )
                if slice_stop == np.inf:
                    idx_stop = sparse_end
                else:
                    idx_stop = idx_start + np.searchsorted(
                        index_array[idx_start: sparse_end],
                        slice_stop,
                        "left"
                    )
                if slice_step == 1:
                    for idx in range(idx_start, idx_stop):
                        result.append(idx)
                else:
                    for idx in range(idx_start, idx_stop):
                        if ((index_array[idx] - slice_start) % slice_step) == 0:
                            result.append(idx)
    return np.array(result)


@alphatims.utils.njit
def tof_slicer(
    index_array,
    slice_start,
    slice_stop,
    slice_step,
    sparse_starts,
    sparse_ends,
    quad_indptr,
    quad_low_values,
    quad_high_values,
    quad_low,
    quad_high,
    precursor_values,
    precursor_low_value,
    precursor_high_value,
):
    result = []
    quad_index = 0
    for sparse_start, sparse_end in zip(sparse_starts, sparse_ends):
        while quad_indptr[quad_index + 1] < sparse_end:
            quad_index += 1
        if quad_high_values[quad_index] < quad_low:
            continue
        if quad_low_values[quad_index] >= quad_high:
            continue
        if precursor_values[quad_index] < precursor_low_value:
            continue
        if precursor_values[quad_index] >= precursor_high_value:
            continue
        if (
            sparse_start == sparse_end
        ) or (
            index_array[sparse_end - 1] < slice_start
        ) or (
            index_array[sparse_start] > slice_stop
        ):
            continue
        if slice_start == -np.inf:
            idx_start = sparse_start
        else:
            idx_start = sparse_start + np.searchsorted(
                index_array[sparse_start: sparse_end],
                slice_start,
                "left"
            )
        if slice_stop == np.inf:
            idx_stop = sparse_end
        else:
            idx_stop = idx_start + np.searchsorted(
                index_array[idx_start: sparse_end],
                slice_stop,
                "left"
            )
        if slice_step == 1:
            for idx in range(idx_start, idx_stop):
                result.append(idx)
        else:
            for idx in range(idx_start, idx_stop):
                if ((index_array[idx] - slice_start) % slice_step) == 0:
                    result.append(idx)
    return np.array(result)


@alphatims.utils.njit
def parse_quad_indptr(
    frame_ids,
    scan_begins,
    scan_ends,
    isolation_mzs,
    isolation_widths,
    precursors,
    scan_max_index,
    frame_max_index,
):
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
            precursor_indices.append(-1)
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
        precursor_indices.append(-1)
    return (
        np.array(quad_indptr),
        np.array(quad_low_values),
        np.array(quad_high_values),
        np.array(precursor_indices),
    )


# Overhead of using multiple threads is slower
@alphatims.utils.pjit(thread_count=1)
def bin_intensities(
    query,
    intensities,
    parsed_indices,
    intensity_bins
):
    intensity = intensities[query]
    if len(parsed_indices) == 1:
        intensity_bins[parsed_indices[0][query]] += intensity
    elif len(parsed_indices) == 2:
        intensity_bins[
            parsed_indices[0][query],
            parsed_indices[1][query]
        ] += intensity
