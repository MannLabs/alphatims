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

# TODO import BRUKER_DLL_FILE_NAME, init_bruker_dll, open_bruker_d_folder from alpharaw

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
        (str, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool).
        The acquisition_mode, global_meta_data, frames, fragment_frames,
        precursors and calibration availability.
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
        calibration_available = BRUKER_DLL_FILE_NAME != ""
        try:
            pd.read_sql_query(
                "SELECT * from CalibrationInfo",
                sql_database_connection
            )
        except pd.io.sql.DatabaseError:
            calibration_available = False
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
            precursors,
            calibration_available,
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
    intensities = buffer[scan_count + 1::2]
    last_scan = len(intensities) - np.sum(scan_indices[1:])
    scan_indices[:-1] = scan_indices[1:]
    scan_indices[-1] = last_scan
    tof_indices = buffer[scan_count::2].copy()
    index = 0
    for size in scan_indices:
        current_sum = 0
        for i in range(size):
            current_sum += tof_indices[index]
            tof_indices[index] = current_sum
            index += 1
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

# TODO: import TimsTOF from alpharaw

class PrecursorFloatError(TypeError):
    """Used to indicate that a precursor value is not an int but a float."""
    pass


@alphatims.utils.pjit(
    # signature_or_function="void(i8,i8[:],i8[:],i8[:],u4[:],u2[:],u4[:],f8[:],i8[:],i8[:])"
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


def save_as_mgf(
    full_file_name,
    spectrum_indptr,
    intensities,
    mobilities,
    average_mzs,
    mono_mzs,
    charges,
    rtinseconds,
    spectrum_tof_indices,
    spectrum_intensity_values,
    precursor_max_index,
    mz_values,
):
    with open(full_file_name, "w") as infile:
        for index in alphatims.utils.progress_callback(
            range(1, precursor_max_index)
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
                mz_values[spectrum_tof_indices[start: end]],
                spectrum_intensity_values[start: end],
            ):
                infile.write(f"{mz:.6f} {intensity}\n")
            infile.write("END IONS\n")


def save_as_spectra(
    full_file_name,
    spectrum_indptr,
    intensities,
    mobilities,
    average_mzs,
    mono_mzs,
    charges,
    rtinseconds,
    spectrum_tof_indices,
    spectrum_intensity_values,
    mz_values,
):
    with h5py.File(full_file_name, "w") as infile:
        infile["indptr"] = spectrum_indptr[1:]
        infile["fragment_mzs"] = mz_values[
            spectrum_tof_indices
        ]
        infile["fragment_intensities"] = spectrum_intensity_values
        infile["precursor_rt"] = rtinseconds
        infile["precursor_charge"] = charges
        infile["precursor_intensity"] = intensities
        infile["precursor_mobility"] = mobilities
        infile["precursor_average_mz"] = average_mzs
        infile["precursor_monoisotopic_mz"] = mono_mzs
