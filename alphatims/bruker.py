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

# TODO import read_bruker_sql from alpharaw

# TODO import parse_decompressed_bruker_binary_type2 from alpharaw


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


# TODO import process_frame, read_bruker_binary from alpharaw
# TODO: import TimsTOF from alpharaw
# TODO: import PrecursorFloatError from alpharaw
# TODO: import set_precursor, centroid_spectra from alpharaw

# TODO import filter_spectra_by_abundant_peaks, trim_spectra from alpharaw

# TODO: import parse_keys, convert_slice_key_to_float_array, convert_slice_key_to_int_array from alpharaw


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



# TODO import valid_quad_mz_values, valid_precursor_index, filter_indices, add_intensity_to_bin from alpharaw
# TODO import indptr_lookup from alpharaw

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
