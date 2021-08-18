#!python

import alphapept.fasta
import alphapept.constants
import alphatims.utils
import alphatims.bruker
import alphatims.plotting
import numpy as np
import pandas as pd
import holoviews as hv
import logging
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.pipeline
import sklearn.model_selection


class Library(object):

    def __init__(
        self,
        file_name: str,
        as_peptide_dict: bool = False,
        decoy: bool = False,
    ):
        self.file_name = file_name
        self.decoy = decoy
        self.peptide_data = pd.read_csv(self.file_name)
        if as_peptide_dict:
            self.peptides = self.convert_to_peptide_dict()
        else:
            self.convert_to_peptide_arrays()

    def __len__(self):
        return len(self.peptide_data)

        def convert_to_peptide_dict(self):
            self.peptide_dict = []
            for rt, mz, im, z, seq_ in alphatims.utils.progress_callback(
                zip(
                    self.peptide_data.rt_apex,
                    self.peptide_data.mz,
                    self.peptide_data.mobility,
                    self.peptide_data.charge,
                    self.peptide_data.sequence,
                    # self.peptide_data.ion_int,
                    # self.peptide_data.ion_types,
                ),
                total=len(self.peptide_data.rt_apex)
            ):
                seq = alphapept.fasta.parse(seq_)
                if self.decoy:
                    seq[:-1] = seq[:-1][::-1]
                peptide = {
                    "sequence": seq_,
                    "mz": mz,
                    "mobility": im,
                    "rt": rt * 60,  # seconds
                    "charge": z,
                    "fragment_mzs": alphapept.fasta.get_frag_dict(
                        seq,
                        alphapept.constants.mass_dict
                    )
                }
                self.peptide_dict.append(peptide)

    def convert_to_peptide_arrays(self):
        self.peptide_rt_apex = self.peptide_data.rt_apex.values * 60
        self.peptide_mzs = self.peptide_data.mz.values
        self.peptide_mobilities = self.peptide_data.mobility.values
        self.peptide_charges = self.peptide_data.charge.values
        self.peptide_sequences = self.peptide_data.sequence.values
        self.peptide_lens = self.peptide_data.n_AA.values * 2 - 2
        self.peptide_offsets = np.empty(
            len(self.peptide_lens) + 1,
            dtype=np.int64
        )
        self.peptide_offsets[0] = 0
        self.peptide_offsets[1:] = np.cumsum(self.peptide_lens)
        self.peptide_fragment_mzs = np.empty(
            self.peptide_offsets[-1],
            dtype=np.float64
        )
        self.peptide_fragment_types = np.empty(
            self.peptide_offsets[-1],
            dtype=np.int8
        )
        set_frags(
            range(len(self.peptide_sequences)),
            self.peptide_sequences,
            self.peptide_fragment_mzs,
            self.peptide_fragment_types,
            self.peptide_offsets,
            self.decoy
        )

    def set_tolerance_arrays(
        self,
        dia_data,
        ppm=50,
        rt_tolerance=30,  # seconds
        mobility_tolerance=0.05,  # 1/k0
    ):
        self.precursor_frame_slices = np.stack(
            [
                dia_data.convert_to_indices(
                    self.peptide_rt_apex - rt_tolerance,
                    return_frame_indices=True
                ),
                dia_data.convert_to_indices(
                    self.peptide_rt_apex + rt_tolerance,
                    return_frame_indices=True
                ),
                np.repeat(1, len(self.peptide_rt_apex))
            ]
        ).T.astype(np.int64)
        self.precursor_scan_slices = np.stack(
            [
                dia_data.convert_to_indices(
                    self.peptide_mobilities + mobility_tolerance,
                    return_scan_indices=True
                ),
                dia_data.convert_to_indices(
                    self.peptide_mobilities - mobility_tolerance,
                    return_scan_indices=True
                ),
                np.repeat(1, len(self.peptide_mobilities))
            ]
        ).T.astype(np.int64)
        self.precursor_tof_slices = np.stack(
            [
                dia_data.convert_to_indices(
                    self.peptide_mzs / (1 + ppm / 10**6),
                    return_tof_indices=True
                ),
                dia_data.convert_to_indices(
                    self.peptide_mzs * (1 + ppm / 10**6),
                    return_tof_indices=True
                ),
                np.repeat(1, len(self.peptide_mzs))
            ]
        ).T.astype(np.int64)
        self.precursor_mz_slices = np.stack(
            [
                self.peptide_mzs / (1 + ppm / 10**6),
                self.peptide_mzs * (1 + ppm / 10**6),
            ]
        ).T
        self.fragment_tof_slices = np.stack(
            [
                dia_data.convert_to_indices(
                    self.peptide_fragment_mzs / (1 + ppm / 10**6),
                    return_tof_indices=True
                ),
                dia_data.convert_to_indices(
                    self.peptide_fragment_mzs * (1 + ppm / 10**6),
                    return_tof_indices=True
                ),
                np.repeat(1, len(self.peptide_fragment_mzs))
            ]
        ).T.astype(np.int64)


@alphatims.utils.threadpool
def set_frags(
    peptide_index,
    peptide_sequences,
    peptide_fragment_mzs,
    peptide_fragment_types,
    peptide_offsets,
    decoy=False
):
    seq_string = peptide_sequences[peptide_index]
    seq = alphapept.fasta.parse(seq_string)
    if decoy:
        seq[:-1] = seq[:-1][::-1]
    # if decoy:
    #     #diaNN style
    #     original = "GAVLIFMPWSCTYHKRQEND"
    #     mutated = "LLLVVLLLLTSSSSLLNDQE"
    #     seq[1] = alphapept.fasta.parse(
    #         mutated[original.index(seq[1][-1])]
    #     )[0]
    #     seq[-2] = alphapept.fasta.parse(
    #         mutated[original.index(seq[-2][-1])]
    #     )[0]
    # if decoy:
    #     seq[-2], seq[0] = seq[0], seq[-2]
    fragment_mzs, fragment_types = alphapept.fasta.get_fragmass(
        seq,
        alphapept.constants.mass_dict
    )
    start = peptide_offsets[peptide_index]
    end = peptide_offsets[peptide_index + 1]
    order = np.argsort(fragment_mzs)
    peptide_fragment_mzs[start: end] = fragment_mzs[order]
    peptide_fragment_types[start: end] = fragment_types[order]


# @alphatims.utils.threadpool
# def process_library_peptide(
#     peptide_index,
#     library,
#     push_peaks,
#     scores,
#     dia_data,
#     left_frame_borders,
#     right_frame_borders,
#     left_scan_borders,
#     right_scan_borders,
#     inflex_threshold,
# ):
#     precursor_frame_slices = library.precursor_frame_slices
#     precursor_scan_slices = library.precursor_scan_slices
#     precursor_tof_slices = library.precursor_tof_slices
#     precursor_mz_slices = library.precursor_mz_slices
#     fragment_tof_slices = library.fragment_tof_slices
#     peptide_offsets = library.peptide_offsets
#     precursor_indices, fragment_indices = get_peptide_raw_indices(
#         peptide_index,
#         precursor_frame_slices,
#         precursor_scan_slices,
#         precursor_tof_slices,
#         precursor_mz_slices,
#         fragment_tof_slices,
#         peptide_offsets,
#         dia_data.frame_max_index,
#         dia_data.scan_max_index,
#         dia_data.push_indptr,
#         dia_data.precursor_indices,
#         dia_data.quad_mz_values,
#         dia_data.quad_indptr,
#         dia_data.tof_indices,
#         dia_data.intensity_values,
#         dia_data.precursor_max_index,
#     )
# #     TODO: Score peptides
# #     TODO: Note the GIL is not yet released for code below
#     fragment_coordinates = dia_data.convert_from_indices(
#         fragment_indices,
#         return_raw_indices=True,
#         return_frame_indices=True,
#         return_scan_indices=True,
#         return_quad_indices=True,
#         return_precursor_indices=True,
#         return_tof_indices=True,
#         return_rt_values=True,
#         return_mobility_values=True,
#         return_quad_mz_values=True,
#         return_push_indices=True,
#         return_mz_values=True,
#         return_intensity_values=True,
#         raw_indices_sorted=True,
#     )
#     # precursor_coordinates = dia_data.convert_from_indices(
#     #     precursor_indices,
#     #     return_raw_indices=True,
#     #     return_frame_indices=True,
#     #     return_scan_indices=True,
#     #     return_quad_indices=True,
#     #     return_precursor_indices=True,
#     #     return_tof_indices=True,
#     #     return_rt_values=True,
#     #     return_mobility_values=True,
#     #     return_quad_mz_values=True,
#     #     return_push_indices=True,
#     #     return_mz_values=True,
#     #     return_intensity_values=True,
#     #     raw_indices_sorted=True,
#     # )
#     (
#         push_peak,
#         score,
#         hit_matrix,
#         unique_push_indices,
#         bpi,
#         left_frame_border,
#         right_frame_border,
#         left_scan_border,
#         right_scan_border,
#     ) = get_apex(
#         peptide_index,
#         fragment_tof_slices,
#         peptide_offsets,
#         fragment_coordinates["tof_indices"],
#         fragment_coordinates["push_indices"],
#         fragment_coordinates["intensity_values"],
#         dia_data.scan_max_index,
#         inflex_threshold,
#     )
#     left_frame_borders[peptide_index] = left_frame_border
#     right_frame_borders[peptide_index] = right_frame_border
#     left_scan_borders[peptide_index] = left_scan_border
#     right_scan_borders[peptide_index] = right_scan_border
#     push_peaks[peptide_index] = push_peak
#     scores[peptide_index] = score
#     return (
#         push_peak,
#         score,
#         hit_matrix,
#         unique_push_indices,
#         bpi,
#         left_frame_border,
#         right_frame_border,
#         left_scan_border,
#         right_scan_border,
#         fragment_indices,
#         precursor_indices,
#     )


# @alphatims.utils.njit(nogil=True)
# def get_peptide_raw_indices(
#     peptide_index,
#     precursor_frame_slices,
#     precursor_scan_slices,
#     precursor_tof_slices,
#     precursor_mz_slices,
#     fragment_tof_slices,
#     peptide_offsets,
#     frame_max_index,
#     scan_max_index,
#     push_indptr,
#     precursor_indices,
#     quad_mz_values,
#     quad_indptr,
#     tof_indices,
#     intensities,
#     precursor_max_index,
# ):
#     frames = precursor_frame_slices[peptide_index].copy().reshape((1,3))
#     scans = precursor_scan_slices[peptide_index].copy().reshape((1,3))
#     tofs = precursor_tof_slices[peptide_index].copy().reshape((1,3))
#     precursor_indices_ = alphatims.bruker.filter_indices(
#         frame_slices=frames,
#         scan_slices=scans,
#         precursor_slices=np.array([[0, 1, 1]]),
#         tof_slices=tofs,
#         quad_slices=np.array([[-np.inf, np.inf]]),
#         intensity_slices=np.array([[-np.inf, np.inf]]),
#         frame_max_index=frame_max_index,
#         scan_max_index=scan_max_index,
#         push_indptr=push_indptr,
#         precursor_indices=precursor_indices,
#         quad_mz_values=quad_mz_values,
#         quad_indptr=quad_indptr,
#         tof_indices=tof_indices,
#         intensities=intensities,
#     )
#     start = peptide_offsets[peptide_index]
#     end = peptide_offsets[peptide_index + 1]
#     fragment_indices_ = alphatims.bruker.filter_indices(
#         frame_slices=frames,
#         scan_slices=scans,
#         precursor_slices=np.array([[1, precursor_max_index, 1]]),
#         tof_slices=fragment_tof_slices[start: end],
#         quad_slices=precursor_mz_slices[peptide_index].copy().reshape((1,2)),
# #         quad_slices=np.array([[-np.inf, np.inf]]),
#         intensity_slices=np.array([[-np.inf, np.inf]]),
#         frame_max_index=frame_max_index,
#         scan_max_index=scan_max_index,
#         push_indptr=push_indptr,
#         precursor_indices=precursor_indices,
#         quad_mz_values=quad_mz_values,
#         quad_indptr=quad_indptr,
#         tof_indices=tof_indices,
#         intensities=intensities,
#     )
#     return precursor_indices_, fragment_indices_
#
#
# @alphatims.utils.njit(nogil=True)
# def get_apex(
#     precursor_index,
#     library_tof_indices,
#     peptide_offsets,
#     fragment_tof_indices,
#     push_indices,
#     intensity_values,
#     scan_max_index,
#     inflex_threshold,
# ):
#     unique_push_indices = []
#     unique_fragment_counts = []
#     unique_fragment_count = 1
#     last_push_index = push_indices[0]
#     for push_index in push_indices:
#         if push_index != last_push_index:
#             unique_push_indices.append(last_push_index)
#             unique_fragment_counts.append(unique_fragment_count)
#             last_push_index = push_index
#             unique_fragment_count = 1
#         else:
#             unique_fragment_count += 1
#     unique_push_indices.append(last_push_index)
#     unique_fragment_counts.append(unique_fragment_count)
#     unique_push_indices = np.array(unique_push_indices)
#     start = peptide_offsets[precursor_index]
#     end = peptide_offsets[precursor_index + 1]
#     hit_matrix = np.zeros((end - start, len(unique_push_indices)))
#     last_push_index = -1
#     push_offset = -1
#     bpi = np.ones((end - start, 1))
#     for push_index, fragment_tof_index, intensity in zip(
#         push_indices,
#         fragment_tof_indices,
#         intensity_values,
#     ):
#         if push_index != last_push_index:
#             library_tof_index = 0
#             last_push_index = push_index
#             push_offset += 1
#         while fragment_tof_index not in range(
#             library_tof_indices[start + library_tof_index][0],
#             library_tof_indices[start + library_tof_index][1],
#         ):
#             library_tof_index += 1
#         if bpi[library_tof_index] < intensity:
#             bpi[library_tof_index] = intensity
#         hit_matrix[library_tof_index, push_offset] += intensity
#     peak_index = np.argmax((hit_matrix / bpi).sum(axis=0))
#     push_peak = unique_push_indices[peak_index]
#
#     scan = push_peak % scan_max_index
#     scan_pushes = (unique_push_indices % scan_max_index)==scan
#     scan_bpi = (
#         hit_matrix[:, scan_pushes] / bpi
#     ).sum(axis=0) / len(bpi)
#     scan_bpi /= np.max(scan_bpi)
#     left, right = get_borders(scan_bpi, inflex_threshold)
#     scan_pushes = unique_push_indices[scan_pushes]
#     left_frame_border = scan_pushes[left] // scan_max_index
#     right_frame_border = scan_pushes[right] // scan_max_index
#
#     frame = push_peak // scan_max_index
#     frame_pushes = (unique_push_indices // scan_max_index)==frame
#     frame_bpi = (
#         hit_matrix[:, frame_pushes] / bpi
#     ).sum(axis=0) / len(bpi)
#     frame_bpi /= np.max(frame_bpi)
#     # TODO start from apex and move down
#     left, right = get_borders(frame_bpi, inflex_threshold)
#     frame_pushes = unique_push_indices[frame_pushes]
#     left_scan_border = frame_pushes[left] % scan_max_index
#     right_scan_border = frame_pushes[right] % scan_max_index
#
#     return (
#         push_peak,
#         np.sum(hit_matrix[:,peak_index] / bpi.ravel()),
#         hit_matrix,
#         unique_push_indices,
#         bpi,
#         left_frame_border,
#         right_frame_border,
#         left_scan_border,
#         right_scan_border,
#     )
#
#
# @alphatims.utils.njit(nogil=True)
# def get_borders(bpi, threshold):
#     apex = np.argmax(bpi)
#     lower = apex
#     upper = apex
#     while lower > 0:
#         lower -= 1
#         if bpi[lower] <= threshold:
#             break
#     while upper < (len(bpi) - 1):
#         upper += 1
#         if bpi[upper] <= threshold:
#             break
#     # lower = 0
#     # upper = 0
#     # last = 0
#     # for i, current_bpi in enumerate(bpi):
#     #     if current_bpi > threshold:
#     #         size = i - last
#     #         if size > (upper - lower):
#     #             lower, upper = last, i + 1
#     #     else:
#     #         last = i
#     return lower, upper


def visualize_peptide(
    dia_data,
    peptide,
    ppm=50,
    rt_tolerance=30, #seconds
    mobility_tolerance=0.05, #1/k0
    heatmap=False,
):
    precursor_mz = peptide["mz"]
    precursor_mobility = peptide["mobility"]
    precursor_rt = peptide["rt"]
    fragment_mzs = peptide["fragment_mzs"]
    rt_slice = slice(
        precursor_rt - rt_tolerance,
        precursor_rt + rt_tolerance
    )
    im_slice = slice(
        precursor_mobility - mobility_tolerance,
        precursor_mobility + mobility_tolerance
    )
    precursor_mz_slice = slice(
        precursor_mz / (1 + ppm / 10**6),
        precursor_mz * (1 + ppm / 10**6)
    )
    precursor_indices = dia_data[
        rt_slice,
        im_slice,
        0, #index 0 means that the quadrupole is not used
        precursor_mz_slice,
        "raw"
    ]
    if heatmap:
        precursor_heatmap = alphatims.plotting.heatmap(
            dia_data.as_dataframe(precursor_indices),
            x_axis_label="rt",
            y_axis_label="mobility",
            title="precursor",
            width=250,
            height=250
        )
        overlay = precursor_heatmap
    else:
        precursor_xic = alphatims.plotting.line_plot(
            dia_data,
            precursor_indices,
            x_axis_label="rt",
            width=900,
            remove_zeros=True,
            label="precursor"
        )
        overlay = precursor_xic
    for fragment_name, mz in fragment_mzs.items():
        fragment_mz_slice = slice(
            mz / (1 + ppm / 10**6),
            mz * (1 + ppm / 10**6)
        )
        fragment_indices = dia_data[
            rt_slice,
            im_slice,
            precursor_mz_slice,
            fragment_mz_slice,
            "raw"
        ]
        if len(fragment_indices) > 0:
            if heatmap:
                fragment_heatmap = alphatims.plotting.heatmap(
                    dia_data.as_dataframe(fragment_indices),
                    x_axis_label="rt",
                    y_axis_label="mobility",
                    title=f"{fragment_name}: {mz:.3f}",
                    width=250,
                    height=250,
                )
                overlay += fragment_heatmap
            else:
                fragment_xic = alphatims.plotting.line_plot(
                    dia_data,
                    fragment_indices,
                    x_axis_label="rt",
                    width=900,
                    remove_zeros=True,
                    label=fragment_name,
                )
                overlay *= fragment_xic.opts(muted=True)
    if not heatmap:
        overlay.opts(hv.opts.Overlay(legend_position='bottom'))
        overlay.opts(hv.opts.Overlay(click_policy='mute'))
        overlay = overlay.opts(show_legend=True)
    return overlay.opts(
        title=f"{peptide['sequence']}_{peptide['charge']}"
    )


# @alphatims.utils.njit(nogil=True)
# def get_feature(
#     fragment_count: int,
#     scan_max_index: int,
#     push_indices: np.ndarray,
#     indptr: np.ndarray,
#     values: np.ndarray,
#     columns: np.ndarray,
#     intensity_values: np.ndarray,
# ):
#     indptr_T = np.bincount(columns, minlength=fragment_count + 1)
#     indptr_T[1:] = np.cumsum(indptr_T[:-1])
#     indptr_T[0] = 0
#     indptr_T_tmp = indptr_T.copy()
#     values_T = np.empty(len(values))
#     max_intensities = np.ones(fragment_count)
#     columns_T = np.empty_like(columns)
#     for i, push_index in enumerate(push_indices):
#         for index in range(indptr[i], indptr[i + 1]):
#             column = columns[index]
#             value = values[index]
#             intensity = intensity_values[value]
#             offset = indptr_T_tmp[column]
#             columns_T[offset] = push_index
#             values_T[offset] = intensity
#             indptr_T_tmp[column] += 1
#             if intensity > max_intensities[column]:
#                 max_intensities[column] = intensity
#     for i, max_intensity in enumerate(max_intensities):
#         start = indptr_T[i]
#         end = indptr_T[i + 1]
#         values_T[start: end] /= max_intensity
#     apex_value = -1
#     relative_intensities_list = []
#     for i, push_index in enumerate(push_indices):
#         ions = values[indptr[i]: indptr[i + 1]]
#         fragments = columns[indptr[i]: indptr[i + 1]]
#         summed_value = np.sum(intensity_values[ions] / max_intensities[fragments])
#         relative_intensities_list.append(summed_value)
#         if summed_value > apex_value:
#             apex_index = i
#             apex_value = summed_value
#     relative_intensities = np.array(relative_intensities_list)
#     # from matplotlib import pyplot as plt
#     # rt_selection = push_indices // scan_max_index == push_indices[apex_index] // scan_max_index
#     # plt.plot(
#     #     push_indices[rt_selection],
#     #     relative_intensities[rt_selection]
#     # )
#     # plt.scatter([push_indices[apex_index]], [apex_value])
#     # for fragment in range(fragment_count):
#     #     start = indptr_T[fragment]
#     #     end = indptr_T[fragment + 1]
#     #     selection = columns_T[start: end] // scan_max_index == push_indices[apex_index] // scan_max_index
#     #     plt.plot(
#     #         columns_T[start: end][selection],
#     #         values_T[start: end][selection]
#     #     )
#     return (
#         indptr_T,
#         values_T,
#         columns_T,
#         max_intensities,
#         push_indices[apex_index],
#         apex_value,
#         relative_intensities,
#     )


@alphatims.utils.njit(nogil=True)
def define_connections(
    push_indices,
    scan_max_index,
    max_rt,
    max_im,
):
    mat = _create_push_matrix(
        push_indices,
        scan_max_index,
    )
    indptr, indices = _get_matrix_connections(
        mat,
        max_rt,
        max_im,
    )
    return indptr, indices


@alphatims.utils.njit(nogil=True)
def _create_push_matrix(
    push_indices,
    scan_max_index,
):
    im = push_indices % scan_max_index
    min_im = np.min(im)
    max_im = np.max(im)
    rt = push_indices // scan_max_index
    min_rt = np.min(rt)
    max_rt = np.max(rt)
    shape = ((max_rt - min_rt + 1), (max_im - min_im + 1))
    mat = np.repeat(-1, shape[0] * shape[1]).reshape(shape)
    for i, push_index in enumerate(push_indices):
        mat[
            push_index // scan_max_index - min_rt,
            push_index % scan_max_index - min_im,
        ] = i
    return mat


@alphatims.utils.njit(nogil=True)
def _get_matrix_connections(
    mat,
    max_rt,
    max_im,
):
    indptr = [0]
    indices = []
    count = 0
    for rt in range(mat.shape[0]):
        for im in range(mat.shape[1]):
            push_index = mat[rt, im]
            if push_index != -1:
                low_rt = max(0, rt - max_rt)
                high_rt = min(mat.shape[0], rt + max_rt + 1)
                low_im = max(0, im - max_im)
                high_im = min(mat.shape[1], im + max_im + 1)
                for other_rt in range(low_rt, high_rt):
                    for other_im in range(low_im, high_im):
                        other_push_index = mat[other_rt, other_im]
                        if other_push_index != -1:
                            if other_push_index != push_index:
                                count += 1
                                indices.append(other_push_index)
                indptr.append(count)
    return np.array(indptr), np.array(indices)

#
# @alphatims.utils.njit(nogil=True)
# def transpose_pushes(
#     fragment_count: int,
#     push_indptr: np.ndarray,
#     fragment_indices: np.ndarray,
# ):
#     fragment_indptr = np.bincount(
#         fragment_indices,
#         minlength=fragment_count + 1
#     )
#     fragment_indptr[1:] = np.cumsum(fragment_indptr[:-1])
#     fragment_offsets = fragment_indptr[:-1].copy()
#     fragment_indptr[0] = 0
#     raw_pointers = np.empty_like(fragment_indices)
#     push_pointers = np.empty_like(fragment_indices)
#     for push_index, start in enumerate(push_indptr[:-1]):
#         end = push_indptr[push_index + 1]
#         for index in range(start, end):
#             fragment_index = fragment_indices[index]
#             offset = fragment_offsets[fragment_index]
#             raw_pointers[offset] = index
#             push_pointers[offset] = push_index
#             fragment_offsets[fragment_index] += 1
#     return fragment_indptr, raw_pointers, push_pointers


@alphatims.utils.njit(nogil=True)
def make_dense_matrix(
    fragment_count: int,
    push_indptr: np.ndarray,
    fragment_indices: np.ndarray,
    raw_indices: np.ndarray,
):
    shape = (push_indptr.shape[0] - 1, fragment_count)
    mat = np.repeat(-1, shape[0] * shape[1]).reshape(shape)
    offset = 0
    for push_index, start in enumerate(push_indptr[:-1]):
        end = push_indptr[push_index + 1]
        for fragment_index in fragment_indices[start: end]:
            mat[push_index, fragment_index] = raw_indices[offset]
            offset += 1
    return mat


@alphatims.utils.njit(nogil=True)
def get_intensity_matrix(
    matrix: np.ndarray,
    intensity_values: np.ndarray,
):
    intensity_matrix = np.zeros(matrix.shape, dtype=np.float64)
    for push_index in range(matrix.shape[0]):
        for fragment_index in range(matrix.shape[1]):
            raw_index = matrix[push_index, fragment_index]
            if raw_index != -1:
                intensity_matrix[
                    push_index,
                    fragment_index
                ] = intensity_values[raw_index]
    return intensity_matrix


@alphatims.utils.njit(nogil=True)
def smoothen_and_normalize_intensity_matrix(
    intensity_matrix: np.ndarray,
    push_connection_indptr: np.ndarray,
    push_connection_indices: np.ndarray,
):
    smooth_intensity_matrix = np.empty_like(intensity_matrix)
    for push_index in range(intensity_matrix.shape[0]):
        start = push_connection_indptr[push_index]
        end = push_connection_indptr[push_index + 1]
        for fragment_index in range(intensity_matrix.shape[1]):
            intensity = intensity_matrix[push_index, fragment_index]
            for connection in push_connection_indices[start: end]:
                intensity += intensity_matrix[connection, fragment_index]
            smooth_intensity_matrix[
                push_index,
                fragment_index
            ] = intensity / (1 + end - start)
    for fragment_index in range(intensity_matrix.shape[1]):
        max_intensity = np.max(smooth_intensity_matrix[:, fragment_index])
        if max_intensity > 0:
            smooth_intensity_matrix[:, fragment_index] /= max_intensity
    return smooth_intensity_matrix


@alphatims.utils.njit(nogil=True)
def peak_descend(
    index,
    peak_mask,
    intensities,
    push_indices,
    rt_lim,
    im_lim,
    im_cycle,
):
    if peak_mask[index]:
        return
    peak_mask[index] = True
    push_index = push_indices[index]
    rt = push_index // im_cycle
    im = push_index % im_cycle
    intensity = intensities[index]
    for i, other_index in enumerate(push_indices):
        other_rt = other_index // im_cycle
        other_im = other_index % im_cycle
        if np.abs(other_rt - rt) > rt_lim:
            continue
        if np.abs(other_im - im) > im_lim:
            continue
        if intensities[i] < intensity:
            peak_descend(
                i,
                peak_mask,
                intensities,
                push_indices,
                rt_lim,
                im_lim,
                im_cycle,
            )


@alphatims.utils.njit
def fdr_to_q_values(fdr_values):
    q_values = np.zeros_like(fdr_values)
    min_q_value = np.max(fdr_values)
    for i in range(len(fdr_values) - 1, -1, -1):
        fdr = fdr_values[i]
        if fdr < min_q_value:
            min_q_value = fdr
        q_values[i] = min_q_value
    return q_values

def get_q_values(_df, score_column, decoy_column):
    _df = _df.reset_index()
    _df = _df.sort_values([score_column,score_column], ascending=False)
    target_values = 1-_df['decoy'].values
    decoy_cumsum = np.cumsum(_df['decoy'].values)
    target_cumsum = np.cumsum(target_values)
    fdr_values = decoy_cumsum/target_cumsum
    _df['q_value'] = fdr_to_q_values(fdr_values)
    return _df

def create_common_df(library, scores, decoy_library, decoy_scores):
    df = {}
    for name, library_, scores_ in [
        ("decoy", decoy_library, decoy_scores),
        ("target", library, scores),
    ]:
        df[name] = pd.DataFrame(
            {
                "library_id": np.arange(len(library_)),
                "peptide": library_.peptide_sequences,
                "charge": library_.peptide_charges,
                "mz": library_.peptide_mzs,
                "mobility": library_.peptide_mobilities,
                "rt_min": library_.peptide_rt_apex/60,
                "rt": library_.peptide_rt_apex,
                "decoy": int(library_.decoy),
                "fragments": library_.peptide_lens,
                "length": library_.peptide_lens // 2 + 1,
                "relative_peak_intensity": scores_[:, 2],
                "push_count": scores_[:, 0].astype(np.int64),
                "fragment_count": scores_[:, 1].astype(np.int64),
                "mobility_error": library_.peptide_mobilities - dia_data.mobility_values[
                    scores_[:, 3].astype(np.int64)%dia_data.scan_max_index
                ],
                "rt_error": library_.peptide_rt_apex - dia_data.rt_values[
                    scores_[:, 3].astype(np.int64)//dia_data.scan_max_index
                ],
                "avg_fragments_per_push": scores_[:, 1] / scores_[:, 0],
                "apex_fragment_enrichment": scores_[:, 2] * scores_[:, 0] / scores_[:, 1]
            }
        )
    df = pd.concat(df.values())
    df = df[np.isfinite(df["apex_fragment_enrichment"])]
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_q_values(df, features, model):
    model.fit(df[features].values, 1-df['decoy'].values)
    df['ML_score'] = model.predict_proba(df[features].values)[:,1]
    new_df = get_q_values(df, 'ML_score', 'decoy')
    return new_df


def train_RF(
    df: pd.DataFrame,
    features: list,
    train_fdr_level:  float = 0.1,
    ini_score: str = 'relative_peak_intensity',
    min_train: int = 1000,
    test_size: float = 0.8,
    max_depth: list = [5,25,50],
    max_leaf_nodes: list = [150,200,250],
    n_jobs: int = -1,
    scoring: str = 'accuracy',
    plot: bool = False,
    random_state: int = 42,
    **kwargs
) -> (sklearn.model_selection.GridSearchCV, list):
    # Setup ML pipeline
    scaler = sklearn.preprocessing.StandardScaler()
    rfc = sklearn.ensemble.RandomForestClassifier(random_state=random_state) # class_weight={False:1,True:5},
    ## Initiate scaling + classification pipeline
    pipeline = sklearn.pipeline .Pipeline([('scaler', scaler), ('clf', rfc)])
    parameters = {
        'clf__max_depth': (max_depth),
        'clf__max_leaf_nodes': (max_leaf_nodes)
    }
    ## Setup grid search framework for parameter selection and internal cross validation
    cv = sklearn.model_selection.GridSearchCV(
        pipeline,
        param_grid=parameters,
        cv=5,
        scoring=scoring,
        verbose=0,
        return_train_score=True,
        n_jobs=n_jobs
    )
    # Prepare target and decoy df
    dfD = df[df.decoy]
    # Select high scoring targets (<= train_fdr_level)
    # df_prescore = filter_score(df)
    # df_prescore = filter_precursor(df_prescore)
    # scored = cut_fdr(df_prescore, fdr_level = train_fdr_level, plot=False)[1]
    # highT = scored[scored.decoy==False]
    # dfT_high = dfT[dfT['query_idx'].isin(highT.query_idx)]
    # dfT_high = dfT_high[dfT_high['db_idx'].isin(highT.db_idx)]
    new_df = get_q_values(df, ini_score, 'decoy')
    dfT_high = df[
        (new_df['q_value'] <= train_fdr_level) & (new_df['decoy'] == 0)
    ]

    # Determine the number of psms for semi-supervised learning
    n_train = int(dfT_high.shape[0])
    if dfD.shape[0] < n_train:
        n_train = int(dfD.shape[0])
        logging.info(
            "The total number of available decoys is lower than "
            "the initial set of high scoring targets."
        )
    if n_train < min_train:
        raise ValueError(
            "There are fewer high scoring targets or decoys than "
            "required by 'min_train'."
        )

    # Subset the targets and decoys datasets to result in a balanced dataset
    df_training = dfT_high.sample(
        n=n_train,
        random_state=random_state
    ).append(dfD.sample(n=n_train, random_state=random_state))

    # Select training and test sets
    X = df_training[features]
    y = df_training['target'].astype(int)
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = sklearn.model_selection.train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values
    )

    # Train the classifier on the training set via 5-fold cross-validation and subsequently test on the test set
    logging.info(
        'Training & cross-validation on {} targets and {} decoys'.format(
            np.sum(y_train), X_train.shape[0] - np.sum(y_train)
        )
    )
    cv.fit(X_train, y_train)

    logging.info(
        'The best parameters selected by 5-fold cross-validation were {}'.format(
            cv.best_params_
        )
    )
    logging.info(
        'The train {} was {}'.format(scoring, cv.score(X_train, y_train))
    )
    logging.info(
        'Testing on {} targets and {} decoys'.format(
            np.sum(y_test),
            X_test.shape[0] - np.sum(y_test)
        )
    )
    logging.info(
        'The test {} was {}'.format(scoring, cv.score(X_test, y_test))
    )

    feature_importances = cv.best_estimator_.named_steps['clf'].feature_importances_
    indices = np.argsort(feature_importances)[::-1][:40]

    top_features = X.columns[indices][:40]
    top_score = feature_importances[indices][:40]

    feature_dict = dict(zip(top_features, top_score))
    logging.info(f"Top features {feature_dict}")

    # Inspect feature importances
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        g = sns.barplot(
            y=X.columns[indices][:40],
            x=feature_importances[indices][:40],
            orient='h',
            palette='RdBu'
        )
        g.set_xlabel("Relative importance", fontsize=12)
        g.set_ylabel("Features", fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title("Feature importance")
        plt.show()

    return cv


@alphatims.utils.threadpool
def process_library_peptide(
    peptide_index,
    library,
    scores,
    dia_data,
    max_scan_difference=3,
    max_cycle_difference=2,
):
    push_indices = alphatims.bruker.get_dia_push_indices(
        library.precursor_frame_slices[peptide_index: peptide_index + 1],
        library.precursor_scan_slices[peptide_index: peptide_index + 1],
        library.precursor_mz_slices[peptide_index: peptide_index + 1],
        dia_data.scan_max_index,
        dia_data.dia_mz_cycle,
    )
    if len(push_indices) == 0:
        return  # Outside quad region?
    fragment_start = library.peptide_offsets[peptide_index]
    fragment_end = library.peptide_offsets[peptide_index + 1]
    (
        push_indptr,
        raw_indices,
        fragment_indices
    ) = alphatims.bruker.filter_tof_to_csr(
        library.fragment_tof_slices[fragment_start: fragment_end],
        push_indices,
        dia_data.tof_indices,
        dia_data.push_indptr,
    )
    if len(raw_indices) == 0:
        return
#     with h5py.File(f"sandbox_images/target_{peptide_index}.hdf", "w") as hdf:
#         push_indptr_ = np.zeros(128 * 128 + 1, dtype=np.int64)
#         scans = push_indices % dia_data.scan_max_index
#         cycles = push_indices // len(dia_data.dia_mz_cycle)
#         min_scan = scans[0]
#         min_cycle = cycles[0]
#         min_frame = push_indices[0] // dia_data.scan_max_index
#         inds = scans - min_scan + 128*(cycles - min_cycle)
#         push_indptr_[inds] = np.diff(push_indptr)
#         push_indptr_[1:] = np.cumsum(push_indptr_[:-1])
#         push_indptr_[0] = 0
#         push_fragment_matrix = alphatims.library.make_dense_matrix(
#             fragment_end - fragment_start,
#             push_indptr_,
#             fragment_indices,
#             raw_indices,
#         )
#         intensity_matrix = alphatims.library.get_intensity_matrix(
#             push_fragment_matrix,
#             dia_data.intensity_values,
#         )
#         hdf.create_dataset(
#             "intensities",
#             data=intensity_matrix.astype(np.uint16),
#             compression="lzf",
#             shuffle=True,
#         )
#         hdf.create_dataset(
#             "im",
#             data=dia_data.mobility_values[min_scan: min_scan + 128],
#             compression="lzf",
#             shuffle=True,
#         )
#         hdf.create_dataset(
#             "rt",
#             data=dia_data.rt_values[min_frame: min_frame + 128: dia_data.precursor_max_index],
#             compression="lzf",
#             shuffle=True,
#         )
#         hdf.create_dataset(
#             "mz",
#             data=library.peptide_fragment_mzs[fragment_start: fragment_end],
#             compression="lzf",
#             shuffle=True,
#         )
#         hdf.create_dataset(
#             "types",
#             data=library.peptide_fragment_types[fragment_start: fragment_end],
#             compression="lzf",
#             shuffle=True,
#         )
#     return
    push_fragment_matrix = alphatims.library.make_dense_matrix(
        fragment_end - fragment_start,
        push_indptr,
        fragment_indices,
        raw_indices,
    )
    intensity_matrix = alphatims.library.get_intensity_matrix(
        push_fragment_matrix,
        dia_data.intensity_values,
    )
    push_connection_indptr, push_connection_indices = alphatims.library.define_connections(
        push_indices,
        dia_data.scan_max_index,
        max_cycle_difference * dia_data.precursor_max_index,
        max_scan_difference,
    )
    smooth_intensity_matrix = alphatims.library.smoothen_and_normalize_intensity_matrix(
        intensity_matrix,
        push_connection_indptr,
        push_connection_indices
    )
    push_connection_indptr, push_connection_indices = alphatims.library.define_connections(
        push_indices,
        dia_data.scan_max_index,
        dia_data.precursor_max_index,
        1,
    )
    smoothed_push_intensities = np.sum(smooth_intensity_matrix, axis=1)
#     best_push = np.argmax(smoothed_push_intensities)
#     peak_mask = np.zeros_like(push_indices, dtype=np.bool_)
#     alphatims.library.peak_descend(
#         best_push,
#         peak_mask,
#         smoothed_push_intensities,
#         push_indices,
#         rt_lim=dia_data.precursor_max_index*1,
#         im_lim=1,
#         im_cycle=dia_data.scan_max_index,
#     )
    fwhm_pushes = np.flatnonzero(
        (smoothed_push_intensities > np.max(smoothed_push_intensities) / 2) #& peak_mask
    )
    corrs = np.corrcoef(smooth_intensity_matrix[fwhm_pushes])
    scores[peptide_index, 0] = len(push_indices)
    scores[peptide_index, 1] = len(raw_indices)
    scores[peptide_index, 2:5] = np.percentile(corrs, [25,50,75])
    scores[peptide_index, 5:8] = np.percentile(smoothed_push_intensities, [50,75, 100])
    scores[peptide_index, 8] = len(fwhm_pushes)
    scores[peptide_index, 9] = push_indices[np.argmax(smoothed_push_intensities)]
    return {
        "push_indices": push_indices,
        "fragment_start": fragment_start,
        "fragment_end": fragment_end,
        "push_indptr": push_indptr,
        "raw_indices": raw_indices,
        "fragment_indices": fragment_indices,
        "push_fragment_matrix": push_fragment_matrix,
        "intensity_matrix": intensity_matrix,
        "push_connection_indptr": push_connection_indptr,
        "push_connection_indices": push_connection_indices,
        "smooth_intensity_matrix": smooth_intensity_matrix,
        "push_connection_indptr": push_connection_indptr,
        "push_connection_indices": push_connection_indices,
        "smoothed_push_intensities": smoothed_push_intensities,
        "fwhm_pushes": fwhm_pushes,
        "corrs": corrs,
    }
