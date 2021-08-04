#!python

import alphapept.fasta
import alphapept.constants
import alphatims.utils
import alphatims.bruker
import alphatims.plotting
import numpy as np
import pandas as pd
import holoviews as hv


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

    def convert_to_peptide_dict(self):
        self.peptide_dict = []
        for rt, mz, im, z, seq_, frag_int, frag_mz in zip(
            self.peptide_data.rt_apex,
            self.peptide_data.mz,
            self.peptide_data.mobility,
            self.peptide_data.charge,
            self.peptide_data.sequence,
            self.peptide_data.ion_int,
            self.peptide_data.ion_types,
        ):
            seq = alphapept.fasta.parse(seq_)
            if self.decoy:
                seq[:-1] = seq[:-1][::-1]
            peptide = {
                "sequence": seq_,
                "mz": mz,
                "mobility": im,
                "rt": rt * 60, #seconds
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
    # if decoy:
    #     seq[:-1] = seq[:-1][::-1]
    if decoy:
        #diaNN style
        original = "GAVLIFMPWSCTYHKRQEND"
        mutated = "LLLVVLLLLTSSSSLLNDQE"
        seq[1] = alphapept.fasta.parse(
            mutated[original.index(seq[1][-1])]
        )[0]
        seq[-2] = alphapept.fasta.parse(
            mutated[original.index(seq[-2][-1])]
        )[0]
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


@alphatims.utils.threadpool
def process_library_peptide(
    peptide_index,
    library,
    push_peaks,
    scores,
    dia_data,
    left_frame_borders,
    right_frame_borders,
    left_scan_borders,
    right_scan_borders,
    inflex_threshold,
):
    precursor_frame_slices = library.precursor_frame_slices
    precursor_scan_slices = library.precursor_scan_slices
    precursor_tof_slices = library.precursor_tof_slices
    precursor_mz_slices = library.precursor_mz_slices
    fragment_tof_slices = library.fragment_tof_slices
    peptide_offsets = library.peptide_offsets
    precursor_indices, fragment_indices = get_peptide_raw_indices(
        peptide_index,
        precursor_frame_slices,
        precursor_scan_slices,
        precursor_tof_slices,
        precursor_mz_slices,
        fragment_tof_slices,
        peptide_offsets,
        dia_data.frame_max_index,
        dia_data.scan_max_index,
        dia_data.push_indptr,
        dia_data.precursor_indices,
        dia_data.quad_mz_values,
        dia_data.quad_indptr,
        dia_data.tof_indices,
        dia_data.intensity_values,
        dia_data.precursor_max_index,
    )
#     TODO: Score peptides
#     TODO: Note the GIL is not yet released for code below
    fragment_coordinates = dia_data.convert_from_indices(
        fragment_indices,
        return_raw_indices=True,
        return_frame_indices=True,
        return_scan_indices=True,
        return_quad_indices=True,
        return_precursor_indices=True,
        return_tof_indices=True,
        return_rt_values=True,
        return_mobility_values=True,
        return_quad_mz_values=True,
        return_push_indices=True,
        return_mz_values=True,
        return_intensity_values=True,
        raw_indices_sorted=True,
    )
    # precursor_coordinates = dia_data.convert_from_indices(
    #     precursor_indices,
    #     return_raw_indices=True,
    #     return_frame_indices=True,
    #     return_scan_indices=True,
    #     return_quad_indices=True,
    #     return_precursor_indices=True,
    #     return_tof_indices=True,
    #     return_rt_values=True,
    #     return_mobility_values=True,
    #     return_quad_mz_values=True,
    #     return_push_indices=True,
    #     return_mz_values=True,
    #     return_intensity_values=True,
    #     raw_indices_sorted=True,
    # )
    (
        push_peak,
        score,
        hit_matrix,
        unique_push_indices,
        bpi,
        left_frame_border,
        right_frame_border,
        left_scan_border,
        right_scan_border,
    ) = get_apex(
        peptide_index,
        fragment_tof_slices,
        peptide_offsets,
        fragment_coordinates["tof_indices"],
        fragment_coordinates["push_indices"],
        fragment_coordinates["intensity_values"],
        dia_data.scan_max_index,
        inflex_threshold,
    )
    left_frame_borders[peptide_index] = left_frame_border
    right_frame_borders[peptide_index] = right_frame_border
    left_scan_borders[peptide_index] = left_scan_border
    right_scan_borders[peptide_index] = right_scan_border
    push_peaks[peptide_index] = push_peak
    scores[peptide_index] = score
    return (
        push_peak,
        score,
        hit_matrix,
        unique_push_indices,
        bpi,
        left_frame_border,
        right_frame_border,
        left_scan_border,
        right_scan_border,
        fragment_indices,
        precursor_indices,
    )


@alphatims.utils.njit(nogil=True)
def get_peptide_raw_indices(
    peptide_index,
    precursor_frame_slices,
    precursor_scan_slices,
    precursor_tof_slices,
    precursor_mz_slices,
    fragment_tof_slices,
    peptide_offsets,
    frame_max_index,
    scan_max_index,
    push_indptr,
    precursor_indices,
    quad_mz_values,
    quad_indptr,
    tof_indices,
    intensities,
    precursor_max_index,
):
    frames = precursor_frame_slices[peptide_index].copy().reshape((1,3))
    scans = precursor_scan_slices[peptide_index].copy().reshape((1,3))
    tofs = precursor_tof_slices[peptide_index].copy().reshape((1,3))
    precursor_indices_ = alphatims.bruker.filter_indices(
        frame_slices=frames,
        scan_slices=scans,
        precursor_slices=np.array([[0, 1, 1]]),
        tof_slices=tofs,
        quad_slices=np.array([[-np.inf, np.inf]]),
        intensity_slices=np.array([[-np.inf, np.inf]]),
        frame_max_index=frame_max_index,
        scan_max_index=scan_max_index,
        push_indptr=push_indptr,
        precursor_indices=precursor_indices,
        quad_mz_values=quad_mz_values,
        quad_indptr=quad_indptr,
        tof_indices=tof_indices,
        intensities=intensities,
    )
    start = peptide_offsets[peptide_index]
    end = peptide_offsets[peptide_index + 1]
    fragment_indices_ = alphatims.bruker.filter_indices(
        frame_slices=frames,
        scan_slices=scans,
        precursor_slices=np.array([[1, precursor_max_index, 1]]),
        tof_slices=fragment_tof_slices[start: end],
        quad_slices=precursor_mz_slices[peptide_index].copy().reshape((1,2)),
#         quad_slices=np.array([[-np.inf, np.inf]]),
        intensity_slices=np.array([[-np.inf, np.inf]]),
        frame_max_index=frame_max_index,
        scan_max_index=scan_max_index,
        push_indptr=push_indptr,
        precursor_indices=precursor_indices,
        quad_mz_values=quad_mz_values,
        quad_indptr=quad_indptr,
        tof_indices=tof_indices,
        intensities=intensities,
    )
    return precursor_indices_, fragment_indices_


@alphatims.utils.njit(nogil=True)
def get_apex(
    precursor_index,
    library_tof_indices,
    peptide_offsets,
    fragment_tof_indices,
    push_indices,
    intensity_values,
    scan_max_index,
    inflex_threshold,
):
    unique_push_indices = []
    unique_fragment_counts = []
    unique_fragment_count = 1
    last_push_index = push_indices[0]
    for push_index in push_indices:
        if push_index != last_push_index:
            unique_push_indices.append(last_push_index)
            unique_fragment_counts.append(unique_fragment_count)
            last_push_index = push_index
            unique_fragment_count = 1
        else:
            unique_fragment_count += 1
    unique_push_indices.append(last_push_index)
    unique_fragment_counts.append(unique_fragment_count)
    unique_push_indices = np.array(unique_push_indices)
    start = peptide_offsets[precursor_index]
    end = peptide_offsets[precursor_index + 1]
    hit_matrix = np.zeros((end - start, len(unique_push_indices)))
    last_push_index = -1
    push_offset = -1
    bpi = np.ones((end - start, 1))
    for push_index, fragment_tof_index, intensity in zip(
        push_indices,
        fragment_tof_indices,
        intensity_values,
    ):
        if push_index != last_push_index:
            library_tof_index = 0
            last_push_index = push_index
            push_offset += 1
        while fragment_tof_index not in range(
            library_tof_indices[start + library_tof_index][0],
            library_tof_indices[start + library_tof_index][1],
        ):
            library_tof_index += 1
        if bpi[library_tof_index] < intensity:
            bpi[library_tof_index] = intensity
        hit_matrix[library_tof_index, push_offset] += intensity
    peak_index = np.argmax((hit_matrix / bpi).sum(axis=0))
    push_peak = unique_push_indices[peak_index]

    scan = push_peak % scan_max_index
    scan_pushes = (unique_push_indices % scan_max_index)==scan
    scan_bpi = (
        hit_matrix[:, scan_pushes] / bpi
    ).sum(axis=0) / len(bpi)
    scan_bpi /= np.max(scan_bpi)
    left, right = get_borders(scan_bpi, inflex_threshold)
    scan_pushes = unique_push_indices[scan_pushes]
    left_frame_border = scan_pushes[left] // scan_max_index
    right_frame_border = scan_pushes[right] // scan_max_index

    frame = push_peak // scan_max_index
    frame_pushes = (unique_push_indices // scan_max_index)==frame
    frame_bpi = (
        hit_matrix[:, frame_pushes] / bpi
    ).sum(axis=0) / len(bpi)
    frame_bpi /= np.max(frame_bpi)
    # TODO start from apex and move down
    left, right = get_borders(frame_bpi, inflex_threshold)
    frame_pushes = unique_push_indices[frame_pushes]
    left_scan_border = frame_pushes[left] % scan_max_index
    right_scan_border = frame_pushes[right] % scan_max_index

    return (
        push_peak,
        np.sum(hit_matrix[:,peak_index] / bpi.ravel()),
        hit_matrix,
        unique_push_indices,
        bpi,
        left_frame_border,
        right_frame_border,
        left_scan_border,
        right_scan_border,
    )


@alphatims.utils.njit(nogil=True)
def get_borders(bpi, threshold):
    apex = np.argmax(bpi)
    lower = apex
    upper = apex
    while lower > 0:
        lower -= 1
        if bpi[lower] <= threshold:
            break
    while upper < (len(bpi) - 1):
        upper += 1
        if bpi[upper] <= threshold:
            break
    # lower = 0
    # upper = 0
    # last = 0
    # for i, current_bpi in enumerate(bpi):
    #     if current_bpi > threshold:
    #         size = i - last
    #         if size > (upper - lower):
    #             lower, upper = last, i + 1
    #     else:
    #         last = i
    return lower, upper


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
        hv.save(overlay, "tutorial_dia_xic_overlay.html")
    return overlay.opts(
        title=f"{peptide['sequence']}_{peptide['charge']}"
    )
