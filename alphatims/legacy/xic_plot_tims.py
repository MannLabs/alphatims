"""Legacy module for plotting XIC on TimsTOF data. Moved from alpharaw."""
import typing
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from alpharaw.viz.plot_utils import plot_line
from alphatims.bruker import (
    TimsTOF,
)
from plotly.subplots import make_subplots

warnings.warn(
    "This module will be deprecated or changed in the future releases",
    category=DeprecationWarning,
)

alphatims_labels = {
    "rt": "rt_values",
    "im": "mobility_values",
}

class XIC_Plot_Tims:
    # hovermode = "x" | "y" | "closest" | False | "x unified" | "y unified"
    hovermode = "closest"
    plot_height = 550
    colorscale_qualitative = "Alphabet"
    colorscale_sequential = "Viridis"
    theme_template = "plotly_white"
    ppm = 20.0
    rt_sec_win = 30.0
    plot_rt_unit: str = "minute"
    im_win = 0.05
    fig: go.Figure = None

    def plot(
        self,
        tims_data: TimsTOF,
        query_df: pd.DataFrame,
        view_dim: typing.Literal["rt", "im"] = "rt",
        title: str = "",
        add_peak_area=True,
    ):
        rt_sec = query_df["rt_sec"].values[0]
        im = 0.0 if "im" not in query_df.columns else query_df["im"].values[0]
        if "precursor_mz" not in query_df.columns:
            precursor_mz = 0.0
        else:
            precursor_mz = query_df.precursor_mz.values[0]
        query_masses = query_df.mz.values
        if "intensity" in query_df.columns:
            query_intensities = query_df.intensity.values
        else:
            query_intensities = None
        ion_names = query_df.ion_name.values

        if "color" not in query_df.columns:
            marker_colors = None
        else:
            marker_colors = query_df.color.values

        return self.plot_query_masses(
            tims_data=tims_data,
            query_masses=query_masses,
            query_ion_names=ion_names,
            query_rt_sec=rt_sec,
            query_im=im,
            precursor_mz=precursor_mz,
            marker_colors=marker_colors,
            view_dim=view_dim,
            query_intensities=query_intensities,
            title=title,
            add_peak_area=add_peak_area,
        )

    def _init_plot(self, view_dim="rt"):
        self.fig = make_subplots(
            cols=1,
            shared_xaxes=True,
            x_title=f"RT ({self.plot_rt_unit})" if view_dim == "rt" else "Mobility",
            y_title="intensity",
            vertical_spacing=0.2,
        )
        self.trace: XIC_Plot_Tims = XIC_Trace_Tims(fig=self.fig, row=1)

    def plot_query_masses(
        self,
        tims_data: TimsTOF,
        query_masses: np.ndarray,
        query_ion_names: typing.List[str],
        query_rt_sec: float,
        query_im: float,
        precursor_mz: float,
        marker_colors: list = None,
        view_dim: typing.Literal["rt", "im"] = "rt",
        query_intensities: np.ndarray = None,
        title="",
        add_peak_area=True,
    ):
        self._init_plot(view_dim=view_dim)
        mass_tols = query_masses * 1e-6 * self.ppm
        if marker_colors is None:
            marker_colors = self._get_color_set(len(query_masses))
        self.trace.add_traces(
            tims_data=tims_data,
            query_masses=query_masses,
            mass_tols=mass_tols,
            ion_names=query_ion_names,
            marker_colors=marker_colors,
            query_rt_sec=query_rt_sec,
            query_im=query_im,
            precursor_left_mz=precursor_mz * (1 - 1e-6 * self.ppm),
            precursor_right_mz=precursor_mz * (1 + 1e-6 * self.ppm),
            view_dim=view_dim,
            rt_sec_win=self.rt_sec_win,
            im_win=self.im_win,
            query_intensities=query_intensities,
            add_peak_area=add_peak_area,
        )
        self.fig.update_layout(
            template=self.theme_template,
            title=dict(text=title, yanchor="bottom"),
            # width=width,
            height=self.plot_height,
            hovermode=self.hovermode,
            showlegend=True,
        )
        return self.fig

    def _get_color_set(self, n_query):
        if n_query <= len(getattr(px.colors.qualitative, self.colorscale_qualitative)):
            color_set = getattr(px.colors.qualitative, self.colorscale_qualitative)
        else:
            color_set = px.colors.sample_colorscale(
                self.colorscale_sequential, samplepoints=n_query
            )
        return color_set


class XIC_Trace_Tims:
    label_format = "{ion_name} {mz:.3f}"
    legend_group = "{ion_name}"  # {ion_name}, {mz} or None
    fig: go.Figure
    row: int = 1
    col: int = 1
    plot_rt_unit: str = "minute"

    def __init__(
        self,
        fig: go.Figure,
        row: int = 1,
        col: int = 1,
        plot_rt_unit: str = "minute",
    ):
        self.fig = fig
        self.row = row
        self.col = col
        self.plot_rt_unit = plot_rt_unit

    def add_traces(
        self,
        tims_data: TimsTOF,
        query_masses: np.ndarray,
        mass_tols: np.ndarray,
        ion_names: typing.List[str],
        marker_colors: typing.List,
        query_rt_sec: float,
        query_im: float,
        precursor_left_mz: float = -1.0,
        precursor_right_mz: float = -1.0,
        view_dim: typing.Literal["rt", "im"] = "rt",
        rt_sec_win=30.0,
        im_win=0.05,
        query_intensities: np.ndarray = None,
        add_peak_area=True,
    ) -> go.Figure:
        """Add traces for the query_masses.

        Args:
            tims_data (TimsTOF): AlphaTims TimsTOF object.
            query_masses (np.ndarray): Query masses.
            ion_names (typing.List[str]): Ion names for query_masses.
            marker_colors (typing.List): Colors for each query mass.
            query_rt_sec (float): Query RT in seconds.
            query_im (float): Query ion mobility.
            precursor_mz (float, optional): Precursor mz, 0 means it is MS1. Defaults to 0.0.
            query_intensities (np.ndarray, optional): Query intensities. Defaults to None.

        Returns:
            go.Figure: self.fig.
        """
        (rt_slice, im_slice, prec_mz_slice, view_indices) = get_plotting_slices(
            tims_data=tims_data,
            rt_sec=query_rt_sec,
            rt_sec_win=rt_sec_win,
            im=query_im,
            im_win=im_win,
            precursor_left_mz=precursor_left_mz,
            precursor_right_mz=precursor_right_mz,
            view_dim=view_dim,
        )

        if query_intensities is None:
            query_intensities = np.zeros_like(query_masses)
        else:
            query_intensities /= query_intensities.max()

        for ion_name, query_mass, query_inten, marker_color, mass_tol in zip(
            ion_names, query_masses, query_intensities, marker_colors, mass_tols
        ):
            self._add_one_trace(
                tims_data=tims_data,
                query_mass=query_mass,
                mass_tol=mass_tol,
                rt_slice=rt_slice,
                im_slice=im_slice,
                prec_mz_slice=prec_mz_slice,
                view_indices=view_indices,
                view_dim=view_dim,
                label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                legend_group=self.legend_group.format(ion_name=ion_name),
                marker_color=marker_color,
                add_peak_area=add_peak_area,
            )
            if query_inten > 0:
                self._add_one_trace(
                    tims_data=tims_data,
                    query_mass=query_mass,
                    mass_tol=mass_tol,
                    rt_slice=rt_slice,
                    im_slice=im_slice,
                    prec_mz_slice=prec_mz_slice,
                    view_indices=view_indices,
                    view_dim=view_dim,
                    label=self.label_format.format(ion_name=ion_name, mz=query_mass),
                    legend_group=self.legend_group.format(ion_name=ion_name),
                    marker_color=marker_color,
                    intensity_scale=-query_inten,
                    add_peak_area=add_peak_area,
                )

    def _add_one_trace(
        self,
        tims_data: TimsTOF,
        query_mass: float,
        mass_tol: float,
        rt_slice: slice,
        im_slice: slice,
        prec_mz_slice: slice,
        view_indices: np.ndarray,
        view_dim: str,
        label: str,
        legend_group: str,
        marker_color: str,
        intensity_scale: float = 1.0,
        add_peak_area=True,
    ):
        frag_indices = tims_data[
            rt_slice,
            im_slice,
            prec_mz_slice,
            slice(
                query_mass - mass_tol,
                query_mass + mass_tol,
            ),
            "raw",
        ]
        if len(frag_indices) == 0:
            return
        self.fig.add_trace(
            plot_line_tims_fast(
                tims_data,
                frag_indices,
                view_indices,
                name=label,
                legend_group=legend_group,
                marker_color=marker_color,
                view_dim=view_dim,
                intensity_scale=intensity_scale,
                add_peak_area=add_peak_area,
            ),
            row=self.row,
            col=self.col,
        )


def get_plotting_slices(
    tims_data: TimsTOF,
    rt_sec: float,
    rt_sec_win: float = 30.0,
    im: float = 0.0,
    im_win: float = 0.05,
    precursor_left_mz: float = -1.0,
    precursor_right_mz: float = -1.0,
    view_dim: str = "rt",
):
    """
    Get plotting slices for target queries in TimsTOF data.
    Args:
        tims_data (TimsTOF): AlphaTims TimsTOF object.
        rt_sec (float): Query RT in seconds.
        rt_sec_win (float, optional): Query RT window in seconds. Defaults to 30.0.
        im (float, optional): Query ion mobility, 0 means no mobility dimension. Defaults to 0..
        im_win (float, optional): Ion mobility window. Defaults to 0.05.
        precursor_mz (float, optional): Precursor mz, 0 means it is MS1. Defaults to 0.0.
        ppm (float, optional): PPM tolerance for `precursor_mz`. Defaults to 20.0.
        view_dim (str, optional): View dimension, "rt" or "im". Defaults to "rt"
    """
    rt_slice = slice(rt_sec - rt_sec_win / 2, rt_sec + rt_sec_win / 2)
    im_slice = slice(im - im_win / 2, im + im_win / 2)

    if precursor_left_mz <= 0:
        prec_mz_slice = 0
        raw_indices = tims_data[rt_slice, im_slice, 0, :, "raw"]
    else:
        prec_mz_slice = slice(
            precursor_left_mz,
            precursor_right_mz,
        )
        raw_indices = tims_data[rt_slice, im_slice, prec_mz_slice, :, "raw"]

    if view_dim == "rt":
        view_indices = np.sort(
            np.array(
                list(
                    set(
                        tims_data.convert_from_indices(
                            raw_indices, return_frame_indices=True
                        )["frame_indices"]
                    )
                ),
                dtype=np.int64,
            )
        )
    else:
        view_indices = np.sort(
            np.array(
                list(
                    set(
                        tims_data.convert_from_indices(
                            raw_indices, return_scan_indices=True
                        )["scan_indices"]
                    )
                ),
                dtype=np.int64,
            )
        )

    return rt_slice, im_slice, prec_mz_slice, view_indices



def plot_line_tims(
    tims_data: TimsTOF,
    tims_raw_indices: np.ndarray,
    tims_view_indices: np.array,
    name: str,
    legend_group: str,
    marker_color: str,
    view_dim: typing.Literal["rt", "im"] = "rt",  # or 'im'
    intensity_scale: float = 1.0,
    rt_unit: str = "minute",
) -> go.Figure:
    """
    Plot an XIC line on alphatims `TimsTOF` data

    Parameters
    ----------
    tims_data : TimsTOF
        The alphatims `TimsTOF` object
    tims_raw_indices : np.ndarray
        Raw indices on `TimsTOF` object
    tims_view_indices : np.array
        View indices on `TimsTOF` object
    name : str
        Display name
    legend_group : str
        Lines will be grouped by `legend_group`
    marker_color : str
        Color of the scatter (x, y)
    view_dim : "rt", "im", optional
        View dimension, "rt" or "im", by default "rt"
    rt_unit : str, optional
        RT unit, by default "minute"

    Returns
    -------
    go.Figure
        Line plot
    """
    x_dimension = alphatims_labels[view_dim]

    intensities = tims_data.bin_intensities(tims_raw_indices, [x_dimension])
    if view_dim == "rt":
        x_ticks = tims_data.rt_values[tims_view_indices]
        intensities = intensities[tims_view_indices]
        if rt_unit == "minute":
            x_ticks /= 60.0
    else:
        x_ticks = tims_data.mobility_values[tims_view_indices]
        intensities = intensities[tims_view_indices]

    return plot_line(
        x_ticks,
        intensities * intensity_scale,
        name=name,
        marker_color=marker_color,
        legend_group=legend_group,
        x_text=view_dim.upper(),
    )


def plot_line_tims_fast(
    tims_data: TimsTOF,
    tims_raw_indices: np.ndarray,
    tims_view_indices: np.array,
    name: str,
    legend_group: str,
    marker_color: str,
    view_dim: typing.Literal["rt", "im"] = "rt",
    intensity_scale: float = 1.0,
    rt_unit: str = "minute",
    add_peak_area=True,
) -> go.Figure:
    """
    Plot an XIC line on alphatims `TimsTOF` data

    Parameters
    ----------
    tims_data : TimsTOF
        The alphatims `TimsTOF` object
    tims_raw_indices : np.ndarray
        Raw indices on `TimsTOF` object
    tims_view_indices : np.array
        View indices on `TimsTOF` object
    name : str
        Display name
    legend_group : str
        Lines will be grouped by `legend_group`
    marker_color : str
        Color of the scatter (x, y)
    view_dim : "rt", "im", optional
        View dimension, "rt" or "im", by default "rt"
    intensity_scale : float, optional
        Intensity scale of mirror plot, by default 1.0
    rt_unit : str, optional
        RT unit, by default "minute"
    add_peak_area : bool, optional
        If add peak area in the hover text, by default True

    Returns
    -------
    go.Figure
        _description_
    """
    x_dimension = alphatims_labels[view_dim]

    intensities = tims_data.bin_intensities(tims_raw_indices, [x_dimension])
    if view_dim == "rt":
        x_ticks = tims_data.rt_values[tims_view_indices]
        intensities = intensities[tims_view_indices]
        if rt_unit == "minute":
            x_ticks /= 60.0
    else:
        x_ticks = tims_data.mobility_values[tims_view_indices]
        intensities = intensities[tims_view_indices]

    if add_peak_area:
        peak_area = abs(np.trapz(y=intensities, x=x_ticks))

    return plot_line(
        x_ticks,
        intensities * intensity_scale,
        name=name,
        marker_color=marker_color,
        legend_group=legend_group,
        x_text=view_dim.upper(),
        other_info=f"Peak area: {peak_area:.2e}" if add_peak_area else "",
    )

