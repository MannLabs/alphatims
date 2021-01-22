#!python
"""This module provides basic LC-TIMS-MSMS plots."""

# external
import numpy as np
import colorcet
import hvplot.pandas
import holoviews as hv
from holoviews import opts
import bokeh.models


def line_plot(
    timstof_data,
    selected_indices,
    x_axis_label: str,
    title: str = "",
    y_axis_label: str = "Intensity",
    remove_zeros: bool = False,
    width: int = 1000,
    height: int = 300,
):
    """Plot an XIC, mobilogram or spectrum as a lineplot.

    Parameters
    ----------
    timstof_data : aphatims.bruker.TimsTOF
        An aphatims.bruker.TimsTOF data object.
    selected_indices : np.int64[:]
        The raw indices that are selected for this plot
    x_axis_label : str
        A label that is used for projection
        (i.e. intensities are summed) on the x-axis. Options are:

            - "m/z, Th"
            - "RT, min"
            - "Inversed IM, V·s·cm\u207B\u00B2"
    title : str
        The title of this plot.
        Will be prepended with "Spectrum", "Mobilogram" or "XIC".
        Default is "".
    y_axis_label : str
        Should not be set for a 1D line plot.
        Default is "Intensity".
    remove_zeros : bool
        If True, zeros are removed.
        Note that a line plot connects consecutive points,
        which can lead to misleading plots if non-zeros are removed.
        If False, use the full range of the appropriate dimension of
        the timstof_data.
        Default is False.
    width : int
        The width of this plot.
        Default is 1000.
    height : int
        The height of this plot.
        Default is 300.

    Returns
    -------
    : hv.Curve
        A curve plot that represents an XIC, mobilogram or spectrum.
    """
    labels = {
        'm/z, Th': "mz_values",
        'RT, min': "rt_values",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
    }
    x_dimension = labels[x_axis_label]
    intensities = timstof_data.bin_intensities(selected_indices, [x_dimension])
    plot_opts = {
        "width": width,
        "height": height,
        "align": 'center',
        "tools": ['hover'],
        "line_width": 1,
        "yformatter": '%.1e',
        "align": 'center',
    }
    if x_dimension == "mz_values":
        x_ticks = timstof_data.mz_values
        plot_opts["title"] = f"Spectrum - {title}"
    elif x_dimension == "mobility_values":
        x_ticks = timstof_data.mobility_values
        plot_opts["title"] = f"Mobilogram - {title}"
    elif x_dimension == "rt_values":
        x_ticks = timstof_data.rt_values / 60
        plot_opts["title"] = f"XIC - {title}"
    if remove_zeros:
        non_zeros = np.flatnonzero(intensities)
        x_ticks = x_ticks[non_zeros]
        intensities = x_ticks[intensities]
    plot = hv.Curve(
        (x_ticks, intensities),
        x_axis_label,
        y_axis_label,
    )
    plot.opts(**plot_opts)
    return plot


def heatmap(
    df,
    x_axis_label: str,
    y_axis_label: str,
    title: str = "",
    z_axis_label: str = "Intensity",
    width: int = 1000,
    height: int = 300,
):
    """Create a scatterplot / heatmap for a dataframe.

    The coordinates of the dataframe are projected
    (i.e. their intensities are summed) on the requested axes.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe wirth coordinates.
        This should be obtained by slicing an alphatims.bruker.TimsTOF object.
    x_axis_label : str
        A label that is used for projection
        (i.e. intensities are summed) on the x-axis. Options are:

            - "m/z, Th"
            - "RT, min"
            - "Inversed IM, V·s·cm\u207B\u00B2"
    y_axis_label : str
        A label that is used for projection
        (i.e. intensities are summed) on the x-axis. Options are:

            - "m/z, Th"
            - "RT, min"
            - "Inversed IM, V·s·cm\u207B\u00B2"
    title : str
        The title of this plot.
        Will be prepended with "Heatmap".
        Default is "".
    z_axis_label : str
        Should not be set for a 2D scatterplot / heatmap.
        Default is "Intensity".
    width : int
        The width of this plot.
        Default is 1000.
    height : int
        The height of this plot.
        Default is 300.

    Returns
    -------
    hv.Scatter
        A scatter plot projected on the 2 dimensions.
    """
    labels = {
        'm/z, Th': "mz_values",
        'RT, min': "rt_values",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
        'Intensity': "intensity_values",
    }
    x_dimension = labels[x_axis_label]
    y_dimension = labels[y_axis_label]
    z_dimension = labels[z_axis_label]
    # hover = bokeh.models.HoverTool(
    #     tooltips=[
    #         (f'{x_axis_label}', f'@{x_dimension}'),
    #         (f'{y_axis_label}', f'@{y_dimension}'),
    #         (f'{z_axis_label}', f'@{z_dimension}'),
    #     ]
    # )
    df["rt_values"] /= 60
    scatter = df.hvplot.scatter(
        x=x_dimension,
        y=y_dimension,
        c=z_dimension,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        clabel=z_axis_label,
        # ylim=(
        #     df[y_dimension].min(),
        #     df[y_dimension].max()
        # ),
        # xlim=(
        #     df[x_dimension].min(),
        #     df[x_dimension].max()
        # ),
        title=f'Heatmap - {title}',
        # tools=[hover],
        datashade=True,
        dynspread=True,
        cmap=colorcet.fire,
        nonselection_color='green',
        selection_color='blue',
        color="white",
        width=width,
        height=height,
    )
    return scatter


def tic_plot(
    timstof_data,
    title: str = "",
    width: int = 1000,
    height: int = 310,
):
    """Create a total ion chromatogram (TIC) for the data.

    Parameters
    ----------
    timstof_data : aphatims.bruker.TimsTOF
        An aphatims.bruker.TimsTOF data object.
    title : str
        The title of this plot.
        Will be prepended with "TIC".
        Default is False
    width : int
        The width of this plot.
        Default is 1000.
    height : int
        The height of this plot.
        Default is 310.

    Returns
    -------
    hv.Curve
        The TIC of the provided dataset.
    """
    hover = bokeh.models.HoverTool(
        tooltips=[('RT, min', '@RT'), ('Intensity', '@SummedIntensities')],
        mode='vline'
    )
    tic_opts = opts.Curve(
        width=width,
        height=height,
        xlabel='RT, min',
        ylabel='Intensity',
        line_width=1,
        yformatter='%.1e',
        shared_axes=True,
        tools=[hover]
    )
    data = timstof_data.frames.query('MsMsType == 0')[[
        'Time', 'SummedIntensities']
    ]
    data['RT'] = data['Time'] / 60
    tic = hv.Curve(
        data=data,
        kdims=['RT'],
        vdims=['SummedIntensities']
    ).opts(
        tic_opts,
        opts.Curve(
            title="TIC - " + title
        )
    )
    return tic
