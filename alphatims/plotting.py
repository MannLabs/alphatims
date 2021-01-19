#!python

# holoviz libraries
import colorcet
import hvplot.pandas
import holoviews as hv
from holoviews import opts


def plot_1d(data, df, selected_indices, x_axis_label, y_axis_label, title):
    # df = DATAFRAME
    # selected_indices = SELECTED_INDICES
    # x_axis_label = plot2_x_axis_label.value
    # y_axis_label =plot2_y_axis_label.value
    # title = title
    labels = {
        'm/z, Th': "mz_values",
        'RT, min': "rt_values",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
        'Intensity': "intensity_values",
    }
    x_coor = labels[x_axis_label]
    if x_coor == 'mz_values':
        mz_intensities = data.bin_intensities(selected_indices, [x_coor])
        spectrum = hv.Spikes(
            (
                sorted(df[x_coor].unique()),
                mz_intensities[mz_intensities > 0]
            ),
            x_axis_label,
            y_axis_label,
        )
        spectrum.opts(
            title='Spectrum - ' + title,
            tools=['hover'],
            color='Intensity',
            cmap=colorcet.kb,
            width=1000,
            height=300,
            align='center',
        )
    elif x_coor == 'mobility_values':
        im_intensities = data.bin_intensities(selected_indices, [x_coor])
        spectrum = hv.Curve(
            (
                sorted(df[x_coor].unique(), reverse=True),
                im_intensities[im_intensities > 0]
            ),
            x_axis_label,
            y_axis_label,
        )
        spectrum.opts(
            title="Mobilogram - " + title,
            width=1000,
            height=300,
            align='center',
            line_width=1,
            yformatter='%.1e',
            tools=['hover']
        )
    elif x_coor == 'rt_values':
        rt_intensities = data.bin_intensities(selected_indices, [x_coor])
        spectrum = hv.Curve(
            (
                sorted(df[x_coor].unique()/60),
                rt_intensities[rt_intensities > 0]
            ),
            x_axis_label,
            y_axis_label,
        )
        spectrum.opts(
            title="XIC - " + title,
            width=1000,
            height=300,
            color='darkred',
            align='center',
            line_width=1,
            yformatter='%.1e',
            tools=['hover']
        )
    return spectrum


def plot_2d(
    df,
    x_axis,
    y_axis,
    title,
):
    # df = DATAFRAME
    # x_axis = plot1_x_axis.value
    # y_axis = plot1_y_axis.value
    # title = WHOLE_TITLE
    labels = {
        'mz_values': 'm/z, Th',
        'rt_values': 'RT, min',
        'mobility_values': 'Inversed IM, V·s·cm\u207B\u00B2',
        'intensity_values': 'Intensity'
    }
    x_coor = [k for k, v in labels.items() if v == x_axis][0]
    y_coor = [k for k, v in labels.items() if v == y_axis][0]
    z_coor = "intensity_values"
    scatter = df.hvplot.scatter(
        x=x_coor,
        y=y_coor,
        c=z_coor,
        xlabel=labels[x_coor],
        ylabel=labels[y_coor],
        ylim=(
            df[y_coor].min(),
            df[y_coor].max()
        ),
        xlim=(
            df[x_coor].min(),
            df[x_coor].max()
        ),
        title='Heatmap - ' + title,
        tools=['hover'],
        datashade=True,
        dynspread=True,
        cmap=colorcet.fire,
        clabel=z_coor,
        nonselection_color='green',
        selection_color='blue',
        color="white",
        width=1000,
        height=300,
    )
    return scatter
