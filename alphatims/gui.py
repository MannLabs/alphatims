#!python

# external
import panel as pn
import time
# local
import alphatims.utils
import alphatims.bruker
import numpy

import pandas as pd
import matplotlib.ticker
from matplotlib import pyplot as plt
import holoviews as hv
from holoviews.operation.datashader import datashade
from holoviews import opts, dim
import hvplot.pandas
import colorcet

hv.extension('matplotlib')
datashade.cmap = colorcet.fire[50:]
opts.defaults(
    opts.Image(cmap="gray_r", axiswise=True),
    opts.Points(cmap="bwr", edgecolors='k', s=50, alpha=1.0), # Remove color_index=2
    opts.RGB(bgcolor="black", show_grid=False),
    opts.Scatter3D(color=dim('c'), fig_size=250, cmap='bwr', edgecolor='k', s=50, alpha=1.0)) #color_index=3
hv.extension('bokeh')


DATASET = None
CONTINUE_RUNNING = True


exit_button = pn.widgets.Button(name='Quit', button_type='primary')
dataset_selection = pn.widgets.FileSelector(
    file_pattern=".d"
)
# dataset_selection = pn.widgets.FileInput(
#     # file_pattern=".d"
# )
frame_slider = pn.widgets.IntRangeSlider(
    name='Frames',
    start=0,
    end=1,
    value=(0, 1),
    step=1
)
scan_slider = pn.widgets.IntRangeSlider(
    name='Scans',
    start=0,
    end=1,
    value=(0, 1),
    step=1
)
quad_slider = pn.widgets.RangeSlider(
    name='Quad',
    start=0,
    end=1,
    value=(-1, 1),
    step=1
)
tof_slider = pn.widgets.IntRangeSlider(
    name='TOF',
    start=0,
    end=1,
    value=(0, 1),
    step=1
)


def run():
    global CONTINUE_RUNNING
    layout = pn.Row(
        dataset_selection,
        pn.Column(
            frame_slider,
            scan_slider,
            quad_slider,
            tof_slider,
            exit_button,
        ),
        browser_pane,
    )
    server = layout.show(threaded=True)
    while CONTINUE_RUNNING:
        time.sleep(1)
    server.stop()


@pn.depends(
    dataset_selection.param.value,
    watch=True
)
def update_dataset(dataset_values):
    global DATASET
    if len(dataset_values) != 1:
        DATASET = None
    elif not dataset_values[0].endswith(".d"):
        DATASET = None
    else:
        DATASET = alphatims.bruker.TimsTOF(
            dataset_values[0]
        )
        frame_slider.end = DATASET.frame_max_index
        scan_slider.end = DATASET.scan_max_index
        quad_slider.end = DATASET.quad_max_index
        tof_slider.end = DATASET.tof_max_index


@pn.depends(
    frame_slider.param.value,
    scan_slider.param.value,
    quad_slider.param.value,
    tof_slider.param.value,
    watch=True
)
def browser_pane(
    frame_values,
    scan_values,
    quad_values,
    tof_values,
):
    global DATASET
    if DATASET is None:
        return None
    else:
        df = DATASET.as_dataframe(
            DATASET[
                slice(*frame_values),
                slice(*scan_values),
                slice(*quad_values),
                slice(*tof_values),
            ]
        )
        labels = {
            'mz_values': 'm/z, Th',
            "rt_values": "Retention time in minutes",
            'mobility_values': 'Inverse Ion Mobility 1/K0, V·s·cm\u207B\u00B2',
            "intensity_values": "Intensity"
        }
        x_coor = 'mz_values'
        y_coor = "mobility_values"
        z_coor = "intensity_values"
        scatter = df.hvplot.scatter(
            x=x_coor,
            y=y_coor,
            c=z_coor,
            xlabel=labels[x_coor],
            width=480,
            height=350,
            ylabel=labels[y_coor],
            tools=['hover', 'box_select'],
        #     agg='mean',
            ylim=(
                df[y_coor].min(),
                df[y_coor].max()
            ),
            xlim=(
                df[x_coor].min(),
                df[x_coor].max()
            ),
        #     rasterize=True,
        #     logz=True,
            datashade=True,
        #     dynspread=True,
            cmap=colorcet.fire,
            colorbar=True,
            clabel=z_coor,
            nonselection_color='green',
            selection_color='blue',
            color="white",
        #     size=10,
        )
        return scatter


@pn.depends(exit_button.param.clicks, watch=True)
def button_event(event):
    global CONTINUE_RUNNING
    print("Quitting server...")
    CONTINUE_RUNNING = False


# x = pn.widgets.Select(value='mpg', options=columns, name='x')
# y = pn.widgets.Select(value='hp', options=columns, name='y')
# color = pn.widgets.ColorPicker(name='Color', value='#AA0505')
#
# @pn.depends(x.param.value, y.param.value, color.param.value)
# def autompg_plot(x, y, color):
#     return autompg.hvplot.scatter(x, y, c=color)
#
# pn.Row(
#     pn.Column('## MPG Explorer', x, y, color),
#     autompg_plot)

# server.io_loop.handlers[18][0]
