#!python

# external
import os
import re
import time

import numpy as np
import pandas as pd

# holoviz libraries
import panel as pn
import colorcet
import hvplot.pandas
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool

# extensions
css = '''
.bk.opt {
    position: relative;
    display: block;
    left: 75px;
    top: 0px;
    width: 80px;
    height: 80px;
}

h1 {
    color: #045082;
    font-size: 45px;
    line-height: 0.6;
    text-align: center;
}

h2 {
    color: #045082;
    text-align: center;
}

.bk.main-part {
    background-color: #EAEAEA;
    font-size: 17px;
    line-height: 23px;
    letter-spacing: 0px;
    font-weight: 500;
    color: #045082;
    text-align: center;
    position: relative !important;
    margin-left: auto;
    margin-right: auto;
    width: 40%;
}

.bk-root .bk-btn-primary {
    background-color: #045082;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.bk.alert-danger {
    background-color: #EAEAEA;
    color: #c72d3b;
    border: 0px #EAEAEA solid;
    padding: 0;
}

.settings {
    background-color: #EAEAEA;
    border: 2px solid #045082;
}

'''
pn.extension(raw_css=[css])
hv.extension('bokeh')


### LOCAL VARIABLES

DATASET = None
CONTINUE_RUNNING = True
whole_title = str()


### HEADER

header_titel = pn.pane.Markdown(
    '# AlphaTims',
    width=1250
)
mpi_biochem_logo = pn.pane.PNG(
    'img/mpi_logo.png',
    link_url='https://www.biochem.mpg.de/en',
    width=60,
    height=60,
    align='start'
)
mpi_logo = pn.pane.JPG(
    'img/max-planck-gesellschaft.jpg',
    link_url='https://www.biochem.mpg.de/en',
    height=62,
    embed=True,
    width=62,
    margin=(5, 0, 0, 5),
    css_classes=['opt']
)
github_logo = pn.pane.PNG(
    'img/github.png',
    link_url='https://github.com/MannLabs/alphatims',
    height=70,
)

header = pn.Row(
    mpi_biochem_logo,
    mpi_logo,
    header_titel,
    github_logo,
    height=73
)

### MAIN PART

project_description = pn.pane.Markdown(
    """### AlphaTIMS is an open-source Python package for fast accessing Bruker TimsTOF data. It provides a very efficient indexed data structure that allows to access four-dimensional TIMS-time of flight data in the standard numerical python (NumPy) manner. AlphaTIMS is a key enabling tool to deal with the large and high dimensional TIMS data.""",
    margin=(10, 0, 0, 0),
    css_classes=['main-part'],
    width=615
)

divider_descr = pn.pane.HTML(
    '<hr style="height: 6px; border:none; background-color: #045082; width: 1010px">',
    width=1510,
    align='center'
)

upload_file = pn.widgets.TextInput(
    name='Specify an experimental file:',
    placeholder='Enter the whole path to Bruker .d folder or .hdf file',
    width=800,
    margin=(15,15,0,15)
)

upload_button = pn.widgets.Button(
    name='Upload  Data',
    button_type='primary',
    height=31,
    width=100,
    margin=(34,20,0,20)
)

upload_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(30,20,0,15),
    width=40,
    height=40
)

upload_error = pn.pane.Alert(
    width=400,
    alert_type="danger",
    margin=(-15,0,10,260),
)

main_part = pn.Column(
    project_description,
    divider_descr,
    pn.Row(
        upload_file,
        upload_button,
        upload_spinner,
        align='center'
    ),
    upload_error,
    background='#eaeaea',
    width=1510,
    height=350,
    margin=(5, 0, 10, 0)
)


### SETTINGS

settings_title = pn.pane.Markdown(
    '## Parameters',
    align='center',
    margin=(10,0,0,0)
)
settings_divider = pn.pane.HTML(
    '<hr style="height: 3.5px; border:none; background-color: #045082; width: 420px;">',
    align='center',
    margin=(0, 10, 0, 20)
)
frame_slider = pn.widgets.IntRangeSlider(
    name='Frames',
    start=1,
    step=1,
    margin=(0, 20)
)
rt_slider = pn.widgets.RangeSlider(
    name='Retention Time',
    start=0,
    step=0.1,
    margin=(0, 20)
)

scan_slider = pn.widgets.IntRangeSlider(
    name='Scans',
    start=1,
    step=1,
    margin=(5, 20)
)

quad_slider = pn.widgets.RangeSlider(
    name='Quad',
    start=-1,
    value=(-1, -1),
    step=1,
    margin=(5, 20)
)

tof_slider = pn.widgets.IntRangeSlider(
    name='TOF',
    start=1,
    step=1,
    margin=(5, 20)
)

exit_button = pn.widgets.Button(
    name='Quit',
    button_type='primary'
)

settings = pn.layout.WidgetBox(
    settings_title,
    settings_divider,
    frame_slider,
    rt_slider,
    scan_slider,
    quad_slider,
    tof_slider,
    exit_button,
    width=460,
    align='start',
    margin=(0,0,0,0),
    css_classes=['settings']
)


### plotting options

hover = HoverTool(tooltips=[('RT, min', '@RT'), ('Intensity', '@SummedIntensities')], mode='vline')
chrom_opts = opts.Curve(
    width=1000,
    height=300,
    xlabel='RT, min',
    ylabel='Intensity',
    line_width=1,
    yformatter='%.1e',
    shared_axes=True,
    tools=[hover]
)


### FUNCTIONS
# preload data
@pn.depends(
    upload_button.param.clicks,
    watch=True
)
def upload_data(_):
    import alphatims.bruker
    global DATASET
    global whole_title
    if upload_file.value.endswith(".d") or upload_file.value.endswith(".hdf"):
        upload_error.object = None
        if not DATASET or DATASET.bruker_d_folder_name != upload_file.value:
            upload_spinner.value = True
            DATASET = alphatims.bruker.TimsTOF(
                upload_file.value
            )
            mode = ''
            if 'DDA' in upload_file.value:
                mode = 'dda-'
            elif 'DIA' in upload_file.value:
                mode = 'dia-'
            whole_title = str(DATASET.meta_data['SampleName']) + '.d: ' + mode + str(DATASET.acquisition_mode)
#             upload_spinner.value = False
    else:
        upload_error.object = '#### Please, specify a path to .d Bruker folder or .hdf file.'

### PLOTTING
def visualize_chrom():
    data = DATASET.frames.query('MsMsType == 0')[['Time', 'SummedIntensities']]
    data['RT'] = data['Time'] / 60
    chrom = hv.Curve(
        data=data,
        kdims=['RT', 'SummedIntensities'],
    ).opts(
        chrom_opts,
        opts.Curve(
            title="Chromatogram - " + whole_title
        )
    )
    return chrom

def visualize_scatter(
    frame_values,
    scan_values,
    quad_values,
    tof_values
):
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
        ylabel=labels[y_coor],
        ylim=(
            df[y_coor].min(),
            df[y_coor].max()
        ),
        xlim=(
            df[x_coor].min(),
            df[x_coor].max()
        ),
        title='Heatmap - ' + whole_title,
        tools=['hover', 'box_select', 'tap'],
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

### SHOW SETTINGS/PLOTS

@pn.depends(
    upload_button.param.clicks
)
def show_settings(_):
    if DATASET:
        frame_slider.end = int(DATASET.frame_max_index + 1)
        rt_slider.end = round(max(DATASET.rt_values) / 60, 3)
        rt_slider.value = (rt_slider.start, rt_slider.end)
        scan_slider.end = int(DATASET.scan_max_index + 1)
        scan_slider.value = (scan_slider.start, scan_slider.end)
        quad_slider.end = round(max(DATASET.quad_high_values), 3)
        tof_slider.end = int(DATASET.tof_max_index + 1)
        tof_slider.value = (tof_slider.start, tof_slider.end)
        return settings
    else:
        return None


@pn.depends(
    frame_slider.param.value,
    rt_slider.param.value,
    scan_slider.param.value,
    quad_slider.param.value,
    tof_slider.param.value,
)
def show_plots(
    frame_values,
    rt_slider,
    scan_values,
    quad_values,
    tof_values,
):
    if DATASET:
        layout_plots = pn.Column(
            visualize_scatter(
                frame_values,
                scan_values,
                quad_values,
                tof_values,
            ),
            visualize_chrom()
        )
        upload_spinner.value = False
        return layout_plots


@pn.depends(
    exit_button.param.clicks,
    watch=True
)
def button_event(_):
    import logging
    global CONTINUE_RUNNING
    logging.info("Quitting server...")
    CONTINUE_RUNNING = False

if __name__ == "__main__":
    def run():
        global CONTINUE_RUNNING
        layout = pn.Column(
            header,
            main_part,
            pn.Row(
                show_settings,
                show_plots,
            ),
        )
        server = layout.show(threaded=True)
        while CONTINUE_RUNNING:
            time.sleep(1)
        server.stop()
