#!python

# external
import os
import numpy as np
import pandas as pd
import panel as pn
import sys
# holoviz libraries
import colorcet
import hvplot.pandas
import holoviews as hv
from holoviews import opts
# local
import alphatims.bruker
import alphatims.utils
import alphatims.plotting


# EXTENSIONS
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

.bk-root .bk-btn-default {
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

.bk.alert-success {
    background-color: #EAEAEA;
    border: 0px #EAEAEA solid;
    padding: 0;
}

.settings {
    background-color: #EAEAEA;
    border: 2px solid #045082;
}

.bk.card-title {
    font-size: 13px;
    font-weight: initial;
}

'''
pn.extension(raw_css=[css])
hv.extension('bokeh')


# LOCAL VARIABLES
DATASET = None
SERVER = None
PLOTS = None
WHOLE_TITLE = str()
SELECTED_INDICES = np.array([])
DATAFRAME = pd.DataFrame()
STACK = alphatims.utils.Global_Stack({})
GLOBAL_INIT_LOCK = True


# PATHS
biochem_logo_path = os.path.join(alphatims.utils.IMG_PATH, "mpi_logo.png")
mpi_logo_path = os.path.join(
    alphatims.utils.IMG_PATH,
    "max-planck-gesellschaft.jpg"
)
github_logo_path = os.path.join(alphatims.utils.IMG_PATH, "github.png")


# HEADER
header_titel = pn.pane.Markdown(
    '# AlphaTims',
    width=1250
)
mpi_biochem_logo = pn.pane.PNG(
    biochem_logo_path,
    link_url='https://www.biochem.mpg.de/en',
    width=60,
    height=60,
    align='start'
)
mpi_logo = pn.pane.JPG(
    mpi_logo_path,
    link_url='https://www.biochem.mpg.de/en',
    height=62,
    embed=True,
    width=62,
    margin=(5, 0, 0, 5),
    css_classes=['opt']
)
github_logo = pn.pane.PNG(
    github_logo_path,
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


# MAIN PART
project_description = pn.pane.Markdown(
    """### AlphaTIMS is an open-source Python package for fast accessing Bruker TimsTOF data. It provides a very efficient indexed data structure that allows to access four-dimensional TIMS-time of flight data in the standard numerical python (NumPy) manner. AlphaTIMS is a key enabling tool to deal with the large and high dimensional TIMS data.""",
    margin=(10, 0, 0, 0),
    css_classes=['main-part'],
    width=615
)

divider_descr = pn.pane.HTML(
    '<hr style="height: 6px; border:none; background-color: #045082; width: 1140px">',
    width=1510,
    align='center'
)
upload_file = pn.widgets.TextInput(
    name='Specify an experimental file:',
    placeholder='Enter the whole path to Bruker .d folder or .hdf file',
    width=800,
    margin=(15, 15, 0, 15)
)
upload_button = pn.widgets.Button(
    name='Upload Data',
    button_type='primary',
    height=31,
    width=100,
    margin=(34, 20, 0, 20)
)
upload_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(30, 15, 0, 15),
    width=40,
    height=40
)
upload_error = pn.pane.Alert(
    width=400,
    alert_type="danger",
    margin=(-15, 0, 10, 200),
)
exit_button = pn.widgets.Button(
    name='Quit',
    button_type='primary',
    height=31,
    width=100,
    margin=(34, 20, 0, 0)
)
main_part = pn.Column(
    project_description,
    divider_descr,
    pn.Row(
        upload_file,
        upload_button,
        upload_spinner,
        exit_button,
        align='center'
    ),
    upload_error,
    background='#eaeaea',
    width=1510,
    height=360,
    margin=(5, 0, 10, 0)
)


# SETTINGS
settings_title = pn.pane.Markdown(
    '## Parameters',
    align='center',
    margin=(10, 0, 0, 0)
)
settings_divider = pn.pane.HTML(
    '<hr style="height: 3.5px; border:none; background-color: #045082; width: 440px;">',
    align='center',
    margin=(0, 10, 0, 10)
)
sliders_divider = pn.pane.HTML(
    '<hr style="height: 1.5px; border:none; background-color: black; width: 425px;">',
    align='center',
    margin=(10, 10, 0, 10)
)
selectors_divider = pn.pane.HTML(
    '<hr style="border-left: 2px solid black; height: 50px;">',
    align='center',
    margin=(-5, 0, 0, 40)
)
card_divider = pn.pane.HTML(
    '<hr style="height: 1.5px; border:none; background-color: black; width: 415px;">',
    align='center',
    margin=(3, 10, 3, 10)
)


# SAVE TO HDF
save_hdf_path = pn.widgets.TextInput(
    name='Specify a path to save .hdf file:',
    placeholder='e.g. D:\Bruker',
    width=240,
    margin=(5, 0, 0, 28)
)
save_hdf_button = pn.widgets.Button(
    name='Save to HDF',
    button_type='default',
    height=31,
    width=100,
    margin=(23, 10, 0, 15)
)
save_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(23, 15, 0, 15),
    width=30,
    height=30
)
save_message = pn.pane.Alert(
    alert_type='success',
    margin=(-10, 15, 20, 30),
    width=300
)


# frame/RT selection
frame_slider = pn.widgets.IntRangeSlider(
    show_value=False,
    bar_color='#045082',
    start=1,
    step=1,
    margin=(10, 20, 10, 20)
)
frame_start = pn.widgets.IntInput(
    name='Start frame',
    value=1,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 14)
)
rt_start = pn.widgets.FloatInput(
    name='Start RT (min)',
    value=0.00,
    step=0.50,
    start=0.00,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)
frame_end = pn.widgets.IntInput(
    name='End frame',
    value=1,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
rt_end = pn.widgets.FloatInput(
    name='End RT (min)',
    step=0.50,
    start=0.00,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)


# scans/IM selection
scan_slider = pn.widgets.IntRangeSlider(
    # name='Scans',
    show_value=False,
    bar_color='#045082',
    width=390,
    start=1,
    step=1,
    margin=(20, 0, 10, 28)
)
scan_start = pn.widgets.IntInput(
    name='Start scan',
    value=1,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
im_start = pn.widgets.FloatInput(
    name='Start IM',
    # value=0.00,
    step=0.10,
    # start=0.00,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)
scan_end = pn.widgets.IntInput(
    name='End scan',
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
im_end = pn.widgets.FloatInput(
    name='End IM',
    step=0.10,
    # start=0.00,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)


# tof and m/z selection
tof_slider = pn.widgets.IntRangeSlider(
    # name='TOF',
    show_value=False,
    bar_color='#045082',
    width=390,
    start=1,
    step=1,
    margin=(20, 0, 10, 28)
)
tof_start = pn.widgets.IntInput(
    name='Start TOF',
    value=1,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
mz_start = pn.widgets.FloatInput(
    name='Start m/z',
    value=0.00,
    step=0.10,
    start=0.00,
    width=80,
    format='0.00',
    margin=(0, 0, 0, 20),
    disabled=False
)
tof_end = pn.widgets.IntInput(
    name='End TOF',
    value=2,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
mz_end = pn.widgets.FloatInput(
    name='End m/z',
    step=0.10,
    start=0.00,
    width=85,
    format='0.00',
    margin=(0, 0, 0, 20),
    disabled=False
)


# Precursor selection
select_ms1_precursors = pn.widgets.Checkbox(
    name='Show MS1 ions (precursors)',
    value=True,
    width=200,
    align="center",
    margin=(20, 0, 10, 0)
)
select_ms2_fragments = pn.widgets.Checkbox(
    name='Show MS2 ions (fragments)',
    value=False,
    width=200,
    align="center",
)

# quad selection
quad_slider = pn.widgets.RangeSlider(
    # name='Quad',
    show_value=False,
    bar_color='#045082',
    align="center",
    start=0,
    value=(0, 0),
    step=1,
    margin=(5, 20),
    disabled=True
)
quad_start = pn.widgets.FloatInput(
    name='Start QUAD',
    value=0.00,
    step=0.50,
    align="center",
    start=0.00,
    width=80,
    format='0.00',
    margin=(0, 0, 0, 0),
    disabled=True
)
quad_end = pn.widgets.FloatInput(
    name='End QUAD',
    value=0.00,
    step=0.50,
    align="center",
    start=0.00,
    width=80,
    format='0.00',
    margin=(0, 0, 0, 0),
    disabled=True
)


#  precursor selection
precursor_slider = pn.widgets.IntRangeSlider(
    # name='TOF',
    show_value=False,
    bar_color='#045082',
    align="center",
    start=1,
    step=1,
    margin=(5, 20),
    disabled=True
)
precursor_start = pn.widgets.IntInput(
    name='Start precursor',
    value=1,
    step=1,
    align="center",
    start=1,
    width=80,
    margin=(0, 0, 0, 0),
    disabled=True
)
precursor_end = pn.widgets.IntInput(
    name='End precursor',
    value=2,
    step=1,
    align="center",
    start=1,
    width=80,
    margin=(0, 0, 0, 0),
    disabled=True
)


# Intensity selection
intensity_slider = pn.widgets.RangeSlider(
    # name='TOF',
    show_value=False,
    bar_color='#045082',
    width=390,
    start=1,
    step=1,
    margin=(20, 0, 10, 28)
)
intensity_start = pn.widgets.IntInput(
    name='Start intensity',
    value=1,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)
intensity_end = pn.widgets.IntInput(
    name='End intensity',
    value=2,
    step=1,
    start=1,
    width=80,
    margin=(0, 0, 0, 0)
)


selection_actions = pn.pane.Markdown(
    'Redo / Undo',
    align='center',
    margin=(-18, 0, 0, 0)
)
undo_button = pn.widgets.Button(
    name='\u21b6',
    # button_type='primary',
    disabled=False,
    height=32,
    width=50,
    margin=(-3, 20, 0, 20),
    align="center"
)
redo_button = pn.widgets.Button(
    name='↷',
    # button_type='primary',
    disabled=False,
    height=32,
    width=50,
    margin=(-3, 20, 0, 0),
    align="center"
)

# Download selected data
def export_sliced_data():
    from io import StringIO
    sio = StringIO()
    DATAFRAME.to_csv(sio, index=False)
    sio.seek(0)
    return sio

download_selection = pn.widgets.FileDownload(
    callback=export_sliced_data,
    filename='sliced_data.csv',
    button_type='default',
    height=31,
    width=250,
    margin=(5, 20, 15, 20),
    align='center'
)

# player
player_title = pn.pane.Markdown(
    "Quick Data Overview",
    align='center',
    margin=(20, 0, -20, 0)
)
player = pn.widgets.DiscretePlayer(
    interval=1800,
    value=1,
    show_loop_controls=True,
    loop_policy='once',
    width=400,
    align='center'
)


# select axis
# plot 1
plot1_title = pn.pane.Markdown(
    '#### Axis for Heatmap',
    margin=(10, 0, -5, 0),
    align='center'
)
plot1_x_axis = pn.widgets.Select(
    name='X axis',
    value='m/z, Th',
    options=['m/z, Th', 'Inversed IM, V·s·cm\u207B\u00B2', 'RT, min'],
    width=180,
    margin=(0, 20, 20, 20),
)
plot1_y_axis = pn.widgets.Select(
    name='Y axis',
    value='Inversed IM, V·s·cm\u207B\u00B2',
    options=['m/z, Th', 'Inversed IM, V·s·cm\u207B\u00B2', 'RT, min'],
    width=180,
    margin=(0, 20, 20, 10),
)

# plot 2
plot2_title = pn.pane.Markdown(
    '#### Axis for XIC/Spectrum/Mobilogram',
    align='center',
    margin=(10, 0, -25, 0),
)
plot2_x_axis = pn.widgets.Select(
    name='X axis',
    value='Inversed IM, V·s·cm\u207B\u00B2',
    options=['RT, min', 'm/z, Th', 'Inversed IM, V·s·cm\u207B\u00B2'],
    width=180,
    margin=(0, 20, 0, 20),
    align='center',
)


# Collapsing all options to cards
frame_selection_card = pn.Card(
    player_title,
    player,
    frame_slider,
    pn.Row(
        frame_start,
        rt_start,
        selectors_divider,
        frame_end,
        rt_end,
#         align='center',
    ),
    title='Select rt_values / frame_indices',
    collapsed=False,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
frame_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': frame_selection_card}
)

scan_selection_card = pn.Card(
    scan_slider,
    pn.Row(
        scan_start,
        im_start,
        selectors_divider,
        scan_end,
        im_end,
        align='center',
    ),
    title='Select mobility_values / scan_indices',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
scan_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': scan_selection_card}
)


quad_selection_card = pn.Card(
    # precursor_fragment_toggle_button,
    select_ms1_precursors,
    select_ms2_fragments,
    quad_slider,
    pn.Row(
        quad_start,
        selectors_divider,
        quad_end,
        align='center',
    ),
    sliders_divider,
    precursor_slider,
    pn.Row(
        precursor_start,
        selectors_divider,
        precursor_end,
        align='center',
    ),
    title='Select quad_values / precursor_indices',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
quad_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.axis_selection_settings');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': quad_selection_card}
)

tof_selection_card = pn.Card(
    tof_slider,
    pn.Row(
        tof_start,
        mz_start,
        selectors_divider,
        tof_end,
        mz_end,
        align='center',
    ),
    title='Select mz_values / tof_indices',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
tof_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': tof_selection_card}
)

intensity_selection_card = pn.Card(
    intensity_slider,
    pn.Row(
        intensity_start,
        selectors_divider,
        intensity_end,
        align='center',
    ),
    title='Select intensity_values',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
intensity_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': intensity_selection_card}
)

axis_selection_card = pn.Card(
    plot1_title,
    pn.Row(
        plot1_x_axis,
        plot1_y_axis
    ),
    plot2_title,
    plot2_x_axis,
    title='Select axis for plots',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
axis_selection_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': axis_selection_card}
)


# putting together all settings widget
settings = pn.Column(
    settings_title,
    card_divider,
    pn.Row(
        save_hdf_path,
        save_hdf_button,
        save_spinner
    ),
    save_message,
    card_divider,
    axis_selection_card,
    card_divider,
    frame_selection_card,
    card_divider,
    scan_selection_card,
    card_divider,
    quad_selection_card,
    card_divider,
    tof_selection_card,
    card_divider,
    intensity_selection_card,
    card_divider,

    pn.Row(
        pn.Column(
            selection_actions,
            pn.Row(
                undo_button,
                redo_button
            ),
            align="center"
        ),
        align="center",
    ),
    card_divider,
    download_selection,
    width=460,
    align='center',
    margin=(0, 0, 0, 0),
    css_classes=['settings']
)


# PLOTTING
def visualize_tic():
    tic = alphatims.plotting.tic_plot(DATASET, WHOLE_TITLE)
    # implement the selection
    bounds_x = hv.streams.BoundsX(
        source=tic,
        boundsx=(rt_start.value, rt_end.value)
    )

    def get_range_func(color):
        def _range(boundsx):
            rt_start.value = boundsx[0]
            rt_end.value = boundsx[1]
            return hv.VSpan(boundsx[0], boundsx[1]).opts(color=color)
        return _range

    dmap = hv.DynamicMap(get_range_func('orange'), streams=[bounds_x])
    fig = tic * dmap
    return fig.opts(responsive=True)


def visualize_scatter():
    return alphatims.plotting.heatmap(
        DATAFRAME,
        plot1_x_axis.value,
        plot1_y_axis.value,
        WHOLE_TITLE,
    )


def visualize_1d_plot():
    return alphatims.plotting.line_plot(
        DATASET,
        SELECTED_INDICES,
        plot2_x_axis.value,
        WHOLE_TITLE,
    )


# Widget dependancies
@pn.depends(
    upload_button.param.clicks,
    watch=True
)
def upload_data(*args):
    sys.path.append('../')
    global DATASET
    global WHOLE_TITLE
    if upload_file.value.endswith(".d") or upload_file.value.endswith(".hdf"):
        ext = os.path.splitext(upload_file.value)[-1]
        if ext == '.d':
            save_hdf_button.disabled = False
            save_message.object = ''
        elif ext == '.hdf':
            save_hdf_button.disabled = True
            save_message.object = ''
        upload_error.object = None
        if DATASET and os.path.basename(
            DATASET.bruker_d_folder_name
        ).split('.')[0] == os.path.basename(upload_file.value).split('.')[0]:
            upload_error.object = '#### This file is already uploaded.'
        elif not DATASET or os.path.basename(
            DATASET.bruker_d_folder_name
        ).split('.')[0] != os.path.basename(upload_file.value).split('.')[0]:
            try:
                upload_spinner.value = True
                DATASET = alphatims.bruker.TimsTOF(
                    upload_file.value,
                    slice_as_dataframe=False
                )
                mode = ''
                if 'DDA' in upload_file.value:
                    mode = 'dda-'
                elif 'DIA' in upload_file.value:
                    mode = 'dia-'
                WHOLE_TITLE = "".join(
                    [
                        str(DATASET.meta_data['SampleName']),
                        ext,
                        ': ',
                        mode,
                        str(DATASET.acquisition_mode)
                    ]
                )
            except ValueError as e:
                print(e)
                upload_error.object = "#### This file is corrupted and can't be uploaded."
    else:
        upload_error.object = '#### Please, specify a path to .d Bruker folder or .hdf file.'


@pn.depends(
    save_hdf_button.param.clicks,
    watch=True
)
def save_hdf(*args):
    save_message.object = ''
    save_spinner.value = True
    file_name = os.path.join(DATASET.directory, f"{DATASET.sample_name}.hdf")
    if save_hdf_path.value:
        directory = save_hdf_path.value
    else:
        directory = DATASET.bruker_d_folder_name
    print(DATASET.bruker_d_folder_name)
    file_name = os.path.join(directory, f"{DATASET.sample_name}.hdf")
    print(directory, file_name)
    DATASET.save_as_hdf(
        overwrite=True,
        directory=directory,
        file_name=file_name,
        compress=False,
    )
    save_spinner.value = False
    if save_hdf_path.value:
        save_message.object = '#### The HDF file is successfully saved in the specified folder.'
    else:
        save_message.object = '#### The HDF file is successfully saved inside original .d folder.'


@pn.depends(
    upload_button.param.clicks
)
def init_settings(*args):
    if DATASET:
        global STACK
        global GLOBAL_INIT_LOCK
        GLOBAL_INIT_LOCK = True
        STACK = alphatims.utils.Global_Stack(
            {
                "intensities": (0, DATASET.intensity_max_value),
                "frames": (1, 2),
                "scans": (0,  DATASET.scan_max_index),
                "tofs": (0,  DATASET.tof_max_index),
                "quads": (0, DATASET.quad_mz_max_value),
                "precursors": (1, DATASET.precursor_max_index),
                "show_fragments": select_ms2_fragments.value,
                "show_precursors": select_ms1_precursors.value,
                "plot_axis": (
                    plot1_y_axis.value,
                    plot2_x_axis.value,
                    plot2_x_axis.value,
                ),
            }
        )

        frames_msmstype = DATASET.frames.query('MsMsType == 0')
        step = len(frames_msmstype) // 10
        player.options = frames_msmstype.loc[1::step, 'Id'].to_list()
        player.start, player.end = STACK["frames"]

        frame_slider.start, frame_slider.end = (0, DATASET.frame_max_index)
        frame_start.start, frame_start.end = (0, DATASET.frame_max_index)
        frame_end.start, frame_end.end = (0, DATASET.frame_max_index)
        rt_start.start, rt_start.end = (0, DATASET.rt_max_value / 60)
        rt_end.start, rt_end.end = (0, DATASET.rt_max_value / 60)

        scan_slider.start, scan_slider.end = STACK["scans"]
        scan_start.start, scan_start.end = STACK["scans"]
        scan_end.start, scan_end.end = STACK["scans"]
        im_start.start, im_start.end = (0, DATASET.mobility_max_value)
        im_end.start, im_end.end = (0, DATASET.mobility_max_value)

        quad_slider.start, quad_slider.end = STACK["quads"]
        quad_start.start, quad_start.end = STACK["quads"]
        quad_end.start, quad_end.end = STACK["quads"]

        precursor_slider.start, precursor_slider.end = STACK["precursors"]
        precursor_start.start, precursor_start.end = STACK["precursors"]
        precursor_end.start, precursor_end.end = STACK["precursors"]

        tof_slider.start, tof_slider.end = STACK["tofs"]
        tof_start.start, tof_start.end = STACK["tofs"]
        tof_end.start, tof_end.end = STACK["tofs"]
        mz_start.start, mz_start.end = (0, DATASET.mz_max_value)
        mz_end.start, mz_end.end = (0, DATASET.mz_max_value)

        intensity_slider.start, intensity_slider.end = STACK["intensities"]
        intensity_start.start, intensity_start.end = STACK["intensities"]
        intensity_end.start, intensity_end.end = STACK["intensities"]

        update_frame_widgets_to_stack()
        update_scan_widgets_to_stack()
        update_quad_widgets_to_stack()
        update_precursor_widgets_to_stack()
        update_tof_widgets_to_stack()
        update_intensity_widgets_to_stack()

        GLOBAL_INIT_LOCK = False
        STACK.is_locked = False
        # first init needed:
        plot2_x_axis.value = 'm/z, Th'

        upload_spinner.value = False
        return settings
    else:
        return None


@pn.depends(
    player.param.value,
    watch=True
)
def update_frame_with_player(*args):
    frame_slider.value = (player.value, player.value + 1)


@pn.depends(
    exit_button.param.clicks,
    watch=True
)
def exit_button_event(*args):
    import logging
    logging.info("Quitting server...")
    exit_button.name = "Server closed"
    exit_button.button_type = "danger"
    SERVER.stop()


@pn.depends(
    frame_slider.param.value,
    frame_start.param.value,
    frame_end.param.value,
    rt_start.param.value,
    rt_end.param.value,

    scan_slider.param.value,
    scan_start.param.value,
    scan_end.param.value,
    im_start.param.value,
    im_end.param.value,

    quad_slider.param.value,
    quad_start.param.value,
    quad_end.param.value,

    precursor_slider.param.value,
    precursor_start.param.value,
    precursor_end.param.value,

    tof_slider.param.value,
    tof_start.param.value,
    tof_end.param.value,
    mz_start.param.value,
    mz_end.param.value,

    intensity_slider.param.value,
    intensity_start.param.value,
    intensity_end.param.value,

    plot1_x_axis.param.value,
    plot1_y_axis.param.value,
    plot2_x_axis.param.value,

    # precursor_fragment_toggle_button.param.value,
    select_ms1_precursors.param.value,
    select_ms2_fragments.param.value,
)
def update_plots_and_settings(*args):
    if DATASET:
        updated_option, updated_value = STACK.update(
            "show_fragments",
            select_ms2_fragments.value
        )
        if updated_value is None:
            updated_option, updated_value = STACK.update(
                "show_precursors",
                select_ms1_precursors.value
            )
        if updated_value is None:
            updated_option, updated_value = check_frames_stack()
        if updated_value is None:
            updated_option, updated_value = check_scans_stack()
        if updated_value is None:
            updated_option, updated_value = check_quads_stack()
        if updated_value is None:
            updated_option, updated_value = check_precursors_stack()
        if updated_value is None:
            updated_option, updated_value = check_tofs_stack()
        if updated_value is None:
            updated_option, updated_value = check_intensities_stack()
        if updated_value is None:
            updated_option, updated_value = STACK.update(
                "plot_axis",
                (
                    plot1_y_axis.value,
                    plot2_x_axis.value,
                    plot2_x_axis.value,
                )
            )
    else:
        updated_option, updated_value = "", None
    return update_global_selection(updated_option, updated_value)


@pn.depends(
    redo_button.param.clicks,
    watch=True,
)
def redo(*args):
    updated_option, updated_value = STACK.redo()
    return update_global_selection(updated_option, updated_value)


@pn.depends(
    undo_button.param.clicks,
    watch=True,
)
def undo(*args):
    updated_option, updated_value = STACK.undo()
    return update_global_selection(updated_option, updated_value)


# Control functions
def update_selected_indices_and_dataframe():
    global SELECTED_INDICES
    global DATAFRAME
    if DATASET:
        frame_values = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*frame_slider.value), "frame_indices"
        )
        scan_values = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*scan_slider.value), "scan_indices"
        )
        if select_ms1_precursors.value:
            quad_values = np.array([[-1, 0]])
            precursor_values = np.array([[0, 1, 1]])
        else:
            quad_values = np.empty(shape=(0, 2), dtype=np.float64)
            precursor_values = np.empty(shape=(0, 3), dtype=np.int64)
        if select_ms2_fragments.value:
            quad_values_ = alphatims.bruker.convert_slice_key_to_float_array(
                DATASET, slice(*quad_slider.value)
            )
            precursor_values_ = alphatims.bruker.convert_slice_key_to_int_array(
                DATASET, slice(*precursor_slider.value), "precursor_indices"
            )
            quad_values = np.vstack([quad_values, quad_values_])
            precursor_values = np.vstack([precursor_values, precursor_values_])
        tof_values = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*tof_slider.value), "tof_indices"
        )
        intensity_values = alphatims.bruker.convert_slice_key_to_float_array(
            DATASET, slice(*intensity_slider.value)
        )
        SELECTED_INDICES = alphatims.bruker.filter_indices(
            frame_slices=frame_values,
            scan_slices=scan_values,
            precursor_slices=precursor_values,
            tof_slices=tof_values,
            quad_slices=quad_values,
            intensity_slices=intensity_values,
            frame_max_index=DATASET.frame_max_index,
            scan_max_index=DATASET.scan_max_index,
            tof_indptr=DATASET.tof_indptr,
            precursor_indices=DATASET.precursor_indices,
            quad_mz_values=DATASET.quad_mz_values,
            quad_indptr=DATASET.quad_indptr,
            tof_indices=DATASET.tof_indices,
            intensities=DATASET.intensity_values
        )
        DATAFRAME = DATASET.as_dataframe(SELECTED_INDICES)


def run():
    global SERVER
    global LAYOUT
    LAYOUT = pn.Column(
        header,
        main_part,
        pn.Row(
            init_settings,
            update_plots_and_settings,
        ),
    )
    SERVER = LAYOUT.show(threaded=True, title='AlphaTims')


def update_global_selection(updated_option, updated_value):
    global PLOTS
    global GLOBAL_INIT_LOCK
    if updated_value is None:
        STACK.is_locked = GLOBAL_INIT_LOCK
        return PLOTS
    if updated_value != "plot_axis":
        print("updating selection")
        GLOBAL_INIT_LOCK = True
        update_widgets(updated_option)
        GLOBAL_INIT_LOCK = False
        update_selected_indices_and_dataframe()
    if DATASET:
        PLOTS = pn.Column(
            visualize_tic(),
            visualize_scatter(),
            visualize_1d_plot()
        )
        STACK.is_locked = GLOBAL_INIT_LOCK
        return PLOTS


def update_widgets(updated_option):
    if updated_option == "show_fragments":
        update_toggle_fragments()
    if updated_option == "frames":
        update_frame_widgets_to_stack()
    if updated_option == "scans":
        update_scan_widgets_to_stack()
    if updated_option == "quads":
        update_quad_widgets_to_stack()
    if updated_option == "precursors":
        update_precursor_widgets_to_stack()
    if updated_option == "tofs":
        update_tof_widgets_to_stack()
    if updated_option == "intensities":
        update_intensity_widgets_to_stack()


def update_toggle_fragments():
    quad_slider.disabled = not quad_slider.disabled
    quad_start.disabled = not quad_start.disabled
    quad_end.disabled = not quad_end.disabled
    precursor_slider.disabled = not precursor_slider.disabled
    precursor_start.disabled = not precursor_start.disabled
    precursor_end.disabled = not precursor_end.disabled


def update_frame_widgets_to_stack():
    frame_slider.value = STACK["frames"]
    frame_start.value, frame_end.value = STACK["frames"]
    rt_start.value = DATASET.rt_values[STACK["frames"][0]] / 60
    index = STACK["frames"][1]
    if index < len(DATASET.rt_values):
        rt_end.value = DATASET.rt_values[index] / 60
    else:
        rt_end.value = DATASET.rt_values[-1] / 60


def update_scan_widgets_to_stack():
    scan_slider.value = STACK["scans"]
    scan_start.value, scan_end.value = STACK["scans"]
    im_start.value = DATASET.mobility_values[STACK["scans"][0]]
    index = STACK["scans"][1]
    if index < len(DATASET.mobility_values):
        im_end.value = DATASET.mobility_values[index]
    else:
        im_end.value = DATASET.mobility_values[-1]


def update_quad_widgets_to_stack():
    quad_slider.value = STACK["quads"]
    quad_start.value, quad_end.value = STACK["quads"]


def update_precursor_widgets_to_stack():
    precursor_slider.value = STACK["precursors"]
    precursor_start.value, precursor_end.value = STACK["precursors"]


def update_tof_widgets_to_stack():
    tof_slider.value = STACK["tofs"]
    tof_start.value, tof_end.value = STACK["tofs"]
    mz_start.value = DATASET.mz_values[STACK["tofs"][0]]
    index = STACK["tofs"][1]
    if index < len(DATASET.mz_values):
        mz_end.value = DATASET.mz_values[index]
    else:
        mz_end.value = DATASET.mz_values[-1]


def update_intensity_widgets_to_stack():
    intensity_slider.value = STACK["intensities"]
    intensity_start.value, intensity_end.value = STACK["intensities"]


def check_frames_stack():
    updated_option, updated_value = STACK.update(
        "frames", frame_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "frames", (frame_start.value, frame_end.value)
        )
    if updated_value is None:
        start_, end_ = DATASET.convert_to_indices(
            np.array([rt_start.value, rt_end.value]) * 60,
            return_frame_indices=True
        )
        updated_option, updated_value = STACK.update(
            "frames", (int(start_), int(end_))
        )
    return updated_option, updated_value


def check_scans_stack():
    updated_option, updated_value = STACK.update(
        "scans", scan_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "scans", (scan_start.value, scan_end.value)
        )
    if updated_value is None:
        start_, end_ = DATASET.convert_to_indices(
            np.array([im_start.value, im_end.value])[::-1],
            return_scan_indices=True
        )[::-1]
        updated_option, updated_value = STACK.update(
            "scans", (int(start_), int(end_))
        )
    return updated_option, updated_value


def check_quads_stack():
    updated_option, updated_value = STACK.update(
        "quads", quad_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "quads", (quad_start.value, quad_end.value)
        )
    return updated_option, updated_value


def check_precursors_stack():
    updated_option, updated_value = STACK.update(
        "precursors", precursor_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "precursors", (precursor_start.value, precursor_end.value)
        )
    return updated_option, updated_value


def check_tofs_stack():
    updated_option, updated_value = STACK.update(
        "tofs", tof_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "tofs", (tof_start.value, tof_end.value)
        )
    if updated_value is None:
        start_, end_ = DATASET.convert_to_indices(
            np.array([mz_start.value, mz_end.value]),
            return_tof_indices=True
        )
        updated_option, updated_value = STACK.update(
            "tofs", (int(start_), int(end_))
        )
    return updated_option, updated_value


def check_intensities_stack():
    updated_option, updated_value = STACK.update(
        "intensities", intensity_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "intensities", (intensity_start.value, intensity_end.value)
        )
    return updated_option, updated_value
