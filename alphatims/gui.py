#!python

# external
import os
import numpy as np
import pandas as pd
import panel as pn
import sys
import warnings
import logging
# holoviz libraries
import holoviews as hv
import bokeh.server.views.ws
# local
import alphatims.bruker
import alphatims.utils
import alphatims.plotting


# TODO: do we want to ignore warnings?
warnings.filterwarnings('ignore')


# GLOBAL VARIABLES
DATASET = None
PLOTS = pn.Column(None, None, None, None, sizing_mode='stretch_width',)
WHOLE_TITLE = str()
SELECTED_INDICES = np.array([])
DATAFRAME = pd.DataFrame()
STACK = alphatims.utils.Global_Stack({})
SERVER = None
TAB_COUNTER = 0
BROWSER = pn.Row(
    None,
    PLOTS,
    sizing_mode='stretch_width',
)
INTENSITY_THRESHOLD = 10**7

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

.bk-root .bk-btn-danger {
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


# PATHS
biochem_logo_path = os.path.join(
    alphatims.utils.IMG_PATH,
    "mpi_logo.png"
)
mpi_logo_path = os.path.join(
    alphatims.utils.IMG_PATH,
    "max-planck-gesellschaft.jpg"
)
github_logo_path = os.path.join(
    alphatims.utils.IMG_PATH,
    "github.png"
)

gui_manual_path = os.path.join(
    alphatims.utils.DOC_PATH,
    "gui_manual.pdf"
)


# HEADER
header_titel = pn.pane.Markdown(
    f'# AlphaTims {alphatims.__version__}',
    sizing_mode='stretch_width',
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
    align='end'
)

header = pn.Row(
    mpi_biochem_logo,
    mpi_logo,
    header_titel,
    github_logo,
    height=73,
    sizing_mode='stretch_width'
)


# MAIN PART
project_description = pn.pane.Markdown(
    """### AlphaTims provides fast accession and visualization of unprocessed LC-TIMS-Q-TOF data from Bruker's timsTOF Pro instruments. It indexes the data such that it can easily be sliced along all five dimensions: LC, TIMS, QUADRUPOLE, TOF and DETECTOR.""",
    # margin=(10, 0, 0, 0),
    css_classes=['main-part'],
    width=690
)

divider_descr = pn.pane.HTML(
    '<hr style="height: 6px; border:none; background-color: #045082">',
    sizing_mode='stretch_width',
    align='center'
)
upload_file = pn.widgets.TextInput(
    name='Specify an experimental file:',
    placeholder='Enter the whole path to Bruker .d folder or .hdf file',
    align="center",
    # width=800,
    sizing_mode="stretch_width",
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
upload_progress = pn.widgets.Progress(
    margin=(40, 15, 0, 15),
    sizing_mode="stretch_width",
    # width=100,
    # height=40,
    max=1,
    value=1,
    active=True,
    bar_color='secondary'
)
upload_error = pn.pane.Alert(
    # width=800,
    sizing_mode="stretch_width",
    alert_type="danger",
    align='center',
    margin=(-15, 0, -5, 0),
)
quit_button = pn.widgets.Button(
    name='Quit',
    button_type='default',
    height=31,
    width=100,
    margin=(34, 20, 0, 0)
)

# if os.path.exists(alphatims.utils.DEMO_FILE_NAME):
#     download_demo = None
#     demo_dataset = pn.widgets.Button(
#         name='Upload demo',
#         button_type='primary',
#         height=31,
#         width=100,
#         margin=(34, 20, 0, 20)
#     )
# else:
#     download_demo = pn.widgets.Button(
#         name='Download demo',
#         button_type='default',
#         height=31,
#         width=100,
#         margin=(34, 20, 0, 20)
#     )
#     demo_dataset = None
#
#     @pn.depends(download_demo.param.clicks, watch=True)
#     def download_demo_data(*args):
#         global demo_dataset
#         import urllib.request
#         import urllib.error
#         import zipfile
#         import io
#         with urllib.request.urlopen(
#             alphatims.utils.DEMO_FILE_NAME_GITHUB
#         ) as sample_file:
#             sample_byte_stream = io.BytesIO(sample_file.read())
#             with zipfile.ZipFile(sample_byte_stream, 'r') as zip_ref:
#                 zip_ref.extractall(
#                     os.path.dirname(alphatims.utils.DEMO_FILE_NAME)
#                 )
#         demo_dataset = pn.widgets.Button(
#             name='Upload demo',
#             button_type='primary',
#             height=31,
#             width=100,
#             margin=(34, 20, 0, 20)
#         )


gui_manual_button = pn.widgets.FileDownload(
    file=gui_manual_path,
    label='Download GUI manual',
    # button_type='default',
    # auto=True,
    # height=31,
    # width=200,
    # margin=(34, 20, 0, 20)
    button_type='primary',
    # height=31,
    # width=100,
    # margin=(34, 20, 0, 0),
)

github_version = alphatims.utils.check_github_version(silent=True)
if github_version == alphatims.__version__:
    download_new_version_text = "AlphaTims version is up-to-date"
    download_new_version_button_type = "primary"
    download_new_version_button = None
else:
    download_new_version_text = f"Download version {github_version}"
    download_new_version_button_type = "danger"
    download_new_version_button = pn.widgets.Button(
        name=download_new_version_text,
        button_type=download_new_version_button_type,
        # height=31,
        # width=100,
        # margin=(34, 20, 0, 0),
        # disabled=True,
        # link_url='https://github.com/MannLabs/alphatims',
    )
    download_new_version_button.js_on_click(
        code=f"""window.open("https://github.com/MannLabs/alphatims#changelog")"""
    )

download_test_data_button = pn.widgets.Button(
    name="Download test data",
    button_type='primary',
    # height=31,
    # width=100,
    # margin=(34, 20, 0, 0),
    # disabled=True,
    # link_url='https://github.com/MannLabs/alphatims',
)
download_test_data_button.js_on_click(
    code="""window.open("https://github.com/MannLabs/alphatims#test-data")"""
)
download_citation_button = pn.widgets.Button(
    name="Download citation",
    button_type='primary',
    # height=31,
    # width=100,
    # margin=(34, 20, 0, 0),
    # disabled=True,
    # link_url='https://github.com/MannLabs/alphatims',
)
download_citation_button.js_on_click(
    code="""window.open("https://github.com/MannLabs/alphatims#citing-alphatims")"""
)


main_part = pn.Column(
    pn.Row(
        project_description,
        pn.layout.HSpacer(width=500),
        pn.Column(
            gui_manual_button,
            download_test_data_button,
            download_citation_button,
            download_new_version_button,
        ),
        background='#eaeaea',
        align='center',
        sizing_mode='stretch_width',
        # height=190,
        # margin=(10, 8, 10, 8),
        css_classes=['background']
    ),
    divider_descr,
    pn.Row(
        upload_file,
        align='center',
        sizing_mode='stretch_width',
    ),
    pn.Row(
        # download_demo,
        # demo_dataset,
        upload_button,
        upload_progress,
        upload_spinner,
        # gui_manual_button,
        # download_new_version_button,
        # quit_button,
        align='center',
        # sizing_mode='stretch_width',
    ),
    pn.Row(
        upload_error,
        align='center',
        width=800,
        margin=(-15, 0, 0, 0)
    ),
    background='#eaeaea',
    sizing_mode='stretch_width',
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
    name='Specify a path to save all data as a portable .hdf file:',
    placeholder='e.g. D:\Bruker',
    margin=(15, 10, 0, 10)
)
save_hdf_overwrite = pn.widgets.Checkbox(
    name='overwrite',
    value=False,
    width=80,
    margin=(10, 10, 0, 0)
)
save_hdf_compress = pn.widgets.Checkbox(
    name='compress',
    value=False,
    width=80,
    margin=(10, 10, 0, 0)
)
save_hdf_button = pn.widgets.Button(
    name='Save as HDF',
    button_type='default',
    height=31,
    width=100,
    margin=(10, 10, 0, 0)
)
save_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(11, 0, 0, 15),
    width=30,
    height=30
)
save_hdf_message = pn.pane.Alert(
    alert_type='success',
    margin=(-15, 5, -10, 15),
    # height=35,
    width=300
)

# SAVE SLICED DATA
save_sliced_data_path = pn.widgets.TextInput(
    name='Specify a path to save the currently selected data as .csv file:',
    placeholder='e.g. D:\Bruker',
    margin=(15, 10, 0, 10)
)
save_sliced_data_overwrite = pn.widgets.Checkbox(
    name='overwrite',
    value=False,
    width=80,
    margin=(10, 10, 0, 0)
)
save_sliced_data_button = pn.widgets.Button(
    name='Save as CSV',
    button_type='default',
    height=31,
    width=100,
    margin=(10, 10, 0, 0)
)
save_sliced_data_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(11, 0, 0, 15),
    width=30,
    height=30
)
save_sliced_data_message = pn.pane.Alert(
    alert_type='success',
    margin=(-15, 5, -10, 15),
    width=300
)

# SAVE MGF
save_mgf_path = pn.widgets.TextInput(
    name='Specify a path to save all MS2 spectra as .mgf file:',
    placeholder='e.g. D:\Bruker',
    margin=(15, 10, 0, 10)
)
save_mgf_overwrite = pn.widgets.Checkbox(
    name='overwrite',
    value=False,
    width=80,
    margin=(10, 10, 0, 0)
)
save_mgf_centroid = pn.widgets.Checkbox(
    name='centroid',
    value=False,
    width=80,
    margin=(10, 10, 0, 0)
)
save_mgf_button = pn.widgets.Button(
    name='Save as MGF',
    button_type='default',
    height=31,
    width=100,
    margin=(10, 10, 0, 0)
)
save_mgf_spinner = pn.indicators.LoadingSpinner(
    value=False,
    bgcolor='light',
    color='secondary',
    margin=(11, 0, 0, 15),
    width=30,
    height=30
)
save_mgf_message = pn.pane.Alert(
    alert_type='success',
    margin=(-15, 5, -10, 15),
    width=300
)

# frame/RT selection
frame_slider = pn.widgets.IntRangeSlider(
    show_value=False,
    bar_color='#045082',
    step=1,
    margin=(10, 20, 10, 20)
)
frame_start = pn.widgets.IntInput(
    name='Start frame',
    step=1,
    width=80,
    margin=(0, 0, 0, 14)
)
rt_start = pn.widgets.FloatInput(
    name='Start RT (min)',
    step=0.50,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)
frame_end = pn.widgets.IntInput(
    name='End frame',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
rt_end = pn.widgets.FloatInput(
    name='End RT (min)',
    step=0.50,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)


# scans/IM selection
scan_slider = pn.widgets.IntRangeSlider(
    show_value=False,
    bar_color='#045082',
    width=390,
    step=1,
    margin=(20, 0, 10, 28)
)
scan_start = pn.widgets.IntInput(
    name='Start scan',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
im_start = pn.widgets.FloatInput(
    name='Start IM',
    step=0.10,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)
scan_end = pn.widgets.IntInput(
    name='End scan',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
im_end = pn.widgets.FloatInput(
    name='End IM',
    step=0.10,
    width=80,
    format='0,0.000',
    margin=(0, 0, 0, 20),
    disabled=False
)


# tof and m/z selection
tof_slider = pn.widgets.IntRangeSlider(
    show_value=False,
    bar_color='#045082',
    width=390,
    step=1,
    margin=(20, 0, 10, 28)
)
tof_start = pn.widgets.IntInput(
    name='Start TOF',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
mz_start = pn.widgets.FloatInput(
    name='Start m/z',
    step=0.10,
    width=80,
    format='0.00',
    margin=(0, 0, 0, 20),
    disabled=False
)
tof_end = pn.widgets.IntInput(
    name='End TOF',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
mz_end = pn.widgets.FloatInput(
    name='End m/z',
    step=0.10,
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
    show_value=False,
    bar_color='#045082',
    align="center",
    step=1,
    margin=(5, 20),
    disabled=True
)
quad_start = pn.widgets.FloatInput(
    name='Start QUAD',
    step=0.50,
    align="center",
    width=80,
    format='0.00',
    margin=(0, 0, 0, 0),
    disabled=True
)
quad_end = pn.widgets.FloatInput(
    name='End QUAD',
    step=0.50,
    align="center",
    width=80,
    format='0.00',
    margin=(0, 0, 0, 0),
    disabled=True
)


#  precursor selection
precursor_slider = pn.widgets.IntRangeSlider(
    show_value=False,
    bar_color='#045082',
    align="center",
    step=1,
    margin=(5, 20),
    disabled=True
)
precursor_start = pn.widgets.IntInput(
    name='Start precursor',
    step=1,
    align="center",
    width=100,
    margin=(0, 0, 0, 0),
    disabled=True
)
precursor_end = pn.widgets.IntInput(
    name='End precursor',
    step=1,
    align="center",
    width=100,
    margin=(0, 0, 0, 0),
    disabled=True
)


# Intensity selection
intensity_slider = pn.widgets.IntRangeSlider(
    # name='TOF',
    show_value=False,
    bar_color='#045082',
    width=390,
    step=1,
    margin=(20, 0, 10, 28)
)
intensity_start = pn.widgets.IntInput(
    name='Start intensity',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)
intensity_end = pn.widgets.IntInput(
    name='End intensity',
    step=1,
    width=80,
    margin=(0, 0, 0, 0)
)


selection_actions = pn.pane.Markdown(
    'Undo | Redo',
    align='center',
    margin=(-18, 0, 0, 0)
)
undo_button = pn.widgets.Button(
    name='\u21b6',
    disabled=False,
    height=32,
    width=50,
    margin=(-3, 5, 0, 0),
    align="center"
)
redo_button = pn.widgets.Button(
    name='↷',
    disabled=False,
    height=32,
    width=50,
    margin=(-3, 0, 0, 5),
    align="center"
)
show_data_title = pn.pane.Markdown(
    'Show table',
    align='center',
    margin=(-18, 10, 0, 0)
)
show_data_checkbox = pn.widgets.Checkbox(
    value=False,
    width=10,
    margin=(10, 10, 0, 0),
    align='center',
)


# player
player_title = pn.pane.Markdown(
    "Quick Data Overview",
    align='center',
    margin=(20, 0, -20, 0)
)
player = pn.widgets.DiscretePlayer(
    interval=2200,
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
    value='RT, min',
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
    title='Select rt_values / frame_indices (LC)',
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
    title='Select mobility_values / scan_indices (TIMS)',
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
    title='Select quad_values / precursor_indices (QUADRUPOLE)',
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
    title='Select mz_values / tof_indices (TOF)',
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
    title='Select intensity_values (DETECTOR)',
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

export_data_card = pn.Card(
    save_hdf_path,
    pn.Row(
        pn.Column(
            save_hdf_overwrite,
            save_hdf_compress,
            margin=(0, 50, 0, -100),
            # align="center"
        ),
        save_hdf_button,
        save_spinner,
        align="center",
    ),
    save_hdf_message,
    save_sliced_data_path,
    pn.Row(
        # TODO: weird spacing?
        pn.Column(
            save_sliced_data_overwrite,
            None,
            # align="center",
            margin=(0, 50, 0, -100),
        ),
        save_sliced_data_button,
        save_sliced_data_spinner,
        align="center",
    ),
    save_sliced_data_message,
    save_mgf_path,
    pn.Row(
        # TODO: weird spacing?
        pn.Column(
            save_mgf_overwrite,
            save_mgf_centroid,
            # align="center",
            margin=(0, 50, 0, -100),
        ),
        save_mgf_button,
        save_mgf_spinner,
        align="center",
    ),
    save_mgf_message,
    title='Export data',
    collapsed=True,
    width=430,
    margin=(10, 10, 10, 15),
    background='#EAEAEA',
    header_background='EAEAEA',
    css_classes=['axis_selection_settings']
)
export_data_card.jscallback(
    collapsed="""
        var $container = $("html,body");
        var $scrollTo = $('.test');

        $container.animate({scrollTop: $container.offset().top + $container.scrollTop(), scrollLeft: 0},300);
        """,
    args={'card': export_data_card}
)

strike_title = pn.pane.Markdown(
    'Strike count (limit / current estimate)',
    align='center',
    margin=(-18, 0, 0, 0)
)
strike_threshold = pn.widgets.IntInput(
    # name="Strike count upper limit",
    step=1,
    width=110,
    value=INTENSITY_THRESHOLD,
    margin=(0, 0, 0, 14)
)
strike_estimate = pn.widgets.IntInput(
    # name='Strike count estimate',
    step=1,
    width=110,
    value=0,
    margin=(0, 0, 0, 14),
    disabled=True
)

# putting together all settings widget
settings = pn.Column(
    settings_title,
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
    axis_selection_card,
    card_divider,
    export_data_card,
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
        pn.Column(
            strike_title,
            pn.Row(
                strike_threshold,
                strike_estimate,
            ),
            align="center"
        ),
        pn.Column(
            show_data_title,
            show_data_checkbox,
            align="center"
        ),
        align="center",
        margin=(0, 0, 20, 0),
    ),
    pn.Spacer(sizing_mode='stretch_height'),
    width=460,
    align='center',
    margin=(0, 0, 20, 0),
    css_classes=['settings']
)


# PLOTTING
def get_range_func(color, boundsx):
    def _range(boundsx):
        return hv.VSpan(boundsx[0], boundsx[1]).opts(color=color)
    return _range


def visualize_tic():
    tic = alphatims.plotting.tic_plot(
        DATASET,
        WHOLE_TITLE,
        width=None,
        height=320
    )
    # implement the selection
    # tic.opts(responsive=True)
    bounds_x = hv.streams.BoundsX(
        source=tic,
        boundsx=(rt_start.value, rt_end.value),
    )
    dmap = hv.DynamicMap(
        get_range_func('orange', bounds_x),
        streams=[bounds_x],
    )
    # tic.remove_tools(bokeh.models.BoxSelectTool)
    fig = tic * dmap
    # TODO: remove "box select (x-axis)" tool as this is not responsive?
    # TODO: plot height does not allows space for hover tool at bottom
    # TODO: both problems can be merged to fix eachother?
    # TODO: similar for 1d_plot if XIC...
    return fig#.opts(responsive=True)


def visualize_scatter():
    axis_dict = {
        "m/z, Th": "mz",
        "RT, min": "rt",
        "Inversed IM, V·s·cm\u207B\u00B2": "mobility",
        "Intensity": "intensity",
    }
    return alphatims.plotting.heatmap(
        DATAFRAME,
        axis_dict[plot1_x_axis.value],
        axis_dict[plot1_y_axis.value],
        WHOLE_TITLE,
        width=None,
        height=320
    )


def visualize_1d_plot():
    axis_dict = {
        "m/z, Th": "mz",
        "RT, min": "rt",
        "Inversed IM, V·s·cm\u207B\u00B2": "mobility",
        "Intensity": "intensity",
    }
    line_plot = alphatims.plotting.line_plot(
        DATASET,
        SELECTED_INDICES,
        axis_dict[plot2_x_axis.value],
        WHOLE_TITLE,
        width=None,
        height=320
    )
    # if plot2_x_axis.value == "RT, min":
    #     bounds_x = hv.streams.BoundsX(
    #         source=line_plot,
    #         boundsx=(rt_start.value, rt_end.value)
    #     )
    #     dmap = hv.DynamicMap(
    #         get_range_func('khaki', bounds_x),
    #         streams=[bounds_x]
    #     )
    #     fig = line_plot * dmap
    #     return fig#.opts(responsive=True)
    # else:
    #     return line_plot
    # NOTE: by default the xlims are now trimmed to the visible selection.
    # Thus, the XIC is always equal to the whole plot,
    # making the stream/ dmap redundant...
    return line_plot


def show_df():
    if show_data_checkbox.value:
        return pn.widgets.DataFrame(
            DATAFRAME,
            show_index=False,
            disabled=True,
            height=400,
            sizing_mode='stretch_width'
        )
    else:
        return None


def upload_data(*args):
    sys.path.append('../')
    global DATASET
    global DATAFRAME
    global SELECTED_INDICES
    global WHOLE_TITLE
    global alphatims
    ext = os.path.splitext(upload_file.value)[-1]
    while upload_file.value.startswith("\""):
        upload_file.value = upload_file.value[1:]
    while upload_file.value.endswith("\""):
        upload_file.value = upload_file.value[:-1]
    if ext in [".d", ".hdf"]:
        save_hdf_message.object = ''
        save_sliced_data_message.object = ''
        upload_error.object = None
        if DATASET and os.path.basename(
            DATASET.bruker_d_folder_name
        ).split('.')[0] == os.path.basename(upload_file.value).split('.')[0]:
            upload_error.object = '#### This file is already uploaded.'
        elif not DATASET or os.path.basename(
            DATASET.bruker_d_folder_name
        ).split('.')[0] != os.path.basename(upload_file.value).split('.')[0]:
            try:
                DATASET = None
                DATAFRAME = None
                SELECTED_INDICES = None
                alphatims.utils.set_progress_callback(upload_progress)
                upload_progress.value = 0
                upload_spinner.value = True
                DATASET = alphatims.bruker.TimsTOF(
                    upload_file.value,
                    slice_as_dataframe=False
                )
                alphatims.utils.set_progress_callback(True)
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
                logging.exception(e)
                upload_error.object = "#### This file is corrupted and can't be uploaded."
    else:
        upload_error.object = '#### Please, specify a correct path to .d Bruker folder or .hdf file.'


@pn.depends(
    save_hdf_button.param.clicks,
    watch=True
)
def save_hdf(*args):
    save_hdf_message.object = ''
    save_spinner.value = True
    directory = os.path.dirname(save_hdf_path.value)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if save_hdf_overwrite.value or not os.path.exists(save_hdf_path.value):
        try:
            DATASET.save_as_hdf(
                overwrite=save_hdf_overwrite.value,
                directory=directory,
                file_name=os.path.basename(save_hdf_path.value),
                compress=save_hdf_compress.value,
            )
            save_hdf_message.alert_type = 'success'
            save_hdf_message.object = '#### The HDF file is successfully saved.'
        except ValueError:
            save_hdf_message.alert_type = 'danger'
            save_hdf_message.object = '#### Could not save the file for unknown reasons.'
    else:
        save_hdf_message.alert_type = 'danger'
        save_hdf_message.object = '#### The file already exists. Specify another name or allow to overwrite the file.'
    save_spinner.value = False


@pn.depends(
    save_sliced_data_button.param.clicks,
    watch=True
)
def save_sliced_data(*args):
    save_sliced_data_message.object = ''
    save_sliced_data_spinner.value = True
    directory = os.path.dirname(save_sliced_data_path.value)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if save_sliced_data_overwrite.value or not os.path.exists(
        save_sliced_data_path.value
    ):
        DATAFRAME.to_csv(save_sliced_data_path.value, index=False)
        save_sliced_data_message.alert_type = 'success'
        save_sliced_data_message.object = '#### The CSV file is successfully saved.'
    else:
        save_sliced_data_message.alert_type = 'danger'
        save_sliced_data_message.object = '#### The file already exists. Specify another name or allow to overwrite the file.'
    save_sliced_data_spinner.value = False


@pn.depends(
    save_mgf_button.param.clicks,
    watch=True
)
def save_mgf(*args):
    save_mgf_message.object = ''
    save_mgf_spinner.value = True
    directory = os.path.dirname(save_mgf_path.value)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if save_mgf_overwrite.value or not os.path.exists(
        save_mgf_path.value
    ):
        DATASET.save_as_mgf(
            overwrite=save_mgf_overwrite.value,
            directory=directory,
            file_name=os.path.basename(save_mgf_path.value),
            centroiding_window=5 if save_mgf_centroid.value else 0
        )
        save_mgf_message.alert_type = 'success'
        save_mgf_message.object = '#### The MGF file is successfully saved.'
    else:
        save_mgf_message.alert_type = 'danger'
        save_mgf_message.object = '#### The file already exists. Specify another name or allow to overwrite the file.'
    save_mgf_spinner.value = False


@pn.depends(
    upload_button.param.clicks,
    watch=True
)
def init_settings(*args):
    global STACK
    upload_data()
    if DATASET:
        with STACK.lock():
            select_ms1_precursors.value = True
            if select_ms2_fragments.value:
                select_ms2_fragments.value = False
                update_toggle_fragments()
            plot1_x_axis.value = 'm/z, Th'
            plot1_y_axis.value = 'Inversed IM, V·s·cm\u207B\u00B2'
            plot2_x_axis.value = 'm/z, Th'
            strike_threshold.value = INTENSITY_THRESHOLD
            show_data_checkbox.value = False
        STACK = alphatims.utils.Global_Stack(
            {
                "intensities": (0, int(DATASET.intensity_max_value)),
                "frames": (0, DATASET.frame_max_index),
                "scans": (0,  DATASET.scan_max_index),
                "tofs": (0,  DATASET.tof_max_index),
                "quads": (DATASET.quad_mz_min_value, DATASET.quad_mz_max_value),
                "precursors": (1, DATASET.precursor_max_index),
                "show_fragments": select_ms2_fragments.value,
                "show_precursors": select_ms1_precursors.value,
                "plot_axis": (
                    plot1_x_axis.value,
                    plot1_y_axis.value,
                    plot2_x_axis.value,
                ),
            }
        )

        update_frame_widgets_to_stack()
        update_scan_widgets_to_stack()
        update_quad_widgets_to_stack()
        update_precursor_widgets_to_stack()
        update_tof_widgets_to_stack()
        update_intensity_widgets_to_stack()

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

        if DATASET.acquisition_mode != "noPASEF":
            quad_slider.start, quad_slider.end = STACK["quads"]
            quad_start.start, quad_start.end = STACK["quads"]
            quad_end.start, quad_end.end = STACK["quads"]

            precursor_slider.start, precursor_slider.end = STACK["precursors"]
            precursor_start.start, precursor_start.end = STACK["precursors"]
            precursor_end.start, precursor_end.end = STACK["precursors"]

            if DATASET.acquisition_mode == "diaPASEF":
                precursor_start.name = "Start window group"
                precursor_end.name = "End window group"
            else:
                precursor_start.name = "Start precursor"
                precursor_end.name = "End precursor"

        else:
            select_ms2_fragments.disabled = True
            select_ms1_precursors.disabled = True

        tof_slider.start, tof_slider.end = STACK["tofs"]
        tof_start.start, tof_start.end = STACK["tofs"]
        tof_end.start, tof_end.end = STACK["tofs"]
        mz_start.start, mz_start.end = (0, DATASET.mz_max_value)
        mz_end.start, mz_end.end = (0, DATASET.mz_max_value)

        intensity_slider.start, intensity_slider.end = STACK["intensities"]
        intensity_start.start, intensity_start.end = STACK["intensities"]
        intensity_end.start, intensity_end.end = STACK["intensities"]

        save_hdf_path.value = os.path.join(
            DATASET.directory,
            f"{DATASET.sample_name}.hdf",
        )
        save_sliced_data_path.value = os.path.join(
            DATASET.directory,
            f"{DATASET.sample_name}_data_slice.csv",
        )
        save_mgf_path.value = os.path.join(
            DATASET.directory,
            f"{DATASET.sample_name}.mgf",
        )
        if not DATASET.acquisition_mode == "ddaPASEF":
            save_mgf_path.disabled = True
            save_mgf_overwrite.disabled = True
            save_mgf_button.disabled = True
            save_mgf_spinner.disabled = True
            save_mgf_message.disabled = True

        # first init needed:
        plot2_x_axis.value = 'RT, min'
        plot2_x_axis.value = 'm/z, Th'

        BROWSER[0] = settings

        upload_spinner.value = False
    else:
        BROWSER[0] = None


@pn.depends(
    player.param.value,
    watch=True
)
def update_frame_with_player(*args):
    frame_slider.value = (player.value, player.value + 1)


@pn.depends(
    # frame_slider.param.value,
    # frame_start.param.value,
    # frame_end.param.value,
    # rt_start.param.value,
    # rt_end.param.value,

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
    watch=True,
)
def update_plots_and_settings(*args):
    if DATASET and not STACK.is_locked:
        updated_option, updated_value = STACK.update(
            "show_fragments",
            select_ms2_fragments.value
        )
        if updated_value is None:
            updated_option, updated_value = STACK.update(
                "show_precursors",
                select_ms1_precursors.value
            )
        # if updated_value is None:
        #     updated_option, updated_value = check_frames_stack()
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
    show_data_checkbox.param.value,
    watch=True,
)
def enable_show_df(*args):
    update_global_selection("plot_axis", "df")


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
        frame_indices = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*frame_slider.value), "frame_indices"
        )
        scan_indices = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*scan_slider.value), "scan_indices"
        )
        if select_ms1_precursors.value:
            quad_values = np.array([[-1, 0]])
            precursor_indices = np.array([[0, 1, 1]])
        else:
            quad_values = np.empty(shape=(0, 2), dtype=np.float64)
            precursor_indices = np.empty(shape=(0, 3), dtype=np.int64)
        if select_ms2_fragments.value:
            quad_values_ = alphatims.bruker.convert_slice_key_to_float_array(
                slice(*quad_slider.value)
            )
            precursor_indices_ = alphatims.bruker.convert_slice_key_to_int_array(
                DATASET, slice(*precursor_slider.value), "precursor_indices"
            )
            quad_values = np.vstack([quad_values, quad_values_])
            precursor_indices = np.vstack([precursor_indices, precursor_indices_])
        tof_indices = alphatims.bruker.convert_slice_key_to_int_array(
            DATASET, slice(*tof_slider.value), "tof_indices"
        )
        intensity_values = alphatims.bruker.convert_slice_key_to_float_array(
            slice(*intensity_slider.value)
        )
        strike_estimate.value = DATASET.estimate_strike_count(
            frame_slices=frame_indices,
            scan_slices=scan_indices,
            precursor_slices=precursor_indices,
            tof_slices=tof_indices,
            quad_slices=quad_values,
        )
        if strike_estimate.value > strike_threshold.value:
            SELECTED_INDICES = np.empty((0,), dtype=np.int64)
        else:
            SELECTED_INDICES = alphatims.bruker.filter_indices(
                frame_slices=frame_indices,
                scan_slices=scan_indices,
                precursor_slices=precursor_indices,
                tof_slices=tof_indices,
                quad_slices=quad_values,
                intensity_slices=intensity_values,
                frame_max_index=DATASET.frame_max_index,
                scan_max_index=DATASET.scan_max_index,
                push_indptr=DATASET.push_indptr,
                precursor_indices=DATASET.precursor_indices,
                quad_mz_values=DATASET.quad_mz_values,
                quad_indptr=DATASET.quad_indptr,
                tof_indices=DATASET.tof_indices,
                intensities=DATASET.intensity_values
            )
        DATAFRAME = DATASET.as_dataframe(SELECTED_INDICES)


def run(port=None, bruker_raw_data=None):
    global LAYOUT
    global PLOTS
    global SERVER
    LAYOUT = pn.Column(
        header,
        main_part,
        BROWSER,
        sizing_mode='stretch_width',
    )
    original_open = bokeh.server.views.ws.WSHandler.open
    bokeh.server.views.ws.WSHandler.open = open_browser_tab(original_open)
    original_on_close = bokeh.server.views.ws.WSHandler.on_close
    bokeh.server.views.ws.WSHandler.on_close = close_browser_tab(
        original_on_close
    )
    if port is not None:
        import socket
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        websocket_origin = f"{ip_address}:{port}"
    else:
        websocket_origin = None
    SERVER = LAYOUT.show(
        title='AlphaTims',
        threaded=True,
        port=port,
        websocket_origin=websocket_origin,
    )
    if bruker_raw_data is not None:
        upload_file.value = bruker_raw_data
        init_settings()
    SERVER.join()


@pn.depends(
    quit_button.param.clicks,
    watch=True
)
def quit_button_event(*args):
    quit_server()


def open_browser_tab(func):
    def wrapper(*args, **kwargs):
        global TAB_COUNTER
        TAB_COUNTER += 1
        return func(*args, **kwargs)
    return wrapper


def close_browser_tab(func):
    def wrapper(*args, **kwargs):
        global TAB_COUNTER
        TAB_COUNTER -= 1
        return_value = func(*args, **kwargs)
        if TAB_COUNTER == 0:
            quit_server()
        return return_value
    return wrapper


def quit_server():
    quit_button.name = "Server closed"
    quit_button.button_type = "danger"
    logging.info("Quitting server...")
    SERVER.stop()


def update_global_selection(updated_option, updated_value):
    if updated_value is not None:
        if updated_option != "plot_axis":
            logging.info(
                f"Updating selection of '{updated_option}' "
                f"with {updated_value}"
            )
            update_widgets(updated_option)
            update_selected_indices_and_dataframe()
        if DATASET:
            if updated_value == "df":
                logging.info("Showing table")
                PLOTS[3] = show_df()
            else:
                logging.info("Updating plots")
                PLOTS[0] = visualize_tic()
                PLOTS[1] = visualize_scatter()
                PLOTS[2] = visualize_1d_plot()
                PLOTS[3] = show_df()


def update_widgets(updated_option):
    with STACK.lock():
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
    with STACK.lock():
        frame_slider.value = STACK["frames"]
        frame_start.value, frame_end.value = STACK["frames"]
        rt_start.value = DATASET.rt_values[STACK["frames"][0]] / 60
        index = STACK["frames"][1]
        if index < len(DATASET.rt_values):
            rt_end.value = DATASET.rt_values[index] / 60
        else:
            rt_end.value = DATASET.rt_values[-1] / 60


def update_scan_widgets_to_stack():
    with STACK.lock():
        scan_slider.value = STACK["scans"]
        scan_start.value, scan_end.value = STACK["scans"]
        im_start.value = DATASET.mobility_values[STACK["scans"][0]]
        index = STACK["scans"][1]
        if index < len(DATASET.mobility_values):
            im_end.value = DATASET.mobility_values[index]
        else:
            im_end.value = DATASET.mobility_values[-1]


def update_quad_widgets_to_stack():
    with STACK.lock():
        quad_slider.value = STACK["quads"]
        quad_start.value, quad_end.value = STACK["quads"]


def update_precursor_widgets_to_stack():
    with STACK.lock():
        precursor_slider.value = STACK["precursors"]
        precursor_start.value, precursor_end.value = STACK["precursors"]


def update_tof_widgets_to_stack():
    with STACK.lock():
        tof_slider.value = STACK["tofs"]
        tof_start.value, tof_end.value = STACK["tofs"]
        mz_start.value = DATASET.mz_values[STACK["tofs"][0]]
        index = STACK["tofs"][1]
        if index < len(DATASET.mz_values):
            mz_end.value = DATASET.mz_values[index]
        else:
            mz_end.value = DATASET.mz_values[-1]


def update_intensity_widgets_to_stack():
    with STACK.lock():
        intensity_slider.value = STACK["intensities"]
        intensity_start.value, intensity_end.value = STACK["intensities"]


@pn.depends(
    frame_slider.param.value,
    frame_start.param.value,
    frame_end.param.value,
    rt_start.param.value,
    rt_end.param.value,
    watch=True
)
def check_frames_stack(*args):
    if STACK.is_locked:
        return
    current_low, current_max = STACK["frames"]
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
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "frames", (updated_value[0], updated_value[0] + 1)
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "frames", (updated_value[1], updated_value[1] + 1)
            )
        update_frame_widgets_to_stack()
        update_global_selection(updated_option, updated_value)


def check_scans_stack():
    current_low, current_max = STACK["scans"]
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
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "scans", (updated_value[0], updated_value[0] + 1)
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "scans", (updated_value[1], updated_value[1] + 1)
            )
    return updated_option, updated_value


def check_quads_stack():
    current_low, current_max = STACK["quads"]
    updated_option, updated_value = STACK.update(
        "quads", quad_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "quads", (quad_start.value, quad_end.value)
        )
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "quads", (updated_value[0], updated_value[0])
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "quads", (updated_value[1], updated_value[1])
            )
    return updated_option, updated_value


def check_precursors_stack():
    current_low, current_max = STACK["precursors"]
    updated_option, updated_value = STACK.update(
        "precursors", precursor_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "precursors", (precursor_start.value, precursor_end.value)
        )
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "precursors", (updated_value[0], updated_value[0] + 1)
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "precursors", (updated_value[1], updated_value[1] + 1)
            )
    return updated_option, updated_value


def check_tofs_stack():
    current_low, current_max = STACK["tofs"]
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
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "tofs", (updated_value[0], updated_value[0] + 1)
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "tofs", (updated_value[1], updated_value[1] + 1)
            )
    return updated_option, updated_value


def check_intensities_stack():
    current_low, current_max = STACK["intensities"]
    updated_option, updated_value = STACK.update(
        "intensities", intensity_slider.value
    )
    if updated_value is None:
        updated_option, updated_value = STACK.update(
            "intensities", (intensity_start.value, intensity_end.value)
        )
    if updated_value is not None:
        if current_low != updated_value[0]:
            if updated_value[0] >= updated_value[1]:
                STACK.undo()
                updated_option, updated_value = STACK.update(
                    "intensities", (updated_value[0], updated_value[0])
                )
        elif updated_value[0] >= updated_value[1]:
            STACK.undo()
            updated_option, updated_value = STACK.update(
                "intensities", (updated_value[1], updated_value[1])
            )
    return updated_option, updated_value
