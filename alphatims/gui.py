#!python

# external
import panel as pn
import time
# local
import alphatims.utils


FRAME_LIMITS = (0, 10000)
SCAN_LIMITS = (0, 927)
QUAD_LIMITS = (-1, 2000)
TOF_LIMITS = (100, 1700)
CONTINUE_RUNNING = True


button = pn.widgets.Button(name='Exit me', button_type='primary')
frame_slider = pn.widgets.IntRangeSlider(
    name='Frame Range Slider',
    start=FRAME_LIMITS[0],
    end=FRAME_LIMITS[1],
    value=(2, 8),
    step=1
)


def run():
    global CONTINUE_RUNNING
    button.on_click(button_event)  # TODO @pn.depends does not work?
    layout = pn.Column(
        button,
        frame_slider,
    )
    server = layout.show(threaded=True)
    while CONTINUE_RUNNING:
        time.sleep(1)
    server.stop()


@pn.depends(frame_slider.param.value)
def frame_slider_event(values):
    print(values)


@pn.depends(button.param.clicks)
def button_event(event):
    global CONTINUE_RUNNING
    print("click!")
    CONTINUE_RUNNING = False
