#!python

# external
import click
# local
import alphatims.utils


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def run():
    alphatims.utils.set_logger()
    overview.add_command(gui_command)
    overview.add_command(convert_command)
    overview()


@click.group(context_settings=CONTEXT_SETTINGS)
def overview(**kwargs):
    pass


@click.command("gui", help="Start graphical user interface.")
def gui_command():
    import alphatims.gui
    alphatims.gui.run()


@click.command("convert", help="Convert raw data to an HDF file.")
def convert_command():
    raise NotImplementedError
