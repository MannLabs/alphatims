#!python

# external
import click
# local
import alphatims.utils


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def run_cli():
    alphatims.utils.set_logger()
    cli_overview.add_command(cli_start_gui)
    cli_overview()


def run_gui(**kwargs):
    raise NotImplementedError


@click.group(
    context_settings=CONTEXT_SETTINGS,
)
def cli_overview():
    pass


@click.command(
    "gui",
    help="Start graphical user interface.",
)
def cli_start_gui(**kwargs):
    run_gui(**kwargs)
