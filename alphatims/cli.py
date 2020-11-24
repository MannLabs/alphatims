#!python


# builtin
import contextlib
# external
import click
# local
import alphatims
import alphatims.utils


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@contextlib.contextmanager
def cli_logging(command_name, **kwargs):
    import logging
    import time
    try:
        start_time = time.time()
        if "threads" in kwargs:
            kwargs["threads"] = alphatims.utils.set_threads(
                kwargs["threads"]
            )
        if ("log_file" in kwargs):
            alphatims.utils.set_logger(
                log_file_name=kwargs["log_file"],
            )
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.FileHandler):
                    kwargs["log_file"] = handler.baseFilename
        logging.info("************************")
        logging.info(f"* AlphaTims {alphatims.__version__} *")
        logging.info("************************")
        logging.info("")
        logging.info(
            f"Running command `alphatims {command_name}` with parameters:"
        )
        max_len = max(len(key) for key in kwargs)
        for key, value in sorted(kwargs.items()):
            logging.info(f"{key:<{max_len}} - {value}")
        logging.info("")
        yield
    finally:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds"
        )
        alphatims.utils.set_logger(log_file_name=None)


def click_option(parameter_name):
    parameters = alphatims.utils.INTERFACE_PARAMETERS[parameter_name]
    if "type" in parameters:
        if parameters["type"] == "int":
            parameters["type"] = int
        elif parameters["type"] == "float":
            parameters["type"] = float
        elif parameters["type"] == "str":
            parameters["type"] = str
        elif isinstance(parameters["type"], dict):
            parameter_type = parameters["type"].pop("name")
            if parameter_type == "path":
                parameters["type"] = click.Path(**parameters["type"])
            elif parameter_type == "choice":
                parameters["type"] = click.Choice(**parameters["type"])
    if "default" in parameters:
        parameters["show_default"] = True
    if "short_name" in parameters:
        short_name = parameters.pop("short_name")
        return click.option(
            short_name,
            f"--{parameter_name}",
            **parameters,
        )
    else:
        return click.option(
            f"--{parameter_name}",
            **parameters,
        )


def run():
    alphatims.utils.set_logger()
    overview.add_command(gui_command)
    overview.add_command(convert_command)
    overview.add_command(featurefind_command)
    overview()


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(alphatims.__version__)
def overview(**kwargs):
    pass


@click.command("gui", help="Start graphical user interface.")
def gui_command():
    import alphatims.gui
    alphatims.gui.run()


@click.command("convert", help="Convert raw data to an HDF file.")
@click_option("bruker_d_folder")
@click_option("threads")
@click_option("log_file")
@click_option("output_folder")
# @click_option("no_log_stream")
def convert_command(**kwargs):
    with cli_logging("convert", **kwargs):
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(kwargs["bruker_d_folder"])
        data.save_as_hdf(
            overwrite=True,
            directory=kwargs["output_folder"]
        )


@click.command("featurefind", help="Find features (NotImplemented yet).")
@click_option("bruker_d_folder")
@click_option("threads")
@click_option("log_file")
@click_option("output_folder")
# @click_option("no_log_stream")
def featurefind_command(**kwargs):
    with cli_logging("convert", **kwargs):
        raise NotImplementedError
