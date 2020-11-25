#!python


# builtin
import contextlib
# external
import click
# local
import alphatims
import alphatims.utils


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def parse_args_with_default_help(self, ctx, args):
    if not args:
        args = ["-h"]
    return self.original_parse_args(ctx, args)


click.Command.original_parse_args = click.Command.parse_args
click.Command.parse_args = parse_args_with_default_help


@contextlib.contextmanager
def cli_logging(command_name, **kwargs):
    import logging
    import time
    import platform
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
        logging.info("Detected platform:")
        logging.info(f"System    - {platform.system()}")
        logging.info(f"Release   - {platform.release()}")
        logging.info(f"Version   - {platform.version()}")
        logging.info(f"Machine   - {platform.machine()}")
        logging.info(f"Processor - {platform.processor()}")
        logging.info("")
        logging.info(
            f"Running command `alphatims {command_name}` with parameters:"
        )
        max_len = max(len(key) for key in kwargs)
        for key, value in sorted(kwargs.items()):
            logging.info(f"{key:<{max_len}} - {value}")
        logging.info("")
        yield
    except Exception:
        logging.exception("Something went wrong, execution incomplete!")
    else:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds"
        )
    finally:
        alphatims.utils.set_logger(log_file_name=None)


def cli_option(parameter_name):
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


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(alphatims.__version__)
def run(**kwargs):
    alphatims.utils.set_logger()
    pass


@run.command("gui", help="Start graphical user interface.")
def gui():
    import alphatims.gui
    alphatims.gui.run()


@run.command("convert")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
# @cli_option("no_log_stream")
def convert(**kwargs):
    """Convert raw data to an HDF file."""
    with cli_logging("convert", **kwargs):
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(kwargs["bruker_d_folder"])
        data.save_as_hdf(
            overwrite=True,
            directory=kwargs["output_folder"]
        )


@run.group(
    "detect",
    help="Detect data structures.",
    context_settings=CONTEXT_SETTINGS
)
def detect(**kwargs):
    pass


@detect.command("ions", help="Detect ions (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
# @cli_option("no_log_stream")
def detect_ions(**kwargs):
    with cli_logging("detect ions", **kwargs):
        raise NotImplementedError

@detect.command("features", help="Detect features (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
# @cli_option("no_log_stream")
def detect_features(**kwargs):
    with cli_logging("detect features", **kwargs):
        raise NotImplementedError


@detect.command("analytes", help="Detect analytes (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
# @cli_option("no_log_stream")
def detect_analytes(**kwargs):
    with cli_logging("detect analytes", **kwargs):
        raise NotImplementedError
