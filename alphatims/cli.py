#!python


# builtin
import contextlib
# external
import click
# local
import alphatims
import alphatims.utils


@contextlib.contextmanager
def parse_cli_parameters(command_name, **kwargs):
    import logging
    import time
    try:
        start_time = time.time()
        if "threads" not in kwargs:
            kwargs["threads"] = alphatims.utils.INTERFACE_PARAMETERS[
                "threads"
            ]["default"]
        kwargs["threads"] = alphatims.utils.set_threads(
            kwargs["threads"]
        )
        if "log_file" not in kwargs:
            kwargs["log_file"] = alphatims.utils.INTERFACE_PARAMETERS[
                "log_file"
            ]["default"]
        if "no_log_stream" not in kwargs:
            kwargs["no_log_stream"] = alphatims.utils.INTERFACE_PARAMETERS[
                "no_log_stream"
            ]["default"]
        kwargs["log_stream"] = not kwargs.pop("no_log_stream")
        kwargs["log_file"] = alphatims.utils.set_logger(
            log_file_name=kwargs["log_file"],
            stream=kwargs["log_stream"],
        )
        alphatims.utils.show_platform_info()
        alphatims.utils.show_python_info()
        if kwargs:
            logging.info(
                f"Running command `alphatims {command_name}` with parameters:"
            )
            max_len = max(len(key) for key in kwargs)
            for key, value in sorted(kwargs.items()):
                logging.info(f"{key:<{max_len}} - {value}")
        else:
            logging.info(f"Running command `alphatims {command_name}`.")
        logging.info("")
        yield kwargs
    except Exception:
        logging.exception("Something went wrong, execution incomplete!")
    else:
        logging.info(
            f"Analysis done in {time.time() - start_time:.2f} seconds"
        )
    finally:
        alphatims.utils.set_logger(log_file_name=None)


def cli_option(parameter_name):
    names = [parameter_name]
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
        names.append(parameters.pop("short_name"))
    return click.option(
        f"--{parameter_name}",
        **parameters,
    )


@click.group(
    context_settings=dict(
        help_option_names=['-h', '--help'],
    ),
    invoke_without_command=True
)
@click.pass_context
@click.version_option(alphatims.__version__, "-v", "--version")
def run(ctx):
    click.echo("************************")
    click.echo(f"* AlphaTims {alphatims.__version__} *")
    click.echo("************************")
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui():
    with parse_cli_parameters("gui") as parameters:
        import logging
        logging.info("Loading GUI..")
        import alphatims.gui
        alphatims.gui.run()


@run.command("convert")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("no_log_stream")
def convert(**kwargs):
    """Convert raw data to an HDF file."""
    with parse_cli_parameters("convert", **kwargs) as parameters:
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(parameters["bruker_d_folder"])
        data.save_as_hdf(
            overwrite=True,
            directory=parameters["output_folder"]
        )


@run.group("detect", help="Detect data structures.")
def detect(**kwargs):
    pass


@detect.command("ions", help="Detect ions (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("no_log_stream")
def detect_ions(**kwargs):
    with parse_cli_parameters("detect ions", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("features", help="Detect features (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("no_log_stream")
def detect_features(**kwargs):
    with parse_cli_parameters("detect features", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("analytes", help="Detect analytes (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("no_log_stream")
def detect_analytes(**kwargs):
    with parse_cli_parameters("detect analytes", **kwargs) as parameters:
        raise NotImplementedError
