#!python


# builtin
import contextlib
import os
# external
import click
# local
import alphatims
import alphatims.utils


@contextlib.contextmanager
def parse_cli_settings(command_name, **kwargs):
    import logging
    import time
    try:
        start_time = time.time()
        kwargs = {key: arg for key, arg in kwargs.items() if arg is not None}
        if ("parameter_file" in kwargs):
            kwargs["parameter_file"] = os.path.abspath(
                kwargs["parameter_file"]
            )
            parameters = alphatims.utils.load_parameters(
                kwargs["parameter_file"]
            )
            kwargs.update(parameters)
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
                f"Running CLI command `alphatims "
                f"{command_name}` with parameters:"
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


def cli_option(parameter_name, **kwargs):
    names = [parameter_name]
    parameters = alphatims.utils.INTERFACE_PARAMETERS[parameter_name]
    parameters.update(kwargs)
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
def run(ctx, **kwargs):
    click.echo("************************")
    click.echo(f"* AlphaTims {alphatims.__version__} *")
    click.echo("************************")
    if ctx.invoked_subcommand is None:
        click.echo(run.get_help(ctx))


@run.command("gui", help="Start graphical user interface.")
def gui():
    with parse_cli_settings("gui") as parameters:
        import logging
        logging.info("Loading GUI..")
        import alphatims.gui
        alphatims.gui.run()


@run.group("export", help="Export information.")
def export(**kwargs):
    pass


@run.group("detect", help="Detect data structures.")
def detect(**kwargs):
    pass


@export.command("raw_as_hdf", help="Export raw file as hdf file.")
@cli_option("bruker_d_folder")
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("no_log_stream")
@cli_option("parameter_file")
def export_raw_as_hdf(**kwargs):
    with parse_cli_settings("export raw_as_hdf", **kwargs) as parameters:
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(parameters["bruker_d_folder"])
        data.save_as_hdf(
            overwrite=True,
            directory=parameters["output_folder"]
        )


@export.command("parameters", help="Export (non-required) parameters as json")
@cli_option(
    "parameter_file",
    required=True,
    type={
      "name": "path",
      "dir_okay": False,
    }
)
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("no_log_stream")
def export_parameters(**kwargs):
    import json
    kwargs["parameter_file"] = os.path.abspath(kwargs["parameter_file"])
    with open(kwargs["parameter_file"], "w") as truncated_file:
        json.dump({}, truncated_file, indent=4, sort_keys=True)
    with parse_cli_settings("export parameters", **kwargs) as parameters:
        parameter_file_name = parameters.pop("parameter_file")
        alphatims.utils.save_parameters(
            parameter_file_name,
            parameters
        )


@detect.command("ions", help="Detect ions (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("no_log_stream")
@cli_option("parameter_file")
def detect_ions(**kwargs):
    with parse_cli_settings("detect ions", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("features", help="Detect features (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("no_log_stream")
@cli_option("parameter_file")
def detect_features(**kwargs):
    with parse_cli_settings("detect features", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("analytes", help="Detect analytes (NotImplemented yet).")
@cli_option("bruker_d_folder")
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("no_log_stream")
@cli_option("parameter_file")
def detect_analytes(**kwargs):
    with parse_cli_settings("detect analytes", **kwargs) as parameters:
        raise NotImplementedError
