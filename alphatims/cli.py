#!python


# builtin
import contextlib
import os
import logging
# external
import click
# local
import alphatims
import alphatims.utils


@contextlib.contextmanager
def parse_cli_settings(command_name: str, **kwargs):
    """A context manager that parses and logs CLI settings.

    Parameters
    ----------
    command_name : str
        The name of the command that utilizes these CLI settings.
    **kwargs
        All values that need to be logged.
        Values (if included) that explicitly will be parsed are:
            output_folder
            parameter_file
            threads
            log_file
            disable_log_stream

    Returns
    -------
    : dict
        A dictionary with parsed parameters.
    """
    import time
    try:
        start_time = time.time()
        kwargs = {key: arg for key, arg in kwargs.items() if arg is not None}
        if "output_folder" in kwargs:
            if kwargs["output_folder"] is not None:
                kwargs["output_folder"] = os.path.abspath(
                    kwargs["output_folder"]
                )
                if not os.path.exists(kwargs["output_folder"]):
                    os.makedirs(kwargs["output_folder"])
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
        if "disable_log_stream" not in kwargs:
            kwargs[
                "disable_log_stream"
            ] = alphatims.utils.INTERFACE_PARAMETERS[
                "disable_log_stream"
            ]["default"]
        kwargs["log_file"] = alphatims.utils.set_logger(
            log_file_name=kwargs["log_file"],
            stream=not kwargs["disable_log_stream"],
        )
        alphatims.utils.show_platform_info()
        alphatims.utils.show_python_info()
        alphatims.utils.check_github_version()
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


def cli_option(
    parameter_name: str,
    as_argument: bool = False,
    **kwargs
):
    """A wrapper for click.options and click.arguments using local defaults.

    Parameters
    ----------
    parameter_name : str
        The name of the parameter or argument.
        It's default values need to be present in
        lib/interface_parameters.json.
    as_argument : bool
        If True, a click.argument is returned.
        If False, a click.option is returned.
        Default is False.
    **kwargs
        Items that overwrite the default values of
        lib/interface_parameters.json.
        These need to be valid items for click.
        A special "type" dict can be used to pass a click.Path or click.Choice,
        that has the following format:
        type = {"name": "path" or "choice", **choice_or_path_kwargs}

    Returns
    -------
    : click.option, click.argument
        A click.option or click.argument decorator.
    """
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
    if not as_argument:
        return click.option(
            f"--{parameter_name}",
            **parameters,
        )
    else:
        return click.argument(
            parameter_name,
            type=parameters["type"],
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
    with parse_cli_settings("gui"):
        logging.info("Loading GUI..")
        import alphatims.gui
        alphatims.gui.run()


@run.group("export", help="Export information.")
def export(**kwargs):
    pass


@run.group("detect", help="Detect data structures.")
def detect(**kwargs):
    pass


@export.command("hdf", help="Export BRUKER_D_FOLDER as hdf file.")
@cli_option("bruker_d_folder", as_argument=True)
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("disable_log_stream")
@cli_option("parameter_file")
@cli_option("compress")
def export_hdf(**kwargs):
    with parse_cli_settings("export hdf", **kwargs) as parameters:
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(parameters["bruker_d_folder"])
        if "output_folder" not in parameters:
            directory = data.directory
        else:
            directory = parameters["output_folder"]
        data.save_as_hdf(
            overwrite=True,
            directory=directory,
            file_name=f"{data.sample_name}.hdf",
            compress=parameters["compress"],
        )


@export.command(
    "parameters",
    help="Export (non-required) parameters as PARAMETER_FILE"
)
@cli_option(
    "parameter_file",
    as_argument=True,
    type={
      "name": "path",
      "dir_okay": False,
    }
)
@cli_option("threads")
@cli_option("log_file")
@cli_option("output_folder")
@cli_option("disable_log_stream")
def export_parameters(**kwargs):
    kwargs["parameter_file"] = os.path.abspath(kwargs["parameter_file"])
    alphatims.utils.save_parameters(kwargs["parameter_file"], {})
    with parse_cli_settings("export parameters", **kwargs) as parameters:
        parameter_file_name = parameters.pop("parameter_file")
        alphatims.utils.save_parameters(
            parameter_file_name,
            parameters
        )


@export.command(
    "slice",
    help="Load a BRUKER_D_FOLDER and export a data slice to a csv file."
)
@cli_option(
    "bruker_d_folder",
    as_argument=True,
    type={
        "name": "path",
        "exists": True,
        "file_okay": True,
        "dir_okay": True
    }
)
@cli_option("slice")
@cli_option("slice_file")
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("disable_log_stream")
@cli_option("parameter_file")
def export_slice(**kwargs):
    with parse_cli_settings("export slice", **kwargs) as parameters:
        slice = parameters["slice"]
        import alphatims.bruker
        data = alphatims.bruker.TimsTOF(parameters["bruker_d_folder"])
        logging.info(f"Slicing data with slice '{slice}'")
        # TODO: Slicing with eval is very unsafe!
        # TODO: update help function
        slice_function = eval(f"lambda d: d[{slice}]")
        data_slice = slice_function(data)
        if "slice_file" not in parameters:
            parameters["slice_file"] = f"{data.sample_name}_slice.csv"
        if "output_folder" not in parameters:
            output_folder = data.directory
        else:
            output_folder = parameters["output_folder"]
        output_file_name = os.path.join(
            output_folder,
            parameters["slice_file"]
        )
        logging.info(
            f"Saving {len(data_slice)} datapoints to {output_file_name}"
        )
        data_slice.to_csv(output_file_name, index=False)


@detect.command("ions", help="Detect ions (NotImplemented yet).")
@cli_option("bruker_d_folder", as_argument=True)
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("disable_log_stream")
@cli_option("parameter_file")
def detect_ions(**kwargs):
    with parse_cli_settings("detect ions", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("features", help="Detect features (NotImplemented yet).")
@cli_option("bruker_d_folder", as_argument=True)
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("disable_log_stream")
@cli_option("parameter_file")
def detect_features(**kwargs):
    with parse_cli_settings("detect features", **kwargs) as parameters:
        raise NotImplementedError


@detect.command("analytes", help="Detect analytes (NotImplemented yet).")
@cli_option("bruker_d_folder", as_argument=True)
@cli_option("output_folder")
@cli_option("log_file")
@cli_option("threads")
@cli_option("disable_log_stream")
@cli_option("parameter_file")
def detect_analytes(**kwargs):
    with parse_cli_settings("detect analytes", **kwargs) as parameters:
        raise NotImplementedError
