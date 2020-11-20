#!python

# builtin
import logging
import os
import sys
import json


BASE_PATH = os.path.dirname(__file__)
EXT_PATH = os.path.join(BASE_PATH, "ext")
LIB_PATH = os.path.join(BASE_PATH, "lib")
LOG_PATH = os.path.join(os.path.dirname(BASE_PATH), "logs")
with open(os.path.join(LIB_PATH, "interface_parameters.json"), "r") as in_file:
    INTERFACE_PARAMETERS = json.load(in_file)
MAX_THREADS = INTERFACE_PARAMETERS["threads"]["default"]
PROGRESS_CALLBACK_STYLE_NONE = 0
PROGRESS_CALLBACK_STYLE_TEXT = 1
PROGRESS_CALLBACK_STYLE_PLOT = 2
PROGRESS_CALLBACK_STYLE = PROGRESS_CALLBACK_STYLE_TEXT


def set_logger(*, log_file_name="", stream=sys.stdout, log_level=logging.INFO):
    import time
    root = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-s - %(message)s', "%Y-%m-%d %H:%M:%S"
    )
    root.setLevel(log_level)
    while root.hasHandlers():
        root.removeHandler(root.handlers[0])
    if stream is not None:
        stream_handler = logging.StreamHandler(stream)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
    if log_file_name is not None:
        if log_file_name == "":
            if not os.path.exists(LOG_PATH):
                os.makedirs(LOG_PATH)
            log_file_name = LOG_PATH
        log_file_name = os.path.abspath(log_file_name)
        if os.path.isdir(log_file_name):
            current_time = time.localtime()
            current_time = "".join(
                [
                    f'{current_time.tm_year:04}',
                    f'{current_time.tm_mon:02}',
                    f'{current_time.tm_mday:02}',
                    f'{current_time.tm_hour:02}',
                    f'{current_time.tm_min:02}',
                    f'{current_time.tm_sec:02}',
                ]
            )
            log_file_name = os.path.join(
                log_file_name,
                f"log_{current_time}.txt"
            )
        directory = os.path.dirname(log_file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_handler = logging.FileHandler(log_file_name, mode="w")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def set_threads(threads, set_global=True):
    import multiprocessing
    if set_global:
        global MAX_THREADS
    max_cpu_count = multiprocessing.cpu_count()
    if threads > max_cpu_count:
        MAX_THREADS = max_cpu_count
    else:
        while threads <= 0:
            threads += max_cpu_count
        MAX_THREADS = threads
    if not set_global:
        return MAX_THREADS


def njit(*args, **kwargs):
    import numba
    return numba.njit(*args, **kwargs)


def pjit(
    _func=None,
    *,
    thread_count=None
):
    import functools
    import threading
    import numba
    import numpy as np

    def parallel_compiled_func_inner(func):
        numba_func = numba.njit(nogil=True)(func)

        def numba_func_parallel(
            iterable,
            start,
            stop,
            step,
            *args,
        ):
            if len(iterable) == 0:
                for i in range(start, stop, step):
                    numba_func(i, *args)
            else:
                for i in iterable:
                    numba_func(i, *args)
        numba_func_parallel = numba.njit(nogil=True)(numba_func_parallel)

        def wrapper(iterable, *args):
            if thread_count is None:
                current_thread_count = MAX_THREADS
            else:
                current_thread_count = set_threads(
                    thread_count,
                    set_global=False
                )
            threads = []
            for thread_id in range(current_thread_count):
                local_iterable = iterable[thread_id::current_thread_count]
                if isinstance(local_iterable, range):
                    start = local_iterable.start
                    stop = local_iterable.stop
                    step = local_iterable.step
                    local_iterable = np.array([], dtype=np.int64)
                else:
                    start = -1
                    stop = -1
                    step = -1
                thread = threading.Thread(
                    target=numba_func_parallel,
                    args=(
                        local_iterable,
                        start,
                        stop,
                        step,
                        *args
                    )
                )
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
                del thread
        return functools.wraps(func)(wrapper)
    if _func is None:
        return parallel_compiled_func_inner
    else:
        return parallel_compiled_func_inner(_func)


def set_progress_callback_style(style=None, set_global=True):
    if set_global:
        global PROGRESS_CALLBACK_STYLE
    if style is not None:
        PROGRESS_CALLBACK_STYLE = style
    if not set_global:
        return PROGRESS_CALLBACK_STYLE


def progress_callback(iterable, style=None):
    import tqdm
    if style is None:
        global PROGRESS_CALLBACK_STYLE
        style = PROGRESS_CALLBACK_STYLE
    if style == PROGRESS_CALLBACK_STYLE_NONE:
        return iterable
    elif style == PROGRESS_CALLBACK_STYLE_TEXT:
        return tqdm.tqdm(iterable)
    elif style == PROGRESS_CALLBACK_STYLE_PLOT:
        # TODO: update?
        return tqdm.gui.tqdm(iterable)
    else:
        raise ValueError("Not a valid progress callback style")


def create_hdf_group_from_dict(hdf_group, data_dict, overwrite=False):
    """
    Save dict to opened hdf_group.
    All keys are expected to be strings.
    Values are converted as follows:
        np.ndarray, pd.core.frame.DataFrame => dataset
        bool, int, float, str => attr
        dict => group (recursive)
    Nothing is overwritten, unless otherwise defined
    """
    import pandas as pd
    import numpy as np
    import h5py
    for key, value in data_dict.items():
        if not isinstance(key, str):
            raise ValueError(f"Key {key} is not a string.")
        if isinstance(value, pd.core.frame.DataFrame):
            new_dict = {key: dict(value)}
            new_dict[key]["is_pd_dataframe"] = True
            create_hdf_group_from_dict(hdf_group, new_dict, overwrite)
        elif isinstance(value, (np.ndarray, pd.core.series.Series)):
            if isinstance(value, (pd.core.series.Series)):
                value = value.values
            if overwrite and (key in hdf_group):
                del hdf_group[key]
            if key not in hdf_group:
                if value.dtype.type == np.str_:
                    value = value.astype(np.dtype('O'))
                if value.dtype == np.dtype('O'):
                    hdf_group.create_dataset(
                        key,
                        data=value,
                        dtype=h5py.string_dtype()
                    )
                else:
                    hdf_group.create_dataset(
                        key,
                        data=value,
                    )
        elif isinstance(value, (bool, int, float, str)):
            if overwrite or (key not in hdf_group.attrs):
                hdf_group.attrs[key] = value
        elif isinstance(value, dict):
            if key not in hdf_group:
                hdf_group.create_group(key)
            create_hdf_group_from_dict(hdf_group[key], value, overwrite)
        else:
            raise ValueError(
                f"The type of {key} is {type(value)}, which "
                "cannot be converted to an HDF value."
            )


def create_dict_from_hdf_group(hdf_group):
    import h5py
    import pandas as pd
    import numpy as np
    result = {}
    for key in hdf_group.attrs:
        value = hdf_group.attrs[key]
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.float64):
            result[key] = float(value)
        elif isinstance(value, (str, bool)):
            result[key] = value
        else:
            raise ValueError(
                f"The type of {key} is {type(value)}, which "
                "cannot be converted properly."
            )
    for key in hdf_group:
        subgroup = hdf_group[key]
        if isinstance(subgroup, h5py.Dataset):
            result[key] = subgroup[:]
        else:
            if "is_pd_dataframe" in subgroup.attrs:
                result[key] = pd.DataFrame(
                    {
                        column: subgroup[column][:] for column in sorted(
                            subgroup
                        )
                    }
                )
            else:
                result[key] = create_dict_from_hdf_group(hdf_group[key])
    return result
