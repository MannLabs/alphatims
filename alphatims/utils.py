#!python

# builtin
import logging
import os
import json


BASE_PATH = os.path.dirname(__file__)
EXT_PATH = os.path.join(BASE_PATH, "ext")
LIB_PATH = os.path.join(BASE_PATH, "lib")
LOG_PATH = os.path.join(os.path.dirname(BASE_PATH), "logs")
with open(os.path.join(LIB_PATH, "interface_parameters.json"), "r") as in_file:
    INTERFACE_PARAMETERS = json.load(in_file)
MAX_THREADS = INTERFACE_PARAMETERS["threads"]["default"]


def set_logger(*, log_file_name="", stream=True, log_level=logging.INFO):
    import time
    import sys
    root = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-s - %(message)s', "%Y-%m-%d %H:%M:%S"
    )
    while root.hasHandlers():
        root.removeHandler(root.handlers[0])
    if stream:
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    if log_file_name is not None:
        if log_file_name == "":
            if not os.path.exists(LOG_PATH):
                os.makedirs(LOG_PATH)
            log_file_name = LOG_PATH
        log_file_name = os.path.abspath(log_file_name)
        if os.path.isdir(log_file_name):
            current_time = "".join([str(i) for i in time.localtime()])
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
