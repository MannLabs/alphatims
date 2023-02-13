# builtin
import ast
import textwrap
import inspect
import types
import weakref

# external
import numba
import pandas as pd
import numpy as np


def precompile_njit_functions_from_object(object_):
    if not is_regular_object_with_dict(object_):
        return
    create_njit_module_for_object(object_)
    for func in iterate_over_callables_from_object(object_):
        try:
            tree = get_source_tree_from_callable(func)
        except OSError:
            pass
        else:
            if tree_has_decorator_containing_name(tree, "njit"):
                add_njit_function_to_object_njit_module(object_, func, tree)
                overwrite_object_function_with_njit_function(object_, func)


def is_regular_object_with_dict(object_):
    if not hasattr(object_, "__dict__"):
        return False
    if isinstance(object_, numba.core.registry.CPUDispatcher):
        return False
    return True


def create_njit_module_for_object(object_):
    module = create_module(object_)
    module.__dict__.update(inspect.getmodule(object_).__dict__)
    module.self = module
    object.__setattr__(object_, "__njit__", module)


def create_module(x):
    if has_njit_module(x):
        return x.__njit__
    if is_pandas_dataframe(x):
        return create_module_from_dataframe(x)
    elif is_regular_object_with_dict(x):
        return create_module_from_object(x)
    else:
        return x


def is_module(x):
    return hasattr(x, "__njit__")


def has_njit_module(x):
    return isinstance(x, types.ModuleType)


def is_pandas_dataframe(x):
    return isinstance(x, pd.DataFrame)


def create_module_from_dataframe(df):
    module = types.ModuleType(f"{df.__class__}_{id(df)}")
    for column in df.columns:
        module.__dict__[column] = np.array(df[column].values, copy=False)
    for key, value in df.__dict__.items():
        if not key.startswith("_"):
            module.__dict__[key] = create_module(value)
    return module


def create_module_from_object(object_):
    module = types.ModuleType(f"{object_.__class__}_{id(object_)}")
    for key, value in object_.__dict__.items():
        if not key.startswith("__"):
            module.__dict__[key] = create_module(value)
    return module


def iterate_over_callables_from_object(object_):
    for key in list(dir(object_)):
        if key.startswith("__"):
            continue
        potential_func = eval(f"object_.{key}")
        if callable(potential_func):
            yield potential_func


def get_source_tree_from_callable(func):
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    return ast.parse(src)


def tree_has_decorator_containing_name(tree, name):
    for decorator in tree.body[0].decorator_list:
        if decorator_contains(decorator, name):
            return True
    return False


def decorator_contains(decorator, name):
    # TODO: Should be properly parsed!
    if name in ast.unparse(decorator):
        return True
    return False


def add_njit_function_to_object_njit_module(object_, func, tree):
    src = create_src_without_self_and_decorators_from_function_tree(tree)
    exec(src, object_.__njit__.__dict__)
    nogil = tree_has_decorator_containing_name(tree, "nogil")
    func_ = numba.njit(nogil=nogil)(object_.__njit__.__dict__[func.__name__])
    src = f"object.__setattr__(object_.__njit__, '{func.__name__}', func_)"
    exec(src)


def create_src_without_self_and_decorators_from_function_tree(tree):
    # TODO: removes the nogil decorator as well!
    origonal_decorators = tree.body[0].decorator_list
    origonal_args = tree.body[0].args.args
    tree.body[0].decorator_list = []
    tree.body[0].args.args = tree.body[0].args.args[1:]
    src = ast.unparse(tree)
    tree.body[0].decorator_list = origonal_decorators
    tree.body[0].args.args = origonal_args
    return src


def overwrite_object_function_with_njit_function(object_, func):
    src = f"object.__setattr__(object_, '{func.__name__}', object_.__njit__.{func.__name__})"
    exec(src)


def njit(*args, **kwargs):
    return numba.njit(*args, **kwargs)


# def njit_class(_func=None, dataclass=False, **decorator_kwargs):
#     import functools
#     def wrapper(_func):
#         @functools.wraps(_func)
#         def inner_func(*func_args, **func_kwargs):
#             _object = _func(*func_args,  **func_kwargs)
#             precompile_njit_functions_from_object(_object)
#             return _object
#         return inner_func
#     if _func is None:
#         return wrapper
#     else:
#         return wrapper(_func)


def njit_class(_cls=None, njit=True):
    def wrapper(_cls):
        def __blank_post_init__(self):
            if hasattr(super(self.__class__.__mro__[self.__class_index__], self), "__post_init__"):
                super(self.__class__.__mro__[self.__class_index__], self).__post_init__()
        def __post_init__(self):
            if not hasattr(self, "__class_index__"):
                object.__setattr__(self, "__class_index__", -1)
            object.__setattr__(self, "__class_index__", self.__class_index__ + 1)
            self.__class__.__mro__[self.__class_index__].__original_post_init__(self)
            object.__setattr__(self, "__class_index__", self.__class_index__ - 1)
            if self.__class_index__ == -1:
                if njit and not hasattr(self, "__njit__"):
                    self.precompile_njit_functions_from_object()
                del self.__dict__["__class_index__"]
        _cls.precompile_njit_functions_from_object = precompile_njit_functions_from_object
        for _super_cls in _cls.__mro__[:-1][::-1]:
            has_post_init = ("__post_init__" in _super_cls.__dict__)
            if not has_post_init:
                _super_cls.__post_init__ = __blank_post_init__
            has_original_post_init = ("__original_post_init__" in _super_cls.__dict__)
            if not has_original_post_init:
                _super_cls.__original_post_init__ = _super_cls.__post_init__
                _super_cls.__post_init__ = __post_init__
        return _cls
    if _cls is None:
        return wrapper
    else:
        return wrapper(_cls)



# def njit_class(_cls=None, dataclass=False, njit=True, hdf=True, **decorator_kwargs):
#     import functools
#     def wrapper(_cls):
#         def __blank_post_init__(self):
#             if hasattr(super(self.__class__.__mro__[self.__class_index__], self), "__post_init__"):
#                 super(self.__class__.__mro__[self.__class_index__], self).__post_init__()
#         def __post_init__(self):
#             if not hasattr(self, "__class_index__"):
#                 object.__setattr__(self, "__class_index__", -1)
#             object.__setattr__(self, "__class_index__", self.__class_index__ + 1)
#             self.__class__.__mro__[self.__class_index__].__original_post_init__(self)
#             object.__setattr__(self, "__class_index__", self.__class_index__ - 1)
#             if self.__class_index__ == -1:
#                 if njit and not hasattr(self, "__njit__"):
#                     tdf2ms2.utils.compiling.precompile_njit_functions_from_object(self)
#                 if hdf and not hasattr(self, "__hdf__"):
#                     save_to_hdf(self)
#                 del self.__dict__["__class_index__"]
#         for _super_cls in _cls.__mro__[:-1][::-1]:
#             has_post_init = ("__post_init__" in _super_cls.__dict__)
#             if not has_post_init:
#                 _super_cls.__post_init__ = __blank_post_init__
#             has_original_post_init = ("__original_post_init__" in _super_cls.__dict__)
#             if not has_original_post_init:
#                 _super_cls.__original_post_init__ = _super_cls.__post_init__
#                 _super_cls.__post_init__ = __post_init__
#         return dataclasses.dataclass(**decorator_kwargs)(_cls)
#     if _cls is None:
#         return wrapper
#     else:
#         return wrapper(_cls)


# def save_to_hdf(self, file_name=None):
#     if file_name is None:
#         file_name = f"sandbox/{hash(self)}.hdf"
#     object.__setattr__(self, "__hdf__", file_name)
#     print(f"Saving object results to {file_name}.")
#     import tdf2ms2.utils.hdf
#     import inspect
#     hdf = tdf2ms2.utils.hdf.HDF_File(
#         file_name,
#         read_only=False,
#         truncate=True,
#     )
#     for key, value in self.__dict__.items():
#         if not callable(value):
#             if hasattr(value, "__hdf__"):
#                 hdf.__setattr__(key, value.__hdf__)
#             elif inspect.ismodule(value):
#                 continue
#             else:
#                 hdf.__setattr__(key, value)
