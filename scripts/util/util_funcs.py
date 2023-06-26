# misc utility functions for autoMBW

# following can only deal with 2 layers of depth or requires import

# Code copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
"""A LazyLoader class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import types

class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies.

  `contrib`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """

  # The lint error here is incorrect.
  def __init__(self, local_name, parent_module_globals, name):  # pylint: disable=super-on-old-class
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals

    super(LazyLoader, self).__init__(name)

  def _load(self):
    # Import the target module and insert it into the parent's namespace
    module = importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)
# end code copy

class LazyDict(dict):
    def __getitem__(self, item):
        value = dict.__getitem__(self, item)
        if type(value) is tuple:
            if callable(value[0]) and type(value[1]) is str:
                function, arg_str = value
                value = eval("function(" + arg_str + ")")
                dict.__setitem__(self, item, value)
        return value

def grouped(iterable, n):
  return zip(*[iter(iterable)]*n)

#context managers (with)
import os, sys
from contextlib import contextmanager

@contextmanager
def change_dir(newdir):
    olddir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(olddir)

@contextmanager
def extend_path(newpath):
    sys.path.append(newpath)
    try:
        yield
    finally:
        sys.path.remove(newpath)

