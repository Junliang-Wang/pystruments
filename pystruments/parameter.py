from copy import deepcopy
from decimal import Decimal

import numpy as np


class Parameter(object):
    """
    This class stores any kind of information from experimental data

    ...

    Parameters
    ----------
    name : str
        name of the dataset
    value : not defined
        stored value
    unit : str, optional
        unit of the value (default is 'a.u.')
    metadata : dict, optional
        extra information (default is {})
    Attributes
    ----------
    name : str
        name of the dataset
    value : not defined
        stored value
    unit : str
        unit of the value (default is 'a.u.')
    metadata : dict
        extra information (default is {})
    """

    def __init__(self, name, value, unit='a.u.', metadata=None,
                 min_value=None, max_value=None,
                 min_step=None, max_step=None, size_step=None,
                 valid_values=None, not_valid_values=None,
                 valid_types=None, valid_multiples=None):
        if metadata is None:
            metadata = {}
        self.name = str(name)
        self.unit = unit
        self.metadata = metadata
        self.min_value = min_value
        self.max_value = max_value
        self.min_step = min_step
        self.max_step = max_step
        self.size_step = size_step
        self.valid_values = valid_values
        self.not_valid_values = not_valid_values
        self.valid_types = valid_types
        self.valid_multiples = valid_multiples

        self._sweep_values_generator = lambda pts: self.value
        self.dim = None
        self.pts = 1
        self.set_value(value)

    def __repr__(self):
        cname = self.__class__.__name__
        out = '{}: {}'.format(cname, str(self))
        return out

    def __str__(self):
        out = '{} = {} ({})'.format(self.name, self.value, self.unit)
        if self.dim is not None:
            out += ', dim = {}'.format(self.dim)
        return out

    def print_info(self):
        attrs = [
            'name', 'unit', 'value', 'dim', 'min_value', 'max_value', 'valid_values', 'not_valid_values',
            'valid_types', 'valid_multiples', 'min_step', 'max_step', 'size_step',
        ]
        values = []
        for attr in attrs:
            value = getattr(self, attr)
            values.append('{}: {}'.format(attr, value))
        info = '\n'.join(values)
        print(info)

    def get_dict(self):
        d = {}
        keys = ['name', 'unit', 'dim', 'metadata']
        for key in keys:
            if hasattr(self, key):
                d[key] = getattr(self, key)
        if isinstance(self.value, np.ndarray):
            value = self.value.tolist()
        else:
            value = self.value
        d['value'] = value
        return d

    def set_value(self, value):
        self.check_value(value)
        self.value = value

    def get_value(self, pts=None):
        if pts is not None:
            self.set_pts(pts)
        return self.value

    def set_pts(self, pts):
        pts = int(pts)
        if pts <= 0:
            raise ValueError('pts must be > 0')
        self.pts = pts
        v = self._sweep_values_generator(self.pts)
        self.set_value(v)

    def set_dim(self, dim):
        if dim is not None:
            dim = int(dim)
            if dim < 1:
                raise ValueError('dim cannot be < 1')
        self.dim = dim

    def is_static(self):
        return self.dim is None

    def static(self, value):
        self._sweep_values_generator = lambda pts: value
        self.dim = None
        self.pts = 1
        self.set_value(value)
        return self

    def sweep_values(self, values, dim):
        values = np.array(values)

        def f(pts):
            if values.size != pts:
                raise ValueError('Parameter "{}" values size != pts'.format(self.name))
            else:
                return values

        self._sweep_values_generator = f
        self.set_dim(dim)
        return self

    def sweep_linear(self, init, final, dim):
        def f(pts):
            values = np.linspace(init, final, pts)
            return values

        self._sweep_values_generator = f
        self.set_dim(dim)
        return self

    def sweep_log(self, init, final, dim):
        def f(pts):
            values = np.logspace(init, final, pts)
            return values

        self._sweep_values_generator = f
        self.set_dim(dim)
        return self

    def sweep_stepsize(self, init, step_size, dim):
        def f(pts):
            values = np.arange(init, init + pts * step_size, step_size)
            return values

        self._sweep_values_generator = f
        self.set_dim(int(dim))
        return self

    def check_value(self, value):
        self._check_value_limits(value)
        self._check_valid_types(value)
        self._check_valid_multiples(value)
        self._check_valid_values(value)
        self._check_step_limits(value)

    def _check_value_limits(self, value):
        valid = verify_min_max(value, self.min_value, self.max_value)
        if not valid:
            raise ValueError(
                'Parameter "{}": value limit exceeded (min:{}, max:{})'.format(self.name, self.min_value,
                                                                               self.max_value))
        return valid

    def _check_step_limits(self, value):
        if self.is_static():
            return True

        value_steps = np.ediff1d(value)
        if value_steps.size == 0:
            return True

        valid_lim = verify_min_max(value_steps, self.min_step, self.max_step)
        if not valid_lim:
            raise ValueError(
                'Parameter "{}": step limit exceeded (min:{}, max:{})'.format(self.name, self.min_step, self.max_step))

        if self.size_step is None:
            return True

        valid_size = True
        for step in value_steps:
            if step % self.size_step != 0:
                raise ValueError('Parameter "{}": step size must be multiple of {}'.format(self.name, self.size_step))
        valid = valid_lim and valid_size
        return valid

    def _check_valid_values(self, value):
        valid = True
        arr = np.array(value)
        if not arr.shape:
            arr = np.array([arr])
        if self.valid_values is not None:
            for value in arr:
                if value not in self.valid_values:
                    valid = False

        if self.not_valid_values is not None:
            for value in self.not_valid_values:
                if value in arr:
                    valid = False
        if not valid:
            raise ValueError('Parameter "{}": Not valid value found'.format(self.name))
        return valid

    def _check_valid_types(self, value):
        if self.valid_types is None:
            return True
        if not isinstance(value, tuple(self.valid_types)):
            raise ValueError(
                'Parameter "{}": Not valid type found. Valid types: {}'.format(self.name, self.valid_types))

    def _check_valid_multiples(self, value):
        if self.valid_multiples is None:
            return True
        for mult in self.valid_multiples:
            reminder = Decimal(str(value)) % Decimal(str(mult))
            reminder = float(reminder)
            if reminder != 0:
                raise ValueError('Parameter "{}": must be multiple of {}'.format(self.name, self.valid_multiples))

    def copy(self):
        """Returns a copy of itself"""
        return deepcopy(self)


def verify_min_max(arr, min_value=None, max_value=None):
    valid_min = True
    valid_max = True
    if min_value is not None:
        valid_min = np.min(arr) >= min_value
    if max_value is not None:
        valid_max = np.max(arr) <= max_value
    valid = valid_min and valid_max
    return valid
