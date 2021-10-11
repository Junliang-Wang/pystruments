import inspect
from copy import deepcopy
from itertools import product, izip

import numpy as np

from funclib import constant, constant_params
from parameter import Parameter


class SingleWaveform(object):
    def __init__(self, name='waveform', func=None, func_params=None, dims=None):
        self.name = name
        dims = dims if dims is not None else [1]
        func_params = func_params if func_params is not None else {}
        if func is None:
            self.set_constant_func()
        else:
            self.set_func(func, func_params)
        self.set_dims(dims)

    def set_constant_func(self):
        params = constant_params(pts=0, ampl=0)
        self.set_func(constant, params)

    def set_func(self, func, params):
        self._check_func(func, params)

        self.func = func
        self.args = get_func_args(func)

        p = {}
        for arg in self.args:
            p[arg] = deepcopy(params[arg])
        self.func_params = p

    def is_static(self):
        unique_dims = self.unique_dims()
        if len(unique_dims) == 0:
            static = True
        else:
            static = False
        return static

    def set_dims(self, dims, reduced=False):
        dims = list(dims)
        if len(dims) < 1:
            raise ValueError('dims must have at least 1 value (waveform pts)')
        static = self.is_static()
        if static and len(dims) == 1:
            dims = dims + [1]
        elif static and reduced:
            dims = [dims[0], 1]
        elif not static:
            unique_dims = self.unique_dims()
            max_udim = max(unique_dims)
            max_dims = len(dims) - 1
            if max_dims < max_udim:
                dims = dims + [1] * (max_udim - max_dims)
            elif max_dims > max_udim and reduced is True:
                dims = dims[:max_udim + 1]
        for arg, param in self.func_params.items():
            if arg == 'pts':
                param.set_value(dims[0])
            elif not param.is_static():
                pts = dims[param.dim]
                param.set_pts(pts)
        self.dims = tuple(dims)

    def unique_dims(self):
        dims = set()
        for key, pi in self.func_params.items():
            if not pi.is_static():
                dims.add(pi.dim)
        return dims

    def unique_dims_size(self):
        unique_dims = self.unique_dims()
        udims_size = [self.dims[i] for i in unique_dims]
        return udims_size

    def unique_waveforms(self, dims=None, reduced=False):
        if dims is not None:
            self.set_dims(dims, reduced=reduced)
        dims = self.dims
        pts = dims[0]
        unique_dims = self.unique_dims()
        udim_pos = {sdi: i for i, sdi in enumerate(unique_dims)}
        udims_size = [dims[i] for i in unique_dims]
        unique_idxs = [range(size) for size in udims_size]
        unique_idxs = unique_idxs[::-1]  # reverse
        for uidx in product(*unique_idxs):
            uidx = uidx[::-1]  # reverse back
            kwargs = {}
            for arg, param in self.func_params.items():
                if arg == 'pts':
                    param.set_value(pts)
                if param.is_static():
                    value = param.get_value()
                else:
                    idx_pos = udim_pos[param.dim]
                    cur_idx = uidx[idx_pos]
                    value = param.value[cur_idx]
                kwargs[arg] = value
            wf = self.func(**kwargs)
            yield wf

    def creation_grid(self, dims=None, reduced=False):
        if dims is not None:
            self.set_dims(dims, reduced=reduced)
        dims = self.dims
        shape = tuple(dims[1:])
        ndim = len(dims[1:])
        grid = np.zeros(shape, dtype=bool)
        sdims = self.unique_dims()
        idxs = [0] * ndim
        for di in sdims:
            di = int(di)
            idxs[di - 1] = slice(None)
        idxs = tuple(idxs)
        grid[idxs] = True
        return grid

    def name_grid(self, dims=None, reduced=False):
        if dims is not None:
            self.set_dims(dims, reduced=reduced)
        dims = self.dims
        shape = tuple(dims[1:])
        ndim = len(dims[1:])
        grid = np.zeros(shape, dtype=object)
        grid[:] = self.name
        unique_dims = self.unique_dims()
        udims_size = [dims[i] for i in unique_dims]
        unique_idxs = [range(size) for size in udims_size]
        unique_idxs = unique_idxs[::-1]  # reverse
        for uidx in product(*unique_idxs):
            uidx = uidx[::-1]  # reverse back
            if not uidx:
                break
            grid_idxs = [slice(None)] * ndim
            name = [self.name]
            for di, j in zip(unique_dims, uidx):
                grid_idxs[di - 1] = j
                name.append('{}d{}'.format(di, j))
            name = '_'.join(name)
            grid[tuple(grid_idxs)] = name
        return grid

    def value_generator(self, dims=None, reduced=False):
        cgrid = self.creation_grid(dims, reduced)
        carr = cgrid.flatten(order='F')
        unique_wfs = self.unique_waveforms()
        last_wf = None
        for create in carr:
            if create:
                last_wf = next(unique_wfs)
            yield last_wf

    def get_dict(self):
        d = {}
        keys = ['name', 'metadata']
        for key in keys:
            if hasattr(self, key):
                d[key] = getattr(self, key)

        fdict = func_to_dict(self.func)
        d['func'] = fdict

        d['func_params'] = {}
        for key, parameter in self.func_params.items():
            d['func_params'][key] = parameter.get_dict()

        return d

    @staticmethod
    def _check_func(f, fparams):
        if not callable(f):
            raise ValueError('func is not callable')
        args = get_func_args(f)
        if 'pts' not in args:
            raise KeyError('"pts" must be an argument for {}"'.format(f))
        for arg in args:
            if arg not in fparams.keys():
                raise KeyError('{} not in func_params'.format(arg))
            if not isinstance(fparams[arg], Parameter):
                raise ValueError('parameter "{}" must be an instance of {}'.format(arg, Parameter))


class Waveform(SingleWaveform):
    def __init__(self, name='waveform', childs=None, **kwargs):
        self.reset_childs()
        childs = childs if childs is not None else []
        for child in childs:
            self.add_child(child)
        super(Waveform, self).__init__(name=name, **kwargs)

    def add_child(self, child):
        if not isinstance(child, SingleWaveform):
            raise TypeError('child has to be an instance of SingleWaveform')
        self.childs.append(child)

    def reset_childs(self):
        self.childs = []

    def unique_dims(self):
        dims = super(Waveform, self).unique_dims()
        for child in self.childs:
            dims = set.union(dims, child.unique_dims())
        return dims

    def set_dims(self, *args, **kwargs):
        super(Waveform, self).set_dims(*args, **kwargs)
        dims = self.dims
        for child in self.childs:
            child.set_dims(dims=dims, reduced=False)

    def childs_waveforms(self):
        childs_wfs = [child.value_generator() for child in self.childs]
        for wfs in izip(*childs_wfs):
            yield wfs

    def get_dict(self):
        d = super(Waveform, self).get_dict()
        d['childs'] = {}
        for i, child in enumerate(self.childs):
            d['childs'][i] = child.get_dict()
        return d

    def save(self, fullpath):
        import json
        with open(fullpath, 'wb') as file:
            json.dump(self.get_dict(), file)


def get_func_args(func):
    args = list(func.__code__.co_varnames)[0:func.__code__.co_argcount]
    return args


def func_to_dict(func):
    name = func.__name__
    code = inspect.getsource(func)
    d = {}
    d['name'] = name
    d['code'] = code
    return d


def load_func(fdict):
    d = {}
    name = fdict['name']
    code = fdict['code']
    exec (code, d)
    return d[name]


if __name__ == '__main__':
    from funclib import pulse, pulse_params

    # dims = [100, 11, 3]
    # waveform.set_dims(dims)
    # # waveform = SingleWaveform('test')
    # ngrid = waveform.wf_name_grid()
    # cgrid = waveform.wf_creation_grid()
    # uwfs = list(waveform.unique_waveforms())
    #
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(1)
    # for i, waveform in enumerate(uwfs):
    #     ax.plot(waveform, label=i)
    # ax.legend()
    func_params = pulse_params(
        pts=100,
        base=0,
        delay=2,
        ampl=1,
        length=10,
    )
    func_params['ampl'].sweep_linear(1, -1, dim=1)
    mk = SingleWaveform(name='marker', func=pulse, func_params=func_params)
    func_params = pulse_params(
        pts=100,
        base=0,
        delay=2,
        ampl=1,
        length=10,
    )
    func_params['base'].max_value = 1
    func_params['base'].sweep_linear(1, -1, dim=1)
    func_params['ampl'].sweep_linear(1, -1, dim=2)
    wf = Waveform('test', func=pulse, func_params=func_params, childs=[mk])
