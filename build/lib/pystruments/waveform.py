import inspect
import json
from copy import deepcopy
from itertools import product, izip

import numpy as np

from pystruments.funclib import constant, constant_params
from pystruments.parameter import Parameter


def split_dims_size(dims_size, dims):
    dims = set(dims)
    dsize = np.array(dims_size, dtype=int)
    if len(dims) == 0:
        n_left = np.prod(dims_size[1:])
        n_mid = 1
        n_right = 1
    else:
        min_dim = min(dims)
        max_dim = max(dims)
        n_left = np.prod(dsize[1:min_dim])
        n_mid = np.prod(dsize[min_dim:max_dim + 1])
        n_right = np.prod(dsize[max_dim + 1:])
    return n_left, n_mid, n_right


def unique_dims(*dims):
    udims = set()
    for d in dims:
        udims.update(d)
    return udims


class Waveform(object):
    def __init__(self, name='waveform', func=None, params=None):
        self.name = name
        params = params if params is not None else {}
        if func is None:
            self.set_constant_func()
        else:
            self.set_func(func, params)

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
        self.params = p

    def is_static(self):
        dims = self.swept_dims()
        return True if len(dims) == 0 else False

    def swept_dims(self):
        dims = set()
        for key, pi in self.params.items():
            if not pi.is_static():
                dims.add(pi.dim)
        return dims

    def get_waveforms(self, dims_size):
        igrid = self.index_grid(dims_size)
        iarr = igrid.flatten(order='F')
        unique_wfs = list(self.unique_waveforms(dims_size))
        for idx in iarr:
            yield unique_wfs[idx]

    def get_waveforms_subset(self, dims_size, dims):
        dims = set(dims)
        self._validate_dims(dims)
        dsize = np.ones(len(dims_size), dtype=int)
        if len(dims) == 0:
            dsize = [dims_size[0]]
        else:
            min_dim = min(dims)
            max_dim = max(dims)
            dsize[0] = dims_size[0]
            dsize[min_dim:max_dim + 1] = dims_size[min_dim:max_dim + 1]
        return self.get_waveforms(dsize)

    def get_waveforms_swept_subset(self, dims_size):
        return self.get_waveforms_subset(dims_size, dims=self.swept_dims())

    def unique_waveforms(self, dims_size):
        self._validate_dims_size(dims_size)
        self._set_dims_size_to_params(dims_size)
        dswept = list(self.swept_dims())
        static_kwargs, swept_params = self._prepare_kwargs()
        for uidx in self._unique_indexes(dims_size):
            kwargs = dict(static_kwargs)
            for arg, param in swept_params.items():
                idx_pos = dswept.index(param.dim)
                cur_idx = uidx[idx_pos]
                value = param.value[cur_idx]
                kwargs[arg] = value
            wf = self.func(**kwargs)
            yield wf

    def get_grids(self, dims_size):
        self._validate_dims_size(dims_size)
        cgrid = self.creation_grid(dims_size)
        igrid = self.index_grid(dims_size)
        ngrid = self.name_grid(dims_size)
        return cgrid, igrid, ngrid

    def creation_grid(self, dims_size):
        self._validate_dims_size(dims_size)
        dsize = list(dims_size)
        shape = tuple(dsize[1:])
        ndim = len(shape)
        grid = np.zeros(shape, dtype=bool)
        dswept = self.swept_dims()
        idxs = [0] * ndim
        for di in dswept:
            di = int(di)
            idxs[di - 1] = slice(None)
        idxs = tuple(idxs)
        grid[idxs] = True
        return grid

    def name_grid(self, dims_size):
        dsize = list(dims_size)
        shape = tuple(dsize[1:])
        ndim = len(shape)
        grid = np.zeros(shape, dtype=object)
        grid[:] = self.name
        dswept = self.swept_dims()
        for uidx in self._unique_indexes(dims_size):
            if not uidx:
                break
            grid_idxs = [slice(None)] * ndim
            name = [self.name]
            for di, j in zip(dswept, uidx):
                grid_idxs[di - 1] = j
                name.append('{}d{}'.format(di, j))
            name = '_'.join(name)
            grid[tuple(grid_idxs)] = name
        return grid

    def index_grid(self, dims_size):
        self._validate_dims_size(dims_size)
        dsize = list(dims_size)
        shape = tuple(dsize[1:])
        ndim = len(shape)
        grid = np.zeros(shape, dtype=int)
        dswept = self.swept_dims()
        counter = 0
        for uidx in self._unique_indexes(dims_size):
            if not uidx:
                break
            grid_idxs = [slice(None)] * ndim
            for di, j in zip(dswept, uidx):
                grid_idxs[di - 1] = j
            grid[tuple(grid_idxs)] = counter
            counter += 1
        return grid

    def split_dims_size(self, dims_size):
        self._validate_dims_size(dims_size)
        return split_dims_size(dims_size, self.swept_dims())

    def get_dict(self):
        d = {}
        keys = ['name']
        for key in keys:
            if not hasattr(self, key):
                continue
            d[key] = getattr(self, key)

        fdict = func_to_dict(self.func)
        d['func'] = fdict

        d['params'] = {}
        for key, parameter in self.params.items():
            d['params'][key] = parameter.get_dict()

        return d

    def set_dict(self, conf):
        keys = ['name']
        for key, value in conf.items():
            if not hasattr(self, key) or key not in keys:
                continue
            setattr(self, key, value)
        func = load_func(conf['func'])
        params = {}
        for key, pdict in conf['params'].items():
            p = Parameter(name=key, value=0)
            p.set_dict(pdict)
            params[key] = p
        self.set_func(func, params)

    def load(self, fullpath):
        with open(fullpath, 'rb') as file:
            conf = json.load(file)
        self.set_dict(conf)

    def save(self, fullpath):
        with open(fullpath, 'wb') as file:
            json.dump(self.get_dict(), file)

    def _validate_dims(self, dims):
        dswept = self.swept_dims()
        dims = set(dims)
        if len(dims) < len(dswept):
            raise ValueError('dims ({}) must include all swept dims ({})'.format(dims, dswept))
        for ds in dswept:
            if ds not in dims:
                raise ValueError('swept dim ({}) not in dims ({})'.format(dswept, dims))

    def _validate_dims_size(self, dims_size):
        dsize = list(dims_size)
        len_dsize = len(dsize)
        max_dim = len_dsize - 1
        swept_dims = self.swept_dims()
        static = len(swept_dims) == 0

        if len_dsize < 1:
            raise ValueError('dims must have at least 1 value (waveform pts)')
        elif not static and max(swept_dims) > max_dim:
            raise ValueError('Max. swept_dims ({}) has no dims_size assigned'.format(swept_dims))

    def _set_dims_size_to_params(self, dims_size):
        dsize = np.array(dims_size)
        for arg, param in self.params.items():
            if arg == 'pts':
                param.set_value(dsize[0])
            elif not param.is_static():
                param.set_pts(dsize[param.dim])

    def _unique_indexes(self, dims_size):
        dswept = self.swept_dims()
        dsize = list(dims_size)
        dswept_size = [dsize[i] for i in dswept]
        dswept_idxs = [range(size) for size in dswept_size]
        dswept_idxs = dswept_idxs[::-1]  # reverse
        for uidx in product(*dswept_idxs):
            yield uidx[::-1]  # reverse back

    def _prepare_kwargs(self):
        static_kwargs = {}
        swept_params = {}
        for arg, param in self.params.items():
            if param.is_static():
                static_kwargs[arg] = param.get_value()
            else:
                swept_params[arg] = param
        return static_kwargs, swept_params

    @staticmethod
    def _check_func(f, fparams):
        if not callable(f):
            raise ValueError('func is not callable')
        args = get_func_args(f)
        if 'pts' not in args:
            raise KeyError('"pts" must be an argument for {}"'.format(f))
        for arg in args:
            if arg not in fparams.keys():
                raise KeyError('{} not in params'.format(arg))
            if not isinstance(fparams[arg], Parameter):
                raise ValueError('parameter "{}" must be an instance of {}'.format(arg, Parameter))


class WaveformGroup(object):
    def __init__(self, waveforms, name='group'):
        self.name = name
        self.reset_waveforms()
        for wf in waveforms:
            self.add_waveform(wf)

    def add_waveform(self, wf):
        if not isinstance(wf, Waveform):
            raise TypeError('waveform has to be an instance of Waveform')
        self.wfs.append(wf)

    def reset_waveforms(self):
        self.wfs = []

    def swept_dims(self):
        dims = [wf.swept_dims() for wf in self.wfs]
        return unique_dims(*dims)

    def is_static(self):
        dims = self.swept_dims()
        return True if len(dims) == 0 else False

    def get_waveforms(self, dims_size):
        wfs_gen = [wf.get_waveforms(dims_size) for wf in self.wfs]
        for wfs_i in izip(*wfs_gen):
            yield wfs_i

    def get_waveforms_subset(self, dims_size, dims):
        wfs_gen = [wf.get_waveforms_subset(dims_size, dims) for wf in self.wfs]
        for wfs_i in izip(*wfs_gen):
            yield wfs_i

    def get_waveforms_swept_subset(self, dims_size):
        if self.is_static():
            dsize = [dims_size[0]]
            return self.get_waveforms(dsize)
        else:
            swept_dims = self.swept_dims()
            return self.get_waveforms_subset(dims_size, swept_dims)

    def split_dims_size(self, dims_size):
        self._validate_dims_size(dims_size)
        return split_dims_size(dims_size, dims=self.swept_dims())

    def get_dict(self):
        d = {}
        keys = ['name']
        for key in keys:
            if hasattr(self, key):
                d[key] = getattr(self, key)

        d['waveforms'] = []
        for i, wf in enumerate(self.wfs):
            d['waveforms'].append(wf.get_dict())
        return d

    def set_dict(self, conf):
        keys = ['name']
        for key, value in conf.items():
            if not hasattr(self, key) or key not in keys:
                continue
            setattr(self, key, value)
        if 'waveforms' in conf.keys():
            for wf_dict in conf['waveforms'].values():
                wf = Waveform()
                wf.set_dict(wf_dict)
                self.add_waveform(wf)

    def save(self, fullpath):
        with open(fullpath, 'wb') as file:
            json.dump(self.get_dict(), file)

    def load(self, fullpath):
        with open(fullpath, 'rb') as file:
            conf = json.load(file)
        self.set_dict(conf)

    def _validate_dims_size(self, dims_size):
        dsize = list(dims_size)
        len_dsize = len(dsize)
        max_dim = len_dsize - 1
        swept_dims = self.swept_dims()
        static = len(swept_dims) == 0

        if len_dsize < 1:
            raise ValueError('dims must have at least 1 value (waveform pts)')
        elif not static and max(swept_dims) > max_dim:
            raise ValueError('Max. swept_dims ({}) has no dims_size assigned'.format(swept_dims))


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
    # # waveform = Waveform('test')
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
    # params = pulse_params(
    #     pts=100,
    #     base=0,
    #     delay=2,
    #     ampl=1,
    #     length=10,
    # )
    # params['ampl'].sweep_linear(1, -1, dim=1)
    # mk = Waveform(name='marker', func=pulse, params=params)
    # params = pulse_params(
    #     pts=100,
    #     base=0,
    #     delay=2,
    #     ampl=1,
    #     length=10,
    # )
    # params['base'].max_value = 1
    # params['base'].sweep_linear(1, -1, dim=1)
    # params['ampl'].sweep_linear(1, -1, dim=2)
    # wf = Waveform('test', func=pulse, params=params, childs=[mk])

    # wf = Waveform()
    # from pystruments.funclib import pulse, pulse_params
    #
    # params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
    # # params['delay'].sweep_stepsize(init=0, step_size=2, dim=1)
    # # params['length'].sweep_stepsize(init=1, step_size=2, dim=2)
    # wf.set_func(pulse, params)
    # dsize = [10, 2, 3, 2]
    # wfs = list(wf.unique_waveforms(dsize))
    # cgrid = wf.creation_grid(dsize)
    # ngrid = wf.name_grid(dsize)
    # # igrid = wf.index_grid(dsize)
    # # values = list(wf.value_generator(dims=[100, 2, 3, 4], reduced=False))
    # # n = wf.name_grid()
    # wfgrid = np.zeros(dsize)
    # w = list(wf.get_waveforms(dsize))
    #
    # ww = list(wf.get_waveforms_subset(dsize, [2]))
    # www = list(wf.get_waveforms_swept_subset(dsize))

    # def f(pts=1, test=2, **kwargs):
    #     pass

    wf1 = Waveform()
    params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
    wf1.set_func(pulse, params)

    wf2 = Waveform()
    params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
    params['delay'].sweep_stepsize(init=0, step_size=2, dim=1)
    wf2.set_func(pulse, params)

    wf3 = Waveform()
    params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
    params['delay'].sweep_stepsize(init=0, step_size=2, dim=1)
    params['length'].sweep_stepsize(init=1, step_size=2, dim=2)
    wf3.set_func(pulse, params)

    wg = WaveformGroup([wf1, wf2, wf3])

    list(wf2.get_waveforms([100, 10, 3]))
    d2 = wf2.get_dict()
    # wf3.set_dict(d2)
    # list(wf3.get_waveforms([100, 10, 3]))
    # print(wg.swept_dims())
    # dsize = [10, 2, 3]
