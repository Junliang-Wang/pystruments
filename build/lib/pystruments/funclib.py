import numpy as np

from pystruments.parameter import Parameter

"""
Predefined functions
"""


def constant(pts, ampl=0):
    return np.ones(pts) * ampl


def constant_params(pts, ampl=0):
    params = {}
    params['pts'] = Parameter('pts', pts, unit='pts')
    params['ampl'] = Parameter('ampl', ampl, unit='V')
    return params


"""
Generators
"""


def nfunc_generator(func):
    """
    Generate n funcs with its respective kwargs: kwargs1_i, kwargs2_i, etc...
    where i is the ith func (1, 2, ...)
    It will normalize the value
    """
    args = list(func.__code__.co_varnames)[0:func.__code__.co_argcount]

    def f(pts, **kwargs):
        keys = args[1:]
        arr = np.zeros(pts)
        i = 1
        valid = True
        while valid:
            keys_i = ['{}_{}'.format(key, i) for key in keys]
            missing_key = False in [ki in kwargs.keys() for ki in keys_i]
            if missing_key:
                valid = False
            else:
                kwargs_i = {}
                for key, key_i in zip(keys, keys_i):
                    kwargs_i[key] = kwargs[key_i]
                arr_i = func(pts, **kwargs_i)
                arr = arr + arr_i
                i += 1
        max_ampl = np.max(np.abs(arr))
        if max_ampl != 0:
            arr = arr / max_ampl
        return arr

    return f


def nparams_generator(func):
    """
    Generate n func_params with its respective kwargs: kwargs1_i, kwargs2_i, etc...
    where i is the ith func (1, 2, ...)
    """
    args = list(func.__code__.co_varnames)[0:func.__code__.co_argcount]

    def f(pts, **kwargs):
        keys = args[1:]
        params = {}
        params['pts'] = Parameter('pts', pts, unit='pts')
        i = 1
        valid = True
        while valid:
            keys_i = ['{}_{}'.format(key, i) for key in keys]
            missing_key = False in [ki in kwargs.keys() for ki in keys_i]
            if missing_key:
                valid = False
            else:
                kwargs_i = {}
                for key, key_i in zip(keys, keys_i):
                    kwargs_i[key] = kwargs[key_i]
                params_i = func(pts, **kwargs_i)
                for key, value in params_i.items():
                    if key == 'pts':
                        continue
                    params['{}_{}'.format(key, i)] = value
                i += 1
        return params

    return f


"""
User-defined functions
"""


def pulse(pts, base, delay, ampl, length=1):
    arr = np.ones(pts) * base
    delay = int(delay)
    length = int(length)
    arr[delay:delay + length] = ampl
    return arr


def pulse_params(pts, base, delay, ampl, length=1):
    params = {}
    params['pts'] = Parameter('pts', pts, unit='pts')
    params['base'] = Parameter('base', base, unit='V')
    params['delay'] = Parameter('delay', delay, unit='pts', size_step=1)
    params['ampl'] = Parameter('ampl', ampl, unit='V')
    params['length'] = Parameter('length', length, unit='pts', size_step=1)
    return params


def pulse_slope(pts, base, delay, ampl, length, left_slope, right_slope):
    arr = np.ones(pts) * base
    delay = int(delay)
    length = int(length)
    left_slope = int(left_slope)
    right_slope = int(right_slope)
    left = np.linspace(base, ampl, left_slope)
    right = np.linspace(ampl, base, right_slope)
    xi = delay
    xl = delay + left_slope
    xr = xl + length
    xf = xr + right_slope
    arr[xi:xl] = left
    arr[xl:xr] = ampl
    arr[xr:xf] = right
    return arr


def pulse_slope_params(pts, base, delay, ampl, length, left_slope, right_slope):
    params = {}
    params['pts'] = Parameter('pts', pts, unit='pts')
    params['base'] = Parameter('base', base, unit='V')
    params['delay'] = Parameter('delay', delay, unit='pts', size_step=1)
    params['ampl'] = Parameter('ampl', ampl, unit='V')
    params['length'] = Parameter('length', length, unit='pts', size_step=1)
    params['left_slope'] = Parameter('left_slope', left_slope, unit='pts', size_step=1)
    params['right_slope'] = Parameter('right_slope', right_slope, unit='pts', size_step=1)
    return params


def npulse(pts, **kwargs):
    """
    Generate n pulses with kwargs: base_i, delay_i, ampl_i, length_i
    where i is the ith pulse (1, 2, ...)
    It will normalize the value
    """
    f = nfunc_generator(pulse)
    return f(pts, **kwargs)


def npulse_params(pts, **kwargs):
    """
    Generate n pulse_params with kwargs: base_i, delay_i, ampl_i, length_i
    where i is the ith pulse (1, 2, ...)
    """
    f = nparams_generator(pulse_params)
    return f(pts, **kwargs)


if __name__ == '__main__':
    # pdict = dict(
    #     pts=100,
    #     base=0,
    #     ampl=1,
    #     max_length=2,
    #     value=1,
    # )
    # y = pulse(**pdict)

    pdict = dict(
        pts=100,

        base_1=0,
        ampl_1=-1,
        length_1=2,
        delay_1=1,

        base_2=0,
        ampl_2=1,
        length_2=5,
        delay_2=3,
    )
    y = npulse(**pdict)
    yp = npulse_params(**pdict)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    ax.plot(y, 'o-')
    plt.show()
