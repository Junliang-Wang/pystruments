import json

import pyvisa

from pystruments.parameter import Parameter


def get_decorator(fget):
    fname = fget.__name__
    param_name = fname[len('get_'):]

    def _f(self, *args, **kwargs):
        value = fget(self, *args, **kwargs)
        param = self.get_parameter(param_name, read_only=False, update=False)
        param.set_value(value)
        return value

    return _f


def set_decorator(fset):
    fname = fset.__name__
    param_name = fname[len('set_'):]

    def _f(self, *args, **kwargs):
        target_value = args[0]
        fget = getattr(self, 'get_{}'.format(param_name))
        cur_value = fget()
        if cur_value != target_value:
            self.validate_parameter_value(param_name, target_value)
            fset(self, *args, **kwargs)
            cur_value = fget()
            param = self.get_parameter(param_name, read_only=False, update=False)
            param.set_value(cur_value)

    return _f


class InstrumentBase(object):
    default_parameters = tuple()

    def __init__(self, address, timeout=2000, write_termination=None, read_termination='\n',
                 parent=None, childs=[], name='instr',
                 **kwargs):
        self.address = address
        self.name = name
        self.timeout = timeout
        self.write_termination = write_termination
        self.read_termination = read_termination
        self.identity = ''
        self._verify_default_parameters()
        self._parameters = {param.name: param.copy() for param in self.default_parameters}
        self.parameters_list = list(self._parameters.keys())
        self.parent = parent
        self.childs = []
        for child in childs:
            self.add_child(child)
        self.reset_safety()

    @property
    def parameters(self):
        params = []
        for param_name in self.parameters_list:
            param = self.get_parameter(param_name, read_only=True, update=True)
            params.append(param)
        return tuple(params)

    def get_identity(self):
        my_identity = self.identity
        return my_identity

    def open_com(self):
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource(self.address)
        self.instrument.write_termination = self.write_termination
        self.instrument.read_termination = self.read_termination
        self.instrument.timeout = self.timeout
        self.identity = self.instrument.query('*IDN?')
        print('Connected to: ' + self.get_identity())
        for child in self.childs:
            child.open_com()

    def close_com(self):
        for child in self.childs:
            child.close_com()
        self.instrument.close()
        print('Disconnected from: ' + self.get_identity())

    def send(self, cmd):
        self.instrument.write(cmd)

    def read(self, msg):
        answer = self.instrument.query(msg)
        return answer

    def _verify_default_parameters(self):
        parameters = self.default_parameters
        for param in parameters:
            if not isinstance(param, Parameter):
                raise TypeError('parameter "{}" is not an instance of {}'.format(param, Parameter))

    def get_parameter(self, name, read_only=True, update=True):
        param = self._parameters[name]
        if update is True:
            fget = getattr(self, 'get_{}'.format(name))
            value = fget()
            param.set_value(value)
        if read_only is True:
            param = param.copy()
        return param

    def get_parameter_validator(self, name):
        param = self.get_parameter(name, read_only=True, update=False)
        return param

    def update_parameters(self):
        for param_name in self.parameters_list:
            self.get_parameter(param_name, read_only=False, update=True)

    def add_safety(self, name, **kwargs):
        param = self.get_parameter_validator(name)
        for key, value in kwargs.items():
            if not hasattr(param, key):
                raise KeyError('parameter "{}": invalid safety property "{}"'.format(name, key))
            setattr(param, key, value)
        self._safe_parameters[name] = param

    def reset_safety(self):
        self._safe_parameters = {}

    def validate_parameter_value(self, name, target_value):
        param = self.get_parameter_validator(name)
        param.set_value(target_value)
        if name in self._safe_parameters.keys():
            param = self._safe_parameters[name]
            param.set_value(target_value)

    def add_child(self, child):
        self.childs.append(child)

    def remove_childs(self):
        self.childs = []

    def set_config(self, config, include_childs=False):
        instr_config = config['config']
        if include_childs and 'childs_config' in config.keys():
            childs_configs = config['childs_config']
        else:
            childs_configs = []

        for key, value in instr_config.items():
            if key not in self.parameters_list:
                raise ValueError('Parameter "{}" is not valid: ', key)
            fset_name = 'set_{}'.format(key)
            if not hasattr(self, fset_name):
                continue
            fset = getattr(self, 'set_{}'.format(key))
            fset(value)
        for i, child_config in enumerate(childs_configs):
            self.childs[i].set_config(config=child_config)

    def get_config(self, include_childs=False):
        config = {}
        config['config'] = {param.name: param.value for param in self.parameters}
        if include_childs:
            config['childs_config'] = [child.get_config() for child in self.childs]
        return config

    def save_config(self, fullpath, include_childs=True):
        with open(fullpath, 'wb') as file:
            json.dump(self.get_config(include_childs=include_childs), file)

    def load_config(self, fullpath, include_childs=True):
        with open(fullpath, 'rb') as file:
            config = json.load(file)
        self.set_config(config, include_childs=include_childs)
        return self.get_config(include_childs=include_childs)

    def __enter__(self):
        self.open_com()
        return self

    def __exit__(self, type, value, traceback):
        self.close_com()

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = '[{}] {} ({})'.format(self.name, self.identity, self.address)
        return out


if __name__ == '__main__':
    instr = InstrumentBase('test')
