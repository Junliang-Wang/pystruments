import unittest

import numpy as np

from pystruments.parameter import Parameter


class Test_Parameter(unittest.TestCase):
    def test_no_safety(self):
        p = Parameter('test', 0)
        valids = [None, True, [1, 2], np.array(2)]
        for v in valids:
            p.set_value(v)

    def test_safety_limits(self):
        p = Parameter('test', 1, min_value=0, max_value=2)
        valids = [0.5, 1.5]
        for v in valids:
            p.set_value(v)

        not_valids = [-1, 3]
        for v in not_valids:
            self.assertRaises(ValueError, p.set_value, v)

    def test_safety_valid_values(self):
        p = Parameter('test', 1, valid_values=[1, True, 'on'])
        valids = [1, True, 'on']
        for v in valids:
            p.set_value(v)

        not_valids = [0, False, 'ON']
        for v in not_valids:
            self.assertRaises(ValueError, p.set_value, v)

    def test_safety_not_valid_values(self):
        p = Parameter('test', [0], not_valid_values=[1, True, 'on'])
        valids = [0, False, 'ON']
        for v in valids:
            p.set_value(v)

        not_valids = [1, True, 'on']
        for v in not_valids:
            self.assertRaises(ValueError, p.set_value, v)

    def test_safety_valid_types(self):
        p = Parameter('test', 0, valid_types=[int, str])
        valids = [0, 'on']
        for v in valids:
            p.set_value(v)

        not_valids = [1.1, []]
        for v in not_valids:
            self.assertRaises(ValueError, p.set_value, v)

    def test_dict(self):
        p = Parameter('test', 0, min_value=0, max_value=2, valid_values=[0, 1, 2])
        d = dict(name='test', value=0, unit='a.u.', metadata={},
                 min_value=0, max_value=2,
                 min_step=None, max_step=None, size_step=None,
                 valid_values=[0, 1, 2], not_valid_values=None,
                 valid_types=None, valid_multiples=None,
                 dim=None)
        for key, value in p.get_dict().items():
            self.assertEqual(value, d[key])


if __name__ == '__main__':
    unittest.main()

    # wf = Waveform()
    # from pystruments.funclib import pulse, pulse_params
    #
    # params = pulse_params(pts=1, base=1, delay=1, ampl=1, length=10)
    # params['delay'].sweep_stepsize(init=0, step_size=5, dim=1)
    # wf.set_func(pulse, params)
