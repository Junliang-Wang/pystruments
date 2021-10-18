import unittest

import numpy as np
import numpy.testing as npt

from pystruments.funclib import pulse, pulse_params
from pystruments.waveform import SingleWaveform


class Test_SingleWaveform(unittest.TestCase):
    def test_dims_static_and_not_reduced(self):
        wf = SingleWaveform()
        self.assertEqual(wf.dims, (1, 1))
        target_expected = [
            [[1], (1, 1)],
            [[10], (10, 1)],
            [[10, 20], (10, 20)],
        ]
        for v in target_expected:
            target, expected = v
            wf.set_dims(target, reduced=False)
            self.assertEqual(wf.dims, expected)

    def test_dims_static_and_reduced(self):
        wf = SingleWaveform()
        target_expected = [
            [[1], (1, 1)],
            [[10], (10, 1)],
            [[10, 20], (10, 1)],
        ]
        for v in target_expected:
            target, expected = v
            wf.set_dims(target, reduced=True)
            self.assertEqual(wf.dims, expected)

    def test_sweep_1d_dims(self):
        fparams = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
        wf = SingleWaveform(func=pulse, func_params=fparams)
        wf.func_params['ampl'].sweep_linear(-1, 1, dim=1)
        target_expected = [
            [[1], (1, 1)],
            [[10], (10, 1)],
            [[10, 20], (10, 20)],
            [[10, 20, 30], (10, 20)],
        ]
        for v in target_expected:
            target, expected = v
            wf.set_dims(target, reduced=True)
            self.assertEqual(wf.dims, expected)

    def test_sweep_1d_values(self):
        fparams = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
        wf = SingleWaveform(func=pulse, func_params=fparams)
        wf.func_params['ampl'].sweep_linear(-1, 1, dim=1)
        arrs = wf.unique_waveforms([100, 3], reduced=True)
        ampls = np.linspace(-1, 1, 3)
        for arr, ai in zip(arrs, ampls):
            npt.assert_equal(arr, pulse(pts=100, base=0, delay=1, ampl=ai, length=10))


if __name__ == '__main__':
    unittest.main()

    # wf = SingleWaveform()
    # from pystruments.funclib import pulse, pulse_params
    #
    # params = pulse_params(pts=1, base=1, delay=1, ampl=1, length=10)
    # params['delay'].sweep_stepsize(init=0, step_size=5, dim=1)
    # wf.set_func(pulse, params)
