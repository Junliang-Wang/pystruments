import unittest

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


if __name__ == '__main__':
    unittest.main()

    # wf = SingleWaveform()
    # from pystruments.funclib import pulse, pulse_params
    #
    # params = pulse_params(pts=1, base=1, delay=1, ampl=1, length=10)
    # params['delay'].sweep_stepsize(init=0, step_size=5, dim=1)
    # wf.set_func(pulse, params)
