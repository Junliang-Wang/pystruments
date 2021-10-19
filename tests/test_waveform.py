import unittest

import numpy as np
import numpy.testing as npt

from pystruments.funclib import pulse, pulse_params
from pystruments.waveform import Waveform, WaveformGroup, split_dims_size, unique_dims


class Teset_Waveform(unittest.TestCase):
    def test_dims_size_static(self):
        wf = Waveform()
        self.assertTrue(wf.is_static())
        valids = [
            [1],
            [100],
            [100, 1],
            [100, 10],
            [100, 10, 20],
        ]
        for d in valids:
            wf._validate_dims_size(d)
        not_valids = [
            [],
        ]
        for d in not_valids:
            self.assertRaises(ValueError, wf._validate_dims_size, d)

    def test_dims_size_swept_1d_1(self):
        wf = Waveform()
        fparams = pulse_params(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        wf.set_func(pulse, fparams)
        self.assertFalse(wf.is_static())
        self.assertEqual(wf.swept_dims(), {1})
        valids = [
            [1, 1],
            [100, 1],
            [100, 10],
            [100, 10, 20],
        ]
        for d in valids:
            wf._validate_dims_size(d)
        not_valids = [
            [],
            [100],
        ]
        for d in not_valids:
            self.assertRaises(ValueError, wf._validate_dims_size, d)

    def test_dims_size_swept_1d_2(self):
        wf = Waveform()
        fparams = pulse_params(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=2)
        wf.set_func(pulse, fparams)
        self.assertFalse(wf.is_static())
        self.assertEqual(wf.swept_dims(), {2})
        valids = [
            [1, 1, 1],
            [100, 1, 1],
            [100, 10, 1],
            [100, 10, 20],
            [100, 10, 20, 30],
        ]
        for d in valids:
            wf._validate_dims_size(d)
        not_valids = [
            [],
            [100],
            [100, 10],
        ]
        for d in not_valids:
            self.assertRaises(ValueError, wf._validate_dims_size, d)

    def test_dims_size_swept_2d(self):
        wf = Waveform()
        fparams = pulse_params(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        fparams['length'].sweep_stepsize(init=1, step_size=2, dim=2)
        wf.set_func(pulse, fparams)
        self.assertFalse(wf.is_static())
        self.assertEqual(wf.swept_dims(), {1, 2})
        valids = [
            [1, 1, 1],
            [100, 1, 1],
            [100, 10, 1],
            [100, 10, 20],
            [100, 10, 20, 30],
        ]
        for d in valids:
            wf._validate_dims_size(d)
        not_valids = [
            [],
            [100],
            [100, 10],
        ]
        for d in not_valids:
            self.assertRaises(ValueError, wf._validate_dims_size, d)

    def test_unique_waveforms_static(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        wf.set_func(pulse, fparams)

        dsizes = [
            [10],
            [10, 3],
        ]
        for dsize in dsizes:
            wf_size = dsize[0]
            wfs = list(wf.unique_waveforms(dsize))
            self.assertEqual(len(wfs), 1)
            self.assertEqual(len(wfs[0]), wf_size)
            kwargs['pts'] = wf_size
            npt.assert_equal(wfs[0], pulse(**kwargs))

    def test_unique_waveforms_swept_1d_1(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        wf.set_func(pulse, fparams)

        dsizes = [
            [10, 2],
            [10, 2, 3],
        ]
        for dsize in dsizes:
            wf_size = dsize[0]
            wfs = list(wf.unique_waveforms(dsize))
            self.assertEqual(len(wfs), dsize[1])
            self.assertEqual(len(wfs[0]), wf_size)
            for i, wfi in enumerate(wfs):
                kwargs['pts'] = wf_size
                kwargs['delay'] = [1, 3][i]
                npt.assert_equal(wfi, pulse(**kwargs))

    def test_unique_waveforms_swept_1d_2(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=2)
        wf.set_func(pulse, fparams)

        dsizes = [
            [10, 2, 3],
            [10, 2, 3, 4],
        ]
        for dsize in dsizes:
            wf_size = dsize[0]
            wfs = list(wf.unique_waveforms(dsize))
            self.assertEqual(len(wfs), dsize[2])
            self.assertEqual(len(wfs[0]), wf_size)
            for i, wfi in enumerate(wfs):
                kwargs['pts'] = wf_size
                kwargs['delay'] = [1, 3, 5][i]
                npt.assert_equal(wfi, pulse(**kwargs))

    def test_unique_waveforms_swept_2d(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        fparams['length'].sweep_stepsize(init=1, step_size=3, dim=2)
        wf.set_func(pulse, fparams)

        dsizes = [
            [10, 2, 3],
            [10, 2, 3, 4],
        ]
        for dsize in dsizes:
            wf_size = dsize[0]
            wfs = list(wf.unique_waveforms(dsize))
            self.assertEqual(len(wfs), np.prod(dsize[1:3]))
            self.assertEqual(len(wfs[0]), wf_size)
            for i, wfi in enumerate(wfs):
                d1 = i % dsize[1]
                d2 = i // dsize[1]
                kwargs['pts'] = wf_size
                kwargs['delay'] = [1, 3, 5][d1]
                kwargs['length'] = [1, 4, 7][d2]
                npt.assert_equal(wfi, pulse(**kwargs))

    def test_get_waveforms_static(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        wf.set_func(pulse, fparams)

        dsizes = [
            [10, 2],
            [10, 2, 3],
        ]
        for dsize in dsizes:
            wf_size = dsize[0]
            n_wfs = np.prod(dsize[1:])
            wfs = list(wf.get_waveforms(dsize))
            kwargs['pts'] = wf_size
            expected = [pulse(**kwargs)] * n_wfs
            npt.assert_equal(np.array(wfs), np.array(expected))

    def test_get_waveforms_swept_1d_1(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        wf.set_func(pulse, fparams)

        dsize = [10, 2]
        wf_size = dsize[0]
        expected = [
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=2),
            pulse(pts=wf_size, base=0, delay=3, ampl=1, length=2),
        ]
        wfs = list(wf.get_waveforms(dsize))
        npt.assert_equal(np.array(wfs), np.array(expected))

        dsize = [10, 2, 3]
        expected = expected * 3
        wfs = list(wf.get_waveforms(dsize))
        npt.assert_equal(np.array(wfs), np.array(expected))

    def test_get_waveforms_swept_1d_2(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['delay'].sweep_stepsize(init=1, step_size=2, dim=1)
        fparams['length'].sweep_stepsize(init=1, step_size=3, dim=2)
        wf.set_func(pulse, fparams)

        dsize = [10, 2, 3]
        wf_size = dsize[0]
        expected = [
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=3, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=3, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=7),
            pulse(pts=wf_size, base=0, delay=3, ampl=1, length=7),
        ]
        wfs = list(wf.get_waveforms(dsize))
        npt.assert_equal(np.array(wfs), np.array(expected))

        dsize = [10, 2, 3, 4]
        expected = expected * 4
        wfs = list(wf.get_waveforms(dsize))
        npt.assert_equal(np.array(wfs), np.array(expected))

    def test_get_waveforms_subset(self):
        wf = Waveform()
        kwargs = dict(pts=1, base=0, delay=1, ampl=1, length=2)
        fparams = pulse_params(**kwargs)
        fparams['length'].sweep_stepsize(init=1, step_size=3, dim=2)
        wf.set_func(pulse, fparams)

        dsize = [10, 2, 3]
        subset_dims = [1, 2]
        wf_size = dsize[0]
        expected = [
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=7),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=7),
        ]
        wfs = list(wf.get_waveforms_subset(dsize, dims=subset_dims))
        npt.assert_equal(np.array(wfs), np.array(expected))

        dsize = [10, 2, 3]
        subset_dims = [2]
        expected = [
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=7),
        ]
        wfs = list(wf.get_waveforms_subset(dsize, dims=subset_dims))
        npt.assert_equal(np.array(wfs), np.array(expected))

        dsize = [10, 2, 3, 4]
        subset_dims = [2, 3]
        expected = [
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=1),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=4),
            pulse(pts=wf_size, base=0, delay=1, ampl=1, length=7),
        ]
        expected = expected * 4
        wfs = list(wf.get_waveforms_subset(dsize, dims=subset_dims))
        npt.assert_equal(np.array(wfs), np.array(expected))

        dsize = [10, 2, 3]
        subset_dims = [1]
        self.assertRaises(ValueError, wf.get_waveforms_subset, dsize, subset_dims)


class Test_methods(unittest.TestCase):
    def test_split_dims_size(self):
        dsize = [10, 2, 3, 4, 5]
        target_expected = [
            [[], (2 * 3 * 4 * 5, 1, 1)],
            [[1], (1, 2, 3 * 4 * 5)],
            [[2], (2, 3, 4 * 5)],
            [[3], (2 * 3, 4, 5)],
            [[1, 2], (1, 2 * 3, 4 * 5)],
            [[1, 3], (1, 2 * 3 * 4, 5)],
            [[4], (2 * 3 * 4, 5, 1)],
            [[5], (2 * 3 * 4 * 5, 1, 1)],
        ]
        for v in target_expected:
            target, expected = v
            self.assertEqual(split_dims_size(dsize, target), expected)

    def test_unique_dims(self):
        target_expected = [
            [[[2, 3], ], {2, 3}],
            [[[2, 3], [2, ]], {2, 3}],
            [[[2, 3], [1, ]], {1, 2, 3}],
            [[[2, 3], [4, ]], {2, 3, 4}],
            [[[2, 3], [1, 4]], {1, 2, 3, 4}],
        ]
        for v in target_expected:
            target, expected = v
            self.assertEqual(unique_dims(*target), expected)


class Test_WaveformGroup(unittest.TestCase):
    def test_add_waveforms(self):
        wg = WaveformGroup([])
        wg.add_waveform(Waveform())

        self.assertRaises(TypeError, wg.add_waveform, 'random')

    def test_unique_dims(self):
        wf1 = Waveform()
        params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
        wf1.set_func(pulse, params)

        wf2 = Waveform()
        params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
        wf2.set_func(pulse, params)

        wf3 = Waveform()
        params = pulse_params(pts=1, base=0, delay=1, ampl=1, length=10)
        wf3.set_func(pulse, params)

        wg = WaveformGroup([wf1, wf2, wf3])
        self.assertEqual(wg.swept_dims(), set([]))

        wf2.params['delay'].sweep_stepsize(init=0, step_size=2, dim=1)
        wg = WaveformGroup([wf1, wf2, wf3])
        self.assertEqual(wg.swept_dims(), {1})

        wf3.params['delay'].sweep_stepsize(init=0, step_size=2, dim=1)
        wf3.params['length'].sweep_stepsize(init=1, step_size=2, dim=2)
        wg = WaveformGroup([wf1, wf2, wf3])
        self.assertEqual(wg.swept_dims(), {1, 2})

    def test_get_waveforms(self):
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

        dsize = [10, 2, 3]
        wfs = list(wg.get_waveforms(dsize))

        wf1_wfs = list(wf1.get_waveforms(dsize))
        wf2_wfs = list(wf2.get_waveforms(dsize))
        wf3_wfs = list(wf3.get_waveforms(dsize))
        for i, wfi in enumerate(wfs):
            npt.assert_equal(wfi[0], wf1_wfs[i])
            npt.assert_equal(wfi[1], wf2_wfs[i])
            npt.assert_equal(wfi[2], wf3_wfs[i])


if __name__ == '__main__':
    unittest.main()

    # wf = Waveform()
    # from pystruments.funclib import pulse, pulse_params
    #
    # params = pulse_params(pts=1, base=1, delay=1, ampl=1, length=10)
    # params['delay'].sweep_stepsize(init=0, step_size=5, dim=1)
    # wf.set_func(pulse, params)
