import unittest
from pystruments.keysight.M8195A import M8195A as AWG
from address import virtual_awg


class Test_AWG_params(unittest.TestCase):
    def test_valids(self):
        param_value = [
            ['sampling_rate_divider', [1, 2, 4]],
            ['sampling_frequency', [53.76, 60, 64]],
            ['awg_mode', ['SING', 'DUAL', 'FOUR', 'MARK', 'DCD', 'DCM']],
            ['sequence_mode', ['ARB', 'STS', 'STSC']],
            ['dynamic_mode', [True, False]],
            ['armed_mode', ['SELF', 'ARM']],
            ['delay', [0, 1e-3, 10e-3, 1, 10]],
            ['trigger_mode', ['TRIG', 'CONT', 'GATED1', 'GATED2']],
            ['trigger_level', [-4, -1, 1, 4]],
            ['trigger_slope', ['POS', 'NEG']],
            ['trigger_sync_mode', ['ASYN', 'SYNC']],
            ['trigger_frequency', [0.1, 1, 10e3, 17857142.8]],
            ['trigger_source', ['TRIG', 'EVEN', 'INT']],
            ['advance_trigger_source', ['TRIG', 'EVEN', 'INT']],
            ['event_trigger_level', [-4, -1, 1, 4]],
            ['event_trigger_slope', ['POS', 'NEG', 'EITH']],
            ['event_trigger_source', ['TRIG', 'EVEN']],
        ]
        address = virtual_awg
        with AWG(address) as instr:
            for pv in param_value:
                param, values = pv
                instr.reset()
                fset = getattr(instr, 'set_{}'.format(param))
                for v in values:
                    fset(v)

    def test_invalid_parameters(self):
        param_value = [
            ['sampling_rate_divider', ['RANDOM', 0.5, -1, 5]],
            ['sampling_frequency', ['RANDOM', 0, 70]],
            ['awg_mode', ['RANDOM', 1]],
            ['sequence_mode', ['RANDOM', 1]],
            ['dynamic_mode', ['TRUE']],
            ['armed_mode', ['RANDOM', 1]],
            ['delay', ['RANDOM', -1, 11]],
            ['trigger_mode', ['RANDOM', 1]],
            ['trigger_level', ['RANDOM', -5, 5]],
            ['trigger_slope', ['RANDOM', 1]],
            ['trigger_sync_mode', ['RANDOM', 1]],
            ['trigger_frequency', ['RANDOM', 0, 1e8]],
            ['trigger_source', ['RANDOM', 1]],
            ['advance_trigger_source', ['RANDOM', 1]],
            ['event_trigger_level', ['RANDOM', -5, 5]],
            ['event_trigger_slope', ['RANDOM', 1]],
            ['event_trigger_source', ['RANDOM', 1]],
        ]
        address = virtual_awg
        with AWG(address) as instr:
            instr.reset()
            for pv in param_value:
                param, values = pv
                fset = getattr(instr, 'set_{}'.format(param))

                def _f(v):
                    try:
                        fset(v)
                        print(param, v)
                    except:
                        raise ValueError()

                for v in values:
                    self.assertRaises(ValueError, _f, v)


class Test_channel_params(unittest.TestCase):
    def test_valids(self):
        param_value = [
            ['memory_mode', ['EXT', 'INT']],
            ['amplitude', [0.076, 0.5, 1]],
            ['offset', [0, 0.75, 100e-6]],
            ['termination_voltage', [-0.75, 0, 0.75]],
            ['clock_delay', [0, 95]],
            ['status', [False, True]],
        ]
        address = virtual_awg
        with AWG(address) as instr:
            for pv in param_value:
                param, values = pv
                instr.reset()
                channel = instr.get_channel(1)
                fset = getattr(channel, 'set_{}'.format(param))
                for v in values:
                    fset(v)

    def test_invalid_parameters(self):
        param_value = [
            ['memory_mode', ['RANDOM', 1]],  # EXTended|INTernal
            ['amplitude', ['RANDOM', 2, -2]],  # 2DO : Vals depend on value and value value...
            ['offset', ['RANDOM', 2, -2]],  # 2DO : Vals depend on value and value value...
            ['termination_voltage', ['RANDOM', 2, -2]],  # 2DO : Vals depend on value and value value...
            ['clock_delay', ['RANDOM', -1, 96]],  # Int between 0 and 95
            ['status', ['RANDOM', 2]],  # Int between 0 and 95
        ]
        address = virtual_awg
        with AWG(address) as instr:
            instr.reset()
            channel = instr.get_channel(1)

            for pv in param_value:
                param, values = pv
                fset = getattr(channel, 'set_{}'.format(param))

                def _f(v):
                    try:
                        fset(v)
                        print(param, v)
                    except:
                        raise ValueError()

                for v in values:
                    self.assertRaises(ValueError, _f, v)


if __name__ == '__main__':
    unittest.main()
