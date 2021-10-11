import unittest

from address import virtual_sync
from pystruments.keysight.M8197A import M8197A as Sync


class Test_sync_params(unittest.TestCase):
    def test_valids(self):
        param_value = [
            ['armed_mode', ['SELF', 'ARM']],
            ['trigger_mode', ['TRIG', 'CONT', 'GATED1', 'GATED2']],
            ['trigger_level', [-4, -1, 1, 4]],
            ['trigger_slope', ['POS', 'NEG', 'EITH']],
            ['trigger_sync_mode', ['ASYN', 'SYNC']],
            ['trigger_frequency', [0.1, 1, 10e3, 17857142.8]],
            ['trigger_source', ['TRIG', 'EVEN', 'INT']],
            ['advance_trigger_source', ['TRIG', 'EVEN', 'INT']],
            ['event_trigger_level', [-4, -1, 1, 4]],
            ['event_trigger_slope', ['POS', 'NEG', 'EITH']],
            ['event_trigger_source', ['TRIG', 'EVEN']],
            ['config_mode', [True, False]],
            ['sampling_frequency', [53.76, 60, 64]],
            ['clock_out_source', ['INT', 'EXT', 'SCLK1', 'SCLK2']],
            ['clock_out_sample_divider', [1, 200, 1024]],
            ['clock_ref_source', ['INT', 'EXT', 'AXI']],
            ['clock_ref_range', [1, 2]],
            ['clock_ref_frequency', [100, 200]],
            ['clock_ref_divider1', [1, 8]],
            ['clock_ref_divider2', [1, 8]],

        ]
        address = virtual_sync
        with Sync(address) as instr:
            for pv in param_value:
                param, values = pv
                instr.reset()
                fset = getattr(instr, 'set_{}'.format(param))
                for v in values:
                    fset(v)

    def test_invalid_parameters(self):
        param_value = [
            ['armed_mode', ['RANDOM', 1]],
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
            ['config_mode', ['RANDOM', 2]],
            ['sampling_frequency', ['RANDOM', 1, 66]],
            ['clock_out_source', ['RANDOM', 1]],
            ['clock_out_sample_divider', ['RANDOM', 0, 1025]],
            ['clock_ref_source', ['RANDOM', 1]],
            ['clock_ref_range', ['RANDOM', 0, 3]],
            ['clock_ref_frequency', ['RANDOM', 9, 18e3]],
            ['clock_ref_divider1', ['RANDOM', 0, 9]],
            ['clock_ref_divider2', ['RANDOM', 0, 9]],
        ]
        address = virtual_sync
        with Sync(address) as instr:
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


# class Test_master(unittest.TestCase):
#     def test_enslave(self):
#         param_value = [
#             ['armed_mode', ['SELF', 'ARM']],  # SELF|ARMed
#             ['trigger_mode', ['TRIG', 'CONT', 'GATED1', 'GATED2']],  # TRIGgered|CONTinuous|GATED1|GATED2
#             ['trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
#             ['trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
#             ['trigger_sync_mode', ['ASYN', 'SYNC']],  # ASYNchronous|SYNChronous
#             ['trigger_frequency', [0.1, 1, 10e3, 17857142.8]],  # Between 0.1Hz and 17857142.9Hz
#             ['trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
#             ['advance_trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
#             ['event_trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
#             ['event_trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
#             ['event_trigger_source', ['TRIG', 'EVEN']],  # TRIGger|EVENt
#             ['config_mode', [True, False]],  # bool
#         ]
#
#         address = virtual_sync
#         with Sync(address) as instr:
#             for pv in param_value:
#                 param, values = pv
#                 instr.reset()
#                 fset = getattr(instr, 'set_{}'.format(param))
#                 for v in values:
#                     fset(v)
#
#     def test_invalid_parameters(self):
#         param_value = [
#             ['armed_mode', ['RANDOM', 1]],  # SELF|ARMed
#             ['trigger_mode', ['RANDOM', 1]],  # TRIGgered|CONTinuous|GATED1|GATED2
#             ['trigger_level', ['RANDOM', -5, 5]],  # Between -4.0V and 4.0V
#             ['trigger_slope', ['RANDOM', 1]],  # POSitive|NEGative|EITHer
#             ['trigger_sync_mode', ['RANDOM', 1]],  # ASYNchronous|SYNChronous
#             ['trigger_frequency', ['RANDOM', 0, 1e8]],  # Between 0.1Hz and 17857142.9Hz
#             ['trigger_source', ['RANDOM', 1]],  # TRIGger|EVENt|INTernal
#             ['advance_trigger_source', ['RANDOM', 1]],  # TRIGger|EVENt|INTernal
#             ['event_trigger_level', ['RANDOM', -5, 5]],  # Between -4.0V and 4.0V
#             ['event_trigger_slope', ['RANDOM', 1]],  # POSitive|NEGative|EITHer
#             ['event_trigger_source', ['RANDOM', 1]],  # TRIGger|EVENt
#             ['config_mode', ['RANDOM', 1]],  # bool
#         ]
#         address = virtual_sync
#         with Sync(address) as instr:
#             instr.reset()
#             for pv in param_value:
#                 param, values = pv
#                 fset = getattr(instr, 'set_{}'.format(param))
#                 for v in values:
#                     self.assertRaises(Exception, fset, v)


if __name__ == '__main__':
    unittest.main()
