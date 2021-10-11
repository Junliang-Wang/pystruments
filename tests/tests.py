import numpy as np

from pystruments.keysight.M8195A import M8195A
from pystruments.keysight.M8197A import M8197A

address_awg_virtual = 'TCPIP0::localhost::hislip4::INSTR'
address_sync_virtual = 'TCPIP0::localhost::hislip0::INSTR'
address_awg1 = 'TCPIP0::localhost::hislip1::INSTR'
address_awg2 = 'TCPIP0::localhost::hislip2::INSTR'
address_sync = 'TCPIP0::localhost::hislip3::INSTR'

d = [
    'sampling_rate_divider',  # 1|2|4
    'sampling_frequency',  # 53.76 GHz to 64 GHz
    'awg_mode',  # SINGle|DUAL|FOUR|MARKer|DCDuplicate|DCMarker
    'sequence_mode',  # ARBitrary|STSequence|STSCenario
    'dynamic_mode',  # Boolean
    'armed_mode',  # SELF|ARMed
    'value',
    'trigger_mode',  # TRIGgered|CONTinuous|GATED1|GATED2
    'trigger_level',  # Between -4.0V and 4.0V
    'trigger_slope',  # POSitive|NEGative|EITHer
    'trigger_sync_mode',  # ASYNchronous|SYNChronous
    'trigger_frequency',  # Between 0.1Hz and 17857142.9Hz
    'trigger_source',  # TRIGger|EVENt|INTernal
    'advance_trigger_source',  # TRIGger|EVENt|INTernal
    'event_trigger_level',  # Between -4.0V and 4.0V
    'event_trigger_slope',  # POSitive|NEGative|EITHer
    'event_trigger_source',  # TRIGger|EVENt
]


def test1():
    param_value = [
        # ['sampling_rate_divider', [1, 2, 4]],  # 1|2|4
        # ['sampling_frequency', [53.76, 60, 64]],  # 53.76 GHz to 64 GHz
        # ['awg_mode', ['SING', 'DUAL', 'FOUR', 'MARK', 'DCD', 'DCM']],
        # SINGle|DUAL|FOUR|MARKer|DCDuplicate|DCMarker
        # ['sequence_mode', ['ARB', 'STS', 'STSC']],  # ARBitrary|STSequence|STSCenario
        # ['dynamic_mode', [True, False]],  # Boolean
        # ['armed_mode', ['SELF', 'ARM']],  # SELF|ARMed
        ['value', [0, 1e-3, 10e-3, 1, 10]],  # 0 to 10 ns
        # ['trigger_mode', ['TRIG', 'CONT', 'GATED1', 'GATED2']],  # TRIGgered|CONTinuous|GATED1|GATED2
        # ['trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
        # ['trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
        # ['trigger_sync_mode', ['ASYN', 'SYNC']],  # ASYNchronous|SYNChronous
        # ['trigger_frequency', [0.1, 1, 10e3, 17857142.8]],  # Between 0.1Hz and 17857142.9Hz
        # ['trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
        # ['advance_trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
        # ['event_trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
        # ['event_trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
        # ['event_trigger_source', ['TRIG', 'EVEN']],  # TRIGger|EVENt
    ]
    address = address_awg_virtual
    with M8195A(address) as awg1:
        for pv in param_value:
            awg1.reset()
            param, values = pv
            fget = getattr(awg1, 'get_{}'.format(param))
            fset = getattr(awg1, 'set_{}'.format(param))
            for v in values:
                fset(v)
                new_v = fget()
                same = v == new_v
                print(same)


def test2():
    param_value = [
        ['armed_mode', ['SELF', 'ARM']],  # SELF|ARMed
        ['trigger_mode', ['TRIG', 'CONT', 'GATED1', 'GATED2']],  # TRIGgered|CONTinuous|GATED1|GATED2
        ['trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
        ['trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
        ['trigger_sync_mode', ['ASYN', 'SYNC']],  # ASYNchronous|SYNChronous
        ['trigger_frequency', [0.1, 1, 10e3, 17857142.8]],  # Between 0.1Hz and 17857142.9Hz
        ['trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
        ['advance_trigger_source', ['TRIG', 'EVEN', 'INT']],  # TRIGger|EVENt|INTernal
        ['event_trigger_level', [-4, -1, 1, 4]],  # Between -4.0V and 4.0V
        ['event_trigger_slope', ['POS', 'NEG', 'EITH']],  # POSitive|NEGative|EITHer
        ['event_trigger_source', ['TRIG', 'EVEN']],  # TRIGger|EVENt
    ]
    address = address_sync_virtual
    with M8197A(address) as awg1:
        for pv in param_value:
            awg1.reset()
            param, values = pv
            fget = getattr(awg1, 'get_{}'.format(param))
            fset = getattr(awg1, 'set_{}'.format(param))
            for v in values:
                fset(v)
                new_v = fget()
                same = v == new_v
                print(same)


def test3():
    param_value = [
        ['memory_mode', ['EXT', 'INT']],  # EXTended|INTernal
        ['value', [0.076, 0.5, 1]],  # 2DO : Vals depend on value and value value...
        ['value', [0, 0.75, 100e-6]],  # 2DO : Vals depend on value and value value...
        ['termination_voltage', [-0.75, 0, 0.75]],  # 2DO : Vals depend on value and value value...
        ['clock_delay', [0, 95]],  # Int between 0 and 95
        ['value', [False, True]],  # Int between 0 and 95
    ]
    channel = 1
    address = address_awg_virtual
    with M8195A(address) as awg1:
        for pv in param_value:
            param, values = pv
            awg1.reset()
            fget = getattr(awg1, 'get_channel_{}'.format(param))
            fset = getattr(awg1, 'set_channel_{}'.format(param))
            for v in values:
                fset(channel, v)
                new_v = fget(channel)
                same = v == new_v
                print(same)


def test4():
    awg_conf = {
        'sampling_rate_divider': 1,  # 1|2|4
        'sampling_frequency': 64,  # 53.76 GHz to 64 GHz
        'awg_mode': 'MARK',  # SINGle|DUAL|FOUR|MARKer|DCDuplicate|DCMarker
        'sequence_mode': 'STS',  # ARBitrary|STSequence|STSCenario
        'dynamic_mode': False,  # Boolean
        'armed_mode': 'SELF',  # SELF|ARMed
        'value': 0,
        'trigger_mode': 'TRIG',  # TRIGgered|CONTinuous|GATED1|GATED2
        'trigger_level': 1,  # Between -4.0V and 4.0V
        'trigger_slope': 'POS',  # POSitive|NEGative|EITHer
        'trigger_sync_mode': 'ASYN',  # ASYNchronous|SYNChronous
        'trigger_frequency': 1,  # Between 0.1Hz and 17857142.9Hz
        'trigger_source': 'TRIG',  # TRIGger|EVENt|INTernal
        'advance_trigger_source': 'TRIG',  # TRIGger|EVENt|INTernal
        'event_trigger_level': 1,  # Between -4.0V and 4.0V
        'event_trigger_slope': 'POS',  # POSitive|NEGative|EITHer
        'event_trigger_source': 'EVEN',  # TRIGger|EVENt
    }
    ch_conf = {
        'memory_mode': 'EXT',  # EXTended|INTernal
        'value': 0.5,  # 2DO : Vals depend on value and value value...
        'value': 0,  # 2DO : Vals depend on value and value value...
        'termination_voltage': 0,  # 2DO : Vals depend on value and value value...
        'clock_delay': 0,  # Int between 0 and 95
        'value': True,  # bool
    }
    size = M8195A.granularity
    pulse1 = np.zeros((size, 3))
    pulse1[1:10, 0] = 1
    pulse1[2:11, 1] = 1
    pulse1[3:12, 2] = 1

    # pulse1 = np.zeros(size)
    # pulse1[1:11] = np.linspace(-1, 1, 10)

    pulse2 = np.zeros_like(pulse1)
    # pulse2[10:20] = 1

    address = address_awg_virtual
    instr = M8195A(address)
    instr.open_com()
    instr.reset()
    instr.set_awg_config(awg_conf)
    instr.set_channel_config(1, ch_conf)
    instr.set_waveform_to_segment(channel=1, segment_number=1, waveform=list(pulse1))
    instr.set_waveform_to_segment(channel=1, segment_number=2, waveform=list(pulse2))
    instr.reset_sequence_table()
    instr.set_sequence_entry(sequence_id=0, segment_id=1, new_sequence=True, segment_advancement_mode='single')
    instr.set_sequence_entry(sequence_id=1, segment_id=2, end_sequence=True, segment_advancement_mode='single',
                             end_scenario=True)
    return instr


if __name__ == '__main__':
    pass
    address_awg_virtual = 'TCPIP0::localhost::hislip4::INSTR'
    address_sync_virtual = 'TCPIP0::localhost::hislip0::INSTR'
    address_awg1 = 'TCPIP0::localhost::hislip1::INSTR'
    address_awg2 = 'TCPIP0::localhost::hislip2::INSTR'
    address_sync = 'TCPIP0::localhost::hislip3::INSTR'

    awg = M8195A(address_awg_virtual)
    awg.open_com()
    awg.reset()
    awg.set_awg_mode('DCM')
    awg.set_sampling_rate_divider(2)
    awg.set_sequence_mode('STS')
    ch1 = awg.channels[1]
    ch2 = awg.channels[2]
    ch1.set_memory_mode('EXT')
    ch2.set_memory_mode('INT')
    awg.delete_all_segments()

    m = 2
    wf = [-1, 1, 0, 0] + [0] * (1280 * m - 4)
    # mk1 = [0, 1, 0, 1] + [0] * (1280*m - 4)
    # mk2 = [0, 0, 1, 1] + [0] * (1280*m - 4)
    mk1 = None
    mk2 = None
    wf_str = awg._waveform_to_str(wf, mk1, mk2)
    print(wf_str)
    for n in [1, 2]:
        awg.set_waveform_to_segment(
            channel=1, waveform=wf, marker1=mk1, marker2=mk2,
            segment_id=n, offset=0,
        )

    # # waveform, mk1, mk2 = awg.get_waveform_from_segment(channel=1, segment_id=1, offset=0, max_length=len(waveform))
    wf_str = awg._get_waveform_from_segment(channel=1, segment_id=1, offset=0)
#
