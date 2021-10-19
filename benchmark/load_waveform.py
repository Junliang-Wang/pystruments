import numpy as np
from pystruments.keysight import M8195A, M8197A
from pystruments.funclib import pulse, pulse_params

address_awg_virtual = 'TCPIP0::localhost::hislip4::INSTR'
address_sync_virtual = 'TCPIP0::localhost::hislip0::INSTR'
address_sync = 'TCPIP0::localhost::hislip1::INSTR'
address_awg1 = 'TCPIP0::localhost::hislip2::INSTR'
address_awg2 = 'TCPIP0::localhost::hislip3::INSTR'


def test1():
    awg = M8195A(address_awg1)
    awg.open_com()
    awg.stop()
    awg.reset()
    awg.delete_all_segments()
    awg.set_awg_mode('DUAL')
    awg.set_sampling_frequency(64)
    awg.set_sampling_rate_divider(2)
    awg.set_armed_mode('SELF')
    awg.set_trigger_mode('TRIG')
    awg.set_advance_trigger_source('TRIG')
    awg.set_event_trigger_source('TRIG')

    awg.channels[1].set_status(True)
    awg.channels[4].set_status(True)
    awg.set_sequence_mode('STS')
    ch1 = awg.channels[1]
    ch2 = awg.channels[4]
    ch1.set_memory_mode('EXT')
    ch2.set_memory_mode('EXT')

    sequencer = awg.get_sequencer()

    ch1_seq = sequencer.channels[1]
    ch4_seq = sequencer.channels[4]

    ch1_seq.name = 'ch1'
    ch4_seq.name = 'ch4'

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=10,
    )
    ch1_seq.set_func(func, func_params)

    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=10,
    )
    func_params['length'].sweep_stepsize(10, 10, dim=1)
    func_params['ampl'].sweep_linear(-1, 1, dim=2)
    ch4_seq.set_func(func, func_params)

    dims = [1280, 5, 5]
    sequencer.generate_sequence(dims, n_empty=0, start_with_empty=True)
    return awg


def test2():
    sync = M8197A(address_sync)
    sync.open_com()
    sync.reset()
    sync.enslave_all()
    sync.set_sampling_frequency(64)

    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    awg1 = sync.awgs[2]
    awg2 = sync.awgs[3]

    func = pulse
    func_params1 = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=100,
    )
    func_params1['length'].sweep_linear(100, 100, dim=1)

    func_params2 = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=100,
    )
    func_params2['length'].sweep_stepsize(10, 100, dim=1)
    # func_params2['ampl'].sweep_linear(-1, 1, dim=2)

    fparams = [func_params1, func_params2]
    dims = [[1280, 5], [1280, 5]]
    for i, awg in enumerate([awg1, awg2]):
        # awg.reset()
        awg.set_awg_mode('SING')
        awg.delete_all_segments()
        awg.reset_sequence_table()
        awg.set_sampling_frequency(64)
        awg.set_sampling_rate_divider(1)

        awg.channels[1].set_status(True)
        awg.channels[1].set_memory_mode('EXT')
        awg.set_sequence_mode('STS')

        sequencer = awg.get_awg_sequencers()
        ch1_seq = sequencer.channels[1]
        ch1_seq.set_func(func, fparams[i])
        sequencer.generate_sequence(dims[i], n_empty=0, start_with_empty=True)
    sync.set_config_mode(False)
    sync.run()
    return sync


def test3():
    sync = M8197A(address_sync)
    sync.open_com()
    sync.reset()
    sync.enslave_all()
    sync.set_sampling_frequency(64)

    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    awg1 = sync.awgs[2]
    awg2 = sync.awgs[3]

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=50,
    )

    awg1.set_awg_mode('SING')
    awg1.delete_all_segments()
    awg1.reset_sequence_table()
    awg1.set_sampling_rate_divider(1)
    awg1.channels[1].set_status(True)
    awg1.channels[1].set_memory_mode('EXT')
    awg1.set_sequence_mode('STS')
    sequencer = awg1.get_awg_sequencers()

    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    # ch1_seq.params['length'].sweep_linear(100, 100, dim=1)
    sequencer.generate_sequence([1280*2, 5], n_empty=0, start_with_empty=True)

    awg2.set_awg_mode('DUAL')
    awg2.delete_all_segments()
    awg2.reset_sequence_table()
    awg2.set_sampling_rate_divider(2)
    awg2.channels[1].set_status(True)
    awg2.channels[4].set_status(True)
    awg2.channels[1].set_memory_mode('EXT')
    awg2.channels[4].set_memory_mode('EXT')
    awg2.set_sequence_mode('STS')

    sequencer = awg2.get_awg_sequencers()
    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    # ch1_seq.params['length'].sweep_stepsize(10, 10, dim=1)
    # sequencer.generate_sequence([1280, 5], n_empty=0, start_with_empty=True)

    ch4_seq = sequencer.channels[4]
    ch4_seq.set_func(func, func_params)
    # ch4_seq.params['ampl'].sweep_linear(-1, 1, dim=1)
    sequencer.generate_sequence([1280, 5], n_empty=0, start_with_empty=True)

    sync.set_config_mode(False)
    # sync.run()
    return sync



def test4():
    sync = M8197A(address_sync)
    sync.open_com()
    sync.reset()
    sync.enslave_all()
    sync.set_sampling_frequency(64)
    sync.set_clock_out_sample_divider(200)
    sync.set_clock_out_source('SCLK1')
    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    awg1 = sync.awgs[2]
    awg2 = sync.awgs[3]

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=50,
    )

    awg1.set_awg_mode('SING')
    awg1.delete_all_segments()
    awg1.reset_sequence_table()
    awg1.set_sampling_rate_divider(1)
    awg1.channels[1].set_status(True)
    awg1.channels[1].set_memory_mode('EXT')
    awg1.set_sequence_mode('STS')
    sequencer = awg1.get_awg_sequencers()

    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    ch1_seq.params['length'].sweep_linear(100, 100, dim=1)
    sequencer.generate_sequence([1280*2, 5], n_empty=0, start_with_empty=True)

    awg2.set_awg_mode('DUAL')
    awg2.delete_all_segments()
    awg2.reset_sequence_table()
    awg2.set_sampling_rate_divider(2)
    awg2.channels[1].set_status(True)
    awg2.channels[4].set_status(True)
    awg2.channels[1].set_memory_mode('EXT')
    awg2.channels[4].set_memory_mode('EXT')
    awg2.set_sequence_mode('STS')

    sequencer = awg2.get_awg_sequencers()
    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    ch1_seq.params['length'].sweep_stepsize(10, 10, dim=1)

    ch4_seq = sequencer.channels[4]
    ch4_seq.set_func(func, func_params)
    ch4_seq.params['ampl'].sweep_linear(-1, 1, dim=1)
    sequencer.generate_sequence([1280, 5], n_empty=0, start_with_empty=True)

    sync.set_config_mode(False)
    sync.run()
    return sync


def test5():
    sync = M8197A(address_sync)
    sync.open_com()
    sync.reset()
    sync.enslave_all()
    sync.set_sampling_frequency(64)
    sync.set_clock_out_sample_divider(200)
    sync.set_clock_out_source('SCLK1')
    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    awg1 = sync.awgs[2]
    awg2 = sync.awgs[3]

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=50,
    )

    awg1.set_awg_mode('SING')
    awg1.delete_all_segments()
    awg1.reset_sequence_table()
    awg1.set_sampling_rate_divider(1)
    awg1.channels[1].set_status(True)
    awg1.channels[1].set_memory_mode('EXT')
    awg1.set_sequence_mode('STS')
    sequencer = awg1.get_awg_sequencers()

    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    ch1_seq.params['length'].sweep_linear(100, 100, dim=1)
    sequencer.generate_sequence([1280, 5*2], n_empty=0, start_with_empty=True)

    awg2.set_awg_mode('DUAL')
    awg2.delete_all_segments()
    awg2.reset_sequence_table()
    awg2.set_sampling_rate_divider(2)
    awg2.channels[1].set_status(True)
    awg2.channels[4].set_status(True)
    awg2.channels[1].set_memory_mode('EXT')
    awg2.channels[4].set_memory_mode('EXT')
    awg2.set_sequence_mode('STS')

    sequencer = awg2.get_awg_sequencers()
    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    ch1_seq.params['length'].sweep_stepsize(10, 10, dim=1)

    ch4_seq = sequencer.channels[4]
    ch4_seq.set_func(func, func_params)
    ch4_seq.params['ampl'].sweep_linear(-1, 1, dim=1)
    sequencer.generate_sequence([1280, 5], n_empty=1, start_with_empty=True)

    sync.set_config_mode(False)
    sync.run()
    return sync


def test6():
    sync = M8197A(address_sync)
    sync.open_com()
    sync.reset()
    sync.enslave_all()
    sync.set_sampling_frequency(64)
    sync.set_clock_out_sample_divider(200)
    sync.set_clock_out_source('SCLK1')
    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    awg1 = sync.awgs[2]
    awg2 = sync.awgs[3]

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=1000,
    )

    awg1.set_awg_mode('SING')
    awg1.delete_all_segments()
    awg1.reset_sequence_table()
    awg1.set_sampling_rate_divider(1)
    awg1.channels[1].set_status(True)
    awg1.channels[1].set_memory_mode('EXT')
    awg1.set_sequence_mode('STS')
    sequencer = awg1.get_awg_sequencers()

    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    sequencer.generate_sequence([1280, 1], n_empty=0, start_with_empty=True)

    awg2.set_awg_mode('DUAL')
    awg2.delete_all_segments()
    awg2.reset_sequence_table()
    awg2.set_sampling_rate_divider(2)
    awg2.channels[1].set_status(False)
    awg2.channels[4].set_status(False)
    awg2.channels[1].set_memory_mode('EXT')
    awg2.channels[4].set_memory_mode('EXT')
    awg2.set_sequence_mode('STS')
    sequencer = awg2.get_awg_sequencers()
    ch1_seq = sequencer.channels[1]
    ch1_seq.set_func(func, func_params)
    sequencer.generate_sequence([1280, 1], n_empty=0, start_with_empty=True)

    sync.set_config_mode(False)
    sync.run()
    return sync

if __name__ == '__main__':
    # awg = test1()
    # sync = test2()
    # sync = test3()
    # sync = test3()
    # sync = test4()
    # sync = test5()
    sync = test6()