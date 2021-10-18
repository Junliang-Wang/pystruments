import numpy as np

from M8195A import M8195A
from waveform import Waveform


class AWGWaveform(Waveform):
    def __init__(self, n_channels=4, *args, **kwargs):
        super(AWGWaveform, self).__init__(*args, **kwargs)
        for n in range(n_channels):
            n = n + 1
            channel = Waveform(name='ch{}'.format(n))
            self.add_child(channel)

    @property
    def channels(self):
        return {i + 1: child for i, child in enumerate(self.childs)}

    def get_dict(self):
        d = super(AWGWaveform, self).get_dict()
        del d['func']
        del d['func_params']
        return d


class SyncWaveform(Waveform):
    def __init__(self, n_awgs, n_channels=4, *args, **kwargs):
        super(SyncWaveform, self).__init__(*args, **kwargs)
        for n in range(n_awgs):
            n = n + 1
            awg = AWGWaveform(name='AWG{}'.format(n), n_channels=n_channels)
            self.add_child(awg)

    def get_dict(self):
        d = super(SyncWaveform, self).get_dict()
        del d['func']
        del d['func_params']
        return d


class M8195A_sequencer(AWGWaveform):
    def __init__(self, awg, *args, **kwargs):
        if not isinstance(awg, M8195A):
            raise ValueError('AWG must be an instance of {}'.format(M8195A))
        self.awg = awg
        n_channels = len(self.awg.channels)
        super(M8195A_sequencer, self).__init__(n_channels=n_channels, *args, **kwargs)

    # # # @profile
    def generate_sequence(self, dims, n_empty=0, start_with_empty=True, **entry_kwargs):
        self._prepare_sequence()
        self.set_dims(dims, reduced=True)
        self._fill_segments()
        self._fill_sequence_table(
            n_empty=n_empty,
            start_with_empty=start_with_empty,
            **entry_kwargs
        )

    def _prepare_sequence(self):
        self.awg.stop()
        self.awg.set_sequence_mode('STS')
        self.awg.reset_sequence_table()
        for ch in self.awg.childs:
            ch.delete_all_segments()

    # # @profile
    def _fill_segments(self):
        wf_channels = self.channels
        last_ids = []
        for n, channel in self.awg.active_channels.items():
            wf_ch = wf_channels[n]
            wf_values = wf_ch.value_generator()

            segment_id = 1  # use 1 for dummy waveform (all 0V)
            channel.define_segment(
                segment_id=segment_id,
                init_value=0.0,
            )
            for values in wf_values:
                segment_id += 1
                channel.set_waveform_to_segment(
                    segment_id=segment_id,
                    waveform=values,
                )
            last_ids.append(segment_id)
        same_last_id = len(set(last_ids)) == 1
        if not same_last_id:
            raise ValueError('Channel waveforms have different sizes')

    def _fill_sequence_table(self, n_empty=0, start_with_empty=True, **entry_kwargs):
        use_marker = self.awg.is_using_markers()
        default_kwargs = dict(
            segment_loop=1,
            segment_advancement_mode='cond',
            sequence_loop=1,
            sequence_advancement_mode='auto',
        )
        for key in default_kwargs.keys():
            if key not in entry_kwargs.keys():
                continue
            default_kwargs[key] = entry_kwargs[key]

        n_segments = int(np.prod(self.dims[1:]))
        seq_size = n_segments * (1 + n_empty)
        last_seq_id = 0
        for i in range(n_segments):
            segment_id = i + 2  # id 1 is a dummy segment

            if start_with_empty:
                segment_ids = [1] * n_empty + [segment_id]
            else:
                segment_ids = [segment_id] + [1] * n_empty

            for seg_id in segment_ids:
                new_sequence = True if last_seq_id == 0 else False
                end_sequence = True if last_seq_id == seq_size - 1 else False
                end_scenario = end_sequence

                self.awg.set_sequence_entry(
                    sequence_id=last_seq_id,
                    segment_id=seg_id,
                    entry_type='data',
                    use_marker=use_marker,
                    new_sequence=new_sequence,
                    end_sequence=end_sequence,
                    end_scenario=end_scenario,
                    **default_kwargs
                )
                last_seq_id += 1


def get_awg_sequencer(awg):
    n_channels = len(awg.channels)
    waveform = AWGWaveform(n_channels=n_channels)
    sequencer = AWGSequencer(awg, waveform)
    return sequencer


class AWGSequencer(object):
    def __init__(self, awg, waveform):
        if not isinstance(awg, M8195A):
            raise ValueError('AWG must be an instance of {}'.format(M8195A))
        if not isinstance(waveform, AWGWaveform):
            raise ValueError('AWG must be an instance of {}'.format(AWGWaveform))
        self.awg = awg
        self.waveform = waveform

    def generate_sequence(self, dims, n_empty=0, start_with_empty=True, **entry_kwargs):
        self._prepare_sequence()
        self.waveform.set_dims(dims, reduced=True)
        self._fill_segments()
        self._fill_sequence_table(
            n_empty=n_empty,
            start_with_empty=start_with_empty,
            **entry_kwargs
        )

    def _prepare_sequence(self):
        self.awg.stop()
        self.awg.set_sequence_mode('STS')
        self.awg.reset_sequence_table()
        for ch in self.awg.childs:
            ch.delete_all_segments()

    def _fill_segments(self):
        wf_channels = self.waveform.channels
        last_ids = []
        for n, channel in self.awg.active_channels.items():
            wf_ch = wf_channels[n]
            wf_values = wf_ch.value_generator()

            segment_id = 1  # use 1 for dummy waveform (all 0V)
            channel.define_segment(
                segment_id=segment_id,
                init_value=0.0,
            )
            for values in wf_values:
                segment_id += 1
                channel.set_waveform_to_segment(
                    segment_id=segment_id,
                    waveform=values,
                )
            last_ids.append(segment_id)
        same_last_id = len(set(last_ids)) == 1
        if not same_last_id:
            raise ValueError('Channel waveforms have different sizes')

    def _fill_sequence_table(self, n_empty=0, start_with_empty=True, **entry_kwargs):
        use_marker = self.awg.is_using_markers()
        default_kwargs = dict(
            segment_loop=1,
            segment_advancement_mode='cond',
            sequence_loop=1,
            sequence_advancement_mode='auto',
        )
        for key in default_kwargs.keys():
            if key not in entry_kwargs.keys():
                continue
            default_kwargs[key] = entry_kwargs[key]

        n_segments = int(np.prod(self.waveform.dims[1:]))
        seq_size = n_segments * (1 + n_empty)
        last_seq_id = 0
        for i in range(n_segments):
            segment_id = i + 2  # id 1 is a dummy segment

            if start_with_empty:
                segment_ids = [1] * n_empty + [segment_id]
            else:
                segment_ids = [segment_id] + [1] * n_empty

            for seg_id in segment_ids:
                new_sequence = True if last_seq_id == 0 else False
                end_sequence = True if last_seq_id == seq_size - 1 else False
                end_scenario = end_sequence

                self.awg.set_sequence_entry(
                    sequence_id=last_seq_id,
                    segment_id=seg_id,
                    entry_type='data',
                    use_marker=use_marker,
                    new_sequence=new_sequence,
                    end_sequence=end_sequence,
                    end_scenario=end_scenario,
                    **default_kwargs
                )
                last_seq_id += 1

    # TODO !!
    def get_dict(self):
        pass
        # d = super(AWGoutput, self).get_dict()
        # d['config_params'] = {}
        # for key, parameter in self.config_params.items():
        #     d['config_params'][key] = parameter.get_dict()
        # d[self._childs_name] = {}
        # for i, child in enumerate(self.childs):
        #     d[self._childs_name][i] = child.get_dict()
        # return d


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
    awg.delete_all_segments()
    awg.set_awg_mode('DUAL')
    awg.set_sampling_rate_divider(2)
    awg.set_trigger_mode('TRIG')
    awg.channels[1].set_status(True)
    awg.channels[4].set_status(True)
    awg.set_sequence_mode('STS')
    ch1 = awg.channels[1]
    ch2 = awg.channels[4]
    ch1.set_memory_mode('EXT')
    ch2.set_memory_mode('EXT')

    sequencer = M8195A_sequencer(awg)
    from funclib import pulse, pulse_params

    ch1 = sequencer.channels[1]
    ch4 = sequencer.channels[4]

    ch1.name = 'ch1'
    ch4.name = 'ch4'

    func = pulse
    func_params = pulse_params(
        pts=1,
        base=0,
        delay=2,
        ampl=1,
        length=10,
    )
    func_params['ampl'].sweep_linear(1, -1, dim=1)
    ch1.set_func(func, func_params)

    func_params = pulse_params(
        pts=1,
        base=0,
        delay=1,
        ampl=1,
        length=3,
    )
    func_params['length'].sweep_stepsize(0, 1, dim=2)
    ch4.set_func(func, func_params)

    import timeit

    num_runs = 5
    x1 = np.arange(1, 102, 10)
    x2 = np.arange(1280, 12801, 1280)
    y = np.ndarray((x1.size, x2.size))
    for i, x1i in enumerate(x1):
        for j, x2i in enumerate(x2):
            dims = [x2i, x1i, 1]


            def f():
                sequencer.generate_sequence(dims, n_empty=0, start_with_empty=True)


            duration = timeit.Timer(f).timeit(number=num_runs)
            avg_duration = duration / num_runs
            # print('On average it took {} seconds'.format(avg_duration))
            y[i, j] = avg_duration
    # f()
    # dims = [12800, 50, 1]

    # sequencer.generate_sequence(dims, n_empty=0, start_with_empty=True)

    # # awg_seq = AWGSequencer(awg=awg, waveform=awg_wf)
    # # awg_seq.generate_sequence(dims, n_empty=1)
    # #
    # # awg = M8195A(...)
    # # sequencer = awg.get_sequencer()
    # # sequencer.channels[1].set_func(func, params)
    # # sequencer.generate_sequence(dims=[1280, 1, 2])
