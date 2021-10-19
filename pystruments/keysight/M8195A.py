import collections

from pystruments.instrument import InstrumentBase, get_decorator, set_decorator
from pystruments.parameter import Parameter
from pystruments.utils import *
from pystruments.waveform import WaveformGroup, Waveform

granularity = 1280
_validators = {
    'segment_id': Parameter('id', value=1, min_value=1, valid_types=[int]),
    'max_length': Parameter('max_length', value=granularity, valid_multiples=[granularity], valid_types=[int]),
    'init_value': Parameter('init_value', value=0, unit='V', min_value=-1, max_value=1),
    'offset': Parameter('offset', value=0, unit='samples', min_value=0, valid_types=[int]),
    'waveform': Parameter('waveform', value=0, unit='V', min_value=-1, max_value=1),
    'entry_type': Parameter('entry_type', value='data', valid_values=['data', 'idle', 'empty']),
    'loop': Parameter('loop', value=1, min_value=1, valid_types=[int]),
    'advancement_mode': Parameter('advancement_mode', value='auto', valid_values=['auto', 'cond', 'repeat', 'single']),
    'sequence_id': Parameter('id', value=1, min_value=0, valid_types=[int]),
}


class M8195A(InstrumentBase):
    granularity = granularity  # Length of segments must be multiple of granularity
    default_parameters = (
        Parameter('sampling_rate_divider', value=1, valid_values=[1, 2, 4], valid_types=[int]),
        Parameter('sampling_frequency', value=64, unit='GHz', min_value=53.76, max_value=65),
        Parameter('awg_mode', value='SING', valid_values=['SING', 'DUAL', 'FOUR', 'MARK', 'DCD', 'DCM']),
        Parameter('sequence_mode', value='ARB', valid_values=['ARB', 'STS', 'STSC']),
        Parameter('dynamic_mode', value=False, valid_values=[True, False]),
        Parameter('armed_mode', value='SELF', valid_values=['SELF', 'ARM']),
        Parameter('delay', value=0, unit='ns', min_value=0, max_value=10),
        Parameter('trigger_mode', value='CONT', valid_values=['TRIG', 'CONT', 'GATED1', 'GATED2']),
        Parameter('trigger_level', value=4, unit='V', min_value=-4.0, max_value=4.0, valid_multiples=[1e-3]),
        Parameter('trigger_slope', value='POS', valid_values=['POS', 'NEG', 'EITH']),
        Parameter('trigger_sync_mode', value='ASYN', valid_values=['ASYN', 'SYNC']),
        Parameter('trigger_frequency', value=1, unit='Hz', min_value=0.1, max_value=17857142.9),
        Parameter('trigger_source', value='INT', valid_values=['TRIG', 'EVEN', 'INT']),
        Parameter('advance_trigger_source', value='INT', valid_values=['TRIG', 'EVEN', 'INT']),
        Parameter('event_trigger_level', value=4, unit='V', min_value=-4.0, max_value=4.0, valid_multiples=[1e-3]),
        Parameter('event_trigger_slope', value='POS', valid_values=['POS', 'NEG', 'EITH']),
        Parameter('event_trigger_source', value='EVEN', valid_values=['TRIG', 'EVEN']),
        Parameter('slave_status', value='NORM', valid_values=['NORM', 'SLAV']),
        Parameter('slot_number', value=1),
    )
    """
    DICTIONARY TO SETUP OR ANALYSE CONTROL PARAMETER OF SEQUENCE ENTRIES

    STRUCTURE:
     - key:    Column-name of sequence table.
               (see soft panel in 'Sequence/Control'-tab of M8195A)
     - value:  List containing:
               1. List of bit-positions to control property defined by 'key'.
               2. Dictionary of settings (key) with corresponding bit sequence (value)
    """
    _control_parameters = collections.OrderedDict({
        'entry_type': [[31], {'idle': [True], 'empty': [True], 'data': [False]}],

        'segment_advancement_mode': [
            [16, 17], {'cond': [True, False], 'repeat': [False, True],
                       'single': [True, True], 'auto': [False, False]}
        ],

        'use_marker': [[24], {True: [True], False: [False]}],

        'new_sequence': [[28], {True: [True], False: [False]}],

        'sequence_advancement_mode': [
            [20, 21], {'cond': [True, False], 'repeat': [False, True],
                       'single': [True, True], 'auto': [False, False]}
        ],

        'end_scenario': [[29], {True: [True], False: [False]}],
        'end_sequence': [[30], {True: [True], False: [False]}],
    })

    def __init__(self, *args, **kwargs):
        if 'timeout' not in kwargs.keys():
            kwargs['timeout'] = 10000  # in ms
        super(M8195A, self).__init__(*args, **kwargs)
        for n in [1, 2, 3, 4]:
            ch = M8195A_channel(n=n, parent=self, name='ch{}'.format(n))
            self.add_child(ch)
        self.create_sequencer()

    def send(self, cmd, wait_to_complete=True):
        super(M8195A, self).send(cmd)
        if wait_to_complete:
            self.read('*OPT?')

    def run(self):
        self.send(':INIT:IMM')

    def stop(self):
        self.send(':ABOR')

    def reset(self):
        self.send('*RST')
        self.send('*CLS')

    def identify(self):
        """
        Identify the Sync by flashing the green LED on the front panel.
        """
        self.send(':INST:IDEN')

    def is_slave(self):
        status = self.get_slave_status()
        _bool = False
        if status == 'SLAV':
            _bool = True
        return _bool

    def force_trigger(self):
        """
        Send a trigger.
        """
        self.send(':TRIG:BEG')

    def force_event_trigger(self):
        """
        Send an advance event.
        """
        self.send(':TRIG:ADV')

    def force_enable_event_trigger(self):
        """
        Send an enabled event.
        """
        self.send(':TRIG:ENAB')

    @property
    def channels(self):
        chs = {child.n: child for child in self.childs}
        return chs

    @property
    def active_channels(self):
        return self.get_active_channels()

    def get_active_channels(self):
        chs = {}
        for n, channel in self.channels.items():
            if channel.is_active():
                chs[n] = channel
        return chs

    def is_using_markers(self):
        awg_mode = self.get_parameter('awg_mode', update=False)
        b = awg_mode in ['MARK', 'DCM']
        return b

    def get_channel(self, channel):
        return self.channels[channel]

    def create_sequencer(self):
        sequencer = M8195A_sequencer(self, name='{}_sequencer'.format(self.name))
        for seq_ch, awg_ch in zip(sequencer.channels.values(), self.channels.values()):
            seq_ch.name = awg_ch.name
        self.sequencer = sequencer

    def get_sequencer(self):
        return self.sequencer

    def get_config(self):
        config = super(M8195A, self).get_config()
        if hasattr(self, 'sequencer'):
            seq_config = self.sequencer.get_dict()
            config['sequencer'] = seq_config
        return config

    """
    segments
    """

    def define_segment(self, channel=1, segment_id=1, length=granularity, init_value=None):
        self.channels[channel].define_segment(
            segment_id=segment_id,
            length=length,
            init_value=init_value,
        )

    def delete_all_segments(self):
        for ch in self.channels.values():
            ch.delete_all_segments()

    def get_segment_catalog(self, channel=1):
        return self.channels[channel].get_segment_catalog()

    def get_waveform_from_segment(self, channel=1, segment_id=1, offset=0):
        return self.channels[channel].get_waveform_from_segment(segment_id, offset)

    def set_waveform_to_segment(self, waveform, marker1=None, marker2=None, channel=1, segment_id=1, offset=0):
        self.channels[channel].set_waveform_to_segment(
            waveform, marker1=marker1, marker2=marker2, segment_id=segment_id, offset=offset)

    """
    sequence
    """

    def reset_sequence_table(self):
        self.send(':STAB:RES')

    def set_sequence_entry(self,
                           sequence_id=0,  # ID of the sequence.
                           entry_type='data',  # Entry could be data, idle or empty.
                           segment_id=1,  # ID of the segment.
                           segment_loop=1,  # Number of loops for this segment.
                           segment_start_offset=0,  # Segment start address, if only part segment is to be used;
                           # Must be a multiple of twice the granularity.
                           segment_end_offset=0xFFFFFFFF,  # Segment end address, if only part segment is to be used;
                           # Must be a multiple of the granularity (minus 1). fffffff for max.
                           segment_advancement_mode='auto',  # Segment advancement mode is related to trigger mode;
                           # could be 'auto', 'cond', 'repeat' or 'single'.
                           use_marker=False,  # Presence of use_marker.
                           new_sequence=False,  # Presence of new sequence.
                           end_sequence=False,  # Presence of end of the sequence.
                           end_scenario=False,  # Presence of end of the scenario. (end of the table)
                           sequence_loop=1,  # Number of loops for this sequence.
                           sequence_advancement_mode='auto',  # Sequence advancement mode is related to trigger mode;
                           # could be 'auto', 'cond', 'repeat' or 'single'.
                           idle_delay=0,  # Idle value in Waveform Sample Clocks.
                           # (Extreme values depend on sampling rate divider 'SRD' : min=2560/SRD; max=2**24*256/SRD-1).
                           idle_sample=0,  # Sample to be played during pause.
                           ):
        """
        This procedure takes specified parameters (or default) and create a sequence "line" in the sequence table.
        It needs segment number to be defined with a waveform.
        """
        _validators['segment_id'].set_value(segment_id)
        _validators['sequence_id'].set_value(sequence_id)
        _validators['entry_type'].set_value(entry_type)
        _validators['loop'].set_value(segment_loop)
        _validators['loop'].set_value(sequence_loop)
        _validators['advancement_mode'].set_value(segment_advancement_mode)
        _validators['advancement_mode'].set_value(sequence_advancement_mode)

        control_parameter = self._setup_control_parameter(
            entry_type=entry_type,
            segment_advancement_mode=segment_advancement_mode,
            use_marker=use_marker,
            new_sequence=new_sequence,
            sequence_advancement_mode=sequence_advancement_mode,
            end_sequence=end_sequence,
            end_scenario=end_scenario,
        )

        if entry_type == 'empty':
            idle_delay = 768  # special code to indicate empty sequence entry

        arg_list = [sequence_id, control_parameter, sequence_loop]
        if entry_type == 'data':
            arg_list.append(segment_loop)
            arg_list.append(segment_id)
            arg_list.append(segment_start_offset)
            arg_list.append(segment_end_offset)
        else:  # idle or empty
            arg_list.append(0)  # Command code.
            arg_list.append(idle_sample)
            arg_list.append(idle_delay)
            arg_list.append(0)  # Nothing. Will always be 0.

        self.send(':STAB:DATA {:d},{:d},{:d},{:d},{:d},{:d},{:d}'.format(*arg_list))

    def get_sequence_entry(self, sequence_id=0):
        """
        This function returns the specified sequence "line" of the table in an ordered dictionary.
        This dictionary is in the same order as the table in the soft panel.
        """
        seq = self._get_sequence(sequence_id)
        true_seq = seq.encode('ASCII', 'ignore')  # Convert unicode to str.
        sequence_list = true_seq.split(',')
        control_parameter = int(sequence_list[0])
        sequence_output = self._analyse_control_parameter(control_parameter)  # Analyse control parameter.

        sequence_output['sequence_id'] = sequence_id  # ID of the sequence.
        sequence_output['sequence_loop'] = sequence_list[1]  # Number of loops for this sequence.
        if sequence_list[5] == '0':  # If last number is 0 then entry is either 'Idle' or 'Empty'.
            if sequence_list[4] == '768':  # SPECIAL CODE TO INDICATE EMPTY SEQUENCE ENTRY.
                entry_type = 'empty'
            else:
                entry_type = 'idle'
            sequence_output['idle_sample'] = sequence_list[3]  # Sample played during pause.
            sequence_output['idle_delay'] = sequence_list[4]  # Idle value in Waveform Sample Clocks.
        else:
            entry_type = 'data'
            sequence_output['segment_loop'] = sequence_list[2]  # Number of loops for this segment.
            sequence_output['segment_id'] = sequence_list[3]  # ID of the segment.
            sequence_output['segment_start_offset'] = sequence_list[4]  # Segment start address, if only use a cut.
            sequence_output['segment_end_offset'] = sequence_list[5]  # Segment end address, if only use a cut.
        sequence_output['entry_type'] = entry_type

        # Order the output according to sequence table
        key_list = ['sequence_id', 'entry_type', 'segment_id', 'segment_loop',
                    'segment_start_offset', 'segment_end_offset', 'segment_advancement_mode',
                    'use_marker', 'new_sequence', 'sequence_loop', 'sequence_advancement_mode',
                    'idle_delay', 'idle_sample', 'end_sequence', 'end_scenario']

        ordered_sequence_output = collections.OrderedDict()
        for key in key_list:
            if key in sequence_output.keys():
                ordered_sequence_output[key] = sequence_output[key]

        return ordered_sequence_output

    def get_sequence_state(self):
        state = self.read(':STAB:SEQ:STAT?')
        return state

    def set_sequence_initial_index(self, idx):
        self.send(':STAB:SEQ:SEL {:d}'.format(idx))

    def get_sequence_initial_index(self):
        idx = self.read(':STAB:SEQ:SEL?')
        return int(idx)

    def _setup_control_parameter(self,
                                 entry_type='data',
                                 segment_advancement_mode='cond',
                                 use_marker=False,
                                 new_sequence=True,
                                 sequence_advancement_mode='cond',
                                 end_sequence=True,
                                 end_scenario=True,
                                 ):
        """
        This function is used in "set_sequence_entry" to compute and return the
        control parameter : a hidden number in bits which contains multiple parameters.
        """
        kwargs = locals()  # Dict of locals
        del kwargs['self']
        # To keep 32bits : set a unsigned-integer constant (else it exceeds max value).
        control_parameter = np.uint32(0)
        for arg_name, arg_val in kwargs.items():
            # Indices of bitwise control parameter value that
            bit_indices = np.array(self._control_parameters[arg_name][0])
            # define a certain property; for instance [16,17] for 'segment_advancement_mode'.
            # Bit-sequence (value) of a certain setting (key); for instance: [True,False] for setting 'cond'.
            bit_sequence = self._control_parameters[arg_name][1][arg_val]
            # Sets up bits accordingly.
            control_parameter += np.sum(2 ** bit_indices[np.where(bit_sequence)])
        return control_parameter

    def _analyse_control_parameter(self, control_parameter):
        """
        This function is used in "get_sequence_entry" to analyse the control
        parameter and return a dictionary of all the parameters and their values.
        """
        # Convert control parameter from integer value to 32-bit representation with correct order in iteration. (ex : 000000010...)
        binary_parameter = str(bin(control_parameter)[2:].zfill(32))[::-1]
        decoded_control_parameter = dict()
        for seq_property, encoder_value in self._control_parameters.items():
            bit_indices = encoder_value[0]  # Indices of bitwise control parameter value that define a certain property;
            # for instance [16,17] for 'segment_advancement_mode'.
            bits = [bool(int(binary_parameter[ind])) for ind in
                    bit_indices]  # Bits of certain property in boolean list.
            settings_dict = encoder_value[1]  # Dictionary of settings (key) with corresponding bit sequence (value).
            settings_dict_values = settings_dict.values()
            settings_dict_keys = settings_dict.keys()
            setting = settings_dict_keys[settings_dict_values.index(bits)]  # Select setting for specific bit-sequence.
            decoded_control_parameter[seq_property] = setting  # Set setting of property.
        return decoded_control_parameter

    def _get_sequence(self,
                      table_index=0,  # Int >=0
                      ):
        """
        Used in the function "get_sequence_entry". (list of numbers that are analysed)
        """
        if table_index >= 0:
            sequence = self.read(':STAB:DATA? {:d},6'.format(table_index))
            return sequence
        else:
            raise Exception('ERROR in M8195A.get_sequence, index not >=0: ', table_index)

    """
    get/set parameters
    """

    @get_decorator
    def get_sampling_rate_divider(self):
        msg = self.read(':INST:MEM:EXT:RDIV?')
        return int(msg[-1])

    @set_decorator
    def set_sampling_rate_divider(self, value):
        self.send(':INST:MEM:EXT:RDIV DIV{:d}'.format(value))

    @get_decorator
    def get_sampling_frequency(self):
        msg = self.read(':FREQ:RAST?')
        return float(msg) * 1e-9  # to GHz

    @set_decorator
    def set_sampling_frequency(self, value):
        value *= 1e9  # to Hz
        self.send(':FREQ:RAST {:f}'.format(value))

    @get_decorator
    def get_delay(self):
        msg = self.read(':ARM:MDEL?')
        return float(msg) * 1e9  # to ns

    @set_decorator
    def set_delay(self, value):
        self.send(':ARM:MDEL {:f} ns'.format(value))

    @get_decorator
    def get_awg_mode(self):
        msg = self.read(':INST:DACM?')
        return str(msg)

    @set_decorator
    def set_awg_mode(self, value):
        self.send(':INST:DACM {:s}'.format(value))

    @get_decorator
    def get_sequence_mode(self):
        msg = self.read(':FUNC:MODE?')
        return str(msg)

    @set_decorator
    def set_sequence_mode(self, value):
        self.send(':FUNC:MODE {:s}'.format(value))

    @get_decorator
    def get_dynamic_mode(self):
        msg = self.read(':STAB:DYN?')
        return bool(int(msg))

    @set_decorator
    def set_dynamic_mode(self, value):
        self.send(':STAB:DYN {:d}'.format(int(value)))

    @get_decorator
    def get_slave_status(self):
        """
        Ask if Sync is enslaved by sync. unit or not.
        """
        msg = self.read(':INST:MMOD:MODE?')
        return str(msg)

    @get_decorator
    def get_trigger_status(self):
        msg = self.read(':TRIG:BEG:HWD?')
        return bool(int(msg))

    @set_decorator
    def set_trigger_status(self, value):
        """
        Enable/Disable the capacity to send a trigger.
        """
        self.send(':TRIG:BEG:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_event_trigger_status(self):
        msg = self.read(':TRIG:ADV:HWD?')
        return bool(int(msg))

    @set_decorator
    def set_event_trigger_status(self, value):
        """
        Enable/Disable the capacity to send an advance event.
        """
        self.send(':TRIG:ADV:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_enable_event_trigger_status(self):
        msg = self.read(':TRIG:ENAB:HWD?')
        return bool(int(msg))

    @set_decorator
    def set_enable_event_trigger_status(self, value):
        """
        Enable/Disable the capacity to send an enabled event.
        """
        self.send(':TRIG:ENAB:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_armed_mode(self):
        msg = self.read(':INIT:CONT:ENAB?')
        return str(msg)

    @set_decorator
    def set_armed_mode(self, value):
        """
        SELF-ARMED : Instrument starts as defined by the selected trigger mode.
        ARMED : if continuous mode, first segment/sequence is played infinitely. Else, treated as self-armed.
        """
        self.send(':INIT:CONT:ENAB {:s}'.format(value))

    @get_decorator
    def get_trigger_mode(self):
        cont_mode = self._get_continuous_mode()
        gate_mode = self._get_gate_mode()
        if not cont_mode and not gate_mode:
            return 'TRIG'
        elif cont_mode and not gate_mode:
            return 'CONT'
        elif not cont_mode and gate_mode:
            return 'GATED1'
        elif cont_mode and gate_mode:
            return 'GATED2'

    @set_decorator
    def set_trigger_mode(self, value):
        """
        HAVE TO DEDUCE TRIGGER MODE FROM CONTINUOUS AND GATE MODES :
        CONT = False, Gate = False : TRIGGERED
        CONT = True, Gate = False : CONTINUOUS
        CONT = False, Gate = True : GATED
        CONT = True, Gate = True : GATED !!!!! (Documentation is wrong..)
        """
        if value == 'TRIG':
            self._set_continuous_mode(False)
            self._set_gate_mode(False)
        elif value == 'CONT':
            self._set_continuous_mode(True)
            self._set_gate_mode(False)
        elif value == 'GATED1':
            self._set_continuous_mode(False)
            self._set_gate_mode(True)
        elif value == 'GATED2':
            self._set_continuous_mode(True)
            self._set_gate_mode(True)
        else:
            raise Exception('ERROR in M8195A.set_trigger_mode, wrong format: ', value)

    def _get_continuous_mode(self):
        """
        Used in the function "get_trigger_mode".
        """
        msg = self.read(':INIT:CONT:STAT?')
        return bool(int(msg))

    def _set_continuous_mode(self, value):
        """
        Used in the function "set_trigger_mode".
        """
        if isbool(value):
            self.send(':INIT:CONT:STAT {:d}'.format(int(value)))
        else:
            raise Exception('ERROR in M8195A._set_continuous_mode, wrong format: ', value)

    def _get_gate_mode(self):
        """
        Used in the function "get_trigger_mode".
        """
        msg = self.read(':INIT:GATE:STAT?')
        return bool(int(msg))

    def _set_gate_mode(self, value=False):
        """
        Used in the function "set_trigger_mode".
        """
        if isbool(value):
            self.send(':INIT:GATE:STAT {:d}'.format(int(value)))
        else:
            raise Exception('ERROR in M8195A._set_gate_mode, wrong format: ', value)

    @get_decorator
    def get_trigger_level(self):
        msg = self.read(':ARM:TRIG:LEV?')
        return float(msg)

    @set_decorator
    def set_trigger_level(self, value):
        """
        Change the threshold level of the trigger.
        """
        self.send(':ARM:TRIG:LEV {:.3f}'.format(value))  # If >1.0: only 2 decimals

    @get_decorator
    def get_trigger_slope(self):
        msg = self.read(':ARM:TRIG:SLOP?')
        return str(msg)

    @set_decorator
    def set_trigger_slope(self, value):
        """
        Change the polarity of the trigger.
        """
        self.send(':ARM:TRIG:SLOP {:s}'.format(value))

    @get_decorator
    def get_trigger_sync_mode(self):
        msg = self.read(':ARM:TRIG:OPER?')
        return str(msg)

    @set_decorator
    def set_trigger_sync_mode(self, value):
        self.send(':ARM:TRIG:OPER {:s}'.format(value))

    @get_decorator
    def get_trigger_frequency(self):
        msg = self.read(':ARM:TRIG:FREQ?')
        return float(msg)

    @set_decorator
    def set_trigger_frequency(self, value):
        self.send(':ARM:TRIG:FREQ {:.1f}'.format(value))

    @get_decorator
    def get_trigger_source(self):
        msg = self.read(':ARM:TRIG:SOUR?')
        return str(msg)

    @set_decorator
    def set_trigger_source(self, value):
        self.send(':ARM:TRIG:SOUR {:s}'.format(value))

    @get_decorator
    def get_advance_trigger_source(self):
        msg = self.read(':TRIG:SOUR:ADV?')
        return str(msg)

    @set_decorator
    def set_advance_trigger_source(self, value):
        self.send(':TRIG:SOUR:ADV {:s}'.format(value))

    @get_decorator
    def get_event_trigger_level(self):
        msg = self.read(':ARM:EVEN:LEV?')
        return float(msg)

    @set_decorator
    def set_event_trigger_level(self, value):
        """
        Change the threshold level of the event trigger.
        """
        self.send(':ARM:EVEN:LEV {:.3f}'.format(value))  # If >1.0: only 2 decimals

    @get_decorator
    def get_event_trigger_slope(self):
        msg = self.read(':ARM:EVEN:SLOP?')
        return str(msg)

    @set_decorator
    def set_event_trigger_slope(self, value):
        """
        Change the polarity of the event trigger.
        """
        self.send(':ARM:EVEN:SLOP {:s}'.format(value))

    @get_decorator
    def get_event_trigger_source(self):
        msg = self.read(':TRIG:SOUR:ENAB?')
        return str(msg)

    @set_decorator
    def set_event_trigger_source(self, value):
        self.send(':TRIG:SOUR:ENAB {:s}'.format(value))

    @get_decorator
    def get_slot_number(self):
        msg = self.read(':INST:SLOT?')
        return int(msg)

    """
    private methods
    """


class M8195A_channel(InstrumentBase):
    granularity = granularity  # Length of segments must be multiple of granularity
    default_parameters = (
        Parameter('memory_mode', value='EXT', valid_values=['EXT', 'INT', 'NONE']),
        Parameter('amplitude', value=0.5, min_value=-1, max_value=1),
        Parameter('offset', value=0.0, min_value=-1, max_value=1),
        Parameter('termination_voltage', value=0.0, min_value=-1, max_value=1),
        Parameter('clock_delay', value=0, unit='samples', min_value=0, max_value=95),
        Parameter('status', value=False, valid_values=[True, False]),
    )

    def __init__(self, n, parent, **kwargs):
        super(M8195A_channel, self).__init__(address=parent.address, parent=parent, **kwargs)
        n = int(n)
        if n not in [1, 2, 3, 4]:
            raise ValueError('channel number must be in [1,2,3,4]')
        self.n = n

    def open_com(self):
        pass

    def close_com(self):
        pass

    def send(self, *args, **kwargs):
        """
        Overwrite self.send method with self.parent.send
        """
        self.parent.send(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.parent.read(*args, **kwargs)

    def is_active(self):
        mmode = self.get_memory_mode()
        return True if mmode != 'NONE' else False

    """
    segments
    """

    def define_segment(self, segment_id=1, length=granularity, init_value=None):
        _validators['segment_id'].set_value(segment_id)
        _validators['max_length'].set_value(length)
        msg = ':TRAC{:d}:DEF {:d},{:d}'.format(self.n, segment_id, length)
        if init_value is not None:
            _validators['init_value'].set_value(init_value)
            msg = '{}, {:f}'.format(msg, init_value)
        self.send(msg)

    def delete_all_segments(self):
        self.send(':TRAC{:d}:DEL:ALL'.format(self.n))

    def get_waveform_from_segment(self, segment_id=1, offset=0):
        _validators['segment_id'].set_value(segment_id)
        _validators['offset'].set_value(offset)
        catalog = self.get_segment_catalog()
        length = catalog[segment_id]
        wf_str = self.read(':TRAC{:d}:DATA? {:d},{:d},{:d}'.format(self.n, segment_id, offset, length))
        using_markers = length == 2 * len(wf_str.split(','))
        wf, mk1, mk2 = self._str_to_waveform(wf_str, using_markers=using_markers)
        return wf, mk1, mk2

    def get_segment_catalog(self):
        msg = self.read(':TRAC{:d}:CAT?'.format(self.n))
        id_length_dict = self._segment_catalog_to_dict(msg)
        return id_length_dict

    def set_waveform_to_segment(self, waveform, marker1=None, marker2=None, segment_id=1, offset=0):
        _validators['segment_id'].set_value(segment_id)
        _validators['offset'].set_value(offset)
        ignore_markers = not self.parent.is_using_markers()
        wf_str = self._waveform_to_str(waveform, marker1=marker1, marker2=marker2, ignore_markers=ignore_markers)
        length = len(waveform)
        self.define_segment(segment_id=segment_id, length=length, init_value=None)
        self.send(':TRAC{:d}:DATA {:d},{:d},{:s}'.format(self.n, segment_id, offset, wf_str))

    """
    set/get parameters
    """

    @get_decorator
    def get_status(self):
        msg = self.read(':OUTP{:d}?'.format(self.n))
        return bool(int(msg))

    @set_decorator
    def set_status(self, value):
        """
        Turn on/off the channel
        """
        self.send(':OUTP{:d} {:d}'.format(self.n, int(value)))

    @get_decorator
    def get_memory_mode(self):
        msg = self.read(':TRAC{:d}:MMOD?'.format(self.n))
        return str(msg)

    @set_decorator
    def set_memory_mode(self, value):
        self.send(':TRAC{:d}:MMOD {:s}'.format(self.n, value))

    @get_decorator
    def get_amplitude(self):
        msg = self.read(':VOLT{:d}?'.format(self.n))
        return float(msg)

    @set_decorator
    def set_amplitude(self, value):
        self.send(':VOLT{:d} {:f}'.format(self.n, value))

    @get_decorator
    def get_offset(self):
        msg = self.read(':VOLT{:d}:OFFS?'.format(self.n))
        return float(msg)

    @set_decorator
    def set_offset(self, value):
        self.send(':VOLT{:d}:OFFS {:f}'.format(self.n, value))

    @get_decorator
    def get_termination_voltage(self):
        msg = self.read(':VOLT{:d}:TERM?'.format(self.n))
        return float(msg)

    @set_decorator
    def set_termination_voltage(self, value):
        self.send(':VOLT{:d}:TERM {:f}'.format(self.n, value))

    @get_decorator
    def get_clock_delay(self):
        msg = self.read(':ARM:SDEL{:d}?'.format(self.n))
        return int(msg)

    @set_decorator
    def set_clock_delay(self, value):
        self.send(':ARM:SDEL{:d} {:d}'.format(self.n, value))

    """
    private methods
    """

    def _waveform_to_str(self, waveform, marker1=None, marker2=None, ignore_markers=False):
        self._check_waveform_values(waveform)
        # wf = [digitise(v) for v in waveform]
        wf = digitise(waveform)
        if ignore_markers:
            wf_str = wf
        else:
            self._check_lengths(waveform, marker1=marker1, marker2=marker2)
            size = len(wf)
            mk1 = marker1 if marker1 is not None else np.zeros(size)
            mk2 = marker2 if marker2 is not None else np.zeros(size)
            self._check_marker_values(mk1)
            self._check_marker_values(mk2)
            mks = np.zeros(size)
            mks[mk1 == 1] += 1
            mks[mk2 == 1] += 1
            wf_str = np.zeros(size * 2)
            wf_str[0::2] = wf
            wf_str[1::2] = mks
        wf_str = wf_str.astype(str)
        wf_str = ','.join(wf_str)
        return wf_str

    @staticmethod
    def _check_waveform_values(values):
        if np.min(values) < -1 or np.max(values) > 1:
            raise ValueError('waveform values must be between -1 to 1')

    @staticmethod
    def _check_marker_values(values):
        comp = map(lambda x: x in [0, 1], values)
        valid = all(comp)
        if not valid:
            raise ValueError('marker values must be 0 or 1')

    @staticmethod
    def _check_lengths(waveform, marker1=None, marker2=None):
        lengths = set([len(wf) for wf in [waveform, marker1, marker2] if wf is not None])
        same_length = len(set(lengths)) == 1
        if not same_length:
            raise ValueError('waveform and markers values must have same max_length')

    @staticmethod
    def _str_to_waveform(wf_str, using_markers=False):
        if not wf_str:
            wf, mk1, mk2 = None, None, None
        else:
            values = wf_str.split(',')
            values = np.array([int(v) for v in values])

            if using_markers:
                wf = values[0::2]
                mks = values[1::2]
                mk1 = np.zeros_like(mks)
                mk2 = np.zeros_like(mks)
                mk1[mks == 1] = 1
                mk1[mks == 3] = 1
                mk2[mks == 2] = 1
                mk2[mks == 3] = 1
            else:
                wf = values
                mk1 = None
                mk2 = None
            wf = np.array([reverse_digitise(v) for v in wf])
        return wf, mk1, mk2

    @staticmethod
    def _segment_catalog_to_dict(str_):
        """
        "0,0" --> empty
        "1, 1280, 2, 1280" = "id, length, id, length..."
        return {1: 1280, ...}
        """
        str_ = str_.split(',')
        ids = str_[0::2]
        lengths = str_[1::2]
        d = {int(id_): int(length) for id_, length in zip(ids, lengths)}
        return d


class M8195A_sequencer(WaveformGroup):
    def __init__(self, awg, name='M8195A_sequencer'):
        if not isinstance(awg, M8195A):
            raise ValueError('AWG must be an instance of {}'.format(M8195A))
        super(M8195A_sequencer, self).__init__(waveforms=[], name=name)
        self.awg = awg
        n_channels = len(self.awg.channels)
        for n in range(n_channels):
            n = n + 1
            channel = Waveform(name='ch{}'.format(n))
            self.add_waveform(channel)

    @property
    def channels(self):
        return {i + 1: wf for i, wf in enumerate(self.wfs)}

    def get_dict(self):
        # TODO
        d = super(M8195A_sequencer, self).get_dict()
        return d

    def generate_sequence(self, dims_size, n_empty=0, start_with_empty=True):
        self._prepare_sequence()
        self._fill_segments(dims_size)
        self._fill_sequence_table(
            dims_size=dims_size,
            n_empty=n_empty,
            start_with_empty=start_with_empty,
        )

    def _prepare_sequence(self):
        self.awg.stop()
        self.awg.set_sequence_mode('STS')
        self.awg.reset_sequence_table()
        for ch in self.awg.channels.values():
            ch.delete_all_segments()

    def _fill_segments(self, dims_size):
        wf_channels = self.channels
        swept_dims = self.swept_dims()

        last_ids = []
        for n, channel in self.awg.active_channels.items():
            wf_ch = wf_channels[n]
            wf_values = wf_ch.get_waveforms_subset(dims_size, dims=swept_dims)

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

    def _fill_sequence_table(self, dims_size, n_empty=0, start_with_empty=True):
        use_marker = self.awg.is_using_markers()
        seg_loop, n_entries, seq_loop = self.split_dims_size(dims_size)
        default_kwargs = dict(
            segment_loop=seg_loop,
            segment_advancement_mode='single',
            sequence_loop=seq_loop,
            sequence_advancement_mode='single',
        )

        seq_size = n_entries * (1 + n_empty)
        last_seq_id = 0
        for i in range(n_entries):
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

    # def _fill_segments(self, dims_size):
    #     wf_channels = self.channels
    #     last_ids = []
    #     for n, channel in self.awg.active_channels.items():
    #         wf_ch = wf_channels[n]
    #         wf_values = wf_ch.get_waveforms(dims_size)
    #
    #         segment_id = 1  # use 1 for dummy waveform (all 0V)
    #         channel.define_segment(
    #             segment_id=segment_id,
    #             init_value=0.0,
    #         )
    #         for values in wf_values:
    #             segment_id += 1
    #             channel.set_waveform_to_segment(
    #                 segment_id=segment_id,
    #                 waveform=values,
    #             )
    #         last_ids.append(segment_id)
    #     same_last_id = len(set(last_ids)) == 1
    #     if not same_last_id:
    #         raise ValueError('Channel waveforms have different sizes')

    # def _fill_sequence_table(self, n_empty=0, start_with_empty=True, **entry_kwargs):
    #     use_marker = self.awg.is_using_markers()
    #     default_kwargs = dict(
    #         segment_loop=1,
    #         segment_advancement_mode='single',
    #         sequence_loop=1,
    #         sequence_advancement_mode='auto',
    #     )
    #     for key in default_kwargs.keys():
    #         if key not in entry_kwargs.keys():
    #             continue
    #         default_kwargs[key] = entry_kwargs[key]
    #
    #     n_segments = int(np.prod(self.dims[1:]))
    #     seq_size = n_segments * (1 + n_empty)
    #     last_seq_id = 0
    #     for i in range(n_segments):
    #         segment_id = i + 2  # id 1 is a dummy segment
    #
    #         if start_with_empty:
    #             segment_ids = [1] * n_empty + [segment_id]
    #         else:
    #             segment_ids = [segment_id] + [1] * n_empty
    #
    #         for seg_id in segment_ids:
    #             new_sequence = True if last_seq_id == 0 else False
    #             end_sequence = True if last_seq_id == seq_size - 1 else False
    #             end_scenario = end_sequence
    #
    #             self.awg.set_sequence_entry(
    #                 sequence_id=last_seq_id,
    #                 segment_id=seg_id,
    #                 entry_type='data',
    #                 use_marker=use_marker,
    #                 new_sequence=new_sequence,
    #                 end_sequence=end_sequence,
    #                 end_scenario=end_scenario,
    #                 **default_kwargs
    #             )
    #             last_seq_id += 1


def digitise(x):
    """
    This function maps a floating-point number in the range [-1.0,1.0]
    on an integer number in the range [-127,...,127].
    """
    x = np.array(x)
    ndiv = 254  # equivalent to 127*2
    dv = 2. / ndiv
    idx = (x + 1) // dv - 127
    idx = idx.astype(int)
    return idx


def reverse_digitise(x):
    """
    This function maps an integer number in the range [-127,...,127] on
    a floating-point number in the range [-1.0,1.0]
    """
    ndiv = len(np.arange(-127, 127.1, 1))  # equivalent to 255 points
    values = np.linspace(-1, 1, ndiv)
    idx = x + 127
    return values[idx]


if __name__ == '__main__':
    import numpy as np
    from pystruments.keysight import M8195A
    from pystruments.funclib import pulse, pulse_params

    address_awg_virtual = 'TCPIP0::localhost::hislip4::INSTR'
    address_sync_virtual = 'TCPIP0::localhost::hislip0::INSTR'
    address_sync = 'TCPIP0::localhost::hislip1::INSTR'
    address_awg1 = 'TCPIP0::localhost::hislip2::INSTR'
    address_awg2 = 'TCPIP0::localhost::hislip3::INSTR'

    awg = M8195A(address_awg_virtual, name='AWG1')
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

    ch4_seq = sequencer.channels[4]

    func = pulse
    params = pulse_params(
        pts=1,
        base=0,
        delay=1,
        ampl=1,
        length=10,
    )
    params['length'].sweep_stepsize(1, 3, dim=2)
    ch4_seq.set_func(func, params)

    sequencer.generate_sequence([1280, 2, 3, 10])
    awg.save_config('keysight_awg.json')
    import json
    with open('keysight_awg.json', 'rb') as file:
        conf = json.load(file)
