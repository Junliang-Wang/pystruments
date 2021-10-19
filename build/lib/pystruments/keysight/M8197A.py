import time

from pystruments.instrument import InstrumentBase, get_decorator, set_decorator
from pystruments.keysight.M8195A import M8195A
from pystruments.parameter import Parameter
from pystruments.utils import *


class M8197A(InstrumentBase):
    default_parameters = (
        Parameter('armed_mode', value='SELF', valid_values=['SELF', 'ARM']),
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
        Parameter('config_mode', value=False, valid_values=[True, False]),
        Parameter('sampling_frequency', value=64, unit='GHz', min_value=53.76, max_value=65),
        Parameter('clock_out_source', value='INT', valid_values=['INT', 'EXT', 'SCLK1', 'SCLK2']),
        Parameter('clock_out_sample_divider', value=1, unit='samples', min_value=1, max_value=1024, valid_types=[int]),
        Parameter('clock_ref_source', value='INT', valid_values=['INT', 'EXT', 'AXI']),
        Parameter('clock_ref_range', value=1, valid_values=[1, 2]),
        Parameter('clock_ref_frequency', value=100, unit='MHz', min_value=10, max_value=17e3),
        Parameter('clock_ref_divider1', value=1, min_value=1, max_value=8, valid_types=[int]),
        Parameter('clock_ref_divider2', value=1, min_value=1, max_value=8, valid_types=[int]),
    )

    def __init__(self, *args, **kwargs):
        if 'timeout' not in kwargs.keys():
            kwargs['timeout'] = 10000  # in ms
        super(M8197A, self).__init__(*args, **kwargs)

    def send(self, cmd, wait_to_complete=True):
        super(M8197A, self).send(cmd)
        if wait_to_complete:
            self.read('*OPT?')

    @property
    def awgs(self):
        return {awg.get_slot_number(): awg for awg in self.childs}

    @property
    def slaves(self):
        return self.childs

    def run(self):
        self.send(':INIT:IMM')

    def stop(self):
        self.send(':ABOR')

    def reset(self):
        self.send('*RST')
        time.sleep(1)

    def identify(self):
        """
        Identify the sync. unit by flashing the green LED on the front panel.
        """
        self.send(':INST:IDEN')

    def discover(self):
        """
        Ask for the list of AWG modules.
        """
        msg = self.read(':INST:MDIS?')
        addresses = self._addresses_to_list(msg)
        return addresses

    def enslave(self, address):
        self.send(':INST:SLAV:ADD \"{:s}\"'.format(address))
        time.sleep(0.1)
        awg = M8195A(address, parent=self)
        awg.open_com()
        self.add_child(awg)

    def enslave_all(self):
        addresses = self.discover()
        for address in addresses:
            self.enslave(address)

    def remove_slave(self, address):
        self.send(':INST:SLAV:DEL {:s}'.format(address))
        time.sleep(0.1)
        for i, awg in enumerate(self.childs):
            if awg.address == address:
                awg.close_com()
                self.childs.pop(i)

    def remove_all_slave(self):
        self.send(':INST:SLAV:DEL:ALL')
        time.sleep(0.1)
        while self.childs:
            awg = self.childs.pop()
            awg.close_com()

    def get_slaves_address(self):
        msg = self.read(':INST:SLAV:LIST?')
        addresses = self._addresses_to_list(msg)
        return addresses

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

    def get_awg_sequencers(self):
        seqs = {}
        for slot_number, awg in self.awgs.items():
            seqs[slot_number] = awg.get_sequencer()
        return seqs

    @staticmethod
    def _addresses_to_list(str_addresses):
        addresses = str(str_addresses).replace('"', '').split(',')
        addresses = filter(bool, addresses)
        return addresses

    """
    get/set parameters
    """

    @set_decorator
    def set_config_mode(self, value):
        """
        Enable/Disable the configuration of the sync. unit.
        """
        self.send(':INST:MMOD:CONF {:d}'.format(int(value)))

    @get_decorator
    def get_config_mode(self):
        config_mode = self.read(':INST:MMOD:CONF?')
        return bool(int(config_mode))

    @get_decorator
    def get_slot_number(self):
        slot = self.read(':INST:SLOT?')
        return int(slot)

    @get_decorator
    def get_trigger_status(self):
        trig_status = self.read(':TRIG:BEG:HWD?')
        return bool(int(trig_status))

    @set_decorator
    def set_trigger_status(self, value):
        """
        Enable/Disable the capacity to send a trigger.
        """
        self.send(':TRIG:BEG:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_event_trigger_status(self):
        event_trig_status = self.read(':TRIG:ADV:HWD?')
        return bool(int(event_trig_status))

    @set_decorator
    def set_event_trigger_status(self, value):
        """
        Enable/Disable the capacity to send an advance event.
        """
        self.send(':TRIG:ADV:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_enable_event_trigger_status(self):
        enable_event_trig_status = self.read(':TRIG:ENAB:HWD?')
        return bool(int(enable_event_trig_status))

    @set_decorator
    def set_enable_event_trigger_status(self, value):
        """
        Enable/Disable the capacity to send an enabled event.
        """
        self.send(':TRIG:ENAB:HWD {:d}'.format(int(value)))

    @get_decorator
    def get_armed_mode(self):
        arm_mode = self.read(':INIT:CONT:ENAB?')
        return str(arm_mode)

    @set_decorator
    def set_armed_mode(self, value):
        """
        SELF-ARMED : Instrument starts as defined by the selected trigger mode.
        ARMED : if continuous mode, first segment/sequence is played infinitely. Else, treated as self-armed.
        """
        self.send(':INIT:CONT:ENAB {:s}'.format(value))

    @set_decorator
    def set_trigger_mode(self, value):
        """
        HAVE TO DEDUCE TRIGGER MODE FROM CONTINUOUS AND GATE MODES :
        CONT = False, Gate = False : TRIGGERED
        CONT = True, Gate = False : CONTINUOUS
        CONT = False, Gate = True : GATED1
        CONT = True, Gate = True : GATED2 !!!!! (Documentation is wrong..)
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
            raise Exception('Trigger mode, wrong format: ', value)

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

    def _get_continuous_mode(self):
        """
        Used in the function "get_trigger_mode".
        """
        cont_mode = self.read(':INIT:CONT:STAT?')
        return bool(int(cont_mode))

    def _set_continuous_mode(self, value):
        """
        Used in the function "set_trigger_mode".
        """
        if isbool(value):
            self.send(':INIT:CONT:STAT {:d}'.format(int(value)))
        else:
            raise Exception('Continuous mode, wrong format: ', value)

    def _get_gate_mode(self):
        """
        Used in the function "get_trigger_mode".
        """
        gate_mode = self.read(':INIT:GATE:STAT?')
        return bool(int(gate_mode))

    def _set_gate_mode(self, value):
        """
        Used in the function "set_trigger_mode".
        """
        if isbool(value):
            self.send(':INIT:GATE:STAT {:d}'.format(int(value)))
        else:
            raise Exception('Gate mode, wrong format: ', value)

    @get_decorator
    def get_trigger_level(self):
        msg = self.read(':ARM:TRIG:LEV?')
        return float(msg)

    @set_decorator
    def set_trigger_level(self, value):
        """
        Change the threshold level of the trigger.
        """
        self.send(':ARM:TRIG:LEV {:f}'.format(value))

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
        self.send(':ARM:TRIG:FREQ {}'.format(float(value)))

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
        self.send(':ARM:EVEN:LEV {}'.format(float(value)))

    @get_decorator
    def get_event_trigger_slope(self):
        msg = self.read(':ARM:EVEN:SLOP?')
        return str(msg)

    @set_decorator
    def set_event_trigger_slope(self, value):
        """
        Change the polarity of the event trigger.
        """
        self.send(':ARM:EVEN:SLOP {}'.format(value))

    @get_decorator
    def get_event_trigger_source(self):
        msg = self.read(':TRIG:SOUR:ENAB?')
        return str(msg)

    @set_decorator
    def set_event_trigger_source(self, value):
        self.send(':TRIG:SOUR:ENAB {}'.format(value))

    @get_decorator
    def get_sampling_frequency(self):
        srd = self.read(':FREQ:RAST?')
        return float(srd) * 1e-9  # to GHz

    @set_decorator
    def set_sampling_frequency(self, value):
        value *= 1e9  # to Hz
        self.send(':FREQ:RAST {:f}'.format(value))

    @get_decorator
    def get_clock_out_source(self):
        msg = self.read(':OUTP:ROSC:SOUR?')
        return str(msg)

    @set_decorator
    def set_clock_out_source(self, value):
        self.send(':OUTP:ROSC:SOUR {}'.format(value))

    @get_decorator
    def get_clock_out_sample_divider(self):
        msg = self.read(':OUTP:ROSC:SCD?')
        return int(msg)

    @set_decorator
    def set_clock_out_sample_divider(self, value):
        self.send(':OUTP:ROSC:SCD {}'.format(int(value)))

    @get_decorator
    def get_clock_ref_source(self):
        msg = self.read(':ROSC:SOUR?')
        return str(msg)

    @set_decorator
    def set_clock_ref_source(self, value):
        self.send(':ROSC:SOUR {}'.format(str(value)))

    @get_decorator
    def get_clock_ref_range(self):
        msg = self.read(':ROSC:RANG?')
        return int(msg[-1])

    @set_decorator
    def set_clock_ref_range(self, value):
        self.send(':ROSC:RANG RANG{}'.format(int(value)))

    @get_decorator
    def get_clock_ref_frequency(self):
        msg = self.read(':ROSC:FREQ?')
        return float(msg) * 1e-6  # Hz to MHz

    @set_decorator
    def set_clock_ref_frequency(self, value):
        value *= 1e6  # MHz to Hz
        self.send(':ROSC:FREQ {:f}'.format(value))

    @get_decorator
    def get_clock_ref_divider1(self):
        msg = self.read(':OUTP:ROSC:RCD1?')
        return int(msg)

    @set_decorator
    def set_clock_ref_divider1(self, value):
        self.send(':OUTP:ROSC:RCD1 {}'.format(int(value)))

    @get_decorator
    def get_clock_ref_divider2(self):
        msg = self.read(':OUTP:ROSC:RCD2?')
        return int(msg)

    @set_decorator
    def set_clock_ref_divider2(self, value):
        self.send(':OUTP:ROSC:RCD2 {}'.format(int(value)))


if __name__ == '__main__':
    from pystruments.funclib import pulse, pulse_params

    address_awg_virtual = 'TCPIP0::localhost::hislip4::INSTR'
    address_sync_virtual = 'TCPIP0::localhost::hislip0::INSTR'
    address_awg1 = 'TCPIP0::localhost::hislip1::INSTR'
    address_awg2 = 'TCPIP0::localhost::hislip2::INSTR'
    address_sync = 'TCPIP0::localhost::hislip3::INSTR'

    sync = M8197A(address_sync_virtual, name='sync')
    sync.open_com()
    sync.enslave_all()
    sync.set_sampling_frequency(64)
    sync.set_armed_mode('SELF')
    sync.set_trigger_mode('TRIG')
    sync.set_advance_trigger_source('TRIG')
    sync.set_event_trigger_source('TRIG')
    for slot_number, awg in sync.awgs.items():
        awg.set_awg_mode('DUAL')
        awg.set_sampling_rate_divider(2)
        awg.channels[1].set_status(True)
        awg.channels[4].set_status(True)
        ch1 = awg.channels[1]
        ch2 = awg.channels[4]
        ch1.set_memory_mode('EXT')
        ch2.set_memory_mode('EXT')
    sequencers = sync.get_awg_sequencers()
    seq0 = sequencers[0]

    func = pulse
    params = pulse_params(
        pts=1,
        base=0,
        delay=1,
        ampl=1,
        length=10,
    )
    ch1_seq = seq0.channels[1]
    ch4_seq = seq0.channels[4]
    ch1_seq.set_func(func, params)

    params = pulse_params(
        pts=1,
        base=0,
        delay=1,
        ampl=1,
        length=10,
    )
    # params['delay'].sweep_stepsize(1, 2, dim=1)
    params['length'].sweep_stepsize(1, 3, dim=2)
    ch4_seq.set_func(func, params)

    dsize = [1280, 2, 3, 10]
    seq0.generate_sequence(dsize, n_empty=0, start_with_empty=True)
    # sync.enslave(address_awg1)
    # sync.configure_sync(sync_dict={'trigger_mode': 'TRIG', 'trigger_level': 0.0})

    sync.save_config('keysight_sync.json')
    import json
    with open('keysight_sync.json', 'rb') as file:
        conf = json.load(file)