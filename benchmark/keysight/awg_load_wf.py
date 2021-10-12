import numpy as np

from pystruments.funclib import pulse, pulse_params
from pystruments.keysight import M8195A

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

# sequencer = M8195A_sequencer(awg)
sequencer = awg.sequencer

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

num_runs = 5
x1 = np.arange(1, 102, 10)
x2 = np.arange(1280, 12801, 1280)
y = np.ndarray((x1.size, x2.size))
# for i, x1i in enumerate(x1):
#     for j, x2i in enumerate(x2):
#         dims = [x2i, x1i, 1]
#
#
#         def f():
#             sequencer.generate_sequence(dims, n_empty=0, start_with_empty=True)
#
#
#         duration = timeit.Timer(f).timeit(number=num_runs)
#         avg_duration = duration / num_runs
#         # print('On average it took {} seconds'.format(avg_duration))
#         y[i, j] = avg_duration
# f()
dims = [1280, 50, 1]

sequencer.generate_sequence(dims, n_empty=0, start_with_empty=True)

# # awg_seq = AWGSequencer(awg=awg, waveform=awg_wf)
# # awg_seq.generate_sequence(dims, n_empty=1)
# #
# # awg = M8195A(...)
# # sequencer = awg.get_sequencer()
# # sequencer.channels[1].set_func(func, params)
# # sequencer.generate_sequence(dims=[1280, 1, 2])
