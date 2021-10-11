import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear(x, m=1, x0=0, y0=0):
    return (x - x0) * m + y0


fullpath = 'awg_load_wf_benchmark.npz'
data = np.load(fullpath)
x1 = data['x1']
x2 = data['x2']
y = data['y']

# color map
if False:
    # plt.close('all')
    fig, ax = plt.subplots(1)
    pcolor = ax.pcolormesh(x2, x1, y)
    ax.set_xlabel('waveform size (pts)')
    ax.set_ylabel('#sequence entries')
    cbar = fig.colorbar(pcolor, ax=ax, extend='both')
    cbar.set_label('Loading time (s)')
    fig.tight_layout()
    fig.savefig('awg_load_wf_bechmark_1.png', dpi=200, bbox_inches='tight')

if True:
    xi, yi = x2 / 1280, y[0]
    popt, perr = curve_fit(linear, xi, yi)
    xfit = np.linspace(0, 12, 101)
    yfit = linear(xfit, *popt)

    # plt.close('all')
    fig, ax = plt.subplots(1)
    ax.plot(xi, yi, 'ko')
    ax.plot(xfit, yfit, 'k--', alpha=0.4)
    t = 'm={:.3f} s/(1280 pts)'.format(popt[0])
    ax.text(0.1, 0.6, t, transform=ax.transAxes)
    ax.set_xlabel('waveform size (1280 pts)')
    ax.set_ylabel('Loading time (s)')
    ax.set_title('sequence entries: {}'.format(x2[0]))
    fig.tight_layout()
    fig.savefig('awg_load_wf_bechmark_2.png', dpi=200, bbox_inches='tight')

if True:
    xi, yi = x1, y[:, 0]
    popt, perr = curve_fit(linear, xi, yi)
    xfit = np.linspace(0, 101, 201)
    yfit = linear(xfit, *popt)

    # plt.close('all')
    fig, ax = plt.subplots(1)
    ax.plot(xi, yi, 'ko')
    ax.plot(xfit, yfit, 'k--', alpha=0.4)
    t = 'm={:.3f} s/seq entry'.format(popt[0])
    ax.text(0.1, 0.6, t, transform=ax.transAxes)
    ax.set_xlabel('#sequence entries')
    ax.set_ylabel('Loading time (s)')
    ax.set_title('waveform size {} pts'.format(x1[0]))
    fig.tight_layout()
    fig.savefig('awg_load_wf_bechmark_3.png', dpi=200, bbox_inches='tight')

xg1, xg2 = np.meshgrid(x1, x2)
xg3 = (xg1 * xg2).T

load_time = y / xg3  # s/point
load_time_block = load_time * 1280  # min block
avg_load_time = np.mean(load_time)
avg_load_time_block = np.mean(load_time_block)

print('average loading time: {} s/point'.format(avg_load_time))
print('average loading time: {} s/block'.format(avg_load_time_block))
