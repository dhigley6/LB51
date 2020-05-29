"""Make plots for thesis
"""

import numpy as np
import matplotlib.pyplot as plt

import LB51_get_cal_data
import LB51FitAbs
import set_plot_params
set_plot_params.init_paper()

####
JOE_INTENSITY = np.array([0.01, 0.03, 0.1, 0.3, 1])
JOE_TRANS = np.array([0.32, 0.35, 0.4, 0.65, 0.9])
JOE_ABS = -np.log(JOE_TRANS)
JOE_ABS = JOE_ABS/JOE_ABS[0]
####

def get_lk30_data():
    """Return processed LK30 data
    """
    lk30_high_plus = np.load('../../LK30/data/proc/high_plus.npz')
    lk30_high_minus = np.load('../../LK30/data/proc/high_minus.npz')
    lk30_high_xas = np.load('../../LK30/data/proc/high_xas.npz')
    lk30_high_xmcd = np.load('../../LK30/data/proc/high_xmcd.npz')
    lk30_const_plus = np.load('../../LK30/data/proc/constant_abs_plus.npz')
    lk30_const_minus = np.load('../../LK30/data/proc/constant_abs_minus.npz')
    lk30_const_xas = np.load('../../LK30/data/proc/constant_abs_xas.npz')
    lk30_const_xmcd = np.load('../../LK30/data/proc/constant_abs_xmcd.npx.npz')
    high_dict = {'plus': lk30_high_plus,
                 'minus': lk30_high_minus,
                 'xas': lk30_high_xas,
                 'xmcd': lk30_high_xmcd}
    const_dict = {'plus': lk30_const_plus,
                  'minus': lk30_const_minus,
                  'xas': lk30_const_xas,
                  'xmcd': lk30_const_xmcd}
    return {'high': high_dict,
            'const': const_dict}

def get_dict(data):
    if 'sum_blown' in data.keys():
        to_return = {'intact': data['sum_intact'],
                     'blown': data['sum_blown']}
    else:
        to_return = {'intact': data['sum_intact']}
    return to_return

def get_short_data():
    """Return short pulse summary data
    """
    data86 = LB51_get_cal_data.get_nonburst_data(86)
    data122 = LB51_get_cal_data.get_nonburst_data(122)
    data99 = LB51_get_cal_data.get_burst_data(99)
    data136 = LB51_get_cal_data.get_burst_data(136)
    data331 = LB51_get_cal_data.get_burst_data(331)
    data99 = LB51_get_cal_data.combine_burst_data([data99, data136, data331])
    data290 = LB51_get_cal_data.get_nonburst_data(290)
    data359 = LB51_get_cal_data.get_nonburst_data(359)
    data388 = LB51_get_cal_data.get_burst_data(388)
    data510 = LB51_get_cal_data.get_burst_data(510)
    data416 = LB51_get_cal_data.get_burst_data(416)
    to_return = {'86': get_dict(data86),
                 '99': get_dict(data99),
                 '122': get_dict(data122),
                 '290': get_dict(data290),
                 '359': get_dict(data359),
                 '388': get_dict(data388),
                 '416': get_dict(data416),
                 '510': get_dict(data510)}
    return to_return

def get_long_data():
    """Return long pulse summary data
    """
    data548 = LB51_get_cal_data.get_nonburst_data(548) 
    data641 = LB51_get_cal_data.get_nonburst_data(641)
    data583 = LB51_get_cal_data.get_burst_data(583)
    data554 = LB51_get_cal_data.get_burst_data(554)
    data603 = LB51_get_cal_data.get_burst_data(603)
    data647 = LB51_get_cal_data.get_burst_data(647)
    data603 = LB51_get_cal_data.combine_burst_data([data603, data647])
    to_return = {'548': get_dict(data548),
                 '641': get_dict(data641),
                 '583': get_dict(data583),
                 '554': get_dict(data554),
                 '603': get_dict(data603)}
    return to_return

def plotting_method():
    long_data = get_long_data()
    axs = panel_plots([long_data['603']['intact']])
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[2].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('../plots/2018_01_23_plotting_method.eps', dpi=600)
    plt.savefig('../plots/2018_01_23_plotting_method.png', dpi=600)

def calc_ints(data):
    stim_window = (772.5, 775.75)
    abs_window = (770, 784)
    phot = data['phot']
    in_abs_range = (phot > 770) & (phot < 784)
    win_phot = phot[(phot > 772.5) & (phot < 775.75)]
    exc_sam = data['exc_sam_spec']
    win_exc_sam = exc_sam[(phot > 772.5) & (phot < 775.75)]
    win_phot_abs = phot[in_abs_range]
    win_no_sam = data['no_sam_spec']*(1-np.exp(-1*data['ssrl_absorption']+0.6))
    win_no_sam = win_no_sam[in_abs_range]
    abs_phots = np.trapz(win_no_sam, x=win_phot_abs)
    stim_phots = np.trapz(win_exc_sam, x=win_phot)
    print(stim_phots/abs_phots)
    plt.figure()
    plt.plot(win_phot_abs, win_no_sam)
    plt.plot(win_phot, win_exc_sam)

def all_short_plot():
    data = get_short_data()
    panel_plots([data['86']['intact'], data['122']['intact'], data['290']['intact'], data['359']['intact'], data['510']['intact'], data['416']['intact'], data['388']['intact'], data['99']['intact']])
    plt.tight_layout()

def all_long_plot():
    data = get_long_data()
    panel_plots([data['548']['intact'], data['641']['intact'], data['583']['intact'], data['554']['intact'], data['603']['intact']])
    plt.tight_layout()

def summary_short_plot():
    data = get_short_data()
    data_sets = [data['359']['intact'], data['388']['intact'], data['99']['intact']]
    axs = panel_plots(data_sets)
    axs[2, 1].set_xlabel('Photon Energy (eV)')
    axs[0, 0].set_title('0.01 J/cm$^2$')
    axs[0, 1].set_title('0.72 J/cm$^2$')
    axs[0, 2].set_title('1.8 J/cm$^2$') 
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.savefig('2018_01_23_short_summary.eps', dpi=600)
    plt.savefig('2018_01_23_short_summary.png', dpi=600)

def summary_long_plot():
    data = get_long_data()
    data_sets = [data['641']['intact'], data['583']['intact'], data['554']['intact'], data['603']['intact']]
    axs = panel_plots(data_sets)
    axs[2, 1].set_xlabel('Photon Energy (eV)')
    axs[0, 0].set_title('0.03 J/cm$^2$')
    axs[0, 1].set_title('0.2 J/cm$^2$')
    axs[0, 2].set_title('0.6 J/cm$^2$')
    axs[0, 3].set_title('9.2 J/cm$^2$')
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.savefig('2018_01_23_long_summary.eps', dpi=600)
    plt.savefig('2018_01_23_long_summary.png', dpi=600)

def abs_fluence_series_plot():
    short_data = get_short_data()
    long_data = get_long_data()
    f, axs = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(6, 4.5))
    in_range359 = get_abs_in_range(short_data['359']['intact'])
    in_range388 = get_abs_in_range(short_data['388']['intact'])
    in_range99 = get_abs_in_range(short_data['99']['intact'])
    axs[1, 0].plot(in_range359['phot'], in_range359['diff'], 'b', label='0.01 J/cm$^2$')
    #axs[1, 0].plot(in_range388['phot'], in_range388['diff'])
    axs[1, 0].plot(in_range99['phot'], in_range99['diff'], 'r', label='1.8 J/cm$^2$')
    in_range641 = get_abs_in_range(long_data['641']['intact'])
    in_range583 = get_abs_in_range(long_data['583']['intact'])
    in_range554 = get_abs_in_range(long_data['554']['intact'])
    in_range603 = get_abs_in_range(long_data['603']['intact'])
    axs[1, 1].plot(in_range641['phot'], in_range641['diff'], 'b', label='0.03 J/cm$^2$')
    #axs[1, 1].plot(in_range583['phot'], in_range583['diff'])
    axs[1, 1].plot(in_range554['phot'], in_range554['diff'], 'g', label='0.6 J/cm$^2$')
    axs[1, 1].plot(in_range603['phot'], in_range603['diff'], 'r', label='9.2 J/cm$^2$')
    axs[0, 0].plot(short_data['99']['intact']['phot'], short_data['99']['intact']['ssrl_absorption'], 'k')
    axs[0, 0].plot(in_range99['phot'], in_range99['abs'], 'r')
    axs[0, 1].plot(short_data['99']['intact']['phot'], short_data['99']['intact']['ssrl_absorption'], 'k')
    axs[0, 1].plot(in_range603['phot'], in_range603['abs'], 'r')
    axs[1, 0].set_xlabel('Photon Energy (eV)')
    axs[1, 1].set_xlabel('Photon Energy (eV)')
    axs[1, 0].set_xlim((774, 782))
    axs[0, 0].set_title('5 fs Pulses')
    axs[0, 1].set_title('25 fs Pulses')
    axs[0, 0].set_ylabel('Absorption (a.u.)')
    axs[1, 0].set_ylabel('Change (a.u.)')
    axs[1, 0].legend(loc='best')
    axs[1, 1].legend(loc='best')
    axs[1, 0].axhline(color='k', linestyle='--', alpha=0.5)
    axs[1, 1].axhline(color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('2018_01_23_abs_series_plot.eps', dpi=600)
    plt.savefig('2018_01_23_abs_series_plot.png', dpi=600)

def abs_change_comp_plot():
    """Compare absorption changes induced by highest fluence short and long pulses
    """
    short_data = get_short_data()
    long_data = get_long_data()
    lk30 = get_lk30_data()
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    data99 = short_data['99']['intact']
    in_range99 = get_abs_in_range(data99)
    test = short_data['99']['intact']
    phot = test['phot']
    axs[0].plot(phot, test['ssrl_absorption'], 'k', label='Linear Absorption')
    in_range603 = get_abs_in_range(long_data['603']['intact'])
    axs[0].plot(in_range99['phot'], in_range99['abs'], label='5 fs, 1.8 J/cm$^2$')
    axs[0].plot(in_range603['phot'], in_range603['abs'], label='25 fs, 9.2 J/cm$^2$')
    axs[1].plot(lk30['high']['xas']['phot'], lk30['high']['xas']['diff']*4, label='4x mono.\n79 mJ/cm$^2$', color='c')
    axs[1].legend(loc='lower left')
    abs_comp_plot(axs[1], [short_data['99']['intact'], long_data['603']['intact']])
    axs[1].set_xlim((774, 782))
    axs[1].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Absorption (a.u.)')
    axs[1].set_ylabel('Change (a.u.)')
    axs[0].legend(loc='upper left')
    axs[0].text(0.9, 0.85, '(a)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.85, '(b)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[1].transAxes)
    plt.tight_layout()
    plt.savefig('2018_02_01_highest_fluence_abs_plot.eps', dpi=600)
    plt.savefig('2018_02_01_highest_fluence_abs_plot.png', dpi=600)

def outrun_cascade_plot():
    """Outrunning electron cascade plot
    """
    short_data = get_short_data()
    long_data = get_long_data()
    in_range388 = get_abs_in_range(short_data['388']['intact'], threshold=0.7)
    in_range554 = get_abs_in_range(long_data['554']['intact'], threshold=0.7)
    f, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
    test = short_data['388']['intact']
    axs[0].plot(test['phot'], test['ssrl_absorption'], 'k', label='Linear Absorption')
    axs[0].plot(in_range388['phot'], in_range388['abs'], 'g', label='5 fs,\n0.7 J/cm$^2$')
    axs[0].plot(in_range554['phot'], in_range554['abs'], 'm', label='25 fs,\n0.6 J/cm$^2$')
    axs[1].plot(in_range388['phot'], in_range388['diff'], 'g')
    axs[1].plot(in_range554['phot'], in_range554['diff'], 'm')
    axs[0].legend(loc='best')
    axs[1].set_xlim((774, 782))
    axs[1].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Absorption (a.u.)')
    axs[1].set_ylabel('Change (a.u.)')
    axs[0].text(0.9, 0.85, '(a)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.85, '(b)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[1].transAxes)
    plt.tight_layout()
    plt.savefig('2018_01_23_outrun_cascade.eps', dpi=600)
    plt.savefig('2018_01_23_outrun_cascade.png', dpi=600)

def fluence_dependence_plot():
    short_data = get_short_data()
    long_data = get_long_data()
    elec_phots = np.linspace(777.2, 778.8, 100)
    short_data_sets = [short_data['86']['intact'], short_data['122']['intact'], short_data['290']['intact'], short_data['359']['intact'], short_data['388']['intact'], short_data['99']['intact']]
    long_data_sets = [long_data['548']['intact'], long_data['641']['intact'], long_data['583']['intact'], long_data['554']['intact'], long_data['603']['intact']]
    short_fluences, short_ints = get_fluence_dep_ints(elec_phots, short_data_sets)
    long_fluences, long_ints = get_fluence_dep_ints(elec_phots, long_data_sets)
    f, axs = plt.subplots(3, 1, figsize=(4, 6))
    short_intensities = short_fluences/5
    long_intensities = long_fluences/25
    data603in_range = get_abs_in_range(long_data['603']['intact'])
    axs[0].plot(short_data['290']['intact']['phot'], short_data['290']['intact']['ssrl_absorption'], 'k--', label='Linear\nAbsorption')
    axs[0].plot(data603in_range['phot'], data603in_range['abs'], 'k-', label='9.2 J/cm$^2$,\n25 fs')
    elec_abs = np.interp(elec_phots, data603in_range['phot'], data603in_range['abs'])
    axs[0].fill_between(elec_phots, [0.6]*len(elec_phots), elec_abs, facecolor='k', edgecolor='w', alpha=0.5)
    axs[1].semilogx(short_intensities/1000, short_ints, marker='o', linestyle='--', color='k', label='5 fs Pulses')
    axs[1].semilogx(long_intensities/1000, long_ints, marker='D', color='k', label='25 fs Pulses')
    #axs[1].semilogx(JOE_INTENSITY, JOE_ABS, 'g')
    axs[2].semilogx(short_fluences/1000, short_ints, linestyle='--', marker='o', color='k', label='5 fs Pulses')
    axs[2].semilogx(long_fluences/1000, long_ints, marker='D', color='k', label='25 fs Pulses')
    axs[0].set_xlim((774, 782))
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Intensity (10$^{15}$ W/cm$^2$)')
    axs[1].set_ylabel('Integral (a.u.)')
    axs[2].set_xlabel('Fluence (J/cm$^2$)')
    axs[2].set_ylabel('Integral (a.u.)')
    axs[0].text(0.9, 0.85, '(a)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.85, '(b)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[1].transAxes)
    axs[2].text(0.9, 0.85, '(c)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[2].transAxes)
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    axs[2].legend(loc='best')
    plt.tight_layout()
    plt.savefig('2018_02_01_trans_fluence_dependence.eps', dpi=600)
    plt.savefig('2018_02_01_trans_fluence_dependence.png', dpi=600)

def comp_joe():
    """Compare to Joe Chen's calculations
    """
    short_data = get_short_data()
    long_data = get_long_data()
    elec_phots = np.linspace(777.2, 778.8, 100)
    short_data_sets = [short_data['86']['intact'], short_data['122']['intact'], short_data['290']['intact'], short_data['359']['intact'], short_data['388']['intact'], short_data['99']['intact']]
    long_data_sets = [long_data['548']['intact'], long_data['641']['intact'], long_data['583']['intact'], long_data['554']['intact'], long_data['603']['intact']]
    short_fluences, short_ints = get_fluence_dep_ints(elec_phots, short_data_sets)
    long_fluences, long_ints = get_fluence_dep_ints(elec_phots, long_data_sets)
    f, axs = plt.subplots(2, 1, figsize=(4, 5))
    short_intensities = short_fluences/5
    long_intensities = long_fluences/25
    data603in_range = get_abs_in_range(long_data['603']['intact'])
    axs[0].plot(short_data['290']['intact']['phot'], short_data['290']['intact']['ssrl_absorption'], 'k--', label='Linear\nAbsorption')
    axs[0].plot(data603in_range['phot'], data603in_range['abs'], 'k-', label='9.2 J/cm$^2$,\n25 fs')
    elec_abs = np.interp(elec_phots, data603in_range['phot'], data603in_range['abs'])
    axs[0].fill_between(elec_phots, [0.6]*len(elec_phots), elec_abs, facecolor='k', edgecolor='w', alpha=0.5)
    axs[1].semilogx(short_intensities/1000, short_ints, marker='o', linestyle='--', color='k', label='5 fs Pulses')
    axs[1].semilogx(long_intensities/1000, long_ints, marker='D', color='k', label='25 fs Pulses')
    axs[1].semilogx(JOE_INTENSITY, JOE_ABS, 'g', label='Superradiative\nStimulated\nEmission')
    axs[0].set_xlim((774, 782))
    axs[0].set_xlabel('Photon Energy (eV)')
    axs[0].set_ylabel('Intensity (a.u.)')
    axs[1].set_xlabel('Intensity (10$^{15}$ W/cm$^2$)')
    axs[1].set_ylabel('Integral (a.u.)')
    axs[0].text(0.9, 0.85, '(a)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[0].transAxes)
    axs[1].text(0.9, 0.85, '(b)', fontsize=10, weight='bold',
                horizontalalignment='center', transform=axs[1].transAxes)
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='best')
    plt.tight_layout()
    plt.savefig('2018_02_01_trans_comp_stohr.eps', dpi=600)
    plt.savefig('2018_02_01_trans_comp_stohr.png', dpi=600)

def get_fluence_dep_ints(phots, data_sets):
    fluences = []
    integrals = []
    norm_xas = np.interp(phots, data_sets[0]['phot'], data_sets[0]['ssrl_absorption'])-0.6
    norm_int = np.trapz(norm_xas, x=phots)
    for data in data_sets:
        in_range = get_abs_in_range(data)
        xas = np.interp(phots, in_range['phot'], in_range['abs'])
        xas = xas-0.6
        integral = np.trapz(xas, x=phots)
        integrals.append(integral)
        fluences.append(data['fluence'])
    return np.array(fluences), np.array(integrals)/norm_int

def make_panel_plot(data_specs):
    data_list = []
    for data_spec in data_specs:
        if data_spec[1] is True:
            data = LB51_get_cal_data.get_burst_data(data_spec[0])
        else:
            data = LB51_get_cal_data.get_nonburst_data(data_spec[0])
        data_list.append(data['sum_intact'])
    panel_plots(data_list)

def panel_plots(data_sets):
    f, axs = plt.subplots(3, len(data_sets), sharex=True, sharey='row', figsize=(6, 5))
    if len(data_sets) > 1:
        for ind, data in enumerate(data_sets):
            abs_plot(axs[0, ind], data)
            spec_plot(axs[1, ind], data)
            diff_plot(axs[2, ind], data)
        axs[2, 0].set_xlim((770, 786))
        axs[0, 0].set_ylim((-1, 1.9))
        axs[0, 0].set_ylabel('Absorption (a.u.)')
        axs[1, 0].set_ylabel('Intensity (a.u.)')
        axs[2, 0].set_ylabel('Intensity (a.u.)')
        #axs[0, 0].legend(loc='best')
        #axs[1, 0].legend(loc='best')
        #axs[2, 0].legend(loc='best')
    else:
        abs_plot(axs[0], data_sets[0])
        spec_plot(axs[1], data_sets[0])
        diff_plot(axs[2], data_sets[0])
        axs[2].set_xlim((768, 788))
        axs[0].set_ylim((-1, 1.9))
        axs[0].set_ylabel('Absorption (a.u.)')
        axs[1].set_ylabel('Intensity (a.u.)')
        axs[2].set_ylabel('Intensity (a.u.)')
        axs[0].text(0.9, 0.85, '(a)', fontsize=10, weight='bold',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[1].text(0.9, 0.85, '(b)', fontsize=10, weight='bold',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[2].text(0.9, 0.85, '(c)', fontsize=10, weight='bold',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.tight_layout()
    return axs

def get_abs_in_range(data, threshold=0.3):
    """Return photon energy range and absorption with significant data
    """
    in_range = data['no_sam_spec'] > threshold*np.amax(data['no_sam_spec'])
    phot_in_range = data['phot'][in_range]
    abs_in_range = data['abs'][in_range]
    diff_in_range = data['abs'][in_range]-data['ssrl_absorption'][in_range]
    to_return = {'phot': phot_in_range,
                 'abs': abs_in_range,
                 'diff': diff_in_range}
    return to_return

def abs_plot(ax, data, threshold=0.27):
    phot = data['phot']
    ax.plot(phot, data['ssrl_absorption'], label='Linear Absorption')
    in_range = data['no_sam_spec'] > threshold*np.amax(data['no_sam_spec'])
    ax.plot(phot[in_range], data['abs'][in_range], label='Measured Absorption')
    ax.plot(phot[in_range], data['abs'][in_range]-data['ssrl_absorption'][in_range], label='Absorption Change')
    ax.axhline(linestyle='--', color='k')

def abs_comp_plot(ax, data_sets, threshold=0.3):
    for data in data_sets:
        phot = data['phot']
        in_range = data['no_sam_spec'] > threshold*np.amax(data['no_sam_spec'])
        ax.plot(phot[in_range], data['abs'][in_range]-data['ssrl_absorption'][in_range])
        ax.axhline(linestyle='--', color='k')

def spec_plot(ax, data):
    phot = data['phot']
    norm = np.amax(data['no_sam_spec'])
    no_sam_spec = data['no_sam_spec']/norm
    sam_spec = data['sam_spec']/norm
    exc_spec = data['exc_sam_spec']/norm
    ax.plot(phot, no_sam_spec*0.55, 'k--', label='No Sample\nSpectrum X 0.55')
    ax.plot(phot, sam_spec, 'k', label='Sample Spectrum')
    ssrl_trans = np.exp(-1*data['ssrl_absorption'])
    lin_sam_spec = no_sam_spec*ssrl_trans
    ax.plot(phot, lin_sam_spec, 'k:', label='Linear Sample\nSpectrum')
    ax.fill_between(phot, lin_sam_spec, sam_spec, where=(exc_spec > 0), facecolor='b', edgecolor='w', alpha=0.65)
    ax.fill_between(phot, lin_sam_spec, sam_spec, where=(exc_spec < 0), facecolor='r', edgecolor='w', alpha=0.65)
    pass

def diff_plot(ax, data):
    phot = data['phot']
    norm = np.amax(data['no_sam_spec'])
    no_sam_spec = data['no_sam_spec']/norm
    exc_spec = data['exc_sam_spec']/norm
    ax.plot(phot, exc_spec*5, color='k', label='Excess Transmitted\nPhotons X 5')
    ax.plot(phot, no_sam_spec, 'k--', label='Incident Spectrum')
    ax.fill_between(phot, np.zeros_like(phot), exc_spec*5, where=(exc_spec>0), facecolor='b', edgecolor='w', alpha=0.65)
    ax.fill_between(phot, np.zeros_like(phot), exc_spec*5, where=(exc_spec<0), facecolor='r', edgecolor='w', alpha=0.65)


#### Below here, plots for testing purposes  ###############################

def spec_int_plot(run_set=583):
    """Plot integrals of spectra vs run number to check for clipping
    """
    data = LB51_get_cal_data.get_burst_data(run_set, correct_split=False)
    f, axs = plt.subplots(2, 1, sharex=True)
    axs[0].scatter(data['intact']['run_num'], np.sum(data['intact']['sam_spec'], axis=1), color='b')
    axs[0].scatter(data['intact']['run_num'], np.sum(data['intact']['no_sam_spec'], axis=1), color='g')
    axs[1].scatter(data['blown']['run_num'], np.sum(data['blown']['sam_spec'], axis=1), color='b')
    axs[1].scatter(data['blown']['run_num'], np.sum(data['blown']['no_sam_spec'], axis=1), color='g')

def noise_level_plot():
    """Check noise levels of data by plotting blown up data along with intact data
    """
    data603intact = LB51_get_cal_data.get_burst_data(603)['sum_intact']
    data603blown = LB51_get_cal_data.get_burst_data(603)['sum_blown']
    data647intact = LB51_get_cal_data.get_burst_data(647)['sum_intact']
    data647blown = LB51_get_cal_data.get_burst_data(647)['sum_blown']
    data554intact = LB51_get_cal_data.get_burst_data(554)['sum_intact']
    data554blown = LB51_get_cal_data.get_burst_data(554)['sum_blown']
    data99 = LB51_get_cal_data.get_burst_data(99)
    data136 = LB51_get_cal_data.get_burst_data(136)
    data331 = LB51_get_cal_data.get_burst_data(331)
    data331intact = data331['sum_intact']
    data331blown = data331['sum_blown']
    data99 = LB51_get_cal_data.combine_burst_data([data99, data136])
    data99intact = data99['sum_intact']
    data99blown = data99['sum_blown']
    f, axs = plt.subplots(4, 1, sharex=True)
    abs_comp_plot(axs[0], [data603intact, data603blown])
    axs[3].set_xlabel('Photon Energy (eV)')
    axs[3].set_ylabel('$\Delta$ XAS')
    axs[2].set_ylabel('$\Delta$ XAS')
    axs[1].set_ylabel('$\Delta$ XAS')
    axs[0].set_ylabel('$\Delta$ XAS')
    abs_comp_plot(axs[1], [data647intact, data647blown])
    #abs_comp_plot(axs[2], [data554intact, data554blown], threshold=0.7)
    abs_comp_plot(axs[2], [data99intact, data99blown])
    abs_comp_plot(axs[3], [data331intact, data331blown])
    axs[0].axvline(775, linestyle='--', color='k')
    axs[1].axvline(775, linestyle='--', color='k')
    axs[2].axvline(775, linestyle='--', color='k')
    axs[3].axvline(775, linestyle='--', color='k')
    f, axs = plt.subplots(2, 1, sharex=True)
    abs_comp_plot(axs[0], [data554intact, data554blown], threshold=0.4)
    abs_comp_plot(axs[1], [data603intact, data603blown])
