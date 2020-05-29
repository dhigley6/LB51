"""Make plots for LB51 spectroscopy paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pickle

import LB51_get_cal_data

#import dummy_ax

import set_plot_params
set_plot_params.init_paper_small()

NO_SAM = 1      # Factor to multiply no sample spectra by when plotting

def summary_plot():
    def spectra_series_plot(ax, data_list):
        for ind, data in enumerate(data_list):
            phot = data['sum_intact']['phot']
            norm = np.amax(data['sum_intact']['no_sam_spec'])
            no_sam_spec = data['sum_intact']['no_sam_spec']/norm
            exc_spec = data['sum_intact']['exc_sam_spec']/norm
            offset = ind*1.5
            ax.plot(phot, offset+exc_spec*5, color='k', label='5 X Diff.')
            ax.plot(phot, offset+no_sam_spec, 'k--', label='Ref.')
            ax.fill_between(phot, offset+np.zeros_like(phot), offset+exc_spec*5, 
                            where=(exc_spec>0), facecolor='b', edgecolor='w')
            ax.fill_between(phot, offset+np.zeros_like(phot), offset+exc_spec*5,
                            where=(exc_spec<0), facecolor='r', edgecolor='w')
            if ind == 0:
                handles, labels = axs[1, 1].get_legend_handles_labels()
                f.legend(handles, labels, loc=(0.45, 0.6), frameon=True)
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_data = LB51_get_cal_data.get_short_pulse_data()
    binned_603 = LB51_get_cal_data.bin_burst_data(long_data['603']['total'], 603)
    binned_99 = LB51_get_cal_data.bin_burst_data(short_data['99']['total'], 99)
    short_summary_data = [short_data['359']]
    short_summary_data.extend(binned_99)
    long_summary_data = [long_data['641']]
    long_summary_data.extend(binned_603)
    f, axs = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(3.37, 5),
                          gridspec_kw={'height_ratios': [1, 2]})
    spectra_series_plot(axs[1, 0], short_summary_data)
    spectra_series_plot(axs[1, 1], long_summary_data)
    sum_data359 = short_data['359']['sum_intact']
    ssrl_res_absorption = sum_data359['ssrl_absorption']-sum_data359['ssrl_absorption'][0]
    axs[0, 0].plot(sum_data359['phot'], ssrl_res_absorption, label='Absorption')
    axs[0, 1].plot(sum_data359['phot'], ssrl_res_absorption, label='Absorption')
    emission = get_emission()
    axs[0, 0].plot(emission['x']-0.4, -1*emission['y']+0.2, label='Emission')
    axs[0, 1].plot(emission['x']-0.4, -1*emission['y']+0.2, label='Emission')
    format_summary_plot(f, axs)
    plt.savefig('../plots/2020_03_22_summary.eps', dpi=600)
    plt.savefig('../plots/2020_03_22_summary.png', dpi=600)
    
def format_summary_plot(f, axs):
    def place_vlines():
        ax_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        vline_loc_list = [774.5, 776.5, 778]
        for ax in ax_list:
            for vline_loc in vline_loc_list:
                ax.axvline(vline_loc, linestyle='--', color='k')
    place_vlines()
    axs[1, 0].set_xlim((769, 787))
    axs[1, 0].set_ylim((-0.25, 4.75))
    #f.add_subplot(111, frameon=False)
    #plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #plt.xlabel('Photon Energy (eV)')
    f.text(0.585, 0.02, 'Photon Energy (eV)', ha='center')
    axs[0, 0].set_ylabel('Intensity (a.u.)')
    axs[1, 0].set_ylabel('Intensity (a.u.)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc=(0.42, 0.9), frameon=True)
    texts = []
    texts.append(axs[1, 0].text(769.5, 3.2, '779 eV,\n1080\nmJ/cm$^2$', transform=axs[1, 0].transData,
                   fontsize=8))
    texts.append(axs[1, 0].text(780.5, 1.8, '777 eV,\n1600\nmJ/cm$^2$', transform=axs[1, 0].transData,
                   fontsize=8))
    texts.append(axs[1, 0].text(769.5, 0.25, '779 eV,\n12\nmJ/cm$^2$', transform=axs[1, 0].transData,
       fontsize=8))
    texts.append(axs[1, 1].text(769.5, 3.2, '780 eV,\n9490\nmJ/cm$^2$', transform=axs[1, 1].transData,
                   fontsize=8))
    texts.append(axs[1, 1].text(780.5, 1.8, '777 eV,\n8950\nmJ/cm$^2$', transform=axs[1, 1].transData,
                   fontsize=8))
    texts.append(axs[1, 1].text(769.5, 0.25, '781 eV,\n30\nmJ/cm$^2$', transform=axs[1, 1].transData,
                   fontsize=8))
    texts.append(axs[1, 0].text(0.3, 0.93, '5 fs', transform=axs[1, 0].transAxes, fontsize=10, backgroundcolor='w'))
    texts.append(axs[1, 1].text(0.3, 0.93, '25 fs', transform=axs[1, 1].transAxes, fontsize=10, backgroundcolor='w'))
    for text in texts:
        text.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                               path_effects.Normal()])
    axs[0, 0].text(0.1, 0.88, 'a', transform=axs[0, 0].transAxes, fontsize=10, fontweight='bold')
    axs[1, 0].text(0.1, 0.94, 'c', transform=axs[1, 0].transAxes, fontsize=10, fontweight='bold')
    axs[0, 1].text(0.8, 0.88, 'b', transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold')
    axs[1, 1].text(0.8, 0.94, 'd', transform=axs[1, 1].transAxes, fontsize=10, fontweight='bold')
    plt.tight_layout(w_pad=0, h_pad=0.1, rect=(0, 0.03, 1, 1))

def quant_plot():
    """Make plot of stimulated scattering strength versus incident intensity
    """
    short_data = LB51_get_cal_data.get_short_pulse_data()
    long_data = LB51_get_cal_data.get_long_pulse_data()
    short_run_sets_list = ['99', '290', '359', '388']
    short_fluences = []
    short_stim_strengths = []
    for run_set in short_run_sets_list:
        fluence = short_data[run_set]['sum_intact']['fluence']
        if run_set == '99':
            fluence = 1898
        stim_strength = get_stim_efficiency(short_data[run_set])
        short_fluences.append(fluence)
        short_stim_strengths.append(stim_strength)
    long_run_sets_list = ['641', '554', '603']
    long_fluences = []
    long_stim_strengths = []
    for run_set in long_run_sets_list:
        fluence = long_data[run_set]['sum_intact']['fluence']
        stim_strength = get_stim_efficiency(long_data[run_set])
        long_fluences.append(fluence)
        long_stim_strengths.append(stim_strength)
    plt.figure(figsize=(3.37, 2.5))
    plt.scatter(np.array(short_fluences)/5, 100*np.array(short_stim_strengths), c='b', label='5 fs')
    plt.scatter(np.array(long_fluences)/25, 100*np.array(long_stim_strengths), c='r', label='25 fs')
    quant_data = {'short_fluences': np.array(short_fluences)*1E12,
                  'long_fluences': np.array(long_fluences)*1E12,
                  'short_efficiencies': 100*np.array(short_stim_strengths),
                  'long_efficiencies': 100*np.array(long_stim_strengths)}
    save_quant_data(quant_data)
    format_quant_plot()
    plt.savefig('../plots/2019_02_03_quant.eps', dpi=600)
    plt.savefig('../plots/2019_02_03_quant.png', dpi=600)
    
def save_quant_data(quant_data):
    """Save quantified data
    """
    with open('../data/proc/stim_efficiency.pickle', 'wb') as f:
        pickle.dump(quant_data, f)
    
def format_quant_plot():
    plt.xlabel('Intensity (10$^{12}$ W/cm$^2$)')
    plt.ylabel('Stimulated Inelastic\nScattering Efficiency (%)')
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

def get_stim_strength(data):
    phot = data['sum_intact']['phot']
    no_sam_spec = data['sum_intact']['no_sam_spec']
    phot_in_range = (phot > 770) & (phot < 790)
    no_sam_spec_sum = np.sum(no_sam_spec[phot_in_range] )
    phot_in_range = (phot > 773.5) & (phot < 775)
    stim = data['sum_intact']['exc_sam_spec']
    stim_sum = np.sum(stim[phot_in_range])
    stim_strength = stim_sum/no_sam_spec_sum
    return stim_strength

def get_stim_efficiency(data):
    ssrl_res_absorption = data['sum_intact']['ssrl_absorption']-data['sum_intact']['ssrl_absorption'][0]
    ssrl_res_trans = np.exp(-1*ssrl_res_absorption)
    res_transmitted = data['sum_intact']['no_sam_spec']*ssrl_res_trans
    res_absorbed = data['sum_intact']['no_sam_spec']-res_transmitted
    phot = data['sum_intact']['phot']
    abs_region = (phot > 774) & (phot < 780)
    res_absorbed_sum = np.sum(res_absorbed[abs_region])
    stim_region = (phot > 773.5) & (phot < 775)
    stim = data['sum_intact']['exc_sam_spec']
    stim_sum = np.sum(stim[stim_region])
    stim_efficiency = stim_sum/res_absorbed_sum
    return stim_efficiency

def overview_plot():
    """Make plots summarizing the obtained data
    """
    def individual_shots_plot(ax, data, shots_to_plot):
        phot = data['sum_intact']['phot']
        for ind, shot in enumerate(shots_to_plot):
            sam_spec = data['intact']['sam_spec'][shot]
            no_sam_spec = data['intact']['no_sam_spec'][shot]
            norm_sam_spec = sam_spec/np.amax(sam_spec)/4
            norm_no_sam_spec = no_sam_spec*NO_SAM/np.amax(sam_spec)/4
            if ind == 0:
                sam_label = 'Sam.'
                no_sam_label = 'Ref.'
            else:
                sam_label = '_nolegend_'
                no_sam_label = '_nolegend_'
            ax.plot(phot, norm_sam_spec+ind, 'k', label=sam_label)
            ax.plot(phot, norm_no_sam_spec+ind, 'k--', label=no_sam_label)
            if ind == 0:
                ax.legend(loc='best')
    
    def summed_spectra_plot(ax, data):
        phot = data['sum_intact']['phot']
        norm = np.amax(data['sum_intact']['no_sam_spec'])
        sam_spec = data['sum_intact']['sam_spec']/norm
        no_sam_spec = data['sum_intact']['no_sam_spec']/norm
        exc_spec = data['sum_intact']['exc_sam_spec']/norm
        ssrl_trans = np.exp(-1*data['sum_intact']['ssrl_absorption'])
        lin_sam_spec = no_sam_spec*ssrl_trans
        ax.plot(phot, no_sam_spec*NO_SAM, 'k--', label='Ref.')
        ax.plot(phot, sam_spec, 'k', label='Sam.')
        ax.plot(phot, lin_sam_spec, 'k:', label='Linear Sam.')
        ax.fill_between(phot, lin_sam_spec, sam_spec, where=(exc_spec > 0),
                        facecolor='b', edgecolor='w')
        ax.fill_between(phot, lin_sam_spec, sam_spec, where=(exc_spec < 0),
                        facecolor='r', edgecolor='w')
        
    def nonlinear_spectra_plot(ax, data):
        phot = data['sum_intact']['phot']
        norm = np.amax(data['sum_intact']['no_sam_spec'])
        no_sam_spec = data['sum_intact']['no_sam_spec']/norm
        exc_spec = data['sum_intact']['exc_sam_spec']/norm
        ax.plot(phot, exc_spec*5, color='k', label='5 X Diff.')
        ax.plot(phot, no_sam_spec, 'k--', label='Ref.')
        ax.fill_between(phot, np.zeros_like(phot), exc_spec*5, 
                        where=(exc_spec>0), facecolor='b', edgecolor='w')
        ax.fill_between(phot, np.zeros_like(phot), exc_spec*5,
                        where=(exc_spec<0), facecolor='r', edgecolor='w')
        
    def xas_plot(ax, low_data, high_data, threshold=0.15):
        high_phot = high_data['sum_intact']['phot']
        high_threshold = threshold*np.amax(high_data['sum_intact']['no_sam_spec'])
        high_in_range = high_data['sum_intact']['no_sam_spec'] > high_threshold
        ax.plot(high_phot[high_in_range], high_data['sum_intact']['abs'][high_in_range],
                color='tab:orange', label='1335\nmJ/cm$^2$')
        low_phot = low_data['sum_intact']['phot']
        low_threshold = threshold*np.amax(low_data['sum_intact']['no_sam_spec'])
        low_in_range = low_data['sum_intact']['no_sam_spec'] > low_threshold
        ax.plot(low_phot[low_in_range], low_data['sum_intact']['abs'][low_in_range],
                'g', label='12\nmJ/cm$^2$')
        ax.plot(low_phot, low_data['sum_intact']['ssrl_absorption'], 'k--', label='Sync.')
        
    short_data = LB51_get_cal_data.get_short_pulse_data()
    f, axs = plt.subplots(2, 2, sharex=True, figsize=(3.37, 4))
    shots_to_plot = [3, 4, 5] # Three consecutive pulses, not at beginning or end
    individual_shots_plot(axs[0, 0], short_data['99'], shots_to_plot)
    summed_spectra_plot(axs[1, 0], short_data['99'])
    nonlinear_spectra_plot(axs[0, 1], short_data['99'])
    xas_plot(axs[1, 1], short_data['359'], short_data['99'])
    format_data_overview_plot(f, axs)
    plt.savefig('../plots/2020_03_23_overview.eps', dpi=600)
    plt.savefig('../plots/2020_03_23_overview.png', dpi=600)

def format_data_overview_plot(f, axs):
    axs[1, 1].set_xlim((770, 783))
    axs[1, 1].set_xticks((770, 775, 780))
    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()
    axs[1, 0].set_xlabel('.', color=(0, 0, 0, 0))
    axs[0, 0].set_yticks([])
    axs[1, 0].set_yticks([])
    axs[0, 1].set_yticks([])
    axs[1, 1].set_yticks([])
    axs[0, 0].set_ylabel('Intensity')
    axs[1, 0].set_ylabel('Intensity')
    axs[0, 1].set_ylabel('Intensity')
    axs[1, 1].set_ylabel('XAS')
    axs[0, 1].yaxis.set_label_position('right')
    axs[1, 1].yaxis.set_label_position('right')
    axs[0, 0].text(0.9, 0.9, 'a', fontsize=10, weight='bold', horizontalalignment='center', transform=axs[0, 0].transAxes)
    axs[1, 0].text(0.9, 0.9, 'b', fontsize=10, weight='bold', horizontalalignment='center', transform=axs[1, 0].transAxes)
    axs[0, 1].text(0.9, 0.9, 'c', fontsize=10, weight='bold', horizontalalignment='center', transform=axs[0, 1].transAxes)
    axs[1, 1].text(0.9, 0.9, 'd', fontsize=10, weight='bold', horizontalalignment='center', transform=axs[1, 1].transAxes)
    f.text(0.5, 0.04, 'Photon Energy (eV)', va='center', ha='center')
    axs[0, 0].legend(loc='lower left', frameon=False, ncol=2)
    axs[0, 1].legend(loc='upper left', frameon=False)
    axs[1, 0].legend(loc='lower center', frameon=False)
    axs[1, 1].legend(loc='best', frameon=False, fontsize=8)
    axs[0, 1].axhline(linestyle=':', color='k')
    axs[0, 0].set_ylim((-0.5, 2.5))
    axs[1, 0].set_ylim((-0.5, 1.1))
    axs[0, 1].set_ylim((-0.3, 1.25))
    axs[0, 1].annotate('i', xy=(774.5, 0.1), xycoords='data', xytext=(772, 0.6), textcoords='data', arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), fontsize=10)
    axs[0, 1].annotate('ii', xy=(776.25, -0.05), xycoords='data', xytext=(780, -0.2), textcoords='data', arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), fontsize=10, verticalalignment='center')
    axs[0, 1].annotate('iii', xy=(778, 0.15), xycoords='data', xytext=(780.5, 0.8), textcoords='data', arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), fontsize=10)
    plt.tight_layout(pad=0.8, w_pad=0, h_pad=0)
    
def get_emission():
    data = np.genfromtxt('../data/measuredEmissionPoints3.txt')
    emission = {'x': data[:, 0],
                'y': data[:, 1]}
    return emission