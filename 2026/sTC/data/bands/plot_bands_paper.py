import sys
import numpy as np

from pyscf import lib

from matplotlib import pyplot as plt
# plt.style.use(['yeplot.yeplot', 'yeplot.times'])
#
# from yeplot import mkdir, get_color, get_lstyls, add_grid_


if __name__ == '__main__':

    # fml = 'C'
    # nocc = 4
    # pp = 'gth'
    # zeta = 'tz'
    # kmesh = '444'
    # emin = -30
    # emax = 55

    fml = 'Si'
    nocc = 4
    pp = 'gth'
    zeta = 'tz'
    kmesh = '444'
    emin = -20
    emax = 45

    fout = f'{fml}_{zeta}_{kmesh}_bands.pdf'

    lw1 = 2
    lw2 = 0.7
    mks1 = 7
    mks2 = 5
    zorder0 = 10
    methods = ['vcut_ws', 'vcut_ws+2', 'vcut_ws+3', 'vcut_ws+4']
    lstyls = [
        {
            'ls':'-', 'lw':lw1, 'color':'k', 'zorder': zorder0,
        },
        {
            'ls':'--', 'lw':lw2, 'color':'red', 'zorder': zorder0+5,
        },
        {
            'ls':'--', 'lw':lw2, 'color':'blue', 'zorder': zorder0+4,
        },
        {
            'ls':'--', 'lw':lw2, 'color':'green', 'zorder': zorder0+3,
        },
    ]
    labels = ['WS-TC', r'WS-sTC ($\eta=2$)', 'WS-sTC ($\eta=3$)', 'WS-sTC ($\eta=4$)']

    def plot1(fbands, lstyls, label, skip=None):
        en_vs_k = lib.chkfile.load(fbands, 'bands')
        kpath = lib.chkfile.load(fbands, 'kpath')

        ehomo = en_vs_k[nocc-1].max()
        elumo = en_vs_k[nocc].min()
        egap_indirect = elumo - ehomo
        egap_direct = en_vs_k[nocc,22] - en_vs_k[nocc-1,22]
        print(f'{label} & ${egap_direct:.2f}$ & ${egap_indirect:.2f}$')

        for i,band in enumerate(en_vs_k):
            label1 = label if i == 0 else None
            if skip is None:
                ax.plot(kpath, band, **lstyls, label=label1)
            else:
                ax.plot(kpath[::skip], band[::skip], **lstyls, label=label1)

    nrow = 1
    ncol = len(methods) - 1

    figsize = (4.5, 2.5)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharey=True, sharex=True)

    for imethod,method in enumerate(methods):
        if imethod == 0: continue
        ax = axs[imethod-1]

        fbands = f'{pp}/{fml}_{zeta}_{kmesh}_{method}.chk'
        plot1(fbands, lstyls[imethod], labels[imethod])

    for icol in range(ncol):
        ax = axs[icol]

        fbands = f'{pp}/{fml}_{zeta}_{kmesh}_{methods[0]}.chk'
        plot1(fbands, lstyls[0], labels[0], skip=2)

        kpath = lib.chkfile.load(fbands, 'kpath')
        latvec = lib.chkfile.load(fbands, 'latvec')
        points = lib.chkfile.load(fbands, 'points')
        sp_points = lib.chkfile.load(fbands, 'sp_points')
        sp_points_name = lib.chkfile.load(fbands, 'sp_points_name').decode('ascii')

        ax.axhline(0, ls='--', lw=0.7, color='k', zorder=zorder0)
        for y in sp_points:
            ax.axvline(y, ls='-', lw=0.7, color='k', zorder=zorder0)

        ax.set_xticks(sp_points)
        ax.set_xticklabels([r'$\Gamma$' if x == 'G' else r'$%s$'%(x) for x in sp_points_name])

        ax.set_xlim((kpath[0], kpath[-1]))

        if icol == 0:
            ax.set_ylabel('Band energy (eV)')
        ax.set_yticks(range(-30,51,10))
        ax.set_ylim([emin, emax])

    handles, labels = axs[0].get_legend_handles_labels()
    handles = []
    labels = []
    for icol in range(ncol):
        handles1, labels1 = axs[icol].get_legend_handles_labels()
        if icol == 0:
            handles.append(handles1[1])
            labels.append(labels1[1])
        handles.append(handles1[0])
        labels.append(labels1[0])
    legend = fig.legend(handles, labels,
                        loc='lower center', bbox_to_anchor=(0.55,1.0), ncol=2,
                        # loc='center left', bbox_to_anchor=(1.0,0.5), ncol=1,
                        # fancybox=False, frameon=True, edgecolor='k', framealpha=1.,
                        fontsize='medium')
                        # )
    legend.get_frame().set_linewidth(0.7)

    plt.savefig(fout)
    plt.close(fig)
